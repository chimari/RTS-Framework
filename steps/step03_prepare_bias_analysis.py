"""Step 03: prepare one dataset for bias and RTS analysis.

Version 3.3.0 adds an exact full-stack median master-bias reference
implementation. It reads each frame once, retains the complete float64
image stack, and returns an immutable read-only median master image.
"""

from __future__ import annotations

__version__ = "3.3.0"

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator

import numpy as np

from common.manifest import FrameManifest
from steps.step02_prepare_frame_groups import (
    DatasetGroup,
    Step02Error,
    Step02Result,
    iter_dataset_images,
    prepare_frame_groups,
)


__all__ = [
    "BiasAnalysisPlan",
    "MeanMasterBiasResult",
    "MedianMasterBiasResult",
    "Step03Error",
    "compute_mean_master_bias",
    "compute_median_master_bias",
    "iter_bias_frames",
    "prepare_bias_analysis",
]


class Step03Error(Exception):
    """Raised when Step 03 cannot prepare a bias-analysis dataset."""


@dataclass(slots=True, frozen=True)
class MeanMasterBiasResult:
    """Immutable arithmetic-mean master-bias result."""

    plan: "BiasAnalysisPlan"
    dataset: str
    n_frames: int
    image_shape: tuple[int, int]
    master_bias: np.ndarray
    minimum: float
    maximum: float
    mean: float



@dataclass(slots=True, frozen=True)
class MedianMasterBiasResult:
    """Immutable exact median master-bias result."""

    plan: "BiasAnalysisPlan"
    dataset: str
    n_frames: int
    image_shape: tuple[int, int]
    master_bias: np.ndarray
    minimum: float
    maximum: float
    median: float



@dataclass(slots=True, frozen=True)
class BiasAnalysisPlan:
    """Immutable metadata plan for one selected bias-analysis dataset."""

    group: DatasetGroup
    dataset: str
    n_frames: int
    image_shape: tuple[int, int]
    pixel_dtype: str
    exposure_s: float
    temperature_min_C: float
    temperature_max_C: float

    @property
    def filepaths(self) -> tuple[Path, ...]:
        """Return frame paths in canonical Step 02 frame order."""
        return self.group.filepaths

    def summary(self) -> str:
        """Return a deterministic human-readable plan summary."""
        height, width = self.image_shape
        return "\n".join(
            [
                "RTS Framework Step 03",
                "=====================",
                "Status      : READY",
                f"Dataset     : {self.dataset}",
                f"Frames      : {self.n_frames}",
                f"Shape       : {height}x{width}",
                f"Pixel dtype : {self.pixel_dtype}",
                f"Exposure    : {self.exposure_s:g} s",
                (
                    "Temperature : "
                    f"{self.temperature_min_C:g}.."
                    f"{self.temperature_max_C:g} C"
                ),
            ]
        )


Step03Source = str | Path | FrameManifest | Step02Result


def prepare_bias_analysis(
    source: Step03Source,
    dataset: str,
    *,
    frame_root: str | Path | None = None,
    min_frames: int = 2,
) -> BiasAnalysisPlan:
    """Select and validate one Step 02 dataset for later bias analysis.

    This milestone reads only manifest metadata. Image arrays are deliberately
    not opened until a later Step 03 milestone.

    Parameters
    ----------
    source
        A normalized manifest path, loaded ``FrameManifest``, or completed
        ``Step02Result``.
    dataset
        Exact dataset name to select.
    frame_root
        Optional relative-path root when ``source`` is a manifest path.
        It cannot be supplied with an existing ``Step02Result``.
    min_frames
        Minimum number of frames required. Must be at least two.

    Raises
    ------
    Step03Error
        If Step 02 preparation fails, the dataset is absent, or its metadata is
        unsuitable for bias analysis.
    """
    dataset = _validate_dataset_name(dataset)
    min_frames = _validate_min_frames(min_frames)
    result = _resolve_step02_result(source, frame_root=frame_root)

    try:
        group = result.get_group(dataset)
    except KeyError as exc:
        available = ", ".join(group.name for group in result.groups)
        if not available:
            available = "(none)"
        raise Step03Error(
            f"Dataset {dataset!r} was not found. Available datasets: {available}."
        ) from exc

    _validate_group(group, min_frames=min_frames)

    return BiasAnalysisPlan(
        group=group,
        dataset=group.name,
        n_frames=group.n_frames,
        image_shape=group.image_shape,
        pixel_dtype=group.pixel_dtype,
        exposure_s=group.exposure_s,
        temperature_min_C=group.temperature_min_C,
        temperature_max_C=group.temperature_max_C,
    )


def compute_median_master_bias(
    plan: BiasAnalysisPlan,
) -> MedianMasterBiasResult:
    """Compute an exact median master bias using a full in-memory stack.

    Images are obtained exclusively from :func:`iter_bias_frames` and are read
    exactly once. Unlike :func:`compute_mean_master_bias`, this reference
    implementation retains every float64 image simultaneously so that
    ``numpy.median(..., axis=0)`` can compute the exact per-pixel median.

    Approximate stack memory usage is::

        n_frames * image_height * image_width * 8 bytes

    This implementation is intended as a correctness reference. Large-format
    detectors or large frame counts may require a later tiled implementation.

    Parameters
    ----------
    plan
        Bias-analysis plan returned by :func:`prepare_bias_analysis`.

    Returns
    -------
    MedianMasterBiasResult
        Immutable result containing a read-only exact median master image.

    Raises
    ------
    Step03Error
        If ``plan`` is invalid, no frames are yielded, the iterator yields an
        unexpected frame count, or an image violates the planned shape.
    """
    if not isinstance(plan, BiasAnalysisPlan):
        raise Step03Error(
            "plan must be a BiasAnalysisPlan returned by "
            "prepare_bias_analysis()."
        )

    stack = np.empty(
        (plan.n_frames, *plan.image_shape),
        dtype=np.float64,
        order="C",
    )
    count = 0

    for image in iter_bias_frames(plan):
        if count >= plan.n_frames:
            raise Step03Error(
                f"Bias dataset {plan.dataset!r} yielded more than "
                f"{plan.n_frames} frame(s)."
            )
        if image.shape != plan.image_shape:
            raise Step03Error(
                f"Bias frame {count} has shape {image.shape!r}; "
                f"expected {plan.image_shape!r}."
            )
        stack[count] = image
        count += 1

    if count == 0:
        raise Step03Error(
            f"Bias dataset {plan.dataset!r} yielded no frames."
        )
    if count != plan.n_frames:
        raise Step03Error(
            f"Bias dataset {plan.dataset!r} yielded {count} frame(s); "
            f"the analysis plan requires {plan.n_frames}."
        )

    master_bias = np.array(
        np.median(stack, axis=0),
        dtype=np.float64,
        order="C",
        copy=True,
    )
    master_bias.setflags(write=False)

    return MedianMasterBiasResult(
        plan=plan,
        dataset=plan.dataset,
        n_frames=count,
        image_shape=plan.image_shape,
        master_bias=master_bias,
        minimum=float(np.min(master_bias)),
        maximum=float(np.max(master_bias)),
        median=float(np.median(master_bias)),
    )


def compute_mean_master_bias(
    plan: BiasAnalysisPlan,
) -> MeanMasterBiasResult:
    """Compute an arithmetic-mean master bias in one streaming pass.

    Images are obtained exclusively from :func:`iter_bias_frames`. The function
    keeps one float64 accumulation array and the current frame in memory; it
    never materializes the complete image stack.

    Parameters
    ----------
    plan
        Bias-analysis plan returned by :func:`prepare_bias_analysis`.

    Returns
    -------
    MeanMasterBiasResult
        Immutable result containing a read-only master-bias image.

    Raises
    ------
    Step03Error
        If ``plan`` is invalid, no frames are yielded, the iterator yields an
        unexpected frame count, or an image violates the planned shape.
    """
    if not isinstance(plan, BiasAnalysisPlan):
        raise Step03Error(
            "plan must be a BiasAnalysisPlan returned by "
            "prepare_bias_analysis()."
        )

    accumulator = np.zeros(plan.image_shape, dtype=np.float64, order="C")
    count = 0

    for image in iter_bias_frames(plan):
        if image.shape != plan.image_shape:
            raise Step03Error(
                f"Bias frame {count} has shape {image.shape!r}; "
                f"expected {plan.image_shape!r}."
            )
        accumulator += image
        count += 1

    if count == 0:
        raise Step03Error(
            f"Bias dataset {plan.dataset!r} yielded no frames."
        )
    if count != plan.n_frames:
        raise Step03Error(
            f"Bias dataset {plan.dataset!r} yielded {count} frame(s); "
            f"the analysis plan requires {plan.n_frames}."
        )

    accumulator /= float(count)
    accumulator.setflags(write=False)

    return MeanMasterBiasResult(
        plan=plan,
        dataset=plan.dataset,
        n_frames=count,
        image_shape=plan.image_shape,
        master_bias=accumulator,
        minimum=float(np.min(accumulator)),
        maximum=float(np.max(accumulator)),
        mean=float(np.mean(accumulator)),
    )


def iter_bias_frames(
    plan: BiasAnalysisPlan,
) -> Iterator[np.ndarray]:
    """Yield analysis-ready bias images in canonical frame order.

    The underlying files are read lazily through the public Step 02 image
    iterator. Each yielded image is a newly allocated array with these
    guarantees:

    - dtype is exactly ``numpy.float64``;
    - layout is C-contiguous;
    - the array is read-only;
    - no previously yielded image is retained by this function.

    Parameters
    ----------
    plan
        Bias-analysis plan returned by :func:`prepare_bias_analysis`.

    Yields
    ------
    numpy.ndarray
        A two-dimensional analysis-ready image.

    Raises
    ------
    Step03Error
        If ``plan`` is invalid or Step 02 cannot read or revalidate a frame.
    """
    if not isinstance(plan, BiasAnalysisPlan):
        raise Step03Error(
            "plan must be a BiasAnalysisPlan returned by "
            "prepare_bias_analysis()."
        )

    try:
        for _frame, image in iter_dataset_images(plan.group):
            converted = np.array(
                image,
                dtype=np.float64,
                order="C",
                copy=True,
            )
            converted.setflags(write=False)
            yield converted
    except Step02Error as exc:
        raise Step03Error(
            f"Unable to iterate bias dataset {plan.dataset!r}: {exc}"
        ) from exc


def _resolve_step02_result(
    source: Step03Source,
    *,
    frame_root: str | Path | None,
) -> Step02Result:
    if isinstance(source, Step02Result):
        if frame_root is not None:
            raise Step03Error(
                "frame_root cannot be used with an existing Step02Result."
            )
        return source

    try:
        return prepare_frame_groups(source, frame_root=frame_root)
    except Step02Error as exc:
        raise Step03Error(f"Step 02 preparation failed: {exc}") from exc


def _validate_dataset_name(dataset: str) -> str:
    if not isinstance(dataset, str):
        raise Step03Error("dataset must be a string.")
    if not dataset:
        raise Step03Error("dataset must not be empty.")
    if dataset != dataset.strip():
        raise Step03Error(
            "dataset must be an exact name without leading or trailing whitespace."
        )
    return dataset


def _validate_min_frames(min_frames: int) -> int:
    if isinstance(min_frames, bool) or not isinstance(min_frames, int):
        raise Step03Error("min_frames must be an integer.")
    if min_frames < 2:
        raise Step03Error("min_frames must be at least 2.")
    return min_frames


def _validate_group(
    group: DatasetGroup,
    *,
    min_frames: int,
) -> None:
    if group.n_frames < min_frames:
        raise Step03Error(
            f"Dataset {group.name!r} contains {group.n_frames} frame(s); "
            f"at least {min_frames} are required."
        )

    height, width = group.image_shape
    if height <= 0 or width <= 0:
        raise Step03Error(
            f"Dataset {group.name!r} has invalid image shape "
            f"{group.image_shape!r}."
        )

    try:
        dtype = np.dtype(group.pixel_dtype)
    except TypeError as exc:
        raise Step03Error(
            f"Dataset {group.name!r} has invalid pixel dtype "
            f"{group.pixel_dtype!r}."
        ) from exc

    if dtype.kind not in {"b", "i", "u", "f"}:
        raise Step03Error(
            f"Dataset {group.name!r} pixel dtype {group.pixel_dtype!r} "
            "is not a supported real numeric dtype."
        )

    if not math.isfinite(group.exposure_s) or group.exposure_s < 0.0:
        raise Step03Error(
            f"Dataset {group.name!r} has invalid exposure_s="
            f"{group.exposure_s!r}."
        )

    if (
        not math.isfinite(group.temperature_min_C)
        or not math.isfinite(group.temperature_max_C)
        or group.temperature_min_C > group.temperature_max_C
    ):
        raise Step03Error(
            f"Dataset {group.name!r} has invalid temperature range "
            f"{group.temperature_min_C!r}..{group.temperature_max_C!r} C."
        )
