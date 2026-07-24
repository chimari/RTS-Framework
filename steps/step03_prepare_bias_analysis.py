"""Step 03: prepare one dataset for bias and RTS analysis.

Version 3.0.0 intentionally performs no image-pixel reads.  It selects one
validated Step 02 dataset and freezes the metadata required by later master-bias
and RTS-analysis milestones.
"""

from __future__ import annotations

__version__ = "3.0.0"

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from common.manifest import FrameManifest
from steps.step02_prepare_frame_groups import (
    DatasetGroup,
    Step02Error,
    Step02Result,
    prepare_frame_groups,
)


__all__ = [
    "BiasAnalysisPlan",
    "Step03Error",
    "prepare_bias_analysis",
]


class Step03Error(Exception):
    """Raised when Step 03 cannot prepare a bias-analysis dataset."""


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
