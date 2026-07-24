"""Step 02: group validated manifest frames by dataset.

This module consumes a canonical manifest produced by Step 01 and creates
immutable dataset groups for later image-statistics and RTS-analysis steps.
No image pixels are read in this step.
"""

from __future__ import annotations

__version__ = "2.4.0"

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import os
import re
import tempfile
import sys
from typing import Callable, Iterator, Sequence

import numpy as np

from common.image_io import read_image

from common.manifest import FrameManifest, FrameRecord, ManifestError


class Step02Error(Exception):
    """Base exception raised when Step 02 cannot prepare frame groups."""


@dataclass(slots=True, frozen=True)
class DatasetGroup:
    """One immutable, internally consistent group of dataset frames."""

    name: str
    frames: tuple[FrameRecord, ...]
    environment: str
    image_shape: tuple[int, int]
    pixel_dtype: str
    exposure_s: float
    temperature_min_C: float
    temperature_max_C: float

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def filepaths(self) -> tuple[Path, ...]:
        return tuple(frame.filepath for frame in self.frames)

    def summary_line(self) -> str:
        height, width = self.image_shape
        return (
            f"{self.name}: frames={self.n_frames}, "
            f"shape={height}x{width}, dtype={self.pixel_dtype}, "
            f"exposure={self.exposure_s:g}s, "
            f"temperature={self.temperature_min_C:g}"
            f"..{self.temperature_max_C:g}C"
        )


@dataclass(slots=True, frozen=True)
class Step02Result:
    """Immutable result returned by :func:`prepare_frame_groups`."""

    manifest: FrameManifest
    groups: tuple[DatasetGroup, ...]

    @property
    def n_datasets(self) -> int:
        return len(self.groups)

    @property
    def n_frames(self) -> int:
        return sum(group.n_frames for group in self.groups)

    def __iter__(self) -> Iterator[DatasetGroup]:
        return iter(self.groups)

    def get_group(self, name: str) -> DatasetGroup:
        """Return one dataset group by exact name."""
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(name)

    def summary(self) -> str:
        lines = [
            "RTS Framework Step 02",
            "=====================",
            "Status   : PASSED",
            f"Datasets : {self.n_datasets}",
            f"Frames   : {self.n_frames}",
        ]
        if self.groups:
            lines.extend(
                ["", "Dataset groups", "--------------"],
            )
            lines.extend(group.summary_line() for group in self.groups)
        return "\n".join(lines)


def prepare_frame_groups(
    manifest_source: str | Path | FrameManifest,
    *,
    frame_root: str | Path | None = None,
) -> Step02Result:
    """Load a manifest and group its frames by dataset.

    Parameters
    ----------
    manifest_source
        Canonical CSV manifest path or an already loaded ``FrameManifest``.
    frame_root
        Optional root used only when loading a CSV containing relative paths.

    Returns
    -------
    Step02Result
        Dataset groups sorted by dataset name. Frames in each group are sorted
        by ``frame_index`` and then by original manifest row.

    Raises
    ------
    Step02Error
        If the manifest cannot be loaded or a dataset is internally
        inconsistent.
    """
    manifest = _load_manifest(manifest_source, frame_root=frame_root)

    buckets: dict[str, list[FrameRecord]] = {}
    for frame in manifest.frames:
        buckets.setdefault(frame.dataset, []).append(frame)

    groups = tuple(
        _build_group(
            dataset,
            sorted(
                frames,
                key=lambda frame: (frame.frame_index, frame.manifest_row),
            ),
        )
        for dataset, frames in sorted(buckets.items())
    )

    return Step02Result(manifest=manifest, groups=groups)


def _load_manifest(
    source: str | Path | FrameManifest,
    *,
    frame_root: str | Path | None,
) -> FrameManifest:
    if isinstance(source, FrameManifest):
        if frame_root is not None:
            raise Step02Error(
                "frame_root cannot be used with an already loaded FrameManifest."
            )
        return source

    try:
        return FrameManifest.from_csv(source, frame_root=frame_root)
    except (ManifestError, OSError, ValueError, TypeError) as exc:
        raise Step02Error(f"Unable to load Step 02 manifest: {source}: {exc}") from exc


def _build_group(
    dataset: str,
    frames: list[FrameRecord],
) -> DatasetGroup:
    if not frames:
        raise Step02Error(f"Dataset {dataset!r} contains no frames.")

    _require_unique_frame_indices(dataset, frames)
    _require_declared_frame_count(dataset, frames)

    first = frames[0]
    environment = _require_single_value(
        dataset,
        frames,
        "environment",
    )
    image_width = _require_single_value(
        dataset,
        frames,
        "image_width",
    )
    image_height = _require_single_value(
        dataset,
        frames,
        "image_height",
    )
    pixel_dtype = _require_single_value(
        dataset,
        frames,
        "pixel_dtype",
    )
    exposure_s = _require_single_value(
        dataset,
        frames,
        "exposure_s",
    )

    temperatures = tuple(frame.temperature_C for frame in frames)

    return DatasetGroup(
        name=dataset,
        frames=tuple(frames),
        environment=environment,
        image_shape=(image_height, image_width),
        pixel_dtype=pixel_dtype,
        exposure_s=exposure_s,
        temperature_min_C=min(temperatures),
        temperature_max_C=max(temperatures),
    )


def _require_unique_frame_indices(
    dataset: str,
    frames: list[FrameRecord],
) -> None:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for frame in frames:
        if frame.frame_index in seen:
            duplicates.add(frame.frame_index)
        seen.add(frame.frame_index)

    if duplicates:
        values = ", ".join(str(value) for value in sorted(duplicates))
        raise Step02Error(
            f"Dataset {dataset!r} has duplicate frame_index value(s): {values}."
        )


def _require_declared_frame_count(
    dataset: str,
    frames: list[FrameRecord],
) -> None:
    declared = {frame.n_frames for frame in frames}
    if len(declared) != 1:
        values = ", ".join(str(value) for value in sorted(declared))
        raise Step02Error(
            f"Dataset {dataset!r} has inconsistent n_frames values: {values}."
        )

    expected = next(iter(declared))
    actual = len(frames)
    if expected != actual:
        raise Step02Error(
            f"Dataset {dataset!r} declares n_frames={expected}, "
            f"but contains {actual} manifest row(s)."
        )

    expected_indices = set(range(expected))
    actual_indices = {frame.frame_index for frame in frames}
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        extra = sorted(actual_indices - expected_indices)
        raise Step02Error(
            f"Dataset {dataset!r} frame_index sequence is incomplete: "
            f"missing={missing}, extra={extra}."
        )


def _require_single_value(
    dataset: str,
    frames: list[FrameRecord],
    attribute: str,
):
    values = {getattr(frame, attribute) for frame in frames}
    if len(values) != 1:
        rendered = ", ".join(repr(value) for value in sorted(values, key=repr))
        raise Step02Error(
            f"Dataset {dataset!r} has inconsistent {attribute}: {rendered}."
        )
    return next(iter(values))




@dataclass(slots=True, frozen=True)
class FrameStatistics:
    """Basic statistics for one image frame."""

    dataset: str
    frame_index: int
    filepath: Path
    temperature_C: float
    exposure_s: float
    finite_pixels: int
    total_pixels: int
    minimum: float
    maximum: float
    mean: float
    median: float
    stddev: float


@dataclass(slots=True, frozen=True)
class DatasetStatistics:
    """Collection of per-frame statistics for one dataset."""

    dataset: str
    frames: tuple[FrameStatistics, ...]

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def frame_mean_min(self) -> float:
        return min(frame.mean for frame in self.frames)

    @property
    def frame_mean_max(self) -> float:
        return max(frame.mean for frame in self.frames)

    @property
    def frame_median_min(self) -> float:
        return min(frame.median for frame in self.frames)

    @property
    def frame_median_max(self) -> float:
        return max(frame.median for frame in self.frames)

    def summary_line(self) -> str:
        return (
            f"{self.dataset}: frames={self.n_frames}, "
            f"mean={self.frame_mean_min:g}..{self.frame_mean_max:g}, "
            f"median={self.frame_median_min:g}..{self.frame_median_max:g}"
        )

ImageProgressCallback = Callable[[int, int, FrameRecord], None]


def iter_dataset_images(
    group: DatasetGroup,
    *,
    progress: ImageProgressCallback | None = None,
) -> Iterator[tuple[FrameRecord, np.ndarray]]:
    """Yield one image at a time for a dataset group.

    Images are loaded lazily and are not retained by this function.  The caller
    controls their lifetime by consuming each yielded array.

    Parameters
    ----------
    group
        Dataset group returned by :func:`prepare_frame_groups`.
    progress
        Optional callback called immediately before each image read as
        ``progress(current, total, frame)``. ``current`` is one-based.

    Yields
    ------
    tuple[FrameRecord, numpy.ndarray]
        The frame record and its two-dimensional image array.

    Raises
    ------
    Step02Error
        If an image cannot be read or if its shape or dtype no longer matches
        the validated dataset metadata.
    """
    total = group.n_frames

    for current, frame in enumerate(group.frames, start=1):
        if progress is not None:
            progress(current, total, frame)

        try:
            image = read_image(frame.filepath)
        except Exception as exc:
            raise Step02Error(
                "Unable to read dataset image: "
                f"dataset={group.name!r}, "
                f"frame_index={frame.frame_index}, "
                f"filepath={frame.filepath}: {exc}"
            ) from exc

        _validate_loaded_image(group, frame, image)
        yield frame, image



def compute_dataset_statistics(
    group: DatasetGroup,
    *,
    progress: ImageProgressCallback | None = None,
) -> DatasetStatistics:
    """Compute basic statistics for each frame in a dataset.

    Frames are read lazily using :func:`iter_dataset_images`. Only the current
    image and the compact statistics records are retained.

    Non-finite floating-point pixels are excluded from all statistics. Integer
    images are treated as fully finite.

    Raises
    ------
    Step02Error
        If a frame cannot be read, its metadata no longer matches, or it
        contains no finite pixels.
    """
    records: list[FrameStatistics] = []

    for frame, image in iter_dataset_images(group, progress=progress):
        total_pixels = int(image.size)

        if np.issubdtype(image.dtype, np.inexact):
            finite_mask = np.isfinite(image)
            finite_pixels = int(np.count_nonzero(finite_mask))
            if finite_pixels == 0:
                raise Step02Error(
                    "Image contains no finite pixels: "
                    f"dataset={group.name!r}, "
                    f"frame_index={frame.frame_index}, "
                    f"filepath={frame.filepath}."
                )
            values = image[finite_mask]
        else:
            finite_pixels = total_pixels
            values = image

        records.append(
            FrameStatistics(
                dataset=group.name,
                frame_index=frame.frame_index,
                filepath=frame.filepath,
                temperature_C=frame.temperature_C,
                exposure_s=frame.exposure_s,
                finite_pixels=finite_pixels,
                total_pixels=total_pixels,
                minimum=float(np.min(values)),
                maximum=float(np.max(values)),
                mean=float(np.mean(values, dtype=np.float64)),
                median=float(np.median(values)),
                stddev=float(np.std(values, dtype=np.float64)),
            )
        )

    if not records:
        raise Step02Error(f"Dataset {group.name!r} contains no frames.")

    return DatasetStatistics(
        dataset=group.name,
        frames=tuple(records),
    )

def _validate_loaded_image(
    group: DatasetGroup,
    frame: FrameRecord,
    image: np.ndarray,
) -> None:
    if image.ndim != 2:
        raise Step02Error(
            "Loaded image is not two-dimensional: "
            f"dataset={group.name!r}, "
            f"frame_index={frame.frame_index}, "
            f"filepath={frame.filepath}, "
            f"ndim={image.ndim}."
        )

    actual_shape = tuple(int(value) for value in image.shape)
    if actual_shape != group.image_shape:
        raise Step02Error(
            "Loaded image shape does not match dataset metadata: "
            f"dataset={group.name!r}, "
            f"frame_index={frame.frame_index}, "
            f"filepath={frame.filepath}, "
            f"expected={group.image_shape}, "
            f"actual={actual_shape}."
        )

    expected_dtype = np.dtype(group.pixel_dtype)
    actual_dtype = image.dtype
    if not _dtype_equivalent(actual_dtype, expected_dtype):
        raise Step02Error(
            "Loaded image dtype does not match dataset metadata: "
            f"dataset={group.name!r}, "
            f"frame_index={frame.frame_index}, "
            f"filepath={frame.filepath}, "
            f"expected={expected_dtype}, "
            f"actual={actual_dtype}."
        )


def _dtype_equivalent(
    actual: np.dtype,
    expected: np.dtype,
) -> bool:
    """Return True when dtype kind and item size match.

    FITS commonly stores multi-byte numeric values in big-endian order, so
    Astropy may return ``>f4`` or ``>i2`` even when the canonical manifest uses
    ``float32`` or ``int16``. Byte order alone does not change the numeric dtype
    expected by later pipeline stages.
    """
    actual = np.dtype(actual)
    expected = np.dtype(expected)
    return (
        actual.kind == expected.kind
        and actual.itemsize == expected.itemsize
    )


STATISTICS_CSV_COLUMNS = (
    "dataset",
    "frame_index",
    "filepath",
    "temperature_C",
    "exposure_s",
    "finite_pixels",
    "total_pixels",
    "minimum",
    "maximum",
    "mean",
    "median",
    "stddev",
)


def write_statistics_csv(
    statistics: DatasetStatistics,
    output_path: str | Path,
) -> Path:
    """Write per-frame dataset statistics as deterministic CSV.

    The output uses a fixed column order, UTF-8 encoding, LF line endings,
    and frame-index ordering. Parent directories are created automatically.
    """
    path = Path(output_path)
    ordered_frames = sorted(
        statistics.frames,
        key=lambda frame: (frame.frame_index, str(frame.filepath)),
    )

    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(
                stream,
                fieldnames=STATISTICS_CSV_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            for frame in ordered_frames:
                writer.writerow(
                    {
                        "dataset": frame.dataset,
                        "frame_index": frame.frame_index,
                        "filepath": str(frame.filepath.resolve()),
                        "temperature_C": _format_float(frame.temperature_C),
                        "exposure_s": _format_float(frame.exposure_s),
                        "finite_pixels": frame.finite_pixels,
                        "total_pixels": frame.total_pixels,
                        "minimum": _format_float(frame.minimum),
                        "maximum": _format_float(frame.maximum),
                        "mean": _format_float(frame.mean),
                        "median": _format_float(frame.median),
                        "stddev": _format_float(frame.stddev),
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())

        temporary_path.replace(path)
        temporary_path = None
    except (OSError, csv.Error, TypeError, ValueError) as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise Step02Error(
            f"Unable to write Step 02 statistics CSV: {path}: {exc}"
        ) from exc

    return path


def _format_float(value: float) -> str:
    """Return a stable round-trippable decimal representation."""
    return format(float(value), ".17g")

def statistics_filename(dataset: str) -> str:
    """Return a deterministic filesystem-safe CSV filename for a dataset."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset.strip())
    stem = stem.strip("._-")
    if not stem:
        stem = "dataset"
    return f"{stem}.csv"


def write_all_statistics_csv(
    result: Step02Result,
    output_dir: str | Path,
    *,
    quiet: bool = False,
) -> tuple[Path, ...]:
    """Compute and write one statistics CSV per dataset.

    All image statistics are computed before any output file is written. This
    prevents read or validation failures from leaving a partially generated
    output set.
    """
    directory = Path(output_dir)
    statistics_results: list[DatasetStatistics] = []
    filenames: dict[str, str] = {}

    for group_index, group in enumerate(result.groups, start=1):
        filename = statistics_filename(group.name)
        previous = filenames.get(filename)
        if previous is not None and previous != group.name:
            raise Step02Error(
                "Dataset names map to the same statistics filename: "
                f"{previous!r} and {group.name!r} -> {filename!r}."
            )
        filenames[filename] = group.name

        if not quiet:
            print(
                f"[{group_index}/{result.n_datasets}] "
                f"{group.name}: {group.n_frames} frames"
            )

        def progress(
            current: int,
            total: int,
            frame: FrameRecord,
        ) -> None:
            if not quiet:
                print(
                    f"  frame {current}/{total}: {frame.filepath.name}",
                    flush=True,
                )

        statistics_results.append(
            compute_dataset_statistics(
                group,
                progress=progress,
            )
        )

    written: list[Path] = []
    for statistics in statistics_results:
        path = directory / statistics_filename(statistics.dataset)
        write_statistics_csv(statistics, path)
        written.append(path)
        if not quiet:
            print(f"  wrote: {path}")

    return tuple(written)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m steps.step02_prepare_frame_groups",
        description="Group a validated RTS manifest by dataset.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--frame-root", type=Path, default=None)
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        default=None,
        help="Compute per-frame statistics and write one CSV per dataset.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress normal summary and progress output.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = prepare_frame_groups(
            args.manifest,
            frame_root=args.frame_root,
        )
        if args.statistics_dir is not None:
            write_all_statistics_csv(
                result,
                args.statistics_dir,
                quiet=args.quiet,
            )
    except Step02Error as exc:
        print(f"Step 02 error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
