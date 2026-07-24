"""Step 02: group validated manifest frames by dataset.

This module consumes a canonical manifest produced by Step 01 and creates
immutable dataset groups for later image-statistics and RTS-analysis steps.
No image pixels are read in this step.
"""

from __future__ import annotations

__version__ = "2.0.0"

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterator, Sequence

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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m steps.step02_prepare_frame_groups",
        description="Group a validated RTS manifest by dataset.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--frame-root", type=Path, default=None)
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
    except Step02Error as exc:
        print(f"Step 02 error: {exc}", file=sys.stderr)
        return 2

    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
