"""Step 01: load and validate an RTS Framework dataset manifest.

This module is the pipeline entry point.  It loads a CSV manifest, validates
its metadata structure, inspects every referenced image, and returns one
immutable result object.  No RTS detection or image statistics are performed.
"""

from __future__ import annotations

__version__ = "1.4.0"

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Callable, Literal, Sequence

from common.image_io import ImageIOError, get_image_shape, validate_image
from common.manifest import FrameManifest, FrameRecord, ManifestError, ManifestValidation


ValidationMode = Literal["shape", "full"]
ProgressCallback = Callable[[int, int, FrameRecord], None]


class Step01Error(Exception):
    """Base exception raised when Step 01 cannot prepare a dataset."""


@dataclass(slots=True, frozen=True)
class ImageValidationIssue:
    """One image-file problem discovered during Step 01."""

    manifest_row: int
    dataset: str
    filepath: Path
    detail: str

    def format(self) -> str:
        return (
            f"row={self.manifest_row}, dataset={self.dataset!r}, "
            f"path={self.filepath}: {self.detail}"
        )


@dataclass(slots=True, frozen=True)
class Step01Result:
    """Immutable result returned by :func:`prepare_dataset`."""

    manifest: FrameManifest
    manifest_validation: ManifestValidation
    image_issues: tuple[ImageValidationIssue, ...]
    validation_mode: ValidationMode

    @property
    def valid(self) -> bool:
        """Return ``True`` when metadata and all referenced images are valid."""
        return self.manifest_validation.valid and not self.image_issues

    @property
    def n_checked_images(self) -> int:
        return self.manifest.n_frames

    def summary(self, *, max_issues: int | None = 20) -> str:
        """Return a human-readable Step 01 report."""
        lines = [
            "RTS Framework Step 01",
            "=====================",
            f"Status          : {'PASSED' if self.valid else 'FAILED'}",
            f"Validation mode : {self.validation_mode}",
            f"Frames checked  : {self.n_checked_images}",
            f"Datasets        : {self.manifest.n_datasets}",
            f"Manifest errors : {self.manifest_validation.error_count}",
            f"Manifest warns  : {self.manifest_validation.warning_count}",
            f"Image errors    : {len(self.image_issues)}",
        ]

        all_issue_lines = [
            issue.format() for issue in self.manifest_validation.issues
        ] + [issue.format() for issue in self.image_issues]

        if all_issue_lines:
            shown = all_issue_lines if max_issues is None else all_issue_lines[:max_issues]
            lines.extend(["", "Issues", "------", *shown])
            omitted = len(all_issue_lines) - len(shown)
            if omitted > 0:
                lines.append(f"... {omitted} additional issue(s) omitted")

        return "\n".join(lines)




NORMALIZED_MANIFEST_COLUMNS = (
    "dataset",
    "directory",
    "environment",
    "frame_index",
    "n_frames",
    "temperature_C",
    "temperature_start_C",
    "temperature_end_C",
    "temperature_fraction",
    "exposure_s",
    "filename",
    "filepath",
    "image_width",
    "image_height",
    "pixel_dtype",
    "byte_order",
)


def write_normalized_manifest(
    result: Step01Result,
    output_path: str | Path,
) -> Path:
    """Write a validated manifest in the canonical RTS CSV representation.

    The output has a fixed column order, absolute frame paths, LF line endings,
    and deterministic row ordering that preserves the source manifest order.
    A normalized manifest is written only when the complete Step 01 result is
    valid.

    Parameters
    ----------
    result
        Result returned by :func:`prepare_dataset`.
    output_path
        Destination CSV path. Parent directories are created automatically.

    Returns
    -------
    Path
        The destination path.

    Raises
    ------
    Step01Error
        If validation failed or the destination cannot be written.
    """
    if not result.valid:
        raise Step01Error(
            "Cannot write normalized manifest because Step 01 validation failed."
        )

    path = Path(output_path)
    ordered_frames = sorted(
        result.manifest.frames,
        key=lambda frame: frame.manifest_row,
    )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=NORMALIZED_MANIFEST_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            for frame in ordered_frames:
                absolute_path = frame.filepath.resolve()
                writer.writerow(
                    {
                        "dataset": frame.dataset,
                        "directory": str(absolute_path.parent),
                        "environment": frame.environment,
                        "frame_index": frame.frame_index,
                        "n_frames": frame.n_frames,
                        "temperature_C": frame.temperature_C,
                        "temperature_start_C": frame.temperature_start_C,
                        "temperature_end_C": frame.temperature_end_C,
                        "temperature_fraction": frame.temperature_fraction,
                        "exposure_s": frame.exposure_s,
                        "filename": absolute_path.name,
                        "filepath": str(absolute_path),
                        "image_width": frame.image_width,
                        "image_height": frame.image_height,
                        "pixel_dtype": frame.pixel_dtype,
                        "byte_order": frame.byte_order,
                    }
                )
    except (OSError, csv.Error, TypeError, ValueError) as exc:
        raise Step01Error(
            f"Unable to write normalized manifest: {path}: {exc}"
        ) from exc

    return path

def write_report(
    result: Step01Result,
    output_path: str | Path,
    *,
    indent: int | None = 2,
) -> Path:
    """Write a machine-readable Step 01 report as UTF-8 JSON.

    The report intentionally omits a generation timestamp so that identical
    validation results produce stable, reproducible JSON content.

    Parameters
    ----------
    result
        Result returned by :func:`prepare_dataset`.
    output_path
        Destination JSON path. Parent directories are created automatically.
    indent
        JSON indentation. Use ``None`` for compact output.

    Returns
    -------
    Path
        The destination path.

    Raises
    ------
    Step01Error
        If the destination cannot be written.
    """
    path = Path(output_path)
    payload = _build_report_payload(result)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=indent,
                sort_keys=True,
            )
            stream.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise Step01Error(f"Unable to write Step 01 report: {path}: {exc}") from exc

    return path


def _build_report_payload(result: Step01Result) -> dict[str, object]:
    """Convert one Step 01 result into the stable JSON report schema."""
    manifest_source = result.manifest.source_path
    frame_root = result.manifest.frame_root

    return {
        "schema_version": "1.0",
        "step": "step01_prepare_dataset",
        "step_version": __version__,
        "status": "passed" if result.valid else "failed",
        "validation_mode": result.validation_mode,
        "manifest": {
            "source_path": (
                str(manifest_source) if manifest_source is not None else None
            ),
            "frame_root": str(frame_root) if frame_root is not None else None,
            "frame_count": result.manifest.n_frames,
            "dataset_count": result.manifest.n_datasets,
        },
        "counts": {
            "frames_checked": result.n_checked_images,
            "manifest_errors": result.manifest_validation.error_count,
            "manifest_warnings": result.manifest_validation.warning_count,
            "image_errors": len(result.image_issues),
        },
        "manifest_issues": [
            {
                "severity": issue.severity,
                "issue_type": issue.issue_type,
                "detail": issue.detail,
                "manifest_row": issue.manifest_row,
                "dataset": issue.dataset,
            }
            for issue in result.manifest_validation.issues
        ],
        "image_issues": [
            {
                "manifest_row": issue.manifest_row,
                "dataset": issue.dataset,
                "filepath": str(issue.filepath),
                "detail": issue.detail,
            }
            for issue in result.image_issues
        ],
    }

def prepare_dataset(
    manifest_path: str | Path,
    *,
    frame_root: str | Path | None = None,
    validation_mode: ValidationMode = "shape",
    progress: ProgressCallback | None = None,
) -> Step01Result:
    """Load and validate one dataset manifest.

    Parameters
    ----------
    manifest_path
        CSV manifest to load.
    frame_root
        Optional root used to resolve relative ``filepath`` values.
    validation_mode
        ``"shape"`` inspects FITS dimensions without loading pixel arrays.
        ``"full"`` reads each image and also validates dtype metadata.
    progress
        Optional callback receiving ``(current, total, frame)`` before each
        image is checked.

    Returns
    -------
    Step01Result
        The loaded manifest and all validation findings.

    Raises
    ------
    Step01Error
        If the CSV cannot be parsed or ``validation_mode`` is unsupported.
    """
    if validation_mode not in ("shape", "full"):
        raise Step01Error(
            f"Unsupported validation mode: {validation_mode!r}. "
            "Expected 'shape' or 'full'."
        )

    try:
        manifest = FrameManifest.from_csv(manifest_path, frame_root=frame_root)
    except ManifestError as exc:
        raise Step01Error(f"Unable to load manifest: {exc}") from exc

    manifest_validation = manifest.validate_structure()
    image_issues: list[ImageValidationIssue] = []
    total = manifest.n_frames

    for current, frame in enumerate(manifest.frames, start=1):
        if progress is not None:
            progress(current, total, frame)

        try:
            if validation_mode == "full":
                validate_image(frame)
            else:
                _validate_shape(frame)
        except (ImageIOError, NotImplementedError) as exc:
            image_issues.append(
                ImageValidationIssue(
                    manifest_row=frame.manifest_row,
                    dataset=frame.dataset,
                    filepath=frame.filepath,
                    detail=str(exc),
                )
            )

    return Step01Result(
        manifest=manifest,
        manifest_validation=manifest_validation,
        image_issues=tuple(image_issues),
        validation_mode=validation_mode,
    )


def _validate_shape(frame: FrameRecord) -> None:
    """Validate readability, dimensionality, and optional geometry metadata."""
    actual_shape = get_image_shape(frame)

    if frame.image_width is None and frame.image_height is None:
        return

    if frame.image_width is None or frame.image_height is None:
        raise ImageIOError(
            "FrameRecord image geometry is incomplete: "
            f"path={frame.filepath}, image_width={frame.image_width}, "
            f"image_height={frame.image_height}"
        )

    expected_shape = (frame.image_height, frame.image_width)
    if actual_shape != expected_shape:
        raise ImageIOError(
            "Image shape does not match FrameRecord metadata: "
            f"path={frame.filepath}, expected={expected_shape}, "
            f"actual={actual_shape}"
        )

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the Step 01 command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m steps.step01_prepare_dataset",
        description=(
            "Load an RTS Framework manifest and validate all referenced images."
        ),
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="CSV manifest to validate.",
    )
    parser.add_argument(
        "--frame-root",
        type=Path,
        default=None,
        help="Root directory used to resolve relative frame paths.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=("shape", "full"),
        default="shape",
        help=(
            "Image validation level: shape reads metadata only; "
            "full also reads pixel arrays and checks dtype."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional destination for the machine-readable JSON report.",
    )
    parser.add_argument(
        "--normalized-manifest",
        type=Path,
        default=None,
        help=(
            "Optional destination for a canonical CSV manifest. "
            "Written only when validation passes."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-frame progress messages.",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=20,
        help=(
            "Maximum issues printed in the text summary. "
            "Use -1 to print all issues."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Step 01 from the command line.

    Exit codes
    ----------
    0
        Manifest and all images passed validation.
    1
        Validation completed but one or more issues were found.
    2
        The command could not run, such as an unreadable manifest or report
        write failure. ``argparse`` also uses exit code 2 for invalid options.
    """
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.max_issues < -1:
        parser.error("--max-issues must be -1 or zero or greater")

    progress: ProgressCallback | None = None
    if not args.quiet:
        def show_progress(
            current: int,
            total: int,
            frame: FrameRecord,
        ) -> None:
            print(
                f"Checking image {current}/{total}: "
                f"{frame.dataset} / {frame.filename}"
            )

        progress = show_progress

    try:
        result = prepare_dataset(
            args.manifest,
            frame_root=args.frame_root,
            validation_mode=args.validation_mode,
            progress=progress,
        )

        if args.report is not None:
            report_path = write_report(result, args.report)
            print(f"JSON report: {report_path}")

        if args.normalized_manifest is not None:
            if result.valid:
                normalized_path = write_normalized_manifest(
                    result,
                    args.normalized_manifest,
                )
                print(f"Normalized manifest: {normalized_path}")
            else:
                print(
                    "Normalized manifest not written: validation failed.",
                    file=sys.stderr,
                )

    except Step01Error as exc:
        print(f"Step 01 error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print()

    max_issues = None if args.max_issues == -1 else args.max_issues
    print(result.summary(max_issues=max_issues))
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

