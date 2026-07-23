#!/usr/bin/env python3
"""
Smoke test and usage example for ``common.manifest``.

This script demonstrates how to:

- load a manifest CSV;
- inspect the manifest and its datasets;
- run structural validation;
- inspect a small number of FrameRecord objects.

It does not open image files. Image existence, shape, dtype, and pixel data
will be tested separately by ``test_image_io.py``.

Examples
--------
Run from the RTS-Framework repository root::

    python tests/test_manifest.py frame_index.csv

Resolve relative image paths below a data directory::

    python tests/test_manifest.py frame_index.csv \
        --frame-root /data/IMX455

Show ten records and all validation issues::

    python tests/test_manifest.py frame_index.csv \
        --show-frames 10 \
        --all-issues
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


# Allow direct execution as ``python tests/test_manifest.py ...``.
# When installed as a package, the ordinary import path is used unchanged.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import manifest as manifest_module  # noqa: E402
from common.manifest import FrameManifest, ManifestError  # noqa: E402


manifest_version = getattr(manifest_module, "__version__", "(not defined)")


SEPARATOR = "=" * 72


def build_argument_parser() -> argparse.ArgumentParser:
    """Create and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Load an RTS Framework manifest CSV, print its summary, "
            "and run structural validation."
        )
    )

    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the frame-manifest CSV file.",
    )
    parser.add_argument(
        "--frame-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory prepended to relative paths in the "
            "CSV filepath column."
        ),
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV text encoding (default: utf-8-sig).",
    )
    parser.add_argument(
        "--show-frames",
        type=non_negative_int,
        default=5,
        metavar="N",
        help="Number of leading FrameRecord objects to display (default: 5).",
    )
    parser.add_argument(
        "--show-datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a summary for every dataset (default: enabled).",
    )
    parser.add_argument(
        "--all-issues",
        action="store_true",
        help="Show every validation issue instead of only the first 20.",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Return a non-zero exit status when validation has warnings.",
    )

    return parser


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {parsed}"
        )

    return parsed


def print_header(csv_path: Path, frame_root: Path | None) -> None:
    """Print test metadata."""
    print(SEPARATOR)
    print("RTS Framework Manifest Test")
    print(SEPARATOR)
    print(f"manifest.py version : {manifest_version}")
    print(f"CSV                 : {csv_path}")
    print(f"Frame root          : {frame_root if frame_root is not None else '(none)'}")


def load_manifest(
    csv_path: Path,
    *,
    frame_root: Path | None,
    encoding: str,
) -> FrameManifest:
    """Load one CSV and return its FrameManifest."""
    print("\n[1/5] Loading CSV")

    manifest = FrameManifest.from_csv(
        csv_path,
        frame_root=frame_root,
        encoding=encoding,
    )

    print(f"PASS: loaded {manifest.n_frames} frame(s)")
    return manifest


def show_manifest_summary(manifest: FrameManifest) -> None:
    """Print the human-readable manifest summary."""
    print("\n[2/5] Manifest summary")
    print(manifest.summary())


def show_validation(
    manifest: FrameManifest,
    *,
    all_issues: bool,
) -> tuple[bool, int]:
    """
    Print structural validation and return ``(valid, warning_count)``.
    """
    print("\n[3/5] Structural validation")

    validation = manifest.validate_structure()
    max_issues = None if all_issues else 20
    print(validation.summary(max_issues=max_issues))

    return validation.valid, validation.warning_count


def show_dataset_summaries(
    manifest: FrameManifest,
    *,
    enabled: bool,
) -> None:
    """Print summaries for all datasets when requested."""
    print("\n[4/5] Dataset inspection")

    if not enabled:
        print("SKIPPED: dataset summaries disabled")
        return

    if manifest.is_empty:
        print("SKIPPED: manifest contains no datasets")
        return

    for index, dataset in enumerate(manifest.datasets, start=1):
        if index > 1:
            print()
        print(dataset.summary())


def show_frame_records(
    manifest: FrameManifest,
    *,
    count: int,
) -> None:
    """Print a compact table of the first ``count`` records."""
    print("\n[5/5] FrameRecord inspection")

    if count == 0:
        print("SKIPPED: --show-frames is 0")
        return

    selected = manifest[:count]
    if not selected:
        print("SKIPPED: manifest contains no frames")
        return

    print(
        f"{'row':>6}  {'dataset':<20}  {'index':>7}  "
        f"{'temp [C]':>10}  {'exposure [s]':>13}  filepath"
    )
    print("-" * 110)

    for frame in selected:
        print(
            f"{frame.manifest_row:6d}  "
            f"{truncate(frame.dataset, 20):<20}  "
            f"{frame.frame_index:7d}  "
            f"{frame.temperature_C:10.3f}  "
            f"{frame.exposure_s:13.9g}  "
            f"{frame.filepath}"
        )

    omitted = manifest.n_frames - len(selected)
    if omitted > 0:
        print(f"... {omitted} additional frame(s) not shown")


def truncate(text: str, width: int) -> str:
    """Truncate text without exceeding ``width`` characters."""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def determine_exit_code(
    *,
    validation_valid: bool,
    warning_count: int,
    strict_warnings: bool,
) -> int:
    """Translate the test result into a process exit status."""
    if not validation_valid:
        return 1

    if strict_warnings and warning_count > 0:
        return 2

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the manifest smoke test and return a process exit status."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    print_header(args.csv_path, args.frame_root)

    try:
        manifest = load_manifest(
            args.csv_path,
            frame_root=args.frame_root,
            encoding=args.encoding,
        )
    except ManifestError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        # This is defensive; from_csv() should normally wrap file-open errors
        # in ManifestError.
        print(f"\nFAILED: operating-system error: {exc}", file=sys.stderr)
        return 1

    show_manifest_summary(manifest)

    validation_valid, warning_count = show_validation(
        manifest,
        all_issues=args.all_issues,
    )

    show_dataset_summaries(
        manifest,
        enabled=args.show_datasets,
    )
    show_frame_records(
        manifest,
        count=args.show_frames,
    )

    exit_code = determine_exit_code(
        validation_valid=validation_valid,
        warning_count=warning_count,
        strict_warnings=args.strict_warnings,
    )

    print("\n" + SEPARATOR)
    if exit_code == 0:
        print("FINISHED: manifest smoke test passed")
    elif exit_code == 2:
        print("FINISHED: validation passed, but strict warning mode failed")
    else:
        print("FINISHED: manifest validation failed")
    print(SEPARATOR)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
