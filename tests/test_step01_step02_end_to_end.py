#!/usr/bin/env python3
"""
Step 01 -> Step 02 end-to-end integration test.

This test accepts an existing RTS Framework input manifest, runs Step 01
to create a normalized manifest, and passes that normalized manifest
directly to Step 02.

The input manifest may describe either a small subset or a full dataset.

Example
-------
python tests/test_step01_step02_end_to_end.py \
    /home/taji/work/IMX455/bias/imx455_subset30.csv \
    --frame-root /home/taji/work/IMX455/bias \
    --output-dir /tmp/imx455_step01_step02_e2e
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class EndToEndError(RuntimeError):
    """Raised when the Step 01 -> Step 02 integration test fails."""


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an existing input manifest through RTS Framework "
            "Step 01 and Step 02."
        )
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Input manifest accepted by Step 01.",
    )
    parser.add_argument(
        "--frame-root",
        type=Path,
        required=True,
        help="Root directory used to resolve relative frame paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which all E2E outputs are written.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=("shape", "full"),
        default="shape",
        help=(
            "Step 01 validation mode. Default: shape. "
            "Step 02 reads the actual image arrays."
        ),
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help=(
            "Do not remove an existing output directory before running. "
            "By default, the output directory is recreated."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress normal Step 01 and Step 02 CLI output.",
    )
    return parser.parse_args(argv)


def run_command(command: list[str]) -> None:
    """Run a command and raise a contextual error on failure."""

    print()
    print("$", " ".join(command))

    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise EndToEndError(
            f"Command failed with exit status {completed.returncode}: "
            f"{' '.join(command)}"
        )


def read_manifest_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read dataset names and rows from a CSV manifest."""

    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)

            if reader.fieldnames is None:
                raise EndToEndError(
                    f"Manifest has no CSV header: {path}"
                )

            if "dataset" not in reader.fieldnames:
                raise EndToEndError(
                    f"Manifest has no 'dataset' column: {path}"
                )

            rows = list(reader)

    except OSError as exc:
        raise EndToEndError(
            f"Could not read manifest {path}: {exc}"
        ) from exc

    if not rows:
        raise EndToEndError(f"Manifest contains no frame rows: {path}")

    datasets = sorted(
        {
            row["dataset"].strip()
            for row in rows
            if row.get("dataset", "").strip()
        }
    )

    if not datasets:
        raise EndToEndError(
            f"Manifest contains no non-empty dataset names: {path}"
        )

    return datasets, rows


def validate_step01_report(path: Path) -> None:
    """Confirm that the Step 01 JSON report exists and is valid JSON."""

    if not path.is_file():
        raise EndToEndError(
            f"Step 01 report was not created: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as stream:
            json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise EndToEndError(
            f"Step 01 report is not valid JSON: {path}: {exc}"
        ) from exc


def validate_step02_outputs(
    *,
    datasets: list[str],
    statistics_dir: Path,
    summary_dir: Path,
) -> None:
    """Confirm that Step 02 produced one CSV and JSON file per dataset."""

    missing: list[Path] = []

    for dataset in datasets:
        statistics_path = statistics_dir / f"{dataset}.csv"
        summary_path = summary_dir / f"{dataset}.json"

        if not statistics_path.is_file():
            missing.append(statistics_path)

        if not summary_path.is_file():
            missing.append(summary_path)

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise EndToEndError(
            "Step 02 did not create all expected outputs:\n"
            f"{formatted}"
        )


def validate_statistics_row_counts(
    *,
    normalized_manifest: Path,
    statistics_dir: Path,
) -> None:
    """Check that each dataset CSV contains one row per manifest frame."""

    _, manifest_rows = read_manifest_rows(normalized_manifest)

    expected_counts: dict[str, int] = {}
    for row in manifest_rows:
        dataset = row["dataset"].strip()
        expected_counts[dataset] = expected_counts.get(dataset, 0) + 1

    for dataset, expected_count in sorted(expected_counts.items()):
        statistics_path = statistics_dir / f"{dataset}.csv"

        try:
            with statistics_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as stream:
                actual_count = sum(1 for _ in csv.DictReader(stream))
        except OSError as exc:
            raise EndToEndError(
                f"Could not read statistics CSV {statistics_path}: {exc}"
            ) from exc

        if actual_count != expected_count:
            raise EndToEndError(
                f"Dataset {dataset!r} statistics row count differs: "
                f"expected {expected_count}, got {actual_count}."
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)

    manifest = args.manifest.expanduser().resolve()
    frame_root = args.frame_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not manifest.is_file():
        print(
            f"ERROR: Input manifest does not exist: {manifest}",
            file=sys.stderr,
        )
        return 2

    if not frame_root.is_dir():
        print(
            f"ERROR: Frame root is not a directory: {frame_root}",
            file=sys.stderr,
        )
        return 2

    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_manifest = output_dir / "normalized_manifest.csv"
    step01_report = output_dir / "step01_report.json"
    statistics_dir = output_dir / "statistics"
    summary_dir = output_dir / "summary"

    input_datasets, input_rows = read_manifest_rows(manifest)

    print("=" * 72)
    print("RTS Framework Step 01 -> Step 02 end-to-end test")
    print("=" * 72)
    print(f"Input manifest : {manifest}")
    print(f"Frame root     : {frame_root}")
    print(f"Output         : {output_dir}")
    print(f"Datasets       : {len(input_datasets)}")
    print(f"Frames         : {len(input_rows)}")
    print(f"Validation     : {args.validation_mode}")

    try:
        step01_command = [
            sys.executable,
            "-m",
            "steps.step01_prepare_dataset",
            str(manifest),
            "--frame-root",
            str(frame_root),
            "--validation-mode",
            args.validation_mode,
            "--normalized-manifest",
            str(normalized_manifest),
            "--report",
            str(step01_report),
        ]

        if args.quiet:
            step01_command.append("--quiet")

        print()
        print("[1/4] Run Step 01")
        run_command(step01_command)

        if not normalized_manifest.is_file():
            raise EndToEndError(
                "Step 01 did not create the normalized manifest: "
                f"{normalized_manifest}"
            )

        validate_step01_report(step01_report)

        normalized_datasets, normalized_rows = read_manifest_rows(
            normalized_manifest
        )

        if normalized_datasets != input_datasets:
            raise EndToEndError(
                "Dataset names changed between the input and normalized "
                "manifests: "
                f"input={input_datasets!r}, "
                f"normalized={normalized_datasets!r}"
            )

        if len(normalized_rows) != len(input_rows):
            raise EndToEndError(
                "Frame count changed between the input and normalized "
                "manifests: "
                f"input={len(input_rows)}, "
                f"normalized={len(normalized_rows)}"
            )

        print("   Normalized manifest : PASS")
        print("   Dataset count       : PASS")
        print("   Frame count         : PASS")

        step02_command = [
            sys.executable,
            "-m",
            "steps.step02_prepare_frame_groups",
            str(normalized_manifest),
            "--frame-root",
            str(frame_root),
            "--statistics-dir",
            str(statistics_dir),
            "--summary-dir",
            str(summary_dir),
        ]

        if args.quiet:
            step02_command.append("--quiet")

        print()
        print("[2/4] Run Step 02")
        run_command(step02_command)

        print()
        print("[3/4] Validate Step 02 artifacts")
        validate_step02_outputs(
            datasets=normalized_datasets,
            statistics_dir=statistics_dir,
            summary_dir=summary_dir,
        )
        print("   Statistics CSV : PASS")
        print("   Summary JSON   : PASS")

        print()
        print("[4/4] Validate per-dataset frame counts")
        validate_statistics_row_counts(
            normalized_manifest=normalized_manifest,
            statistics_dir=statistics_dir,
        )
        print("   Statistics rows: PASS")

    except EndToEndError as exc:
        print()
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    print()
    print("=" * 72)
    print("FINISHED: Step 01 -> Step 02 end-to-end test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
