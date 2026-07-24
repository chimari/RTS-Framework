"""Integration test for the Step 01 command-line interface."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIDTH = 9576
HEIGHT = 6388


def make_row(
    image: Path,
    *,
    dataset: str,
    image_width: int = WIDTH,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "directory": str(image.parent),
        "environment": "test",
        "frame_index": 0,
        "n_frames": 1,
        "temperature_C": -12.1,
        "temperature_start_C": -12.1,
        "temperature_end_C": -12.1,
        "temperature_fraction": 0.0,
        "exposure_s": 0.0,
        "filename": image.name,
        "filepath": str(image),
        "image_width": image_width,
        "image_height": HEIGHT,
        "pixel_dtype": "uint16",
        "byte_order": "not-applicable",
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "steps.step01_prepare_dataset",
            *args,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def require(condition: bool, message: str, process=None) -> None:
    if condition:
        return
    print(f"FAIL: {message}")
    if process is not None:
        print("--- stdout ---")
        print(process.stdout)
        print("--- stderr ---")
        print(process.stderr)
    raise SystemExit(1)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} IMAGE")
        return 2

    source = Path(sys.argv[1]).resolve()
    require(source.is_file(), f"image does not exist: {source}")

    print("=" * 72)
    print("RTS Framework Step 01 CLI integration test")
    print("=" * 72)
    print(f"image : {source}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step01_cli_") as temp_dir:
        root = Path(temp_dir)
        valid_image = root / f"valid{source.suffix}"
        invalid_image = root / f"invalid{source.suffix}"
        valid_image.symlink_to(source)
        invalid_image.symlink_to(source)

        valid_manifest = root / "valid.csv"
        invalid_manifest = root / "invalid.csv"
        valid_report = root / "reports" / "valid.json"
        invalid_report = root / "reports" / "invalid.json"

        write_manifest(
            valid_manifest,
            [make_row(valid_image, dataset="bias-valid")],
        )
        write_manifest(
            invalid_manifest,
            [
                make_row(
                    invalid_image,
                    dataset="bias-invalid",
                    image_width=WIDTH + 1,
                )
            ],
        )

        print("[1/4] Passing validation returns exit code 0")
        passed = run_cli(
            str(valid_manifest),
            "--report",
            str(valid_report),
        )
        require(passed.returncode == 0, "expected exit code 0", passed)
        require("Status          : PASSED" in passed.stdout, "missing PASSED")
        require("Checking image 1/1" in passed.stdout, "missing progress")
        require(valid_report.is_file(), "valid JSON report was not created")
        payload = json.loads(valid_report.read_text(encoding="utf-8"))
        require(payload["status"] == "passed", "report status is not passed")
        print("   Exit code : 0")
        print("   Report    : created")
        print("   Result    : PASS")
        print()

        print("[2/4] Failed validation returns exit code 1")
        failed = run_cli(
            str(invalid_manifest),
            "--report",
            str(invalid_report),
        )
        require(failed.returncode == 1, "expected exit code 1", failed)
        require("Status          : FAILED" in failed.stdout, "missing FAILED")
        require(invalid_report.is_file(), "failed JSON report was not created")
        payload = json.loads(invalid_report.read_text(encoding="utf-8"))
        require(payload["status"] == "failed", "report status is not failed")
        require(payload["counts"]["image_errors"] == 1, "wrong issue count")
        print("   Exit code : 1")
        print("   Report    : created")
        print("   Result    : PASS")
        print()

        print("[3/4] Operational error returns exit code 2")
        missing = run_cli(str(root / "missing_manifest.csv"))
        require(missing.returncode == 2, "expected exit code 2", missing)
        require("Step 01 error:" in missing.stderr, "missing error message")
        print("   Exit code : 2")
        print("   stderr    : Step 01 error")
        print("   Result    : PASS")
        print()

        print("[4/4] Quiet mode suppresses progress")
        quiet = run_cli(str(valid_manifest), "--quiet")
        require(quiet.returncode == 0, "quiet command failed", quiet)
        require("Checking image" not in quiet.stdout, "progress was not hidden")
        require("Status          : PASSED" in quiet.stdout, "summary was hidden")
        print("   Progress suppressed : YES")
        print("   Summary retained    : YES")
        print("   Result              : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 01 CLI integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
