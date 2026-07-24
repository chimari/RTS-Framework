"""Integration test for Step 01 normalized manifest output."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIDTH = 9576
HEIGHT = 6388

EXPECTED_COLUMNS = [
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
]


def make_row(
    image: Path,
    *,
    dataset: str,
    frame_index: int,
    n_frames: int,
    image_width: int = WIDTH,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "directory": str(image.parent),
        "environment": "test",
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": -12.1,
        "temperature_start_C": -12.1,
        "temperature_end_C": -12.1,
        "temperature_fraction": (
            0.0 if n_frames == 1 else frame_index / (n_frames - 1)
        ),
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
        [sys.executable, "-m", "steps.step01_prepare_dataset", *args],
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


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} IMAGE")
        return 2

    source = Path(sys.argv[1]).resolve()
    require(source.is_file(), f"image does not exist: {source}")

    print("=" * 72)
    print("RTS Framework Step 01 normalized manifest test")
    print("=" * 72)
    print(f"image : {source}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step01_norm_") as temp_dir:
        root = Path(temp_dir)

        frame_b1 = root / f"b_frame_0001{source.suffix}"
        frame_a1 = root / f"a_frame_0001{source.suffix}"
        frame_b0 = root / f"b_frame_0000{source.suffix}"
        bad_frame = root / f"bad_frame{source.suffix}"
        for path in (frame_b1, frame_a1, frame_b0, bad_frame):
            path.symlink_to(source)

        # Intentionally unordered input rows.
        valid_manifest = root / "input_valid.csv"
        write_manifest(
            valid_manifest,
            [
                make_row(
                    frame_b1,
                    dataset="dataset-b",
                    frame_index=1,
                    n_frames=2,
                ),
                make_row(
                    frame_a1,
                    dataset="dataset-a",
                    frame_index=0,
                    n_frames=1,
                ),
                make_row(
                    frame_b0,
                    dataset="dataset-b",
                    frame_index=0,
                    n_frames=2,
                ),
            ],
        )

        normalized = root / "output" / "manifest.normalized.csv"

        print("[1/4] Valid result creates normalized CSV")
        passed = run_cli(
            str(valid_manifest),
            "--normalized-manifest",
            str(normalized),
            "--quiet",
        )
        require(passed.returncode == 0, "expected exit code 0", passed)
        require(normalized.is_file(), "normalized manifest was not created")
        require(
            f"Normalized manifest: {normalized}" in passed.stdout,
            "output path was not reported",
            passed,
        )
        print("   Exit code : 0")
        print("   CSV       : created")
        print("   Result    : PASS")
        print()

        print("[2/4] Canonical columns, absolute paths, and stable ordering")
        columns, rows = read_rows(normalized)
        require(columns == EXPECTED_COLUMNS, "column order is not canonical")
        require(len(rows) == 3, "expected 3 normalized rows")
        require(
            [(row["dataset"], row["frame_index"]) for row in rows]
            == [
                ("dataset-a", "0"),
                ("dataset-b", "0"),
                ("dataset-b", "1"),
            ],
            "rows are not deterministically ordered",
        )
        require(
            all(Path(row["filepath"]).is_absolute() for row in rows),
            "filepath is not absolute",
        )
        require(
            all(
                row["directory"] == str(Path(row["filepath"]).parent)
                for row in rows
            ),
            "directory does not match normalized filepath",
        )
        require(
            all(row["filename"] == Path(row["filepath"]).name for row in rows),
            "filename does not match normalized filepath",
        )
        print("   Columns        : canonical")
        print("   Paths          : absolute")
        print("   Dataset order  : stable")
        print("   Frame order    : stable")
        print("   Result         : PASS")
        print()

        print("[3/4] Repeated output is byte-for-byte identical")
        normalized_repeat = root / "output" / "manifest.repeat.csv"
        repeated = run_cli(
            str(valid_manifest),
            "--normalized-manifest",
            str(normalized_repeat),
            "--quiet",
        )
        require(repeated.returncode == 0, "repeat command failed", repeated)
        require(
            normalized.read_bytes() == normalized_repeat.read_bytes(),
            "normalized outputs differ",
        )
        print("   Deterministic : YES")
        print("   Result        : PASS")
        print()

        print("[4/4] Failed validation does not create normalized CSV")
        invalid_manifest = root / "input_invalid.csv"
        write_manifest(
            invalid_manifest,
            [
                make_row(
                    bad_frame,
                    dataset="dataset-invalid",
                    frame_index=0,
                    n_frames=1,
                    image_width=WIDTH + 1,
                )
            ],
        )
        rejected_output = root / "output" / "rejected.csv"
        failed = run_cli(
            str(invalid_manifest),
            "--normalized-manifest",
            str(rejected_output),
            "--quiet",
        )
        require(failed.returncode == 1, "expected exit code 1", failed)
        require(
            not rejected_output.exists(),
            "normalized CSV was created from invalid input",
        )
        require(
            "Normalized manifest not written: validation failed."
            in failed.stderr,
            "skip reason was not reported",
            failed,
        )
        print("   Exit code       : 1")
        print("   CSV created     : NO")
        print("   Reason reported : YES")
        print("   Result          : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 01 normalized manifest test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
