"""Integration test for the Step 02 statistics CLI."""

from __future__ import annotations

import csv
import io
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steps import step02_prepare_frame_groups as step02


def write_fits(path: Path, data: np.ndarray) -> None:
    fits.PrimaryHDU(data=data).writeto(path, overwrite=True)


def make_row(
    dataset: str,
    image: Path,
    *,
    frame_index: int,
    n_frames: int,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "directory": str(image.parent),
        "environment": "test",
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": -10.0 + frame_index,
        "temperature_start_C": -10.0,
        "temperature_end_C": -9.0,
        "temperature_fraction": (
            0.0 if n_frames == 1 else frame_index / (n_frames - 1)
        ),
        "exposure_s": 0.0,
        "filename": image.name,
        "filepath": str(image),
        "image_width": 3,
        "image_height": 2,
        "pixel_dtype": "uint16",
        "byte_order": "not-applicable",
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def run_main(arguments: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = step02.main(arguments)
    return code, stdout.getvalue(), stderr.getvalue()


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 02 statistics CLI integration test")
    print("=" * 72)
    print(f"step02 version : {step02.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step02_cli_") as temp_dir:
        root = Path(temp_dir)
        frames = root / "frames"
        frames.mkdir()

        cold0 = frames / "cold_0000.fit"
        cold1 = frames / "cold_0001.fit"
        room0 = frames / "room_0000.fit"
        write_fits(cold0, np.arange(6, dtype=np.uint16).reshape(2, 3))
        write_fits(cold1, np.arange(10, 16, dtype=np.uint16).reshape(2, 3))
        write_fits(room0, np.arange(20, 26, dtype=np.uint16).reshape(2, 3))

        manifest = root / "manifest.csv"
        write_manifest(
            manifest,
            [
                make_row("cold set", cold0, frame_index=0, n_frames=2),
                make_row("cold set", cold1, frame_index=1, n_frames=2),
                make_row("room", room0, frame_index=0, n_frames=1),
            ],
        )

        print("[1/4] CLI writes one CSV per dataset")
        output_dir = root / "statistics"
        code, stdout, stderr = run_main(
            [str(manifest), "--statistics-dir", str(output_dir)]
        )
        require(code == 0, f"unexpected exit code: {code}")
        require(stderr == "", f"unexpected stderr: {stderr!r}")
        require((output_dir / "cold_set.csv").is_file(), "cold CSV missing")
        require((output_dir / "room.csv").is_file(), "room CSV missing")
        require("[1/2] cold set: 2 frames" in stdout, "dataset progress missing")
        require("frame 1/2: cold_0000.fit" in stdout, "frame progress missing")
        require("Status   : PASSED" in stdout, "summary missing")
        print("   Files    : cold_set.csv, room.csv")
        print("   Progress : shown")
        print("   Result   : PASS")
        print()

        print("[2/4] --quiet suppresses normal output")
        quiet_dir = root / "quiet_statistics"
        code, stdout, stderr = run_main(
            [
                str(manifest),
                "--statistics-dir",
                str(quiet_dir),
                "--quiet",
            ]
        )
        require(code == 0, "quiet CLI failed")
        require(stdout == "", f"quiet stdout is not empty: {stdout!r}")
        require(stderr == "", f"quiet stderr is not empty: {stderr!r}")
        require((quiet_dir / "cold_set.csv").is_file(), "quiet CSV missing")
        print("   stdout : empty")
        print("   stderr : empty")
        print("   Result : PASS")
        print()

        print("[3/4] Dataset filenames are deterministic and safe")
        require(
            step02.statistics_filename("cold set") == "cold_set.csv",
            "space sanitization failed",
        )
        require(
            step02.statistics_filename("../") == "dataset.csv",
            "unsafe dataset fallback failed",
        )
        require(
            step02.statistics_filename("bias-01") == "bias-01.csv",
            "safe name changed unexpectedly",
        )
        print("   cold set -> cold_set.csv")
        print("   ../      -> dataset.csv")
        print("   Result   : PASS")
        print()

        print("[4/4] Read failure leaves no output CSV")
        missing_manifest = root / "missing.csv"
        missing_frame = frames / "missing.fit"
        write_manifest(
            missing_manifest,
            [make_row("broken", missing_frame, frame_index=0, n_frames=1)],
        )
        failed_dir = root / "failed_statistics"
        code, stdout, stderr = run_main(
            [
                str(missing_manifest),
                "--statistics-dir",
                str(failed_dir),
                "--quiet",
            ]
        )
        require(code == 2, f"wrong failure exit code: {code}")
        require(stdout == "", "failure emitted quiet stdout")
        require("Step 02 error:" in stderr, "failure message missing")
        require(
            not failed_dir.exists()
            or not list(failed_dir.glob("*.csv")),
            "partial CSV was left after read failure",
        )
        print("   Exit code   : 2")
        print("   Output CSVs : none")
        print("   Result      : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 02 statistics CLI integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
