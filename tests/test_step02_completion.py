"""Completion integration test for RTS Framework Step 02 v2.7.0."""

from __future__ import annotations

import csv
import io
import json
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


EXPECTED_PUBLIC_API = {
    "DatasetGroup",
    "DatasetStatistics",
    "FrameStatistics",
    "STATISTICS_CSV_COLUMNS",
    "Step02Error",
    "Step02Result",
    "build_argument_parser",
    "build_statistics_summary",
    "compute_dataset_statistics",
    "iter_dataset_images",
    "main",
    "prepare_frame_groups",
    "statistics_filename",
    "statistics_summary_filename",
    "write_all_statistics_csv",
    "write_all_statistics_summary_json",
    "write_statistics_csv",
    "write_statistics_summary_json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def write_fits(path: Path, data: np.ndarray) -> None:
    fits.PrimaryHDU(data=data).writeto(path, overwrite=True)


def make_row(
    dataset: str,
    image: Path,
    *,
    frame_index: int,
    n_frames: int,
    temperature_C: float,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "directory": str(image.parent),
        "environment": "completion-test",
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": temperature_C,
        "temperature_start_C": -20.0,
        "temperature_end_C": -19.0,
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
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run_main(arguments: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = step02.main(arguments)
    return code, stdout.getvalue(), stderr.getvalue()


def capture_parser_exit(arguments: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            step02.main(arguments)
    except SystemExit as exc:
        code = int(exc.code)
    else:
        raise AssertionError("argparse did not exit")
    return code, stdout.getvalue(), stderr.getvalue()


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 02 completion integration test")
    print("=" * 72)
    print(f"step02 version : {step02.__version__}")
    print()

    print("[1/5] Public API is explicit and complete")
    require(set(step02.__all__) == EXPECTED_PUBLIC_API, "unexpected public API")
    for name in EXPECTED_PUBLIC_API:
        require(hasattr(step02, name), f"missing public API: {name}")
    print(f"   Public symbols : {len(EXPECTED_PUBLIC_API)}")
    print("   Result         : PASS")
    print()

    print("[2/5] CLI help and version are self-contained")
    code, stdout, stderr = capture_parser_exit(["--help"])
    require(code == 0, "help exit code is not 0")
    require(stderr == "", f"help wrote stderr: {stderr!r}")
    for token in (
        "MANIFEST",
        "--frame-root DIRECTORY",
        "--statistics-dir DIRECTORY",
        "--summary-dir DIRECTORY",
        "--quiet",
        "Exit status: 0 on success, 2",
    ):
        require(token in stdout, f"help text missing: {token!r}")

    code, stdout, stderr = capture_parser_exit(["--version"])
    require(code == 0, "version exit code is not 0")
    require(step02.__version__ in stdout, "version string missing")
    require(stderr == "", "version wrote stderr")
    print("   --help    : PASS")
    print("   --version : PASS")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step02_completion_") as temp_dir:
        root = Path(temp_dir)
        frames = root / "frames"
        frames.mkdir()

        image0 = frames / "bias_0000.fit"
        image1 = frames / "bias_0001.fit"
        write_fits(image0, np.arange(6, dtype=np.uint16).reshape(2, 3))
        write_fits(image1, np.arange(10, 16, dtype=np.uint16).reshape(2, 3))

        manifest = root / "manifest.normalized.csv"
        write_manifest(
            manifest,
            [
                make_row(
                    "bias",
                    image0,
                    frame_index=0,
                    n_frames=2,
                    temperature_C=-20.0,
                ),
                make_row(
                    "bias",
                    image1,
                    frame_index=1,
                    n_frames=2,
                    temperature_C=-19.0,
                ),
            ],
        )

        print("[3/5] Combined CLI export writes canonical CSV and JSON")
        csv_dir = root / "statistics"
        json_dir = root / "summaries"
        code, stdout, stderr = run_main(
            [
                str(manifest),
                "--statistics-dir",
                str(csv_dir),
                "--summary-dir",
                str(json_dir),
            ]
        )
        require(code == 0, f"combined CLI exit code: {code}")
        require(stderr == "", f"combined CLI stderr: {stderr!r}")
        csv_path = csv_dir / "bias.csv"
        json_path = json_dir / "bias.json"
        require(csv_path.is_file(), "statistics CSV missing")
        require(json_path.is_file(), "summary JSON missing")
        require(csv_path.read_bytes().endswith(b"\n"), "CSV final newline missing")
        require(json_path.read_bytes().endswith(b"\n"), "JSON final newline missing")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        require(payload["dataset"] == "bias", "wrong JSON dataset")
        require(payload["n_frames"] == 2, "wrong JSON frame count")
        require(payload["image"]["shape"] == [2, 3], "wrong JSON shape")
        require("Status   : PASSED" in stdout, "final summary missing")
        print("   bias.csv  : written")
        print("   bias.json : written")
        print("   Result    : PASS")
        print()

        print("[4/5] Quiet combined export is silent and deterministic")
        quiet_csv = root / "quiet_statistics"
        quiet_json = root / "quiet_summaries"
        code, stdout, stderr = run_main(
            [
                str(manifest),
                "--statistics-dir",
                str(quiet_csv),
                "--summary-dir",
                str(quiet_json),
                "--quiet",
            ]
        )
        require(code == 0, "quiet combined export failed")
        require(stdout == "", f"quiet stdout is not empty: {stdout!r}")
        require(stderr == "", f"quiet stderr is not empty: {stderr!r}")
        require(
            (quiet_csv / "bias.csv").read_bytes() == csv_path.read_bytes(),
            "CSV output is not deterministic",
        )
        require(
            (quiet_json / "bias.json").read_bytes() == json_path.read_bytes(),
            "JSON output is not deterministic",
        )
        print("   stdout       : empty")
        print("   stderr       : empty")
        print("   Byte identity: YES")
        print("   Result       : PASS")
        print()

        print("[5/5] Failure paths use exit status 2")
        missing = root / "missing.csv"
        code, stdout, stderr = run_main([str(missing), "--quiet"])
        require(code == 2, f"Step02Error exit code is {code}, not 2")
        require(stdout == "", "quiet failure emitted stdout")
        require("Step 02 error:" in stderr, "Step02Error prefix missing")

        code, stdout, stderr = capture_parser_exit([])
        require(code == 2, f"argument error exit code is {code}, not 2")
        require(stdout == "", "argument error emitted stdout")
        require("usage:" in stderr, "argument error usage missing")
        print("   Step02Error   : 2")
        print("   Argument error: 2")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: RTS Framework Step 02 completion test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
