"""Integration test for the Step 05 CLI v5.5.0."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steps import step03_prepare_bias_analysis as step03
from steps import step04_prepare_rts_dictionary_analysis as step04
from steps import step05_apply_rts_correction as step05


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def write_manifest(root: Path) -> Path:
    paths = []
    for index in range(8):
        data = np.array(
            [
                [0 if index % 4 < 2 else 10, 20, 30],
                [10 if index % 4 < 2 else 0, 5, 50],
            ],
            dtype=np.uint16,
        )
        path = root / f"bias_{index:04d}.fit"
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows = []
    for index, path in enumerate(paths):
        rows.append({
            "dataset": "bias",
            "directory": str(root),
            "environment": "step05-v5.5-test",
            "frame_index": index,
            "n_frames": len(paths),
            "temperature_C": -10.0,
            "temperature_start_C": -10.0,
            "temperature_end_C": -10.0,
            "temperature_fraction": index / (len(paths) - 1),
            "exposure_s": 0.0,
            "filename": path.name,
            "filepath": str(path),
            "image_width": 3,
            "image_height": 2,
            "pixel_dtype": "uint16",
            "byte_order": "not-applicable",
        })

    manifest = root / "manifest.normalized.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def prepare_artifacts(root: Path) -> tuple[Path, Path]:
    manifest = write_manifest(root)
    analysis = step03.prepare_bias_analysis(manifest, "bias")
    plan04 = step04.prepare_rts_dictionary_analysis(analysis)
    built = step04.build_rts_dictionary_artifacts(
        plan04,
        root / "dictionary.csv",
        minimum_score=0.9,
        minimum_state_count=2,
        minimum_separation=5.0,
        minimum_transition_count=3,
        minimum_lower_run=2,
        minimum_upper_run=2,
    )
    step04.audit_rts_dictionary_input_files(built.metadata_path)

    source = root / "target.fit"
    data = np.full((2, 3), 20, dtype=np.float32)
    fits.PrimaryHDU(data=data).writeto(source, overwrite=True)
    plan = step05.prepare_rts_correction(built.metadata_path, source)

    for index, candidate in enumerate(plan.candidates):
        data[candidate.row, candidate.column] = (
            candidate.lower_state_center + 0.1 * candidate.state_separation
            if index % 2 == 0
            else candidate.upper_state_center - 0.1 * candidate.state_separation
        )
    fits.PrimaryHDU(data=data).writeto(source, overwrite=True)
    return built.metadata_path, source


def call_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = step05.run_rts_correction_cli(args)
    return code, stdout.getvalue(), stderr.getvalue()


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 05 CLI test")
    print("=" * 72)
    print(f"step05 version : {step05.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step05_cli_") as temp:
        root = Path(temp)
        metadata, source = prepare_artifacts(root)

        print("[1/4] Normal CLI runs the complete correction pipeline")
        output = root / "normal.fit"
        code, stdout, stderr = call_cli([
            "--metadata", str(metadata),
            "--input", str(source),
            "--output", str(output),
        ])
        require(code == 0, f"normal CLI exit code changed: {code}")
        require(output.exists(), "normal CLI output missing")
        require("RTS correction completed" in stdout, "success text missing")
        require("Verified       : True" in stdout, "verification text missing")
        require(stderr == "", f"unexpected stderr: {stderr}")
        print("   Exit code     : 0")
        print("   Output FITS   : created")
        print("   Human report  : emitted")
        print("   Result        : PASS")
        print()

        print("[2/4] Quiet and JSON modes are deterministic")
        quiet_output = root / "quiet.fit"
        code, stdout, stderr = call_cli([
            "--metadata", str(metadata),
            "--input", str(source),
            "--output", str(quiet_output),
            "--quiet",
        ])
        require(code == 0, "quiet CLI failed")
        require(stdout == "", "quiet mode wrote stdout")
        require(stderr == "", "quiet mode wrote stderr")

        json_output = root / "json.fit"
        code, stdout, stderr = call_cli([
            "--metadata", str(metadata),
            "--input", str(source),
            "--output", str(json_output),
            "--json",
        ])
        require(code == 0, "JSON CLI failed")
        require(stderr == "", "JSON mode wrote stderr")
        payload = json.loads(stdout)
        require(payload["status"] == "OK", "JSON status changed")
        require(payload["exit_code"] == 0, "JSON exit code changed")
        require(payload["verified"] is True, "JSON verified changed")
        require(
            payload["output_path"] == str(json_output.resolve()),
            "JSON output path changed",
        )
        require(
            payload["candidate_count"]
            == payload["applied_count"] + payload["preserved_count"],
            "JSON counts disagree",
        )
        print("   Quiet mode    : no output")
        print("   JSON mode     : valid object")
        print("   Result        : PASS")
        print()

        print("[3/4] Operational failures return exit code 1")
        existing = root / "existing.fit"
        existing.write_bytes(b"do-not-overwrite")
        code, stdout, stderr = call_cli([
            "--metadata", str(metadata),
            "--input", str(source),
            "--output", str(existing),
        ])
        require(code == 1, f"failure exit code changed: {code}")
        require(stdout == "", "failure wrote stdout")
        require("ERROR:" in stderr, "failure error text missing")
        require(existing.read_bytes() == b"do-not-overwrite", "file overwritten")

        code, stdout, stderr = call_cli([
            "--metadata", str(metadata),
            "--input", str(source),
            "--output", str(existing),
            "--json",
        ])
        require(code == 1, "JSON failure exit code changed")
        require(stderr == "", "JSON failure wrote stderr")
        payload = json.loads(stdout)
        require(payload["status"] == "ERROR", "JSON error status changed")
        require(payload["exit_code"] == 1, "JSON error code changed")
        require("error" in payload, "JSON error message missing")
        print("   Existing file: rejected")
        print("   Text failure  : stderr + exit 1")
        print("   JSON failure  : JSON + exit 1")
        print("   Result        : PASS")
        print()

        print("[4/4] Script entry point and --version work")
        module_path = Path(step05.__file__).resolve()
        completed = subprocess.run(
            [sys.executable, str(module_path), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        require(completed.returncode == 0, "--version failed")
        require(step05.__version__ in completed.stdout, "version missing")
        require(completed.stderr == "", "--version wrote stderr")
        print("   Script entry  : executable")
        print(f"   Version       : {step05.__version__}")
        print("   Exit code     : 0")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 05 CLI test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
