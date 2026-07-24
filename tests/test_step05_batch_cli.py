"""Integration test for the Step 05 batch CLI v5.7.0."""

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
            "environment": "step05-v5.7-test",
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


def prepare_artifacts(root: Path) -> tuple[Path, tuple[Path, Path], Path]:
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

    inputs = []
    for image_index in range(2):
        source = root / f"target_{image_index}.fit"
        data = np.full((2, 3), 20 + image_index, dtype=np.float32)
        fits.PrimaryHDU(data=data).writeto(source, overwrite=True)
        plan = step05.prepare_rts_correction(built.metadata_path, source)
        for index, candidate in enumerate(plan.candidates):
            data[candidate.row, candidate.column] = (
                candidate.lower_state_center
                + 0.1 * candidate.state_separation
                if (index + image_index) % 2 == 0
                else candidate.upper_state_center
                - 0.1 * candidate.state_separation
            )
        fits.PrimaryHDU(data=data).writeto(source, overwrite=True)
        inputs.append(source)

    invalid = root / "invalid.fit"
    fits.PrimaryHDU(
        data=np.zeros((3, 3), dtype=np.float32)
    ).writeto(invalid, overwrite=True)

    return built.metadata_path, tuple(inputs), invalid


def call_cli(args: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = step05.run_rts_correction_batch_cli(args)
    return code, stdout.getvalue(), stderr.getvalue()


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 05 batch CLI test")
    print("=" * 72)
    print(f"step05 version : {step05.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step05_batch_cli_") as temp:
        root = Path(temp)
        metadata, inputs, invalid = prepare_artifacts(root)

        print("[1/4] Repeated --input processes a complete successful batch")
        output_dir = root / "repeat_outputs"
        output_dir.mkdir()
        code, stdout, stderr = call_cli([
            "--metadata", str(metadata),
            "--input", str(inputs[0]),
            "--input", str(inputs[1]),
            "--output-directory", str(output_dir),
        ])
        require(code == 0, f"repeated-input exit code changed: {code}")
        require(stderr == "", f"unexpected stderr: {stderr}")
        require("RTS batch correction completed" in stdout, "report missing")
        require("Succeeded      : 2" in stdout, "success count missing")
        require(
            (output_dir / "target_0_rts_corrected.fit").exists(),
            "first batch output missing",
        )
        require(
            (output_dir / "target_1_rts_corrected.fit").exists(),
            "second batch output missing",
        )
        print("   Inputs        : 2")
        print("   Exit code     : 0")
        print("   Outputs       : 2")
        print("   Result        : PASS")
        print()

        print("[2/4] Input list, comments, quiet, and JSON modes work")
        list_file = root / "inputs.txt"
        list_file.write_text(
            "\n".join([
                "# Step 05 batch input list",
                "",
                str(inputs[0]),
                f"  {inputs[1]}  ",
                "",
            ]),
            encoding="utf-8",
        )

        quiet_dir = root / "quiet_outputs"
        quiet_dir.mkdir()
        code, stdout, stderr = call_cli([
            "--metadata", str(metadata),
            "--input-list", str(list_file),
            "--output-directory", str(quiet_dir),
            "--quiet",
        ])
        require(code == 0, "quiet input-list CLI failed")
        require(stdout == "", "quiet mode wrote stdout")
        require(stderr == "", "quiet mode wrote stderr")

        json_dir = root / "json_outputs"
        json_dir.mkdir()
        code, stdout, stderr = call_cli([
            "--metadata", str(metadata),
            "--input-list", str(list_file),
            "--output-directory", str(json_dir),
            "--json",
        ])
        require(code == 0, "JSON input-list CLI failed")
        require(stderr == "", "JSON mode wrote stderr")
        payload = json.loads(stdout)
        require(payload["status"] == "OK", "JSON status changed")
        require(payload["exit_code"] == 0, "JSON exit code changed")
        require(payload["total_count"] == 2, "JSON total changed")
        require(payload["failed_count"] == 0, "JSON failures changed")
        print("   Input list    : parsed")
        print("   Quiet mode    : no output")
        print("   JSON mode     : valid aggregate")
        print("   Result        : PASS")
        print()

        print("[3/4] Partial batches return exit code 1 with aggregate details")
        partial_dir = root / "partial_outputs"
        partial_dir.mkdir()
        code, stdout, stderr = call_cli([
            "--metadata", str(metadata),
            "--input", str(inputs[0]),
            "--input", str(invalid),
            "--input", str(inputs[1]),
            "--output-directory", str(partial_dir),
            "--continue-on-error",
            "--json",
        ])
        require(code == 1, f"partial exit code changed: {code}")
        require(stderr == "", "partial JSON wrote stderr")
        payload = json.loads(stdout)
        require(payload["status"] == "PARTIAL", "partial status changed")
        require(payload["exit_code"] == 1, "partial JSON code changed")
        require(payload["succeeded_count"] == 2, "partial success changed")
        require(payload["failed_count"] == 1, "partial failure changed")
        require(len(payload["items"]) == 3, "partial items changed")
        require(payload["items"][1]["succeeded"] is False, "failure missing")
        require("error" in payload["items"][1], "failure message missing")
        print("   Successes     : 2")
        print("   Failures      : 1")
        print("   Exit code     : 1")
        print("   Result        : PASS")
        print()

        print("[4/4] Missing inputs and fail-fast errors are reported safely")
        empty_dir = root / "empty_outputs"
        empty_dir.mkdir()
        code, stdout, stderr = call_cli([
            "--metadata", str(metadata),
            "--output-directory", str(empty_dir),
        ])
        require(code == 1, "missing-input exit code changed")
        require(stdout == "", "missing-input wrote stdout")
        require("ERROR:" in stderr, "missing-input error missing")

        failfast_dir = root / "failfast_outputs"
        failfast_dir.mkdir()
        code, stdout, stderr = call_cli([
            "--metadata", str(metadata),
            "--input", str(invalid),
            "--input", str(inputs[0]),
            "--output-directory", str(failfast_dir),
        ])
        require(code == 1, "fail-fast exit code changed")
        require(stdout == "", "fail-fast wrote stdout")
        require("ERROR:" in stderr, "fail-fast error missing")
        require(
            not (failfast_dir / "target_0_rts_corrected.fit").exists(),
            "fail-fast continued after failure",
        )
        print("   Missing input : rejected")
        print("   Fail-fast     : exit 1")
        print("   Later inputs  : not processed")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 05 batch CLI test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
