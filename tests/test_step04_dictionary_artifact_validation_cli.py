"""Integration test for Step 04 artifact-validation CLI v4.37.0."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steps import step03_prepare_bias_analysis as step03
from steps import step04_prepare_rts_dictionary_analysis as step04


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def write_dataset(root: Path) -> Path:
    paths = []
    for index in range(8):
        data = np.array(
            [
                [0 if index % 4 < 2 else 10, 20, 30],
                [0 if index % 2 == 0 else 10, 5, 50],
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
            "environment": "step04-v4.37-test",
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


def build_kwargs() -> dict[str, object]:
    return {
        "minimum_score": 0.9,
        "minimum_state_count": 2,
        "minimum_separation": 5.0,
        "minimum_transition_count": 3,
        "minimum_lower_run": 2,
        "minimum_upper_run": 2,
    }


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 artifact-validation CLI test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_validation_cli_") as temp:
        root = Path(temp)
        manifest = write_dataset(root)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(manifest, "bias")
        )
        built = step04.build_rts_dictionary_artifacts(
            plan,
            root / "dictionary.csv",
            **build_kwargs(),
        )
        audit = step04.audit_rts_dictionary_input_files(
            built.metadata_path
        )

        print("[1/5] CLI validation succeeds with human-readable output")
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = step04.run_rts_input_audit_cli(
            [str(built.metadata_path), "--validate"],
            stdout=stdout,
            stderr=stderr,
        )
        require(exit_code == 0, "VALID exit code changed")
        require("Status                : VALID\n" in stdout.getvalue(),
                "VALID status missing")
        require("Validation PASSED\n" in stdout.getvalue(),
                "success footer missing")
        require(stderr.getvalue() == "", "VALID wrote to stderr")
        print("   Exit code     : 0")
        print("   Human report  : stable")
        print("   Result        : PASS")
        print()

        print("[2/5] CLI validation failure uses exit code 4")
        comparison = audit.comparison_json_path
        original = comparison.read_bytes()
        comparison.write_text("{", encoding="utf-8")
        bad_stdout = io.StringIO()
        bad_stderr = io.StringIO()
        bad_exit = step04.run_rts_input_audit_cli(
            [str(built.metadata_path), "--validate"],
            stdout=bad_stdout,
            stderr=bad_stderr,
        )
        require(bad_exit == 4, "INVALID exit code changed")
        require(bad_stdout.getvalue() == "", "INVALID wrote to stdout")
        require("Status : INVALID\n" in bad_stderr.getvalue(),
                "INVALID status missing")
        comparison.write_bytes(original)
        print("   Exit code     : 4")
        print("   Failure report: stable")
        print("   Result        : PASS")
        print()

        print("[3/5] --validate --json emits one stable JSON object")
        json_stdout = io.StringIO()
        json_stderr = io.StringIO()
        json_exit = step04.run_rts_input_audit_cli(
            [str(built.metadata_path), "--validate", "--json"],
            stdout=json_stdout,
            stderr=json_stderr,
        )
        payload = json.loads(json_stdout.getvalue())
        require(json_exit == 0, "JSON validation exit changed")
        require(payload["status"] == "VALID", "JSON status changed")
        require(payload["ok"] is True, "JSON ok changed")
        require(payload["exit_code"] == 0, "JSON exit code changed")
        require(payload["dataset"] == "bias", "JSON dataset changed")
        require(json_stderr.getvalue() == "", "JSON wrote to stderr")
        print("   JSON payload  : valid")
        print("   Extra text    : none")
        print("   Result        : PASS")
        print()

        print("[4/5] JSON file export and quiet mode are supported")
        destination = root / "reports" / "validation.json"
        quiet_stdout = io.StringIO()
        quiet_stderr = io.StringIO()
        quiet_exit = step04.run_rts_input_audit_cli(
            [
                str(built.metadata_path),
                "--validate",
                "--quiet",
                "--json-output",
                str(destination),
            ],
            stdout=quiet_stdout,
            stderr=quiet_stderr,
        )
        require(quiet_exit == 0, "quiet validation exit changed")
        require(quiet_stdout.getvalue() == "", "quiet wrote stdout")
        require(quiet_stderr.getvalue() == "", "quiet wrote stderr")
        require(destination.is_file(), "validation JSON file missing")
        require(
            json.loads(destination.read_text(encoding="utf-8"))
            == payload,
            "validation file payload changed",
        )
        print("   JSON file     : created")
        print("   Quiet output  : none")
        print("   Result        : PASS")
        print()

        print("[5/5] Output is deterministic and audit mode is preserved")
        first_stdout = io.StringIO()
        second_stdout = io.StringIO()
        first_exit = step04.run_rts_input_audit_cli(
            [str(built.metadata_path), "--validate", "--json"],
            stdout=first_stdout,
            stderr=io.StringIO(),
        )
        second_exit = step04.run_rts_input_audit_cli(
            [str(built.metadata_path), "--validate", "--json"],
            stdout=second_stdout,
            stderr=io.StringIO(),
        )
        require(first_exit == second_exit == 0, "deterministic exits changed")
        require(first_stdout.getvalue() == second_stdout.getvalue(),
                "validation JSON text changed")

        audit_stdout = io.StringIO()
        audit_exit = step04.run_rts_input_audit_cli(
            [str(built.metadata_path), "--json"],
            stdout=audit_stdout,
            stderr=io.StringIO(),
        )
        audit_payload = json.loads(audit_stdout.getvalue())
        require(audit_exit == 0, "audit mode exit changed")
        require(audit_payload["status"] == "MATCH",
                "audit mode was not preserved")
        print("   JSON text     : deterministic")
        print("   Audit CLI     : preserved")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 artifact-validation CLI test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
