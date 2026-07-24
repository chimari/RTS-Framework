"""Integration test for Step 04 audit JSON report export v4.35.0."""

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


def write_dataset(root: Path) -> tuple[Path, list[Path]]:
    paths = []
    for frame_index in range(8):
        data = np.array(
            [
                [0 if frame_index % 4 < 2 else 10, 20, 30],
                [0 if frame_index % 2 == 0 else 10, 5, 50],
            ],
            dtype=np.uint16,
        )
        path = root / f"bias_{frame_index:04d}.fit"
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows = []
    for frame_index, path in enumerate(paths):
        rows.append({
            "dataset": "bias",
            "directory": str(root),
            "environment": "step04-v4.35-test",
            "frame_index": frame_index,
            "n_frames": len(paths),
            "temperature_C": -10.0,
            "temperature_start_C": -10.0,
            "temperature_end_C": -10.0,
            "temperature_fraction": frame_index / (len(paths) - 1),
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
    return manifest, paths


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
    print("RTS Framework Step 04 audit JSON report export test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_json_output_") as temp:
        root = Path(temp)
        manifest, input_paths = write_dataset(root)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(manifest, "bias")
        )
        artifacts = step04.build_rts_dictionary_artifacts(
            plan,
            root / "dictionary.csv",
            **build_kwargs(),
        )
        audit = step04.audit_rts_dictionary_input_files(
            artifacts.metadata_path
        )
        status = step04.evaluate_rts_input_audit(audit)
        expected_payload = audit.to_json_summary(status)

        print("[1/4] Library API writes the deterministic JSON report")
        destination = root / "reports" / "audit.json"
        returned = step04.write_rts_input_audit_json(
            audit,
            destination,
            status=status,
        )
        require(returned == destination.resolve(), "returned path changed")
        require(destination.is_file(), "audit JSON was not written")
        payload = json.loads(destination.read_text(encoding="utf-8"))
        require(payload == expected_payload, "saved payload changed")
        require(
            destination.read_bytes().endswith(b"\n"),
            "saved JSON lacks final newline",
        )
        print("   Parent creation : complete")
        print("   Payload         : valid")
        print("   Final newline   : present")
        print("   Result          : PASS")
        print()

        print("[2/4] CLI --json-output writes a report")
        cli_destination = root / "cli" / "audit.json"
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = step04.run_rts_input_audit_cli(
            [
                str(artifacts.metadata_path),
                "--json-output",
                str(cli_destination),
            ],
            stdout=stdout,
            stderr=stderr,
        )
        require(exit_code == 0, "CLI JSON output did not return exit 0")
        require(cli_destination.is_file(), "CLI report missing")
        require(
            json.loads(cli_destination.read_text(encoding="utf-8"))
            == expected_payload,
            "CLI report payload changed",
        )
        require(
            "RTS input audit: MATCH\n" in stdout.getvalue(),
            "normal human output disappeared",
        )
        require(stderr.getvalue() == "", "CLI wrote to stderr")
        print("   Report file   : created")
        print("   Human output  : preserved")
        print("   Exit code     : 0")
        print("   Result        : PASS")
        print()

        print("[3/4] --json and --json-output work together")
        both_destination = root / "both" / "audit.json"
        both_stdout = io.StringIO()
        exit_code = step04.run_rts_input_audit_cli(
            [
                str(artifacts.metadata_path),
                "--json",
                "--json-output",
                str(both_destination),
            ],
            stdout=both_stdout,
            stderr=io.StringIO(),
        )
        stdout_payload = json.loads(both_stdout.getvalue())
        file_payload = json.loads(
            both_destination.read_text(encoding="utf-8")
        )
        require(exit_code == 0, "combined JSON mode exit changed")
        require(stdout_payload == expected_payload, "stdout JSON changed")
        require(file_payload == expected_payload, "file JSON changed")
        require(stdout_payload == file_payload, "JSON outputs differ")
        print("   Stdout JSON   : valid")
        print("   File JSON     : valid")
        print("   Payloads      : identical")
        print("   Result        : PASS")
        print()

        print("[4/4] Determinism, UTF-8, LF, and errors are preserved")
        first = root / "determinism" / "first.json"
        second = root / "determinism" / "second.json"
        step04.write_rts_input_audit_json(audit, first)
        step04.write_rts_input_audit_json(audit, second)
        require(first.read_bytes() == second.read_bytes(), "bytes changed")
        require(b"\r\n" not in first.read_bytes(), "CRLF was written")
        first.read_text(encoding="utf-8")

        try:
            step04.write_rts_input_audit_json(object(), root / "bad.json")
        except step04.Step04Error:
            pass
        else:
            require(False, "invalid audit was accepted")

        try:
            step04.write_rts_input_audit_json(
                audit,
                root / "bad-status.json",
                status=object(),
            )
        except step04.Step04Error:
            pass
        else:
            require(False, "invalid status was accepted")

        original = input_paths[0].read_bytes()
        input_paths[0].write_bytes(original + b"\x00")
        changed_destination = root / "changed" / "audit.json"
        changed_exit = step04.run_rts_input_audit_cli(
            [
                str(artifacts.metadata_path),
                "--json-output",
                str(changed_destination),
                "--quiet",
            ],
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        changed_payload = json.loads(
            changed_destination.read_text(encoding="utf-8")
        )
        require(changed_exit == 1, "quiet changed exit code changed")
        require(
            changed_payload["status"] == "CHANGED",
            "changed report status missing",
        )
        input_paths[0].write_bytes(original)
        print("   Byte output   : deterministic")
        print("   Encoding/LF   : stable")
        print("   Invalid input : rejected")
        print("   Quiet export  : supported")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 audit JSON report export test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
