"""Integration test for Step 04 deterministic JSON output v4.34.0."""

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


EXPECTED_KEYS = [
    "status",
    "ok",
    "exit_code",
    "message",
    "changed_count",
    "missing_count",
    "additional_count",
    "metadata_json",
    "fingerprint_json",
    "comparison_json",
]


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
            "environment": "step04-v4.34-test",
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
    print("RTS Framework Step 04 input-audit JSON test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_json_") as temp:
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

        print("[1/4] Status exposes stable external JSON fields")
        status_json = status.to_json_summary()
        require(
            list(status_json) == EXPECTED_KEYS[:7],
            "status JSON key order changed",
        )
        require(status_json["status"] == "MATCH", "status name changed")
        require(status_json["ok"] is True, "status ok changed")
        require(status_json["exit_code"] == 0, "status code changed")
        require(
            status_json == status.to_json_summary(),
            "status JSON is not deterministic",
        )
        print("   Keys          : stable")
        print("   Values        : JSON-compatible")
        print("   Determinism   : preserved")
        print("   Result        : PASS")
        print()

        print("[2/4] Audit exposes the complete fixed-schema JSON summary")
        audit_json = audit.to_json_summary(status)
        require(
            list(audit_json) == EXPECTED_KEYS,
            "audit JSON key order changed",
        )
        require(
            audit_json["metadata_json"]
            == str(audit.metadata_path.resolve()),
            "metadata path changed",
        )
        require(
            audit_json["fingerprint_json"]
            == str(audit.fingerprint_json_path.resolve()),
            "fingerprint path changed",
        )
        require(
            audit_json["comparison_json"]
            == str(audit.comparison_json_path.resolve()),
            "comparison path changed",
        )
        require(
            audit.to_json_summary() == audit_json,
            "automatic status evaluation changed the summary",
        )
        try:
            audit.to_json_summary(object())
        except step04.Step04Error:
            pass
        else:
            require(False, "invalid status was accepted")
        print("   Fixed schema  : complete")
        print("   Artifact paths: normalized")
        print("   Auto status   : equivalent")
        print("   Result        : PASS")
        print()

        print("[3/4] CLI --json emits only one parseable JSON object")
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = step04.run_rts_input_audit_cli(
            [str(artifacts.metadata_path), "--json"],
            stdout=stdout,
            stderr=stderr,
        )
        payload = json.loads(stdout.getvalue())
        require(exit_code == 0, "JSON MATCH did not return exit 0")
        require(payload == audit_json, "CLI JSON payload changed")
        require(stdout.getvalue().endswith("\n"), "missing final newline")
        require(stderr.getvalue() == "", "CLI JSON wrote to stderr")

        original = input_paths[0].read_bytes()
        input_paths[0].write_bytes(original + b"\x00")
        changed_stdout = io.StringIO()
        changed_exit = step04.run_rts_input_audit_cli(
            [str(artifacts.metadata_path), "--json"],
            stdout=changed_stdout,
            stderr=io.StringIO(),
        )
        changed_payload = json.loads(changed_stdout.getvalue())
        require(changed_exit == 1, "JSON CHANGED did not return exit 1")
        require(
            changed_payload["status"] == "CHANGED",
            "JSON CHANGED status missing",
        )
        require(
            changed_payload["changed_count"] == 1,
            "JSON changed count missing",
        )
        input_paths[0].write_bytes(original)
        print("   MATCH payload : valid")
        print("   CHANGED       : exit and count preserved")
        print("   Extra text    : none")
        print("   Result        : PASS")
        print()

        print("[4/4] JSON order is deterministic and quiet is exclusive")
        first = io.StringIO()
        second = io.StringIO()
        step04.run_rts_input_audit_cli(
            [str(artifacts.metadata_path), "--json"],
            stdout=first,
            stderr=io.StringIO(),
        )
        step04.run_rts_input_audit_cli(
            [str(artifacts.metadata_path), "--json"],
            stdout=second,
            stderr=io.StringIO(),
        )
        require(first.getvalue() == second.getvalue(), "JSON text changed")

        error_stderr = io.StringIO()
        try:
            step04.run_rts_input_audit_cli(
                [
                    str(artifacts.metadata_path),
                    "--json",
                    "--quiet",
                ],
                stdout=io.StringIO(),
                stderr=error_stderr,
            )
        except SystemExit as exc:
            require(exc.code == 2, "exclusive option exit code changed")
        else:
            require(False, "--json and --quiet were accepted together")
        require(
            "not allowed with argument" in error_stderr.getvalue(),
            "exclusive option error is missing",
        )
        print("   JSON text     : deterministic")
        print("   Key order     : stable")
        print("   quiet + JSON  : rejected")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 input-audit JSON test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
