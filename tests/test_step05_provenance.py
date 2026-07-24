"""Integration test for Step 05 provenance v5.9.0."""

from __future__ import annotations

import csv
import hashlib
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(root: Path) -> Path:
    files = []
    for index in range(8):
        data = np.array([
            [0 if index % 4 < 2 else 10, 20, 30],
            [10 if index % 4 < 2 else 0, 5, 50],
        ], dtype=np.uint16)
        path = root / f"bias_{index:04d}.fit"
        fits.PrimaryHDU(data=data).writeto(path)
        files.append(path)

    rows = []
    for index, path in enumerate(files):
        rows.append({
            "dataset": "bias",
            "directory": str(root),
            "environment": "step05-v5.9-test",
            "frame_index": index,
            "n_frames": len(files),
            "temperature_C": -10.0,
            "temperature_start_C": -10.0,
            "temperature_end_C": -10.0,
            "temperature_fraction": index / (len(files) - 1),
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


def prepare(root: Path):
    analysis = step03.prepare_bias_analysis(source_manifest(root), "bias")
    built = step04.build_rts_dictionary_artifacts(
        step04.prepare_rts_dictionary_analysis(analysis),
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
        path = root / f"target_{image_index}.fit"
        data = np.full((2, 3), 20 + image_index, dtype=np.float32)
        fits.PrimaryHDU(data=data).writeto(path)
        plan = step05.prepare_rts_correction(built.metadata_path, path)
        for index, candidate in enumerate(plan.candidates):
            data[candidate.row, candidate.column] = (
                candidate.lower_state_center
                + 0.1 * candidate.state_separation
                if (index + image_index) % 2 == 0
                else candidate.upper_state_center
                - 0.1 * candidate.state_separation
            )
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        inputs.append(path)

    invalid = root / "invalid.fit"
    fits.PrimaryHDU(data=np.zeros((3, 3))).writeto(invalid)
    return built.metadata_path, built.output_path, tuple(inputs), invalid


def call_cli(args):
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = step05.run_rts_correction_batch_cli(args)
    return code, stdout.getvalue(), stderr.getvalue()


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 05 provenance test")
    print("=" * 72)
    print(f"step05 version : {step05.__version__}")
    print()

    fixed_time = "2026-07-25T00:00:00Z"

    with tempfile.TemporaryDirectory(prefix="rts_step05_provenance_") as temp:
        root = Path(temp)
        metadata, dictionary, inputs, invalid = prepare(root)

        output_dir = root / "outputs"
        output_dir.mkdir()
        result = step05.run_rts_correction_batch(
            metadata, inputs, output_dir
        )

        print("[1/4] Environment and configuration are recorded")
        provenance = root / "provenance.json"
        step05.write_rts_batch_provenance_json(
            result,
            provenance,
            execution_time_utc=fixed_time,
            cli_arguments=("--input", "example.fit"),
        )
        payload = json.loads(provenance.read_text(encoding="utf-8"))
        require(payload["provenance_version"] == 1, "version missing")
        require(payload["execution_time_utc"] == fixed_time, "time changed")
        require(payload["software"]["step05_version"] == step05.__version__,
                "Step05 version missing")
        require(payload["software"]["python_version"], "Python missing")
        require(payload["software"]["numpy_version"], "NumPy missing")
        require(payload["software"]["astropy_version"], "Astropy missing")
        require(payload["platform"]["system"], "platform missing")
        require(payload["configuration"]["state_tolerance_fraction"] == 0.25,
                "tolerance changed")
        require(payload["configuration"]["cli_arguments"]
                == ["--input", "example.fit"], "arguments changed")
        print("   Software      : recorded")
        print("   Platform      : recorded")
        print("   Configuration : recorded")
        print("   Result        : PASS")
        print()

        print("[2/4] Artifact and FITS SHA256 values are exact")
        require(payload["artifacts"]["metadata_sha256"] == sha256(metadata),
                "metadata SHA256 changed")
        require(payload["artifacts"]["dictionary_sha256"] == sha256(dictionary),
                "dictionary SHA256 changed")
        for index, item in enumerate(payload["items"]):
            require(item["input_sha256"] == sha256(inputs[index]),
                    "input SHA256 changed")
            require(item["output_sha256"]
                    == sha256(Path(item["output_path"])),
                    "output SHA256 changed")
            require(item["output_verified"] is True, "verification lost")
        print("   Metadata      : exact")
        print("   Dictionary    : exact")
        print("   Inputs/outputs: exact")
        print("   Result        : PASS")
        print()

        print("[3/4] Partial provenance preserves failure state")
        partial_dir = root / "partial_outputs"
        partial_dir.mkdir()
        partial = step05.run_rts_correction_batch(
            metadata,
            [inputs[0], invalid, inputs[1]],
            partial_dir,
            continue_on_error=True,
        )
        partial_path = root / "partial_provenance.json"
        step05.write_rts_batch_provenance_json(
            partial,
            partial_path,
            execution_time_utc=fixed_time,
        )
        partial_payload = json.loads(
            partial_path.read_text(encoding="utf-8")
        )
        failed = partial_payload["items"][1]
        require(partial_payload["summary"]["failed_count"] == 1,
                "failure count changed")
        require(failed["succeeded"] is False, "failed status lost")
        require(failed["error"], "failure message lost")
        require(failed["output_sha256"] is None, "failed output SHA not null")
        require(failed["output_verified"] is False,
                "failed verification changed")
        print("   Successes     : 2")
        print("   Failures      : 1")
        print("   Failure detail: preserved")
        print("   Result        : PASS")
        print()

        print("[4/4] CLI output and deterministic rewrite are protected")
        cli_dir = root / "cli_outputs"
        cli_dir.mkdir()
        cli_provenance = root / "cli_provenance.json"
        args = [
            "--metadata", str(metadata),
            "--input", str(inputs[0]),
            "--input", str(inputs[1]),
            "--output-directory", str(cli_dir),
            "--provenance-json", str(cli_provenance),
            "--json",
        ]
        code, stdout, stderr = call_cli(args)
        require(code == 0, f"CLI exit changed: {code}")
        require(stderr == "", f"unexpected stderr: {stderr}")
        require(cli_provenance.exists(), "CLI provenance missing")
        require(json.loads(stdout)["status"] == "OK", "CLI status changed")
        cli_payload = json.loads(cli_provenance.read_text(encoding="utf-8"))
        require(cli_payload["configuration"]["cli_arguments"] == args,
                "CLI arguments not preserved")

        deterministic = root / "deterministic.json"
        step05.write_rts_batch_provenance_json(
            result,
            deterministic,
            execution_time_utc=fixed_time,
        )
        first = deterministic.read_bytes()
        step05.write_rts_batch_provenance_json(
            result,
            deterministic,
            execution_time_utc=fixed_time,
            overwrite=True,
        )
        require(first == deterministic.read_bytes(),
                "fixed-time rewrite is not byte-identical")

        protected_dir = root / "protected_outputs"
        protected_dir.mkdir()
        code, stdout, stderr = call_cli([
            "--metadata", str(metadata),
            "--input", str(inputs[0]),
            "--output-directory", str(protected_dir),
            "--provenance-json", str(cli_provenance),
        ])
        require(code == 1, "existing provenance was overwritten")
        require("ERROR:" in stderr, "protection error missing")
        print("   CLI provenance: written")
        print("   Rewrite       : byte-identical")
        print("   Existing file : protected")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 05 provenance test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
