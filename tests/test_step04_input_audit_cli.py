"""Integration test for the Step 04 input-audit CLI v4.33.0."""

from __future__ import annotations

import csv
import io
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
            "environment": "step04-v4.33-test",
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
    print("RTS Framework Step 04 input-audit CLI test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_cli_") as temp:
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
        metadata_path = artifacts.metadata_path

        print("[1/4] MATCH prints a complete report and returns exit 0")
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = step04.run_rts_input_audit_cli(
            [str(metadata_path)],
            stdout=stdout,
            stderr=stderr,
        )
        report = stdout.getvalue()
        require(exit_code == 0, "MATCH did not return exit 0")
        require(
            "RTS input audit: MATCH\n" in report,
            "MATCH status line is missing",
        )
        require(
            "Input audit passed: all expected files match.\n" in report,
            "MATCH message is missing",
        )
        require(
            "Fingerprint JSON :" in report
            and "Comparison JSON  :" in report,
            "artifact paths are missing",
        )
        require(stderr.getvalue() == "", "MATCH wrote to stderr")
        print("   Status line : present")
        print("   Paths       : present")
        print("   Exit code   : 0")
        print("   Result      : PASS")
        print()

        print("[2/4] CHANGED prints failure status and returns exit 1")
        original = input_paths[0].read_bytes()
        input_paths[0].write_bytes(original + b"\x00")
        stdout = io.StringIO()
        exit_code = step04.run_rts_input_audit_cli(
            [str(metadata_path)],
            stdout=stdout,
            stderr=io.StringIO(),
        )
        report = stdout.getvalue()
        require(exit_code == 1, "CHANGED did not return exit 1")
        require(
            "RTS input audit: CHANGED\n" in report,
            "CHANGED status line is missing",
        )
        require(
            "1 file changed" in report,
            "CHANGED count is missing",
        )
        input_paths[0].write_bytes(original)
        print("   Status line : CHANGED")
        print("   Changed     : detected")
        print("   Exit code   : 1")
        print("   Result      : PASS")
        print()

        print("[3/4] Custom artifact paths and quiet mode are supported")
        custom_fingerprint = root / "custom" / "baseline.json"
        custom_comparison = root / "custom" / "comparison.json"
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = step04.run_rts_input_audit_cli(
            [
                str(metadata_path),
                "--fingerprint-json",
                str(custom_fingerprint),
                "--comparison-json",
                str(custom_comparison),
                "--quiet",
            ],
            stdout=stdout,
            stderr=stderr,
        )
        require(exit_code == 0, "quiet MATCH did not return exit 0")
        require(custom_fingerprint.is_file(), "custom baseline missing")
        require(custom_comparison.is_file(), "custom comparison missing")
        require(stdout.getvalue() == "", "quiet mode wrote to stdout")
        require(stderr.getvalue() == "", "quiet mode wrote to stderr")
        print("   Custom paths : created")
        print("   Quiet output : empty")
        print("   Exit code    : preserved")
        print("   Result       : PASS")
        print()

        print("[4/4] Version, argument errors, and runtime errors are stable")
        version_stdout = io.StringIO()
        try:
            step04.run_rts_input_audit_cli(
                ["--version"],
                stdout=version_stdout,
                stderr=io.StringIO(),
            )
        except SystemExit as exc:
            require(exc.code == 0, "--version exit code changed")
        else:
            require(False, "--version did not raise SystemExit")
        require(
            version_stdout.getvalue()
            == f"rts-step04-input-audit {step04.__version__}\n",
            "--version output changed",
        )

        try:
            step04.run_rts_input_audit_cli(
                [],
                stdout=io.StringIO(),
                stderr=io.StringIO(),
            )
        except SystemExit as exc:
            require(exc.code == 2, "missing argument exit code changed")
        else:
            require(False, "missing metadata argument was accepted")

        runtime_stderr = io.StringIO()
        runtime_exit = step04.run_rts_input_audit_cli(
            [str(root / "does-not-exist.metadata.json")],
            stdout=io.StringIO(),
            stderr=runtime_stderr,
        )
        require(runtime_exit == 64, "runtime error code changed")
        require(
            runtime_stderr.getvalue().startswith(
                "RTS input audit error:"
            ),
            "runtime error message changed",
        )
        print("   Version       : deterministic")
        print("   Argument error: exit 2")
        print("   Runtime error : exit 64")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 input-audit CLI test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
