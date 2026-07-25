#!/usr/bin/env python3
"""Integration test for Step06 v6.0.0 numerical RTS quality assessment."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError
from io import StringIO
from pathlib import Path
import json
import sys
import tempfile

import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steps import step06_evaluate_rts_correction as step06


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_step06_error(function, text: str) -> None:
    try:
        function()
    except step06.Step06Error as exc:
        check(text.lower() in str(exc).lower(), f"Unexpected error: {exc}")
    else:
        raise AssertionError("Expected Step06Error")


def write_fits(path: Path, data: np.ndarray) -> None:
    fits.PrimaryHDU(data=data).writeto(path, overwrite=False)


def run_cli(arguments: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = step06.run_rts_evaluation_cli(arguments)
    return status, stdout.getvalue(), stderr.getvalue()


def main() -> None:
    print("=" * 72)
    print("RTS Framework Step 06 numerical evaluation test")
    print("=" * 72)
    print(f"step06 version : {step06.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step06_test_") as temp_name:
        root = Path(temp_name)
        original_path = root / "original.fits"
        corrected_path = root / "corrected.fits"

        original = np.array(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32
        )
        corrected = original.copy()
        corrected[0, 1] -= 4.0
        corrected[1, 2] += 2.0
        write_fits(original_path, original)
        write_fits(corrected_path, corrected)

        print("[1/4] Numerical comparison is correct and immutable")
        evaluation = step06.evaluate_rts_correction(
            original_path, corrected_path
        )
        expected_difference = corrected.astype(np.float64) - original
        check(evaluation.image_shape == (2, 3), "Unexpected shape")
        check(evaluation.changed_pixel_count == 2, "Changed count mismatch")
        check(evaluation.unchanged_pixel_count == 4, "Unchanged count mismatch")
        check(
            evaluation.changed_pixel_fraction == 2 / 6,
            "Changed fraction mismatch",
        )
        check(
            np.isclose(evaluation.difference_sum, expected_difference.sum()),
            "Signed sum mismatch",
        )
        check(
            np.isclose(
                evaluation.absolute_difference_sum,
                np.abs(expected_difference).sum(),
            ),
            "Absolute sum mismatch",
        )
        check(
            np.isclose(
                evaluation.difference_rms,
                np.sqrt(np.mean(expected_difference**2)),
            ),
            "RMS mismatch",
        )
        try:
            evaluation.changed_pixel_count = 99
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            raise AssertionError("Evaluation must be immutable")
        print("   Changed pixels: 2")
        print("   Signed/absolute: verified")
        print("   Result object  : immutable")
        print("   Result         : PASS")
        print()

        print("[2/4] Tolerance and invalid pairs are handled safely")
        tolerant = step06.evaluate_rts_correction(
            original_path, corrected_path, change_tolerance=2.0
        )
        check(tolerant.changed_pixel_count == 1, "Tolerance rule mismatch")
        shape_path = root / "shape.fits"
        write_fits(shape_path, np.zeros((3, 3), dtype=np.float32))
        expect_step06_error(
            lambda: step06.evaluate_rts_correction(original_path, shape_path),
            "shapes differ",
        )
        expect_step06_error(
            lambda: step06.evaluate_rts_correction(
                original_path, corrected_path, change_tolerance=-1
            ),
            "non-negative",
        )
        nan_path = root / "nan.fits"
        nan_data = corrected.copy()
        nan_data[0, 0] = np.nan
        write_fits(nan_path, nan_data)
        expect_step06_error(
            lambda: step06.evaluate_rts_correction(original_path, nan_path),
            "finite-pixel masks",
        )
        print("   Tolerance      : strict > threshold")
        print("   Shape mismatch : rejected")
        print("   Finite masks   : validated")
        print("   Result         : PASS")
        print()

        print("[3/4] JSON output is deterministic and protected")
        report_path = root / "evaluation.json"
        step06.write_rts_evaluation_json(evaluation, report_path)
        first_bytes = report_path.read_bytes()
        payload = json.loads(first_bytes)
        check(payload["changed_pixel_count"] == 2, "JSON count mismatch")
        expect_step06_error(
            lambda: step06.write_rts_evaluation_json(evaluation, report_path),
            "already exists",
        )
        step06.write_rts_evaluation_json(
            evaluation, report_path, overwrite=True
        )
        check(report_path.read_bytes() == first_bytes, "JSON is not deterministic")
        print("   JSON schema    : valid")
        print("   Existing file : protected")
        print("   Re-write      : byte-identical")
        print("   Result         : PASS")
        print()

        print("[4/4] CLI text, JSON, quiet mode, and failures are stable")
        common = [
            "--original",
            str(original_path),
            "--corrected",
            str(corrected_path),
        ]
        status1, stdout1, stderr1 = run_cli(common + ["--json"])
        status2, stdout2, stderr2 = run_cli(common + ["--json"])
        check(status1 == status2 == 0, "JSON CLI should succeed")
        check(stdout1 == stdout2, "CLI JSON should be deterministic")
        check(stderr1 == stderr2 == "", "Unexpected CLI stderr")
        json.loads(stdout1)

        quiet_report = root / "quiet.json"
        status, stdout, stderr = run_cli(
            common + ["--quiet", "--output-json", str(quiet_report)]
        )
        check(status == 0, "Quiet CLI should succeed")
        check(stdout == stderr == "", "Quiet CLI must be silent")
        check(quiet_report.is_file(), "Quiet report missing")

        status, stdout, stderr = run_cli(
            [
                "--original",
                str(original_path),
                "--corrected",
                str(shape_path),
                "--json",
            ]
        )
        check(status == 1, "Invalid CLI should return 1")
        check(stdout == "", "Invalid CLI should not write stdout")
        check("Step06 error:" in stderr, "Missing CLI error message")
        print("   JSON output   : deterministic")
        print("   Quiet mode    : silent")
        print("   Invalid exit  : 1")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 06 numerical evaluation test passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
