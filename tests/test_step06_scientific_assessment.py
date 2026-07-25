#!/usr/bin/env python3
"""Integration test for Step06 v6.3.0 scientific assessment."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
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


def write_fits(path: Path, data: np.ndarray) -> None:
    fits.PrimaryHDU(data=data).writeto(path, overwrite=False)


def run_cli(arguments: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = step06.run_rts_evaluation_cli(arguments)
    return status, stdout.getvalue(), stderr.getvalue()


def pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def synthetic_evaluation(*, ratio: float | None, shift_sigma: float, changed_fraction: float = 0.01):
    original_std = 10.0
    corrected_std = original_std if ratio is None else original_std * ratio
    return step06.RTSCorrectionEvaluation(
        original_path=Path("/original.fits"), corrected_path=Path("/corrected.fits"),
        image_shape=(10, 10), original_dtype="float32", corrected_dtype="float32",
        pixel_count=100, finite_pixel_count=100,
        changed_pixel_count=int(round(100 * changed_fraction)),
        changed_pixel_fraction=changed_fraction, change_tolerance=0.0,
        difference_sum=shift_sigma * original_std * 100,
        difference_mean=shift_sigma * original_std,
        difference_median=0.0, difference_minimum=-1.0, difference_maximum=1.0,
        absolute_difference_sum=1.0, absolute_difference_mean=0.01,
        absolute_difference_median=0.0, absolute_difference_maximum=1.0,
        difference_rms=0.1, original_mean=1000.0,
        corrected_mean=1000.0 + shift_sigma * original_std,
        original_std=original_std, corrected_std=corrected_std,
        noise_std_change=corrected_std - original_std, noise_std_ratio=ratio,
    )


def main() -> None:
    print("=" * 72)
    print("RTS Framework Step 06 scientific assessment test")
    print("=" * 72)
    print(f"step06 version : {step06.__version__}")
    print()

    print("[1/4] Fixed grading thresholds are stable")
    cases = [
        (0.85, 0.02, "Excellent"),
        (0.95, 0.08, "Good"),
        (1.03, 0.20, "Acceptable"),
        (1.08, 0.30, "Warning"),
    ]
    for ratio, shift, expected in cases:
        result = step06.assess_rts_correction(
            synthetic_evaluation(ratio=ratio, shift_sigma=shift)
        )
        check(result.grade == expected, f"Expected {expected}, got {result.grade}")
    print("   Grades         : Excellent / Good / Acceptable / Warning")
    print("   Thresholds     : deterministic")
    print("   Result         : PASS")
    print()

    print("[2/4] Concerns and summary are deterministic")
    warning = step06.assess_rts_correction(
        synthetic_evaluation(ratio=1.08, shift_sigma=0.30, changed_fraction=0.20)
    )
    check(len(warning.concerns) >= 3, "Expected multiple warning concerns")
    summary = warning.summary()
    check(summary["grade"] == "Warning", "Assessment summary grade mismatch")
    check(summary["criteria_version"] == "1.0", "Criteria version mismatch")
    check(isinstance(summary["concerns"], list), "Concerns must serialize as a list")
    print("   Concern rules  : stable")
    print("   JSON summary   : serializable")
    print("   Result         : PASS")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step06_assessment_test_") as temp_name:
        root = Path(temp_name)
        original_path = root / "original.fits"
        corrected_path = root / "corrected.fits"
        rng = np.random.default_rng(12345)
        original = rng.normal(1000.0, 10.0, size=(48, 64)).astype(np.float32)
        corrected = (1000.0 + 0.85 * (original - 1000.0)).astype(np.float32)
        write_fits(original_path, original)
        write_fits(corrected_path, corrected)

        print("[3/4] PDF contains scientific assessment and caveat")
        report_path = root / "scientific_report.pdf"
        step06.generate_rts_evaluation_pdf(
            original_path, corrected_path, report_path,
            change_tolerance=0.01, bins=32, plot_dpi=80,
        )
        text = pdf_text(report_path)
        for required in (
            "Automatic scientific assessment",
            "Grade",
            "Excellent",
            "Automatic concerns",
            "does not replace inspection",
        ):
            check(required in text, f"Missing PDF text: {required}")
        print("   Assessment     : embedded")
        print("   Concerns       : embedded")
        print("   Scientific note: embedded")
        print("   Result         : PASS")
        print()

        print("[4/4] CLI text exposes the assessment without changing JSON")
        status, stdout, stderr = run_cli([
            "--original", str(original_path),
            "--corrected", str(corrected_path),
            "--change-tolerance", "0.01",
        ])
        check(status == 0 and stderr == "", "Text CLI should succeed")
        check("Assessment grade" in stdout, "Text CLI missing assessment")

        status, stdout, stderr = run_cli([
            "--original", str(original_path),
            "--corrected", str(corrected_path),
            "--change-tolerance", "0.01",
            "--json",
        ])
        check(status == 0 and stderr == "", "JSON CLI should succeed")
        check('"scientific_assessment"' not in stdout, "Legacy JSON schema changed")
        print("   Text output    : assessment included")
        print("   JSON schema    : backward compatible")
        print("   Result         : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 06 scientific assessment test passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
