#!/usr/bin/env python3
"""Integration test for Step06 v6.2.0 deterministic PDF report generation."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import hashlib
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


def expect_step06_error(function, text: str) -> None:
    try:
        function()
    except step06.Step06Error as exc:
        check(text.lower() in str(exc).lower(), f"Unexpected error: {exc}")
    else:
        raise AssertionError("Expected Step06Error")


def run_cli(arguments: list[str]) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = step06.run_rts_evaluation_cli(arguments)
    return status, stdout.getvalue(), stderr.getvalue()


def pdf_page_count(path: Path) -> int:
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    return len(PdfReader(str(path)).pages)


def main() -> None:
    print("=" * 72)
    print("RTS Framework Step 06 PDF report test")
    print("=" * 72)
    print(f"step06 version : {step06.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step06_pdf_test_") as temp_name:
        root = Path(temp_name)
        original_path = root / "original.fits"
        corrected_path = root / "corrected.fits"

        y, x = np.mgrid[:48, :64]
        original = (1200.0 + 0.1 * x + 0.25 * y).astype(np.float32)
        corrected = original.copy()
        corrected[5, 6] -= 15.0
        corrected[16, 24] += 9.0
        corrected[33, 51] -= 6.0
        write_fits(original_path, original)
        write_fits(corrected_path, corrected)

        print("[1/4] Multi-page PDF report is generated")
        report_path = root / "evaluation.pdf"
        result = step06.generate_rts_evaluation_pdf(
            original_path,
            corrected_path,
            report_path,
            change_tolerance=1.0,
            bins=32,
            plot_dpi=80,
        )
        check(result == report_path.resolve(), "Unexpected returned PDF path")
        check(report_path.is_file(), "PDF report missing")
        check(report_path.read_bytes().startswith(b"%PDF-"), "Invalid PDF header")
        check(report_path.stat().st_size > 10000, "PDF report unexpectedly small")
        pages = pdf_page_count(report_path)
        check(pages >= 4, f"Expected at least four pages, got {pages}")
        check(not list(root.glob("*.png")), "Temporary PNG files leaked")
        print(f"   PDF pages      : {pages}")
        print("   File format    : valid")
        print("   Temporary plots: cleaned")
        print("   Result         : PASS")
        print()

        print("[2/4] Existing output and invalid paths are rejected")
        original_bytes = report_path.read_bytes()
        expect_step06_error(
            lambda: step06.generate_rts_evaluation_pdf(
                original_path, corrected_path, report_path
            ),
            "already exists",
        )
        check(report_path.read_bytes() == original_bytes, "Protected PDF changed")
        expect_step06_error(
            lambda: step06.generate_rts_evaluation_pdf(
                original_path, corrected_path, root / "report.txt"
            ),
            "end with .pdf",
        )
        expect_step06_error(
            lambda: step06.generate_rts_evaluation_pdf(
                original_path, corrected_path, root / "missing" / "report.pdf"
            ),
            "does not exist",
        )
        print("   Existing PDF   : protected")
        print("   Extension      : validated")
        print("   Missing dir    : rejected")
        print("   Result         : PASS")
        print()

        print("[3/4] Overwrite regeneration is deterministic")
        first_digest = hashlib.sha256(original_bytes).hexdigest()
        step06.generate_rts_evaluation_pdf(
            original_path,
            corrected_path,
            report_path,
            change_tolerance=1.0,
            bins=32,
            plot_dpi=80,
            overwrite=True,
        )
        second_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
        check(first_digest == second_digest, "PDF bytes are not deterministic")
        print("   Regeneration   : successful")
        print("   PDF bytes      : identical")
        print("   Result         : PASS")
        print()

        print("[4/4] CLI PDF generation and quiet mode are stable")
        cli_path = root / "cli_report.pdf"
        common = [
            "--original", str(original_path),
            "--corrected", str(corrected_path),
            "--output-pdf", str(cli_path),
            "--histogram-bins", "24",
            "--plot-dpi", "80",
            "--quiet",
        ]
        status, stdout, stderr = run_cli(common)
        check(status == 0, "Quiet CLI PDF generation should succeed")
        check(stdout == stderr == "", "Quiet CLI must be silent")
        check(cli_path.is_file(), "CLI PDF missing")

        status, stdout, stderr = run_cli(common)
        check(status == 1, "Existing CLI PDF should fail")
        check(stdout == "", "Failed CLI should not write stdout")
        check("already exists" in stderr.lower(), "Missing overwrite error")

        status, stdout, stderr = run_cli(common + ["--overwrite"])
        check(status == 0, "CLI overwrite should succeed")
        check(stdout == stderr == "", "Quiet overwrite must be silent")
        print("   CLI PDF        : generated")
        print("   Quiet mode     : silent")
        print("   Overwrite      : enforced")
        print("   Result         : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 06 PDF report test passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
