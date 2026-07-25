#!/usr/bin/env python3
"""Integration test for Step06 v6.4.0 mask-based science metrics."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
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


def main() -> None:
    print("=" * 72)
    print("RTS Framework Step 06 science metrics test")
    print("=" * 72)
    print(f"step06 version : {step06.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step06_science_metrics_") as temp_name:
        root = Path(temp_name)
        original_path = root / "original.fits"
        corrected_path = root / "corrected.fits"
        mask_path = root / "rts_mask.fits"

        original = np.zeros((4, 5), dtype=np.float32)
        corrected = original.copy()
        corrected[0, 0] = 1.0
        corrected[0, 1] = 1.0
        corrected[3, 4] = 1.0
        corrected[1, 4] = 1.0  # deliberately outside the RTS mask

        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[0, 0] = 1
        mask[0, 1] = 1
        mask[2, 2] = 1  # residual candidate: not changed
        mask[3, 4] = 1

        write_fits(original_path, original)
        write_fits(corrected_path, corrected)
        write_fits(mask_path, mask)

        print("[1/4] Mask-based fractions and cluster statistics are exact")
        metrics = step06.calculate_rts_science_metrics(
            original_path, corrected_path, mask_path, change_tolerance=0.0
        )
        check(metrics.reference_pixel_count == 4, "Reference count mismatch")
        check(metrics.rts_pixel_fraction == 0.20, "RTS fraction mismatch")
        check(metrics.changed_pixel_count == 4, "Changed count mismatch")
        check(metrics.changed_reference_pixel_count == 3, "Changed reference mismatch")
        check(metrics.correction_coverage_fraction == 0.75, "Coverage mismatch")
        check(metrics.residual_candidate_count == 1, "Residual count mismatch")
        check(metrics.residual_candidate_fraction == 0.25, "Residual fraction mismatch")
        check(metrics.off_mask_changed_pixel_count == 1, "Off-mask count mismatch")
        check(metrics.off_mask_changed_fraction == 0.05, "Off-mask fraction mismatch")
        check(metrics.correction_selectivity_fraction == 0.75, "Selectivity mismatch")
        check(metrics.reference_cluster_count == 3, "Reference cluster count mismatch")
        check(metrics.largest_reference_cluster_size == 2, "Largest cluster mismatch")
        check(metrics.changed_reference_cluster_count == 2, "Changed cluster count mismatch")
        check(metrics.largest_changed_reference_cluster_size == 2, "Largest changed cluster mismatch")
        print("   RTS fraction   : exact")
        print("   Coverage       : exact")
        print("   Residual proxy : explicitly candidate-only")
        print("   Clusters       : deterministic 8-connectivity")
        print("   Result         : PASS")
        print()

        print("[2/4] Mask validation and immutable JSON summary are stable")
        summary = metrics.summary()
        check(summary["definition_version"] == "1.0", "Definition version mismatch")
        check("candidates only" in summary["residual_definition"], "Residual caveat missing")
        check(isinstance(summary["image_shape"], list), "Image shape must serialize as list")

        zero_mask_path = root / "zero_mask.fits"
        bad_shape_path = root / "bad_shape.fits"
        write_fits(zero_mask_path, np.zeros((4, 5), dtype=np.uint8))
        write_fits(bad_shape_path, np.ones((2, 2), dtype=np.uint8))
        for invalid_path in (zero_mask_path, bad_shape_path):
            try:
                step06.calculate_rts_science_metrics(
                    original_path, corrected_path, invalid_path
                )
            except step06.Step06Error:
                pass
            else:
                raise AssertionError(f"Invalid mask was accepted: {invalid_path}")
        print("   Summary        : serializable")
        print("   Zero mask      : rejected")
        print("   Shape mismatch : rejected")
        print("   Result         : PASS")
        print()

        print("[3/4] Science JSON and PDF report include mask-based metrics")
        json_path = root / "science_metrics.json"
        step06.write_rts_science_metrics_json(metrics, json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        check(payload["correction_coverage_fraction"] == 0.75, "Science JSON mismatch")
        check(payload["reference_cluster_count"] == 3, "Cluster JSON mismatch")

        pdf_path = root / "science_report.pdf"
        step06.generate_rts_evaluation_pdf(
            original_path,
            corrected_path,
            pdf_path,
            change_tolerance=0.0,
            bins=16,
            plot_dpi=80,
            rts_mask_path=mask_path,
        )
        text = pdf_text(pdf_path)
        for required in (
            "Mask-based science metrics",
            "RTS pixel fraction",
            "Correction coverage",
            "Residual candidates",
            "not proof of residual RTS behavior",
        ):
            check(required in text, f"Missing PDF text: {required}")
        print("   Science JSON   : deterministic")
        print("   PDF metrics    : embedded")
        print("   Caveat         : embedded")
        print("   Result         : PASS")
        print()

        print("[4/4] CLI exposes optional science metrics without changing legacy JSON")
        cli_json = root / "cli_science.json"
        status, stdout, stderr = run_cli([
            "--original", str(original_path),
            "--corrected", str(corrected_path),
            "--rts-mask", str(mask_path),
            "--science-json", str(cli_json),
        ])
        check(status == 0 and stderr == "", "Science CLI should succeed")
        check("Correction coverage" in stdout, "Text CLI missing science metrics")
        check(cli_json.is_file(), "CLI science JSON missing")

        status, stdout, stderr = run_cli([
            "--original", str(original_path),
            "--corrected", str(corrected_path),
            "--rts-mask", str(mask_path),
            "--json",
        ])
        check(status == 0 and stderr == "", "Legacy JSON CLI should succeed")
        check('"rts_pixel_fraction"' not in stdout, "Legacy evaluation JSON schema changed")

        status, stdout, stderr = run_cli([
            "--original", str(original_path),
            "--corrected", str(corrected_path),
            "--science-json", str(root / "invalid.json"),
        ])
        check(status == 1, "--science-json without --rts-mask must fail")
        print("   Text output    : science metrics included")
        print("   Legacy JSON    : backward compatible")
        print("   CLI dependency : enforced")
        print("   Result         : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 06 science metrics test passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
