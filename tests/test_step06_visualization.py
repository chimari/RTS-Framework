#!/usr/bin/env python3
"""Integration test for Step06 v6.1.0 deterministic PNG visualization."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError
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


def output_paths(outputs) -> list[Path]:
    return [
        outputs.difference_image,
        outputs.correction_map,
        outputs.original_histogram,
        outputs.corrected_histogram,
        outputs.difference_histogram,
    ]


def main() -> None:
    print("=" * 72)
    print("RTS Framework Step 06 visualization test")
    print("=" * 72)
    print(f"step06 version : {step06.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step06_plot_test_") as temp_name:
        root = Path(temp_name)
        original_path = root / "original.fits"
        corrected_path = root / "corrected.fits"

        y, x = np.mgrid[:32, :40]
        original = (1000.0 + 0.25 * x + 0.5 * y).astype(np.float32)
        corrected = original.copy()
        corrected[4, 5] -= 12.0
        corrected[12, 21] += 8.0
        corrected[24, 31] -= 4.0
        write_fits(original_path, original)
        write_fits(corrected_path, corrected)

        print("[1/4] Five diagnostic PNG files are generated")
        plot_dir = root / "plots"
        plot_dir.mkdir()
        outputs = step06.generate_rts_evaluation_plots(
            original_path,
            corrected_path,
            plot_dir,
            change_tolerance=1.0,
            bins=32,
            dpi=80,
        )
        paths = output_paths(outputs)
        check(len(paths) == 5, "Expected five outputs")
        check(len({path.name for path in paths}) == 5, "Output names differ")
        for path in paths:
            check(path.is_file(), f"Missing plot: {path}")
            check(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), "Not PNG")
            check(path.stat().st_size > 1000, f"Plot unexpectedly small: {path}")
        try:
            outputs.difference_image = root / "changed.png"
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            raise AssertionError("Plot output result must be immutable")
        print("   PNG outputs   : 5")
        print("   File format   : valid")
        print("   Result object : immutable")
        print("   Result        : PASS")
        print()

        print("[2/4] Existing outputs and invalid options are rejected")
        first_bytes = {path.name: path.read_bytes() for path in paths}
        expect_step06_error(
            lambda: step06.generate_rts_evaluation_plots(
                original_path, corrected_path, plot_dir
            ),
            "already exists",
        )
        check(
            all(path.read_bytes() == first_bytes[path.name] for path in paths),
            "Protected plots changed",
        )
        expect_step06_error(
            lambda: step06.generate_rts_evaluation_plots(
                original_path, corrected_path, root / "missing"
            ),
            "does not exist",
        )
        expect_step06_error(
            lambda: step06.generate_rts_evaluation_plots(
                original_path, corrected_path, plot_dir, bins=1, overwrite=True
            ),
            "bins",
        )
        expect_step06_error(
            lambda: step06.generate_rts_evaluation_plots(
                original_path, corrected_path, plot_dir, dpi=20, overwrite=True
            ),
            "dpi",
        )
        print("   Existing files: protected")
        print("   Missing dir   : rejected")
        print("   Plot options  : validated")
        print("   Result        : PASS")
        print()

        print("[3/4] Overwrite regeneration is deterministic")
        regenerated = step06.generate_rts_evaluation_plots(
            original_path,
            corrected_path,
            plot_dir,
            change_tolerance=1.0,
            bins=32,
            dpi=80,
            overwrite=True,
        )
        regenerated_paths = output_paths(regenerated)
        for path in regenerated_paths:
            check(
                path.read_bytes() == first_bytes[path.name],
                f"Plot is not byte-deterministic: {path.name}",
            )
        print("   Regeneration  : successful")
        print("   PNG bytes     : identical")
        print("   Output names  : stable")
        print("   Result        : PASS")
        print()

        print("[4/4] CLI plot generation and quiet mode are stable")
        cli_dir = root / "cli_plots"
        cli_dir.mkdir()
        common = [
            "--original", str(original_path),
            "--corrected", str(corrected_path),
            "--plot-directory", str(cli_dir),
            "--histogram-bins", "24",
            "--plot-dpi", "80",
        ]
        status, stdout, stderr = run_cli(common + ["--quiet"])
        check(status == 0, "Quiet CLI plot generation should succeed")
        check(stdout == stderr == "", "Quiet CLI must be silent")
        check(len(list(cli_dir.glob("*.png"))) == 5, "CLI plots missing")

        status, stdout, stderr = run_cli(common + ["--quiet"])
        check(status == 1, "Existing CLI plot outputs should fail")
        check(stdout == "", "Failed CLI should not write stdout")
        check("already exists" in stderr.lower(), "Missing overwrite error")

        status, stdout, stderr = run_cli(common + ["--quiet", "--overwrite"])
        check(status == 0, "CLI overwrite should succeed")
        check(stdout == stderr == "", "Quiet overwrite must be silent")
        print("   CLI plots     : 5")
        print("   Quiet mode    : silent")
        print("   Overwrite     : enforced")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 06 visualization test passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
