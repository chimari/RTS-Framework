"""Integration test for Step 04 deterministic two-state score v4.3.0."""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from dataclasses import FrozenInstanceError, replace
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


def require_close(
    actual: float,
    expected: float,
    message: str,
    tolerance: float = 1.0e-12,
) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        print(f"FAIL: {message}: actual={actual!r}, expected={expected!r}")
        raise SystemExit(1)


def write_dataset(root: Path, values: list[int]) -> Path:
    paths: list[Path] = []
    for index, value in enumerate(values):
        path = root / f"bias_{index:04d}.fit"
        data = np.array(
            [
                [value, 100 + index],
                [200 + index, 300 + index],
            ],
            dtype=np.uint16,
        )
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows: list[dict[str, object]] = []
    n_frames = len(paths)
    for frame_index, path in enumerate(paths):
        rows.append(
            {
                "dataset": "bias",
                "directory": str(root),
                "environment": "step04-v4.3-test",
                "frame_index": frame_index,
                "n_frames": n_frames,
                "temperature_C": -10.0,
                "temperature_start_C": -10.0,
                "temperature_end_C": -10.0,
                "temperature_fraction": frame_index / (n_frames - 1),
                "exposure_s": 0.0,
                "filename": path.name,
                "filepath": str(path),
                "image_width": 2,
                "image_height": 2,
                "pixel_dtype": "uint16",
                "byte_order": "not-applicable",
            }
        )

    manifest = root / "manifest.normalized.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def expect_step04_error(callable_, contains: str) -> None:
    try:
        callable_()
    except step04.Step04Error as exc:
        require(contains in str(exc), f"wrong error message: {exc}")
    else:
        require(False, "Step04Error was not raised")


def readonly(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array.setflags(write=False)
    return array


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 two-state score integration test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_score_") as temp_dir:
        root = Path(temp_dir)
        manifest = write_dataset(root, [0, 1, 10, 11])
        bias_plan = step03.prepare_bias_analysis(manifest, "bias")
        plan = step04.prepare_rts_dictionary_analysis(bias_plan)
        series = step04.load_pixel_timeseries(plan, row=0, column=0)

        print("[1/4] Exact two-state fit finds the known optimal split")
        result = step04.compute_two_state_score(series)
        require(result.series is series, "source series was not retained")
        require(result.n_frames == 4, "wrong frame count")
        require(result.lower_state_count == 2, "wrong lower-state count")
        require(result.upper_state_count == 2, "wrong upper-state count")
        require_close(result.lower_state_center, 0.5, "wrong lower center")
        require_close(result.upper_state_center, 10.5, "wrong upper center")
        require_close(result.state_separation, 10.0, "wrong separation")
        require_close(result.single_state_residual, 101.0, "wrong single SSE")
        require_close(result.two_state_residual, 1.0, "wrong two-state SSE")
        require_close(result.score, 100.0 / 101.0, "wrong score")
        print("   Values          : [0, 1, 10, 11]")
        print("   Centers         : 0.5, 10.5")
        print("   Residuals       : 101.0 -> 1.0")
        print("   Score           : 100/101")
        print("   Result          : PASS")
        print()

        print("[2/4] Input order does not change the deterministic result")
        shuffled = replace(
            series,
            values=readonly([10.0, 0.0, 11.0, 1.0]),
        )
        repeated = step04.compute_two_state_score(shuffled)
        require_close(
            repeated.lower_state_center,
            result.lower_state_center,
            "lower center depends on input order",
        )
        require_close(
            repeated.upper_state_center,
            result.upper_state_center,
            "upper center depends on input order",
        )
        require_close(
            repeated.two_state_residual,
            result.two_state_residual,
            "residual depends on input order",
        )
        require_close(
            repeated.score,
            result.score,
            "score depends on input order",
        )

        constant = replace(series, values=readonly([5.0, 5.0, 5.0, 5.0]))
        constant_result = step04.compute_two_state_score(constant)
        require_close(
            constant_result.single_state_residual,
            0.0,
            "constant single-state residual is not zero",
        )
        require_close(
            constant_result.two_state_residual,
            0.0,
            "constant two-state residual is not zero",
        )
        require_close(
            constant_result.score,
            0.0,
            "constant score is not zero",
        )
        require(
            constant_result.lower_state_count == 1,
            "equal-residual tie did not retain the first split",
        )
        print("   Order invariant : YES")
        print("   Constant score  : 0.0")
        print("   Tie rule        : smallest split index")
        print("   Result          : PASS")
        print()

        print("[3/4] Result and summary are immutable and deterministic")
        expected_summary = {
            "dataset": "bias",
            "row": 0,
            "column": 0,
            "n_frames": 4,
            "lower_state_count": 2,
            "upper_state_count": 2,
            "lower_state_center": 0.5,
            "upper_state_center": 10.5,
            "state_separation": 10.0,
            "single_state_residual": 101.0,
            "two_state_residual": 1.0,
            "score": 100.0 / 101.0,
        }
        require(result.summary() == expected_summary, "summary content changed")
        require(result.summary() == result.summary(), "summary is not deterministic")
        try:
            result.score = 0.0
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "frozen score result accepted modification")
        print("   Frozen        : YES")
        print("   Deterministic : YES")
        print("   JSON-ready    : YES")
        print("   Result        : PASS")
        print()

        print("[4/4] Invalid, short, inconsistent, and non-finite inputs fail")
        expect_step04_error(
            lambda: step04.compute_two_state_score(object()),
            "series must be a PixelTimeSeries",
        )

        short = replace(
            series,
            values=readonly([1.0, 2.0]),
            n_frames=2,
        )
        expect_step04_error(
            lambda: step04.compute_two_state_score(short),
            "requires at least 3",
        )

        two_dimensional = np.zeros((2, 2), dtype=np.float64)
        two_dimensional.setflags(write=False)
        expect_step04_error(
            lambda: step04.compute_two_state_score(
                replace(series, values=two_dimensional)
            ),
            "must be one-dimensional",
        )

        mismatched = replace(
            series,
            values=readonly([1.0, 2.0, 3.0]),
        )
        expect_step04_error(
            lambda: step04.compute_two_state_score(mismatched),
            "metadata requires 4",
        )

        non_finite = replace(
            series,
            values=readonly([1.0, np.inf, 3.0, 4.0]),
        )
        expect_step04_error(
            lambda: step04.compute_two_state_score(non_finite),
            "must all be finite",
        )
        print("   Invalid type      : rejected")
        print("   Fewer than 3      : rejected")
        print("   Shape/count error : rejected")
        print("   NaN/Inf           : rejected")
        print("   Result            : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 two-state score integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
