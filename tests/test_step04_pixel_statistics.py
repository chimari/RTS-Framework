"""Integration test for Step 04 pixel time-series statistics v4.2.0."""

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


def require_close(actual: float, expected: float, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
        print(f"FAIL: {message}: actual={actual!r}, expected={expected!r}")
        raise SystemExit(1)


def write_dataset(root: Path, values: list[int]) -> Path:
    paths: list[Path] = []
    for index, value in enumerate(values):
        path = root / f"bias_{index:04d}.fit"
        data = np.array(
            [[100 + index, value], [200 + index, 300 + index]],
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
                "environment": "step04-v4.2-test",
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


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 pixel statistics integration test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_stats_") as temp_dir:
        root = Path(temp_dir)
        manifest = write_dataset(root, [1, 2, 4, 8])
        bias_plan = step03.prepare_bias_analysis(manifest, "bias")
        plan = step04.prepare_rts_dictionary_analysis(bias_plan)
        series = step04.load_pixel_timeseries(plan, row=0, column=1)

        print("[1/4] Canonical statistics are computed exactly")
        result = step04.compute_pixel_timeseries_statistics(series)
        expected_values = np.array([1.0, 2.0, 4.0, 8.0])
        require(result.series is series, "source series was not retained")
        require(result.n_frames == 4, "wrong frame count")
        require_close(result.minimum, 1.0, "wrong minimum")
        require_close(result.maximum, 8.0, "wrong maximum")
        require_close(result.mean, 3.75, "wrong mean")
        require_close(result.median, 3.0, "wrong median")
        require_close(
            result.standard_deviation,
            float(np.std(expected_values, ddof=0)),
            "wrong population standard deviation",
        )
        require_close(
            result.median_absolute_deviation,
            1.5,
            "wrong raw MAD",
        )
        require_close(result.peak_to_peak, 7.0, "wrong peak-to-peak")
        print("   Values : [1, 2, 4, 8]")
        print("   Mean   : 3.75")
        print("   Median : 3.0")
        print("   MAD    : 1.5 (raw, unscaled)")
        print("   Result : PASS")
        print()

        print("[2/4] Definitions also hold for a constant time series")
        constant_values = np.full(4, 12.5, dtype=np.float64)
        constant_values.setflags(write=False)
        constant_series = replace(series, values=constant_values)
        constant = step04.compute_pixel_timeseries_statistics(constant_series)
        require_close(constant.minimum, 12.5, "constant minimum changed")
        require_close(constant.maximum, 12.5, "constant maximum changed")
        require_close(constant.mean, 12.5, "constant mean changed")
        require_close(constant.median, 12.5, "constant median changed")
        require_close(
            constant.standard_deviation,
            0.0,
            "constant standard deviation is not zero",
        )
        require_close(
            constant.median_absolute_deviation,
            0.0,
            "constant MAD is not zero",
        )
        require_close(
            constant.peak_to_peak,
            0.0,
            "constant peak-to-peak is not zero",
        )
        print("   Population std : 0.0")
        print("   Raw MAD        : 0.0")
        print("   Peak-to-peak   : 0.0")
        print("   Result         : PASS")
        print()

        print("[3/4] Result and canonical summary are immutable/deterministic")
        expected_summary = {
            "dataset": "bias",
            "row": 0,
            "column": 1,
            "n_frames": 4,
            "minimum": 1.0,
            "maximum": 8.0,
            "mean": 3.75,
            "median": 3.0,
            "standard_deviation": float(np.std(expected_values, ddof=0)),
            "median_absolute_deviation": 1.5,
            "peak_to_peak": 7.0,
        }
        require(result.summary() == expected_summary, "summary content changed")
        require(result.summary() == result.summary(), "summary is not deterministic")
        try:
            result.mean = 99.0
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "frozen statistics accepted modification")
        print("   Frozen        : YES")
        print("   Deterministic : YES")
        print("   JSON-ready    : YES")
        print("   Result        : PASS")
        print()

        print("[4/4] Invalid, inconsistent, and non-finite inputs are rejected")
        expect_step04_error(
            lambda: step04.compute_pixel_timeseries_statistics(object()),
            "series must be a PixelTimeSeries",
        )

        empty = np.empty(0, dtype=np.float64)
        empty.setflags(write=False)
        expect_step04_error(
            lambda: step04.compute_pixel_timeseries_statistics(
                replace(series, values=empty, n_frames=0)
            ),
            "must not be empty",
        )

        two_dimensional = np.zeros((2, 2), dtype=np.float64)
        two_dimensional.setflags(write=False)
        expect_step04_error(
            lambda: step04.compute_pixel_timeseries_statistics(
                replace(series, values=two_dimensional)
            ),
            "must be one-dimensional",
        )

        short = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        short.setflags(write=False)
        expect_step04_error(
            lambda: step04.compute_pixel_timeseries_statistics(
                replace(series, values=short)
            ),
            "metadata requires 4",
        )

        non_finite = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float64)
        non_finite.setflags(write=False)
        expect_step04_error(
            lambda: step04.compute_pixel_timeseries_statistics(
                replace(series, values=non_finite)
            ),
            "must all be finite",
        )
        print("   Invalid type       : rejected")
        print("   Empty/2-D input    : rejected")
        print("   Metadata mismatch  : rejected")
        print("   NaN/Inf            : rejected")
        print("   Result             : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 pixel statistics integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
