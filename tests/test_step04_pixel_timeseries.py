"""Integration test for Step 04 one-pixel time-series loading v4.1.0."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

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


def readonly_frame(array: np.ndarray) -> np.ndarray:
    frame = np.array(array, dtype=np.float64, order="C", copy=True)
    frame.setflags(write=False)
    return frame


def write_dataset(root: Path, arrays: list[np.ndarray]) -> Path:
    paths = [root / f"bias_{index:04d}.fit" for index in range(len(arrays))]
    for path, array in zip(paths, arrays, strict=True):
        fits.PrimaryHDU(data=array).writeto(path, overwrite=True)

    rows: list[dict[str, object]] = []
    n_frames = len(paths)
    height, width = arrays[0].shape
    for frame_index, path in enumerate(paths):
        rows.append(
            {
                "dataset": "bias",
                "directory": str(root),
                "environment": "step04-v4.1-test",
                "frame_index": frame_index,
                "n_frames": n_frames,
                "temperature_C": -10.0 + 0.1 * frame_index,
                "temperature_start_C": -10.0,
                "temperature_end_C": -10.0 + 0.1 * (n_frames - 1),
                "temperature_fraction": frame_index / (n_frames - 1),
                "exposure_s": 0.0,
                "filename": path.name,
                "filepath": str(path),
                "image_width": width,
                "image_height": height,
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
        writer.writerows(list(reversed(rows)))
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
    print("RTS Framework Step 04 pixel time-series integration test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    arrays = [
        np.array([[10, 11, 12], [20, 21, 22]], dtype=np.uint16),
        np.array([[30, 31, 32], [40, 41, 42]], dtype=np.uint16),
        np.array([[50, 51, 52], [60, 61, 62]], dtype=np.uint16),
    ]

    with tempfile.TemporaryDirectory(prefix="rts_step04_pixel_") as temp_dir:
        root = Path(temp_dir)
        manifest = write_dataset(root, arrays)
        bias_plan = step03.prepare_bias_analysis(manifest, "bias")
        plan = step04.prepare_rts_dictionary_analysis(bias_plan)

        print("[1/4] One coordinate is loaded in canonical frame order")
        result = step04.load_pixel_timeseries(plan, row=1, column=2)
        expected = np.array([22.0, 42.0, 62.0], dtype=np.float64)
        require(
            np.array_equal(result.values, expected),
            "time-series values or order are incorrect",
        )
        require(result.plan is plan, "RTS plan was not retained")
        require(result.dataset == "bias", "wrong dataset")
        require(result.row == 1, "wrong row")
        require(result.column == 2, "wrong column")
        require(result.n_frames == 3, "wrong frame count")
        print("   Coordinate : (1, 2)")
        print("   Values     : [22, 42, 62]")
        print("   Result     : PASS")
        print()

        print("[2/4] The result vector is independent and immutable")
        values = result.values
        require(values.dtype == np.dtype(np.float64), "dtype is not float64")
        require(values.ndim == 1, "result is not one-dimensional")
        require(values.flags.c_contiguous, "result is not C-contiguous")
        require(values.flags.owndata, "result does not own its data")
        require(not values.flags.writeable, "result is writable")
        for source in arrays:
            require(
                not np.shares_memory(values, source),
                "result shares source-image memory",
            )
        try:
            values[0] = 999.0
        except ValueError:
            pass
        else:
            require(False, "read-only result accepted modification")
        expected_summary = {
            "dataset": "bias",
            "row": 1,
            "column": 2,
            "n_frames": 3,
            "dtype": "float64",
        }
        require(result.summary() == expected_summary, "summary content changed")
        print("   dtype        : float64")
        print("   C-contiguous : YES")
        print("   Writable     : NO")
        print("   Result       : PASS")
        print()

        print("[3/4] Loading performs exactly one lazy frame pass")
        calls = {"iterator": 0, "yielded": 0}

        def fake_iterator(received_plan):
            require(
                received_plan is bias_plan,
                "wrong BiasAnalysisPlan passed to Step 03",
            )
            calls["iterator"] += 1
            for array in arrays:
                calls["yielded"] += 1
                yield readonly_frame(array)

        with patch.object(step04, "iter_bias_frames", fake_iterator):
            repeated = step04.load_pixel_timeseries(plan, 0, 1)

        require(calls["iterator"] == 1, "iterator was requested more than once")
        require(calls["yielded"] == 3, "wrong number of frames was consumed")
        require(
            np.array_equal(repeated.values, [11.0, 31.0, 51.0]),
            "mocked time series is incorrect",
        )
        print("   Iterator calls : 1")
        print("   Frames read    : 3")
        print("   Full cube      : NO")
        print("   Result         : PASS")
        print()

        print("[4/4] Invalid coordinates and iterator failures are rejected")
        expect_step04_error(
            lambda: step04.load_pixel_timeseries(object(), 0, 0),
            "plan must be an RTSDictionaryPlan",
        )
        for invalid in (True, 1.0, "1"):
            expect_step04_error(
                lambda invalid=invalid: step04.load_pixel_timeseries(
                    plan,
                    invalid,
                    0,
                ),
                "row must be an integer",
            )
        expect_step04_error(
            lambda: step04.load_pixel_timeseries(plan, -1, 0),
            "row=-1 is outside",
        )
        expect_step04_error(
            lambda: step04.load_pixel_timeseries(plan, 2, 0),
            "row=2 is outside",
        )
        expect_step04_error(
            lambda: step04.load_pixel_timeseries(plan, 0, 3),
            "column=3 is outside",
        )

        def empty_iterator(_plan):
            if False:
                yield np.empty((0, 0))

        with patch.object(step04, "iter_bias_frames", empty_iterator):
            expect_step04_error(
                lambda: step04.load_pixel_timeseries(plan, 0, 0),
                "yielded no frames",
            )

        def short_iterator(_plan):
            for array in arrays[:2]:
                yield readonly_frame(array)

        with patch.object(step04, "iter_bias_frames", short_iterator):
            expect_step04_error(
                lambda: step04.load_pixel_timeseries(plan, 0, 0),
                "plan requires 3",
            )

        def long_iterator(_plan):
            for array in arrays + arrays[:1]:
                yield readonly_frame(array)

        with patch.object(step04, "iter_bias_frames", long_iterator):
            expect_step04_error(
                lambda: step04.load_pixel_timeseries(plan, 0, 0),
                "more than 3",
            )

        def bad_shape_iterator(_plan):
            yield readonly_frame(np.zeros((1, 1)))
            yield from ()

        with patch.object(step04, "iter_bias_frames", bad_shape_iterator):
            expect_step04_error(
                lambda: step04.load_pixel_timeseries(plan, 0, 0),
                "expected (2, 3)",
            )

        with patch.object(
            step04,
            "iter_bias_frames",
            side_effect=step03.Step03Error("synthetic read failure"),
        ):
            expect_step04_error(
                lambda: step04.load_pixel_timeseries(plan, 0, 0),
                "synthetic read failure",
            )
        print("   Invalid plan/coordinates : rejected")
        print("   Bad frame counts/shapes  : rejected")
        print("   Step 03 failures         : translated")
        print("   Result                   : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 pixel time-series integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
