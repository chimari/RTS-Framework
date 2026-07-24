"""Integration test for Step 03 exact median master bias v3.3.0."""

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


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def write_fits(path: Path, data: np.ndarray) -> None:
    fits.PrimaryHDU(data=data).writeto(path, overwrite=True)


def make_row(
    image: Path,
    *,
    frame_index: int,
    n_frames: int,
) -> dict[str, object]:
    fraction = 0.0 if n_frames == 1 else frame_index / (n_frames - 1)
    return {
        "dataset": "bias",
        "directory": str(image.parent),
        "environment": "step03-v3.3-test",
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": -10.0 + 0.1 * frame_index,
        "temperature_start_C": -10.0,
        "temperature_end_C": -10.0 + 0.1 * (n_frames - 1),
        "temperature_fraction": fraction,
        "exposure_s": 0.0,
        "filename": image.name,
        "filepath": str(image),
        "image_width": 3,
        "image_height": 2,
        "pixel_dtype": "uint16",
        "byte_order": "not-applicable",
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_plan(
    root: Path,
    name: str,
    arrays: list[np.ndarray],
):
    directory = root / name
    directory.mkdir()
    paths = [
        directory / f"bias_{index:04d}.fit"
        for index in range(len(arrays))
    ]
    for path, array in zip(paths, arrays, strict=True):
        write_fits(path, array)

    manifest = directory / "manifest.normalized.csv"
    rows = [
        make_row(path, frame_index=index, n_frames=len(arrays))
        for index, path in enumerate(paths)
    ]
    write_manifest(manifest, list(reversed(rows)))
    return step03.prepare_bias_analysis(manifest, "bias")


def expect_step03_error(callable_, contains: str) -> None:
    try:
        callable_()
    except step03.Step03Error as exc:
        require(contains in str(exc), f"wrong error message: {exc}")
    else:
        require(False, "Step03Error was not raised")


def readonly_frame(array: np.ndarray) -> np.ndarray:
    frame = np.array(array, dtype=np.float64, order="C", copy=True)
    frame.setflags(write=False)
    return frame


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 03 median master-bias integration test")
    print("=" * 72)
    print(f"step03 version : {step03.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step03_median_") as temp_dir:
        root = Path(temp_dir)

        odd_arrays = [
            np.array([[100, 1, 50], [8, 7, 30]], dtype=np.uint16),
            np.array([[2, 20, 6], [40, 9, 10]], dtype=np.uint16),
            np.array([[4, 3, 8], [6, 50, 20]], dtype=np.uint16),
        ]
        odd_plan = build_plan(root, "odd", odd_arrays)

        print("[1/4] Odd frame count produces the exact per-pixel median")
        odd_result = step03.compute_median_master_bias(odd_plan)
        odd_expected = np.array(
            [[4.0, 3.0, 8.0], [8.0, 9.0, 20.0]],
            dtype=np.float64,
        )
        require(
            np.array_equal(odd_result.master_bias, odd_expected),
            "odd-count median values are incorrect",
        )
        print("   Frames : 3")
        print("   Method : exact per-pixel median")
        print("   Result : PASS")
        print()

        print("[2/4] Even frame count averages the two central values")
        even_arrays = [
            np.array([[0, 10, 20], [30, 40, 50]], dtype=np.uint16),
            np.array([[2, 12, 22], [32, 42, 52]], dtype=np.uint16),
            np.array([[100, 14, 24], [34, 44, 54]], dtype=np.uint16),
            np.array([[102, 16, 26], [36, 46, 56]], dtype=np.uint16),
        ]
        even_plan = build_plan(root, "even", even_arrays)
        even_result = step03.compute_median_master_bias(even_plan)
        even_expected = np.array(
            [[51.0, 13.0, 23.0], [33.0, 43.0, 53.0]],
            dtype=np.float64,
        )
        require(
            np.array_equal(even_result.master_bias, even_expected),
            "even-count median values are incorrect",
        )
        print("   Frames         : 4")
        print("   Central values : averaged")
        print("   Result         : PASS")
        print()

        print("[3/4] Metadata, statistics, and immutability are correct")
        result = odd_result
        master = result.master_bias
        require(result.plan is odd_plan, "plan was not retained")
        require(result.dataset == "bias", "wrong dataset")
        require(result.n_frames == 3, "wrong frame count")
        require(result.image_shape == (2, 3), "wrong image shape")
        require(result.minimum == 3.0, "wrong minimum")
        require(result.maximum == 20.0, "wrong maximum")
        require(result.median == 8.0, "wrong image median")
        require(master.dtype == np.dtype(np.float64), "dtype is not float64")
        require(master.flags.c_contiguous, "master is not C-contiguous")
        require(master.flags.owndata, "master does not own its data")
        require(not master.flags.writeable, "master is writable")
        try:
            master[0, 0] = 999.0
        except ValueError:
            pass
        else:
            require(False, "read-only master accepted modification")
        for source in odd_arrays:
            require(
                not np.shares_memory(master, source),
                "master shares source memory",
            )
        print("   Minimum      : 3")
        print("   Maximum      : 20")
        print("   Image median : 8")
        print("   Writable     : NO")
        print("   Result       : PASS")
        print()

        print("[4/4] Computation reads one pass and rejects bad counts")
        calls = {"iterator": 0, "yielded": 0}

        def fake_iterator(received_plan):
            require(received_plan is odd_plan, "wrong plan passed to iterator")
            calls["iterator"] += 1
            for array in odd_arrays:
                calls["yielded"] += 1
                yield readonly_frame(array)

        with patch.object(step03, "iter_bias_frames", fake_iterator):
            repeated = step03.compute_median_master_bias(odd_plan)

        require(calls["iterator"] == 1, "iterator was requested more than once")
        require(calls["yielded"] == 3, "unexpected yielded frame count")
        require(
            np.array_equal(repeated.master_bias, odd_expected),
            "mocked streaming result is incorrect",
        )

        def empty_iterator(_plan):
            if False:
                yield np.empty((0, 0))

        with patch.object(step03, "iter_bias_frames", empty_iterator):
            expect_step03_error(
                lambda: step03.compute_median_master_bias(odd_plan),
                "yielded no frames",
            )

        def short_iterator(_plan):
            for array in odd_arrays[:2]:
                yield readonly_frame(array)

        with patch.object(step03, "iter_bias_frames", short_iterator):
            expect_step03_error(
                lambda: step03.compute_median_master_bias(odd_plan),
                "analysis plan requires 3",
            )

        def long_iterator(_plan):
            for array in odd_arrays + odd_arrays[:1]:
                yield readonly_frame(array)

        with patch.object(step03, "iter_bias_frames", long_iterator):
            expect_step03_error(
                lambda: step03.compute_median_master_bias(odd_plan),
                "yielded more than 3",
            )

        expect_step03_error(
            lambda: step03.compute_median_master_bias(object()),
            "plan must be a BiasAnalysisPlan",
        )
        print("   Iterator calls  : 1")
        print("   Frames yielded  : 3")
        print("   Empty/short/long: rejected")
        print("   Result          : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 03 median master-bias integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
