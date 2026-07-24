"""Integration test for Step 03 mean master bias v3.2.0."""

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
    return {
        "dataset": "bias",
        "directory": str(image.parent),
        "environment": "step03-v3.2-test",
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": -10.0 + 0.1 * frame_index,
        "temperature_start_C": -10.0,
        "temperature_end_C": -9.8,
        "temperature_fraction": frame_index / (n_frames - 1),
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


def expect_step03_error(callable_, contains: str) -> None:
    try:
        callable_()
    except step03.Step03Error as exc:
        require(contains in str(exc), f"wrong error message: {exc}")
    else:
        require(False, "Step03Error was not raised")


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 03 mean master-bias integration test")
    print("=" * 72)
    print(f"step03 version : {step03.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step03_mean_") as temp_dir:
        root = Path(temp_dir)
        paths = [root / f"bias_{index:04d}.fit" for index in range(3)]
        arrays = [
            np.array([[0, 2, 4], [6, 8, 10]], dtype=np.uint16),
            np.array([[2, 4, 6], [8, 10, 12]], dtype=np.uint16),
            np.array([[4, 6, 8], [10, 12, 14]], dtype=np.uint16),
        ]
        for path, data in zip(paths, arrays, strict=True):
            write_fits(path, data)

        manifest = root / "manifest.normalized.csv"
        write_manifest(
            manifest,
            [
                make_row(paths[2], frame_index=2, n_frames=3),
                make_row(paths[0], frame_index=0, n_frames=3),
                make_row(paths[1], frame_index=1, n_frames=3),
            ],
        )
        plan = step03.prepare_bias_analysis(manifest, "bias")

        print("[1/4] Known frames produce the expected arithmetic mean")
        result = step03.compute_mean_master_bias(plan)
        expected = np.array(
            [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]],
            dtype=np.float64,
        )
        require(
            np.array_equal(result.master_bias, expected),
            "master bias values are incorrect",
        )
        print("   Frames : 3")
        print("   Method : arithmetic mean")
        print("   Result : PASS")
        print()

        print("[2/4] Result metadata and scalar statistics are correct")
        require(result.plan is plan, "plan was not retained")
        require(result.dataset == "bias", "wrong dataset")
        require(result.n_frames == 3, "wrong frame count")
        require(result.image_shape == (2, 3), "wrong image shape")
        require(result.minimum == 2.0, "wrong minimum")
        require(result.maximum == 12.0, "wrong maximum")
        require(result.mean == 7.0, "wrong mean")
        print("   Minimum : 2")
        print("   Maximum : 12")
        print("   Mean    : 7")
        print("   Result  : PASS")
        print()

        print("[3/4] Master image is independent, float64, and immutable")
        master = result.master_bias
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
        for source in arrays:
            require(
                not np.shares_memory(master, source),
                "master shares source memory",
            )
        print("   dtype       : float64")
        print("   C-contiguous: YES")
        print("   Writable    : NO")
        print("   Result      : PASS")
        print()

        print("[4/4] Computation consumes exactly one streaming pass")
        calls = {"iterator": 0, "yielded": 0}

        def fake_iter_bias_frames(received_plan):
            require(received_plan is plan, "wrong plan passed to iterator")
            calls["iterator"] += 1
            for source in arrays:
                calls["yielded"] += 1
                frame = np.array(source, dtype=np.float64, order="C", copy=True)
                frame.setflags(write=False)
                yield frame

        with patch.object(
            step03,
            "iter_bias_frames",
            fake_iter_bias_frames,
        ):
            repeated = step03.compute_mean_master_bias(plan)

        require(calls["iterator"] == 1, "iterator was requested more than once")
        require(calls["yielded"] == 3, "unexpected number of yielded frames")
        require(
            np.array_equal(repeated.master_bias, expected),
            "mocked streaming result is incorrect",
        )

        def empty_iterator(_plan):
            if False:
                yield np.empty((0, 0))

        with patch.object(step03, "iter_bias_frames", empty_iterator):
            expect_step03_error(
                lambda: step03.compute_mean_master_bias(plan),
                "yielded no frames",
            )

        def short_iterator(_plan):
            for source in arrays[:2]:
                frame = np.array(source, dtype=np.float64, copy=True)
                frame.setflags(write=False)
                yield frame

        with patch.object(step03, "iter_bias_frames", short_iterator):
            expect_step03_error(
                lambda: step03.compute_mean_master_bias(plan),
                "analysis plan requires 3",
            )

        expect_step03_error(
            lambda: step03.compute_mean_master_bias(object()),
            "plan must be a BiasAnalysisPlan",
        )
        print("   Iterator calls : 1")
        print("   Frames yielded : 3")
        print("   Empty/short run: rejected")
        print("   Result         : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 03 mean master-bias integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
