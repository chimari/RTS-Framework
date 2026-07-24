"""Integration test for Step 04 image-coordinate iteration v4.9.0."""

from __future__ import annotations

import csv
import sys
import tempfile
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


def write_dataset(root: Path) -> Path:
    paths: list[Path] = []
    for index in range(3):
        path = root / f"bias_{index:04d}.fit"
        data = np.arange(12, dtype=np.uint16).reshape(3, 4) + index
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows: list[dict[str, object]] = []
    n_frames = len(paths)
    for frame_index, path in enumerate(paths):
        rows.append(
            {
                "dataset": "bias",
                "directory": str(root),
                "environment": "step04-v4.9-test",
                "frame_index": frame_index,
                "n_frames": n_frames,
                "temperature_C": -10.0,
                "temperature_start_C": -10.0,
                "temperature_end_C": -10.0,
                "temperature_fraction": frame_index / (n_frames - 1),
                "exposure_s": 0.0,
                "filename": path.name,
                "filepath": str(path),
                "image_width": 4,
                "image_height": 3,
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
    print("RTS Framework Step 04 image-coordinate iteration test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_coords_") as temp_dir:
        root = Path(temp_dir)
        manifest = write_dataset(root)
        bias_plan = step03.prepare_bias_analysis(manifest, "bias")
        plan = step04.prepare_rts_dictionary_analysis(bias_plan)

        print("[1/4] Full image uses deterministic row-major order")
        full = list(step04.iter_image_coordinates(plan))
        expected_full = [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 3),
        ]
        require(full == expected_full, f"full-image order changed: {full}")
        require(len(full) == 12, "wrong full-image coordinate count")
        print("   Image shape : 3 x 4")
        print("   Coordinates : 12")
        print("   Order       : row-major")
        print("   Result      : PASS")
        print()

        print("[2/4] Rectangular ROI uses exclusive stop bounds")
        roi = list(
            step04.iter_image_coordinates(
                plan,
                row_start=1,
                row_stop=3,
                column_start=1,
                column_stop=4,
            )
        )
        expected_roi = [
            (1, 1), (1, 2), (1, 3),
            (2, 1), (2, 2), (2, 3),
        ]
        require(roi == expected_roi, f"ROI coordinates changed: {roi}")
        print("   Rows        : [1, 3)")
        print("   Columns     : [1, 4)")
        print("   Coordinates : 6")
        print("   Result      : PASS")
        print()

        print("[3/4] Empty regions are valid and iteration is lazy")
        require(
            list(
                step04.iter_image_coordinates(
                    plan,
                    row_start=2,
                    row_stop=2,
                )
            )
            == [],
            "empty row range did not yield an empty iterator",
        )
        require(
            list(
                step04.iter_image_coordinates(
                    plan,
                    column_start=4,
                    column_stop=4,
                )
            )
            == [],
            "empty column range did not yield an empty iterator",
        )

        iterator = step04.iter_image_coordinates(
            plan,
            row_start=1,
            row_stop=3,
            column_start=2,
            column_stop=4,
        )
        require(iter(iterator) is iterator, "result is not an iterator")
        require(next(iterator) == (1, 2), "wrong first lazy coordinate")
        require(next(iterator) == (1, 3), "wrong second lazy coordinate")
        require(list(iterator) == [(2, 2), (2, 3)], "wrong remaining coordinates")
        print("   Empty row ROI    : valid")
        print("   Empty column ROI : valid")
        print("   Lazy iteration   : confirmed")
        print("   Result           : PASS")
        print()

        print("[4/4] Invalid plans and ROI bounds are rejected")
        expect_step04_error(
            lambda: next(step04.iter_image_coordinates(object())),
            "plan must be an RTSDictionaryPlan",
        )

        invalid_cases = [
            ({"row_start": -1}, "row_start"),
            ({"row_start": 4}, "row_start"),
            ({"row_stop": 4}, "row_stop"),
            ({"column_start": -1}, "column_start"),
            ({"column_start": 5}, "column_start"),
            ({"column_stop": 5}, "column_stop"),
            ({"row_start": 2, "row_stop": 1}, "row_start"),
            (
                {"column_start": 3, "column_stop": 2},
                "column_start",
            ),
            ({"row_start": True}, "row_start"),
            ({"row_stop": 2.0}, "row_stop"),
            ({"column_start": "1"}, "column_start"),
            ({"column_stop": False}, "column_stop"),
        ]

        for kwargs, expected in invalid_cases:
            expect_step04_error(
                lambda kwargs=kwargs: next(
                    step04.iter_image_coordinates(plan, **kwargs)
                ),
                expected,
            )

        print("   Invalid plan       : rejected")
        print("   Out-of-range bound : rejected")
        print("   Reversed range     : rejected")
        print("   Invalid bound type : rejected")
        print("   Result             : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 image-coordinate iteration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
