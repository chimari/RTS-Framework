"""Integration test for Step 04 image/ROI RTS analysis iteration v4.10.0."""

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
    bases = np.array([[0, 20, 40], [60, 80, 100]], dtype=np.uint16)
    offsets = [0, 1, 10, 11, 9, 2, 1, 10]

    paths: list[Path] = []
    for index, offset in enumerate(offsets):
        path = root / f"bias_{index:04d}.fit"
        fits.PrimaryHDU(data=bases + offset).writeto(path, overwrite=True)
        paths.append(path)

    rows: list[dict[str, object]] = []
    n_frames = len(paths)
    for frame_index, path in enumerate(paths):
        rows.append(
            {
                "dataset": "bias",
                "directory": str(root),
                "environment": "step04-v4.10-test",
                "frame_index": frame_index,
                "n_frames": n_frames,
                "temperature_C": -10.0,
                "temperature_start_C": -10.0,
                "temperature_end_C": -10.0,
                "temperature_fraction": frame_index / (n_frames - 1),
                "exposure_s": 0.0,
                "filename": path.name,
                "filepath": str(path),
                "image_width": 3,
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
    print("RTS Framework Step 04 image/ROI analysis iteration test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_image_analysis_") as temp_dir:
        root = Path(temp_dir)
        manifest = write_dataset(root)
        bias_plan = step03.prepare_bias_analysis(manifest, "bias")
        plan = step04.prepare_rts_dictionary_analysis(bias_plan)

        thresholds = {
            "minimum_score": 0.9,
            "minimum_state_count": 2,
            "minimum_separation": 5.0,
            "minimum_transition_count": 3,
            "minimum_lower_run": 2,
            "minimum_upper_run": 3,
        }

        print("[1/4] Full-image analysis uses deterministic row-major order")
        results = list(
            step04.iter_image_rts_analyses(
                plan,
                **thresholds,
            )
        )
        observed = [(item.series.row, item.series.column) for item in results]
        expected = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        require(observed == expected, f"full-image order changed: {observed}")
        require(len(results) == 6, "wrong full-image result count")
        require(all(item.is_candidate for item in results), "expected all candidates")
        print("   Image shape : 2 x 3")
        print("   Results     : 6")
        print("   Order       : row-major")
        print("   Result      : PASS")
        print()

        print("[2/4] ROI results match explicit-coordinate iteration")
        roi_kwargs = {
            "row_start": 0,
            "row_stop": 2,
            "column_start": 1,
            "column_stop": 3,
        }
        roi_results = list(
            step04.iter_image_rts_analyses(
                plan,
                **roi_kwargs,
                **thresholds,
            )
        )
        coordinates = list(
            step04.iter_image_coordinates(
                plan,
                **roi_kwargs,
            )
        )
        explicit_results = list(
            step04.iter_rts_pixel_analyses(
                plan,
                coordinates,
                **thresholds,
            )
        )
        require(
            [(item.series.row, item.series.column) for item in roi_results]
            == [(0, 1), (0, 2), (1, 1), (1, 2)],
            "wrong ROI order",
        )
        require(
            [item.summary() for item in roi_results]
            == [item.summary() for item in explicit_results],
            "ROI composition differs from explicit pipeline",
        )
        print("   ROI coordinates : 4")
        print("   Coordinate API  : MATCH")
        print("   Pixel API       : MATCH")
        print("   Result          : PASS")
        print()

        print("[3/4] Iteration is lazy and empty ROIs remain empty")
        iterator = step04.iter_image_rts_analyses(
            plan,
            row_start=1,
            row_stop=2,
            column_start=1,
            column_stop=3,
            **thresholds,
        )
        require(iter(iterator) is iterator, "result is not an iterator")
        first = next(iterator)
        require(
            (first.series.row, first.series.column) == (1, 1),
            "wrong first lazy result",
        )
        remaining = list(iterator)
        require(
            [(item.series.row, item.series.column) for item in remaining]
            == [(1, 2)],
            "wrong remaining lazy results",
        )

        empty = list(
            step04.iter_image_rts_analyses(
                plan,
                row_start=1,
                row_stop=1,
                **thresholds,
            )
        )
        require(empty == [], "empty ROI produced analysis results")
        print("   First next() : one pixel")
        print("   Remaining    : preserved")
        print("   Empty ROI    : empty")
        print("   Result       : PASS")
        print()

        print("[4/4] Coordinate and analysis validation errors propagate lazily")
        bad_roi = step04.iter_image_rts_analyses(
            plan,
            row_start=3,
            **thresholds,
        )
        expect_step04_error(
            lambda: next(bad_roi),
            "row_start",
        )

        bad_thresholds = dict(thresholds)
        bad_thresholds["minimum_transition_count"] = -1
        bad_analysis = step04.iter_image_rts_analyses(
            plan,
            row_start=0,
            row_stop=1,
            column_start=0,
            column_stop=1,
            **bad_thresholds,
        )
        expect_step04_error(
            lambda: next(bad_analysis),
            "minimum_transition_count",
        )

        invalid_plan = step04.iter_image_rts_analyses(
            object(),
            **thresholds,
        )
        expect_step04_error(
            lambda: next(invalid_plan),
            "plan must be an RTSDictionaryPlan",
        )
        print("   Invalid ROI       : propagated")
        print("   Invalid threshold : propagated")
        print("   Invalid plan      : propagated")
        print("   Result            : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 image/ROI analysis iteration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
