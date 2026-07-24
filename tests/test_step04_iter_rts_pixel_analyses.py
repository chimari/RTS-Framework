"""Integration test for Step 04 explicit-coordinate iteration v4.8.0."""

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
    frame_values = [
        np.array([[0, 20], [40, 60]], dtype=np.uint16),
        np.array([[1, 21], [41, 61]], dtype=np.uint16),
        np.array([[10, 30], [50, 70]], dtype=np.uint16),
        np.array([[11, 31], [51, 71]], dtype=np.uint16),
        np.array([[9, 29], [49, 69]], dtype=np.uint16),
        np.array([[2, 22], [42, 62]], dtype=np.uint16),
        np.array([[1, 21], [41, 61]], dtype=np.uint16),
        np.array([[10, 30], [50, 70]], dtype=np.uint16),
    ]

    paths: list[Path] = []
    for index, data in enumerate(frame_values):
        path = root / f"bias_{index:04d}.fit"
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows: list[dict[str, object]] = []
    n_frames = len(paths)
    for frame_index, path in enumerate(paths):
        rows.append(
            {
                "dataset": "bias",
                "directory": str(root),
                "environment": "step04-v4.8-test",
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
    print("RTS Framework Step 04 explicit-coordinate iteration test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_iter_") as temp_dir:
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

        print("[1/4] Results preserve exact coordinate order and duplicates")
        coordinates = [(1, 1), (0, 0), (1, 1), [0, 1]]
        results = list(
            step04.iter_rts_pixel_analyses(
                plan,
                coordinates,
                **thresholds,
            )
        )
        observed = [(result.series.row, result.series.column) for result in results]
        require(
            observed == [(1, 1), (0, 0), (1, 1), (0, 1)],
            f"coordinate order changed: {observed}",
        )
        require(len(results) == 4, "duplicate coordinate was removed")
        require(
            results[0] is not results[2],
            "duplicate coordinate reused the same aggregate object",
        )
        require(
            results[0].summary() == results[2].summary(),
            "duplicate coordinate produced non-deterministic result",
        )
        print("   Input order : preserved")
        print("   Duplicates  : preserved")
        print("   Re-analysis : deterministic")
        print("   Result      : PASS")
        print()

        print("[2/4] Iterator is lazy and supports one-shot generators")
        consumed: list[tuple[int, int]] = []

        def coordinate_source():
            for coordinate in [(0, 0), (0, 1), (1, 0)]:
                consumed.append(coordinate)
                yield coordinate

        iterator = step04.iter_rts_pixel_analyses(
            plan,
            coordinate_source(),
            **thresholds,
        )
        require(consumed == [], "coordinate generator was consumed eagerly")

        first = next(iterator)
        require(consumed == [(0, 0)], "iterator consumed more than first coordinate")
        require(
            (first.series.row, first.series.column) == (0, 0),
            "wrong first lazy result",
        )

        remaining = list(iterator)
        require(
            consumed == [(0, 0), (0, 1), (1, 0)],
            "one-shot generator was not consumed exactly once",
        )
        require(
            [(item.series.row, item.series.column) for item in remaining]
            == [(0, 1), (1, 0)],
            "remaining lazy result order changed",
        )
        print("   Construction : no consumption")
        print("   First next() : one coordinate")
        print("   Generator    : consumed once")
        print("   Result       : PASS")
        print()

        print("[3/4] Each yielded result matches analyze_rts_pixel()")
        for coordinate, yielded in zip(
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            step04.iter_rts_pixel_analyses(
                plan,
                [(0, 0), (0, 1), (1, 0), (1, 1)],
                **thresholds,
            ),
        ):
            direct = step04.analyze_rts_pixel(
                plan,
                row=coordinate[0],
                column=coordinate[1],
                **thresholds,
            )
            require(
                yielded.summary() == direct.summary(),
                f"iterator result differs at coordinate {coordinate}",
            )
        print("   Four coordinates : MATCH")
        print("   Public pipeline   : MATCH")
        print("   Result            : PASS")
        print()

        print("[4/4] Malformed coordinates and downstream errors are lazy")
        invalid_iterator = step04.iter_rts_pixel_analyses(
            plan,
            [(0, 0), (0,)],
            **thresholds,
        )
        first = next(invalid_iterator)
        require(
            (first.series.row, first.series.column) == (0, 0),
            "valid item before malformed coordinate did not yield",
        )
        expect_step04_error(
            lambda: next(invalid_iterator),
            "coordinates[1]",
        )

        expect_step04_error(
            lambda: next(
                step04.iter_rts_pixel_analyses(
                    plan,
                    42,
                    **thresholds,
                )
            ),
            "coordinates must be an iterable",
        )

        for invalid_coordinate, expected in (
            ([(True, 0)], "row must be an integer"),
            ([(0, False)], "column must be an integer"),
            ([(0.0, 0)], "row must be an integer"),
            ([(0, "1")], "column must be an integer"),
        ):
            expect_step04_error(
                lambda invalid_coordinate=invalid_coordinate: next(
                    step04.iter_rts_pixel_analyses(
                        plan,
                        invalid_coordinate,
                        **thresholds,
                    )
                ),
                expected,
            )

        out_of_bounds = step04.iter_rts_pixel_analyses(
            plan,
            [(0, 0), (99, 0)],
            **thresholds,
        )
        next(out_of_bounds)
        expect_step04_error(
            lambda: next(out_of_bounds),
            "row",
        )
        print("   Malformed pair      : rejected lazily")
        print("   Non-iterable input  : rejected lazily")
        print("   Invalid value types : rejected lazily")
        print("   Bounds error        : propagated lazily")
        print("   Result              : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 explicit-coordinate iteration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
