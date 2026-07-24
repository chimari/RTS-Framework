"""Integration test for Step 04 timed progress state v4.18.0."""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from dataclasses import FrozenInstanceError
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
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        print(f"FAIL: {message}: actual={actual}, expected={expected}")
        raise SystemExit(1)


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def write_dataset(root: Path) -> Path:
    values = [
        [0, 0, 10, 10, 0, 0, 10, 10],
        [20, 21, 20, 21, 20, 21, 20, 21],
        [30, 30, 45, 45, 30, 30, 45, 45],
        [0, 10, 0, 10, 0, 10, 0, 10],
        [5, 5, 15, 15, 5, 5, 15, 15],
        [50, 51, 50, 51, 50, 51, 50, 51],
    ]

    paths: list[Path] = []
    for frame_index in range(8):
        flat = [pixel_series[frame_index] for pixel_series in values]
        data = np.array([flat[:3], flat[3:]], dtype=np.uint16)
        path = root / f"bias_{frame_index:04d}.fit"
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows: list[dict[str, object]] = []
    for frame_index, path in enumerate(paths):
        rows.append(
            {
                "dataset": "bias",
                "directory": str(root),
                "environment": "step04-v4.18-test",
                "frame_index": frame_index,
                "n_frames": len(paths),
                "temperature_C": -10.0,
                "temperature_start_C": -10.0,
                "temperature_end_C": -10.0,
                "temperature_fraction": frame_index / (len(paths) - 1),
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
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
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
    print("RTS Framework Step 04 timed progress state test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    print("[1/4] Timed state computes deterministic rate and ETA")
    states: list[step04.RTSProgressState] = []
    callback = step04.make_timed_progress_callback(
        states.append,
        clock=FakeClock([100.0, 102.0, 105.0]),
    )
    callback(0, 10)
    callback(4, 10)
    callback(10, 10)

    require(len(states) == 3, "wrong state count")
    require(states[0].elapsed_seconds == 0.0, "wrong initial elapsed")
    require(states[0].pixels_per_second is None, "initial rate must be None")
    require(states[0].remaining_seconds is None, "initial ETA must be None")

    require_close(states[1].elapsed_seconds, 2.0, "wrong middle elapsed")
    require_close(states[1].pixels_per_second, 2.0, "wrong middle rate")
    require_close(states[1].remaining_seconds, 3.0, "wrong middle ETA")

    require_close(states[2].elapsed_seconds, 5.0, "wrong final elapsed")
    require_close(states[2].pixels_per_second, 2.0, "wrong final rate")
    require_close(states[2].remaining_seconds, 0.0, "wrong final ETA")
    print("   Elapsed time : deterministic")
    print("   Processing rate : computed")
    print("   Remaining time  : computed")
    print("   Result          : PASS")
    print()

    print("[2/4] Progress state is immutable and exposes canonical properties")
    state = states[1]
    try:
        state.completed = 99
    except (FrozenInstanceError, AttributeError):
        pass
    else:
        require(False, "progress state is mutable")

    require_close(state.fraction_complete, 0.4, "wrong fraction")
    require_close(state.percent_complete, 40.0, "wrong percent")
    require(not state.is_complete, "middle state incorrectly complete")
    require(states[2].is_complete, "final state not complete")

    empty_states: list[step04.RTSProgressState] = []
    empty = step04.make_timed_progress_callback(
        empty_states.append,
        clock=FakeClock([10.0]),
    )
    empty(0, 0)
    require(empty_states[0].fraction_complete == 1.0, "empty fraction")
    require(empty_states[0].percent_complete == 100.0, "empty percent")
    require(empty_states[0].is_complete, "empty state must be complete")
    require(
        state.summary()["remaining_seconds"] == 3.0,
        "summary changed timing value",
    )
    print("   Dataclass   : frozen")
    print("   Fraction    : canonical")
    print("   Empty total : complete")
    print("   Summary     : deterministic")
    print("   Result      : PASS")
    print()

    print("[3/4] Timed and interval callbacks compose with the build API")
    with tempfile.TemporaryDirectory(prefix="rts_step04_timed_") as temp_dir:
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
            "minimum_upper_run": 2,
        }

        build_states: list[step04.RTSProgressState] = []
        timed = step04.make_timed_progress_callback(
            build_states.append,
            clock=FakeClock([20.0, 24.0, 26.0]),
        )
        interval = step04.make_interval_progress_callback(timed, every=4)

        result = step04.build_rts_dictionary_csv_result(
            plan,
            root / "dictionary.csv",
            progress_callback=interval,
            **thresholds,
        )
        require(
            [(item.completed, item.total) for item in build_states]
            == [(0, 6), (4, 6), (6, 6)],
            "wrong composed progress sequence",
        )
        require_close(build_states[1].pixels_per_second, 1.0, "wrong build rate")
        require_close(build_states[1].remaining_seconds, 2.0, "wrong build ETA")
        require(result.analyzed_pixel_count == 6, "wrong analyzed count")
        require(result.candidate_count == 3, "wrong candidate count")
        print("   Event sequence : 0/6, 4/6, 6/6")
        print("   Composition    : interval -> timed")
        print("   Build result   : unchanged")
        print("   Result         : PASS")
        print()

        print("[4/4] Invalid timing events and callbacks are rejected")
        expect_step04_error(
            lambda: step04.make_timed_progress_callback(123),
            "callback must be callable",
        )
        expect_step04_error(
            lambda: step04.make_timed_progress_callback(lambda _: None, clock=123),
            "clock must be callable",
        )

        validating = step04.make_timed_progress_callback(
            lambda _: None,
            clock=FakeClock([0.0, 1.0, 2.0]),
        )
        validating(0, 6)
        expect_step04_error(lambda: validating(7, 6), "must not exceed")
        expect_step04_error(lambda: validating(1, 7), "total must remain unchanged")

        decreasing = step04.make_timed_progress_callback(
            lambda _: None,
            clock=FakeClock([0.0, 1.0, 2.0]),
        )
        decreasing(0, 6)
        decreasing(4, 6)
        expect_step04_error(lambda: decreasing(3, 6), "must not decrease")
        print("   Invalid callback : rejected")
        print("   Invalid clock    : rejected")
        print("   Invalid totals   : rejected")
        print("   Decreasing count : rejected")
        print("   Result           : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 timed progress state test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
