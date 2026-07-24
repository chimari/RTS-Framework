"""Integration test for Step 04 interval progress callback v4.17.0."""

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
        data = np.array(
            [flat[:3], flat[3:]],
            dtype=np.uint16,
        )
        path = root / f"bias_{frame_index:04d}.fit"
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows: list[dict[str, object]] = []
    for frame_index, path in enumerate(paths):
        rows.append(
            {
                "dataset": "bias",
                "directory": str(root),
                "environment": "step04-v4.17-test",
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
    print("RTS Framework Step 04 interval progress callback test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    print("[1/4] Wrapper forwards start, intervals, and final event")
    events: list[tuple[int, int]] = []
    callback = step04.make_interval_progress_callback(
        lambda completed, total: events.append((completed, total)),
        every=2,
    )
    for completed in range(7):
        callback(completed, 6)
    require(
        events == [(0, 6), (2, 6), (4, 6), (6, 6)],
        f"wrong interval sequence: {events}",
    )
    print("   Start event    : preserved")
    print("   Interval       : every 2 pixels")
    print("   Final event    : preserved")
    print("   Result         : PASS")
    print()

    print("[2/4] Final event is preserved off interval and duplicates are suppressed")
    events = []
    callback = step04.make_interval_progress_callback(
        lambda completed, total: events.append((completed, total)),
        every=4,
    )
    for event in [(0, 6), (0, 6), (1, 6), (4, 6), (6, 6), (6, 6)]:
        callback(*event)
    require(
        events == [(0, 6), (4, 6), (6, 6)],
        f"wrong duplicate/final handling: {events}",
    )

    empty_events: list[tuple[int, int]] = []
    empty_callback = step04.make_interval_progress_callback(
        lambda completed, total: empty_events.append((completed, total)),
        every=1000,
    )
    empty_callback(0, 0)
    require(empty_events == [(0, 0)], f"wrong empty event: {empty_events}")
    print("   Off-interval final : preserved")
    print("   Duplicate events   : suppressed")
    print("   Empty ROI          : one 0/0 event")
    print("   Result             : PASS")
    print()

    print("[3/4] High-level build uses the wrapper without changing results")
    with tempfile.TemporaryDirectory(prefix="rts_step04_interval_") as temp_dir:
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

        full_events: list[tuple[int, int]] = []
        interval = step04.make_interval_progress_callback(
            lambda completed, total: full_events.append((completed, total)),
            every=4,
        )
        result = step04.build_rts_dictionary_csv_result(
            plan,
            root / "dictionary.csv",
            progress_callback=interval,
            **thresholds,
        )
        require(
            full_events == [(0, 6), (4, 6), (6, 6)],
            f"wrong build interval events: {full_events}",
        )
        require(result.analyzed_pixel_count == 6, "wrong analyzed pixel count")
        require(result.candidate_count == 3, "wrong candidate count")
        print("   Build events    : 0/6, 4/6, 6/6")
        print("   Analyzed pixels : 6")
        print("   Candidates      : 3")
        print("   Result          : PASS")
        print()

        print("[4/4] Validation and wrapped exceptions remain explicit")
        expect_step04_error(
            lambda: step04.make_interval_progress_callback(123, every=1),
            "callback must be callable",
        )
        expect_step04_error(
            lambda: step04.make_interval_progress_callback(lambda *_: None, every=True),
            "every must be an integer",
        )
        expect_step04_error(
            lambda: step04.make_interval_progress_callback(lambda *_: None, every=0),
            "every must be greater than zero",
        )

        protected = root / "protected.csv"
        original = "ORIGINAL\n"
        protected.write_text(original, encoding="utf-8")

        def failing_callback(completed: int, total: int) -> None:
            if completed == 4:
                raise RuntimeError("intentional interval failure")

        failing_interval = step04.make_interval_progress_callback(
            failing_callback,
            every=4,
        )
        expect_step04_error(
            lambda: step04.build_rts_dictionary_csv_result(
                plan,
                protected,
                progress_callback=failing_interval,
                **thresholds,
            ),
            "progress_callback failed at 4/6",
        )
        require(
            protected.read_text(encoding="utf-8") == original,
            "destination changed after wrapped callback failure",
        )
        print("   Invalid callback : rejected")
        print("   Invalid interval : rejected")
        print("   Callback failure : propagated through build API")
        print("   Destination      : preserved")
        print("   Result           : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 interval progress callback test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
