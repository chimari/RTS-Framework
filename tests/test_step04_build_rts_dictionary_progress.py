"""Integration test for Step 04 RTS dictionary build progress v4.16.0."""

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
    candidate_a = [0, 0, 10, 10, 0, 0, 10, 10]
    stable = [20, 21, 20, 21, 20, 21, 20, 21]
    candidate_b = [30, 30, 45, 45, 30, 30, 45, 45]
    rejected_temporal = [0, 10, 0, 10, 0, 10, 0, 10]

    paths: list[Path] = []
    for index, values in enumerate(
        zip(candidate_a, stable, candidate_b, rejected_temporal, strict=True)
    ):
        path = root / f"bias_{index:04d}.fit"
        a, s, b, t = values
        data = np.array([[a, s], [b, t]], dtype=np.uint16)
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows: list[dict[str, object]] = []
    for frame_index, path in enumerate(paths):
        rows.append(
            {
                "dataset": "bias",
                "directory": str(root),
                "environment": "step04-v4.16-test",
                "frame_index": frame_index,
                "n_frames": len(paths),
                "temperature_C": -10.0,
                "temperature_start_C": -10.0,
                "temperature_end_C": -10.0,
                "temperature_fraction": frame_index / (len(paths) - 1),
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
    print("RTS Framework Step 04 dictionary build progress test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_progress_") as temp_dir:
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

        print("[1/4] Full-image progress is deterministic")
        events: list[tuple[int, int]] = []
        output = root / "full.csv"
        result = step04.build_rts_dictionary_csv_result(
            plan,
            output,
            progress_callback=lambda completed, total: events.append(
                (completed, total)
            ),
            **thresholds,
        )
        require(
            events == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)],
            f"wrong progress sequence: {events}",
        )
        require(result.analyzed_pixel_count == 4, "wrong analyzed count")
        require(result.candidate_count == 2, "wrong candidate count")
        print("   Initial event : 0/4")
        print("   Final event   : 4/4")
        print("   Event order   : deterministic")
        print("   Result        : PASS")
        print()

        print("[2/4] ROI and empty ROI report exact totals")
        roi_events: list[tuple[int, int]] = []
        step04.build_rts_dictionary_csv_result(
            plan,
            root / "roi.csv",
            row_start=1,
            row_stop=2,
            column_start=0,
            column_stop=2,
            progress_callback=lambda completed, total: roi_events.append(
                (completed, total)
            ),
            **thresholds,
        )
        require(
            roi_events == [(0, 2), (1, 2), (2, 2)],
            f"wrong ROI progress: {roi_events}",
        )

        empty_events: list[tuple[int, int]] = []
        empty = step04.build_rts_dictionary_csv_result(
            plan,
            root / "empty.csv",
            row_start=1,
            row_stop=1,
            progress_callback=lambda completed, total: empty_events.append(
                (completed, total)
            ),
            **thresholds,
        )
        require(empty_events == [(0, 0)], f"wrong empty progress: {empty_events}")
        require(empty.analyzed_pixel_count == 0, "empty ROI analyzed pixels")
        print("   Selected ROI : 0/2 through 2/2")
        print("   Empty ROI    : one 0/0 event")
        print("   Result       : PASS")
        print()

        print("[3/4] Legacy API forwards the callback unchanged")
        legacy_events: list[tuple[int, int]] = []
        legacy_output = root / "legacy.csv"
        returned = step04.build_rts_dictionary_csv(
            plan,
            legacy_output,
            progress_callback=lambda completed, total: legacy_events.append(
                (completed, total)
            ),
            **thresholds,
        )
        require(returned == legacy_output, "legacy returned wrong path")
        require(
            legacy_events == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)],
            f"legacy callback changed: {legacy_events}",
        )
        print("   Return type : pathlib.Path")
        print("   Callback    : forwarded")
        print("   Sequence    : identical")
        print("   Result      : PASS")
        print()

        print("[4/4] Callback validation and failures preserve the destination")
        protected = root / "protected.csv"
        original = "ORIGINAL\n"
        protected.write_text(original, encoding="utf-8")

        expect_step04_error(
            lambda: step04.build_rts_dictionary_csv_result(
                plan,
                protected,
                progress_callback=123,
                **thresholds,
            ),
            "progress_callback must be callable",
        )
        require(
            protected.read_text(encoding="utf-8") == original,
            "destination changed after invalid callback",
        )

        def failing_callback(completed: int, total: int) -> None:
            if completed == 2:
                raise RuntimeError("intentional callback failure")

        expect_step04_error(
            lambda: step04.build_rts_dictionary_csv_result(
                plan,
                protected,
                progress_callback=failing_callback,
                **thresholds,
            ),
            "progress_callback failed at 2/4",
        )
        require(
            protected.read_text(encoding="utf-8") == original,
            "destination changed after callback failure",
        )
        leftovers = list(root.glob(f".{protected.name}.*.tmp"))
        require(leftovers == [], f"temporary files remain: {leftovers}")
        print("   Invalid callback : rejected")
        print("   Raised exception : wrapped")
        print("   Existing output  : preserved")
        print("   Temporary files  : removed")
        print("   Result           : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 dictionary build progress test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
