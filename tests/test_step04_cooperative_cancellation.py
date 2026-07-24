"""Integration test for Step 04 cooperative cancellation v4.19.0."""

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
    paths = []
    for frame_index in range(8):
        flat = [series[frame_index] for series in values]
        data = np.array([flat[:3], flat[3:]], dtype=np.uint16)
        path = root / f"bias_{frame_index:04d}.fit"
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows = []
    for frame_index, path in enumerate(paths):
        rows.append({
            "dataset": "bias",
            "directory": str(root),
            "environment": "step04-v4.19-test",
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
        })

    manifest = root / "manifest.normalized.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def thresholds():
    return {
        "minimum_score": 0.9,
        "minimum_state_count": 2,
        "minimum_separation": 5.0,
        "minimum_transition_count": 3,
        "minimum_lower_run": 2,
        "minimum_upper_run": 2,
    }


def expect_cancelled(callable_, contains: str) -> None:
    try:
        callable_()
    except step04.Step04Cancelled as exc:
        require(contains in str(exc), f"wrong cancellation message: {exc}")
    else:
        require(False, "Step04Cancelled was not raised")


def expect_step04_error(callable_, contains: str) -> None:
    try:
        callable_()
    except step04.Step04Error as exc:
        require(not isinstance(exc, step04.Step04Cancelled), "unexpected cancellation")
        require(contains in str(exc), f"wrong error message: {exc}")
    else:
        require(False, "Step04Error was not raised")


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 cooperative cancellation test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_cancel_") as temp_dir:
        root = Path(temp_dir)
        manifest = write_dataset(root)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(manifest, "bias")
        )

        print("[1/4] Immediate cancellation occurs before output creation")
        output = root / "immediate.csv"
        progress_events = []
        expect_cancelled(
            lambda: step04.build_rts_dictionary_csv_result(
                plan, output,
                progress_callback=lambda completed, total: progress_events.append((completed, total)),
                cancel_requested=lambda: True,
                **thresholds(),
            ),
            "after 0 completed pixels",
        )
        require(not output.exists(), "output created")
        require(progress_events == [], "progress emitted")
        require(list(root.glob(f".{output.name}.*.tmp")) == [], "temporary file remains")
        print("   Cancellation point : before first pixel")
        print("   Progress events    : none")
        print("   Output             : not created")
        print("   Result             : PASS")
        print()

        print("[2/4] Mid-build cancellation preserves existing destination")
        protected = root / "protected.csv"
        original = "ORIGINAL\n"
        protected.write_text(original, encoding="utf-8")
        checks = 0
        events = []

        def cancel_after_three() -> bool:
            nonlocal checks
            checks += 1
            return checks >= 5

        expect_cancelled(
            lambda: step04.build_rts_dictionary_csv_result(
                plan, protected,
                progress_callback=lambda completed, total: events.append((completed, total)),
                cancel_requested=cancel_after_three,
                **thresholds(),
            ),
            "after 3 completed pixels",
        )
        require(events == [(0, 6), (1, 6), (2, 6), (3, 6)], f"events={events}")
        require(protected.read_text(encoding="utf-8") == original, "destination changed")
        require(list(root.glob(f".{protected.name}.*.tmp")) == [], "temporary file remains")
        print("   Completed pixels : 3")
        print("   Progress         : through 3/6")
        print("   Existing output  : preserved")
        print("   Temporary files  : removed")
        print("   Result           : PASS")
        print()

        print("[3/4] False cancellation callback leaves build unchanged")
        normal_output = root / "normal.csv"
        calls = 0

        def never_cancel() -> bool:
            nonlocal calls
            calls += 1
            return False

        result = step04.build_rts_dictionary_csv_result(
            plan, normal_output, cancel_requested=never_cancel, **thresholds()
        )
        require(result.analyzed_pixel_count == 6, "wrong analyzed count")
        require(result.candidate_count == 3, "wrong candidate count")
        require(calls == 8, f"wrong check count: {calls}")

        legacy_output = root / "legacy.csv"
        returned = step04.build_rts_dictionary_csv(
            plan, legacy_output, cancel_requested=lambda: False, **thresholds()
        )
        require(returned == legacy_output, "legacy returned wrong path")
        require(legacy_output.read_bytes() == normal_output.read_bytes(), "legacy output changed")
        print("   Cancellation checks : initial, each pixel, completion")
        print("   Build result        : unchanged")
        print("   Legacy API          : forwards callback")
        print("   Result              : PASS")
        print()

        print("[4/4] Invalid cancellation callbacks are rejected safely")
        invalid = root / "invalid.csv"
        invalid.write_text(original, encoding="utf-8")
        expect_step04_error(
            lambda: step04.build_rts_dictionary_csv_result(
                plan, invalid, cancel_requested=123, **thresholds()
            ),
            "cancel_requested must be callable",
        )
        expect_step04_error(
            lambda: step04.build_rts_dictionary_csv_result(
                plan, invalid, cancel_requested=lambda: 1, **thresholds()
            ),
            "cancel_requested must return bool",
        )

        def failing_cancel() -> bool:
            raise RuntimeError("intentional failure")

        expect_step04_error(
            lambda: step04.build_rts_dictionary_csv_result(
                plan, invalid, cancel_requested=failing_cancel, **thresholds()
            ),
            "cancel_requested failed after 0 completed pixels",
        )
        require(invalid.read_text(encoding="utf-8") == original, "destination changed")
        print("   Non-callable       : rejected")
        print("   Non-bool return    : rejected")
        print("   Callback exception : wrapped")
        print("   Destination        : preserved")
        print("   Result             : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 cooperative cancellation test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
