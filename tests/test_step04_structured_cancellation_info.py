"""Integration test for Step 04 structured cancellation info v4.20.0."""

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
            "environment": "step04-v4.20-test",
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


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 structured cancellation info test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_cancel_info_") as temp_dir:
        root = Path(temp_dir)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(write_dataset(root), "bias")
        )

        print("[1/4] Immediate cancellation exposes complete structured context")
        output = root / "immediate.csv"
        try:
            step04.build_rts_dictionary_csv_result(
                plan, output, cancel_requested=lambda: True, **thresholds()
            )
        except step04.Step04Cancelled as exc:
            info = exc.info
            exception_message = str(exc)
        else:
            require(False, "Step04Cancelled was not raised")

        require(isinstance(info, step04.RTSCancellationInfo), "wrong info type")
        require(info.completed_pixel_count == 0, "wrong completed count")
        require(info.total_pixel_count == 6, "wrong total count")
        require(info.remaining_pixel_count == 6, "wrong remaining count")
        require(info.output_path == output, "wrong output path")
        require_close(info.fraction_complete, 0.0, "wrong fraction")
        require_close(info.percent_complete, 0.0, "wrong percent")
        require(
            exception_message
            == "RTS dictionary build cancelled after 0 completed pixels.",
            "legacy exception message changed",
        )
        print("   Completed pixels : 0")
        print("   Total pixels     : 6")
        print("   Output path      : available")
        print("   Message          : compatible")
        print("   Result           : PASS")
        print()

        print("[2/4] Mid-build cancellation reports exact completion state")
        checks = 0
        output = root / "mid.csv"

        def cancel_after_three() -> bool:
            nonlocal checks
            checks += 1
            return checks >= 5

        try:
            step04.build_rts_dictionary_csv_result(
                plan, output, cancel_requested=cancel_after_three, **thresholds()
            )
        except step04.Step04Cancelled as exc:
            info = exc.info
        else:
            require(False, "mid-build cancellation was not raised")

        require(info.completed_pixel_count == 3, "wrong completed count")
        require(info.total_pixel_count == 6, "wrong total count")
        require(info.remaining_pixel_count == 3, "wrong remaining count")
        require_close(info.fraction_complete, 0.5, "wrong fraction")
        require_close(info.percent_complete, 50.0, "wrong percent")
        require(info.output_path == output, "wrong output path")
        print("   Completed pixels : 3")
        print("   Remaining pixels : 3")
        print("   Completion       : 50 percent")
        print("   Result           : PASS")
        print()

        print("[3/4] Cancellation info is immutable and serializes deterministically")
        try:
            info.completed_pixel_count = 99
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "cancellation info is mutable")

        require(info.summary() == {
            "completed_pixel_count": 3,
            "total_pixel_count": 6,
            "remaining_pixel_count": 3,
            "fraction_complete": 0.5,
            "percent_complete": 50.0,
            "output_path": str(output),
        }, "summary is not canonical")

        empty = step04.RTSCancellationInfo(
            completed_pixel_count=0,
            total_pixel_count=0,
            output_path=root / "empty.csv",
        )
        require(empty.fraction_complete == 1.0, "empty fraction")
        require(empty.percent_complete == 100.0, "empty percent")
        require(empty.remaining_pixel_count == 0, "empty remaining")
        print("   Dataclass   : frozen")
        print("   Summary     : deterministic")
        print("   Empty total : complete")
        print("   Result      : PASS")
        print()

        print("[4/4] Non-cancelled builds and both public APIs remain unchanged")
        normal = root / "normal.csv"
        result = step04.build_rts_dictionary_csv_result(
            plan, normal, cancel_requested=lambda: False, **thresholds()
        )
        require(result.analyzed_pixel_count == 6, "wrong analyzed count")
        require(result.candidate_count == 3, "wrong candidate count")

        legacy = root / "legacy.csv"
        returned = step04.build_rts_dictionary_csv(
            plan, legacy, cancel_requested=lambda: False, **thresholds()
        )
        require(returned == legacy, "legacy returned wrong path")
        require(legacy.read_bytes() == normal.read_bytes(), "CSV output changed")
        print("   Build result : unchanged")
        print("   Legacy API   : unchanged")
        print("   CSV bytes    : identical")
        print("   Result       : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 structured cancellation info test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
