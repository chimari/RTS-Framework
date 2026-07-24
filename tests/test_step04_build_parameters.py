"""Integration test for Step 04 build parameters v4.21.0."""

from __future__ import annotations

import csv
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
            "environment": "step04-v4.21-test",
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
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def kwargs() -> dict[str, object]:
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
    print("RTS Framework Step 04 dictionary build parameters test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_parameters_") as temp_dir:
        root = Path(temp_dir)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(write_dataset(root), "bias")
        )

        print("[1/4] Full-image build exposes normalized immutable parameters")
        output = root / "full.csv"
        result = step04.build_rts_dictionary_csv_result(
            plan, output, **kwargs()
        )
        p = result.parameters
        require(isinstance(p, step04.RTSDictionaryBuildParameters),
                "wrong parameter type")
        require((p.row_start, p.row_stop) == (0, 2), "wrong row range")
        require((p.column_start, p.column_stop) == (0, 3),
                "wrong column range")
        require((p.row_count, p.column_count, p.pixel_count) == (2, 3, 6),
                "wrong derived counts")
        require(result.analyzed_pixel_count == p.pixel_count,
                "count mismatch")
        print("   Image range : rows 0:2, columns 0:3")
        print("   Pixel count : 6")
        print("   Result      : PASS")
        print()

        print("[2/4] Explicit subregion preserves exact build conditions")
        sub_output = root / "sub.csv"
        sub_result = step04.build_rts_dictionary_csv_result(
            plan,
            sub_output,
            row_start=1,
            row_stop=2,
            column_start=1,
            column_stop=3,
            **kwargs(),
        )
        p = sub_result.parameters
        require((p.row_start, p.row_stop) == (1, 2), "wrong sub rows")
        require((p.column_start, p.column_stop) == (1, 3),
                "wrong sub columns")
        require((p.row_count, p.column_count, p.pixel_count) == (1, 2, 2),
                "wrong sub counts")
        require(p.minimum_score == 0.9, "wrong score")
        require(p.minimum_state_count == 2, "wrong state count")
        require(p.minimum_separation == 5.0, "wrong separation")
        require(p.minimum_transition_count == 3, "wrong transitions")
        require(p.minimum_lower_run == 2, "wrong lower run")
        require(p.minimum_upper_run == 2, "wrong upper run")
        print("   Image range : rows 1:2, columns 1:3")
        print("   Pixel count : 2")
        print("   Thresholds  : preserved")
        print("   Result      : PASS")
        print()

        print("[3/4] Parameters are frozen and summarize deterministically")
        try:
            p.minimum_score = 999.0
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "parameters are mutable")

        expected = {
            "minimum_score": 0.9,
            "minimum_state_count": 2,
            "minimum_separation": 5.0,
            "minimum_transition_count": 3,
            "minimum_lower_run": 2,
            "minimum_upper_run": 2,
            "row_start": 1,
            "row_stop": 2,
            "column_start": 1,
            "column_stop": 3,
            "row_count": 1,
            "column_count": 2,
            "pixel_count": 2,
        }
        require(p.summary() == expected, "summary mismatch")
        require(
            sub_result.summary()["parameters"] == expected,
            "result summary does not include parameters",
        )
        print("   Dataclass      : frozen")
        print("   Summary        : deterministic")
        print("   Result summary : includes parameters")
        print("   Result         : PASS")
        print()

        print("[4/4] CSV, legacy API, and cancellation remain unchanged")
        legacy = root / "legacy.csv"
        returned = step04.build_rts_dictionary_csv(plan, legacy, **kwargs())
        require(returned == legacy, "legacy API returned wrong path")
        require(legacy.read_bytes() == output.read_bytes(), "CSV bytes changed")

        try:
            step04.build_rts_dictionary_csv_result(
                plan,
                root / "cancel.csv",
                cancel_requested=lambda: True,
                **kwargs(),
            )
        except step04.Step04Cancelled as exc:
            require(exc.info.total_pixel_count == 6,
                    "cancellation total changed")
        else:
            require(False, "cancellation no longer raised")
        print("   Legacy API   : unchanged")
        print("   CSV bytes    : identical")
        print("   Cancellation : unchanged")
        print("   Result       : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 dictionary build parameters test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
