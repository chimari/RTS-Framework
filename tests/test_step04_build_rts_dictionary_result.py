"""Integration test for Step 04 RTS dictionary build result v4.15.0."""

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
                "environment": "step04-v4.15-test",
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
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def expect_step04_error(callable_, contains: str) -> None:
    try:
        callable_()
    except step04.Step04Error as exc:
        require(contains in str(exc), f"wrong error message: {exc}")
    else:
        require(False, "Step04Error was not raised")


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 dictionary build result test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_result_") as temp_dir:
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

        print("[1/4] Full-image result reports canonical counts and metadata")
        output = root / "full.csv"
        result = step04.build_rts_dictionary_csv_result(
            plan, output, **thresholds
        )
        require(
            isinstance(result, step04.RTSDictionaryBuildResult),
            "wrong result type",
        )
        require(result.output_path == output, "wrong output path")
        require(result.dataset == "bias", "wrong dataset")
        require(
            (result.row_start, result.row_stop, result.column_start, result.column_stop)
            == (0, 2, 0, 2),
            "wrong resolved full-image ROI",
        )
        require(result.region_shape == (2, 2), "wrong region shape")
        require(result.analyzed_pixel_count == 4, "wrong analyzed pixel count")
        require(result.candidate_count == 2, "wrong candidate count")
        require(len(read_rows(output)) == 2, "CSV count differs from result")
        print("   Region          : 2x2")
        print("   Analyzed pixels : 4")
        print("   Candidates      : 2")
        print("   Result          : PASS")
        print()

        print("[2/4] ROI and empty ROI counts are exact")
        roi_output = root / "roi.csv"
        roi = step04.build_rts_dictionary_csv_result(
            plan,
            roi_output,
            row_start=1,
            row_stop=2,
            column_start=0,
            column_stop=2,
            **thresholds,
        )
        require(roi.region_shape == (1, 2), "wrong ROI shape")
        require(roi.analyzed_pixel_count == 2, "wrong ROI pixel count")
        require(roi.candidate_count == 1, "wrong ROI candidate count")

        empty_output = root / "empty.csv"
        empty = step04.build_rts_dictionary_csv_result(
            plan,
            empty_output,
            row_start=1,
            row_stop=1,
            **thresholds,
        )
        require(empty.region_shape == (0, 2), "wrong empty ROI shape")
        require(empty.analyzed_pixel_count == 0, "empty ROI analyzed pixels")
        require(empty.candidate_count == 0, "empty ROI candidates")
        require(read_rows(empty_output) == [], "empty ROI CSV has rows")
        print("   Selected ROI : exact counts")
        print("   Empty ROI    : zero counts")
        print("   Result       : PASS")
        print()

        print("[3/4] Result is immutable and legacy API remains compatible")
        try:
            result.candidate_count = 99
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "build result is mutable")

        summary = result.summary()
        require(summary["output_path"] == str(output), "wrong summary path")
        require(summary["region_shape"] == (2, 2), "wrong summary shape")
        require(summary["candidate_count"] == 2, "wrong summary count")

        legacy_output = root / "legacy.csv"
        legacy_return = step04.build_rts_dictionary_csv(
            plan, legacy_output, **thresholds
        )
        require(isinstance(legacy_return, Path), "legacy return type changed")
        require(legacy_return == legacy_output, "legacy returned wrong path")
        require(
            legacy_output.read_bytes() == output.read_bytes(),
            "legacy CSV output changed",
        )
        print("   Dataclass       : frozen")
        print("   Summary         : deterministic")
        print("   Legacy return   : pathlib.Path")
        print("   Legacy CSV      : identical")
        print("   Result          : PASS")
        print()

        print("[4/4] Validation failures create no result and preserve output")
        protected = root / "protected.csv"
        original = "ORIGINAL\n"
        protected.write_text(original, encoding="utf-8")

        expect_step04_error(
            lambda: step04.build_rts_dictionary_csv_result(
                plan,
                protected,
                row_start=-1,
                **thresholds,
            ),
            "row_start",
        )
        require(
            protected.read_text(encoding="utf-8") == original,
            "destination changed after invalid ROI",
        )

        expect_step04_error(
            lambda: step04.build_rts_dictionary_csv_result(
                object(),
                protected,
                **thresholds,
            ),
            "plan",
        )
        require(
            protected.read_text(encoding="utf-8") == original,
            "destination changed after invalid plan",
        )
        print("   Invalid ROI  : rejected")
        print("   Invalid plan : rejected")
        print("   Destination  : preserved")
        print("   Result       : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 dictionary build result test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
