"""Integration test for Step 04 RTS candidate row serialization v4.12.0."""

from __future__ import annotations

import csv
import math
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


EXPECTED_COLUMNS = [
    "dataset",
    "row",
    "column",
    "n_frames",
    "minimum",
    "maximum",
    "mean",
    "median",
    "standard_deviation",
    "median_absolute_deviation",
    "peak_to_peak",
    "lower_state_count",
    "upper_state_count",
    "lower_state_center",
    "upper_state_center",
    "state_separation",
    "single_state_residual",
    "two_state_residual",
    "two_state_score",
    "minimum_score",
    "minimum_state_count",
    "minimum_separation",
    "transition_count",
    "lower_to_upper_count",
    "upper_to_lower_count",
    "longest_lower_run",
    "longest_upper_run",
    "minimum_transition_count",
    "minimum_lower_run",
    "minimum_upper_run",
    "is_candidate",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def write_dataset(root: Path) -> Path:
    candidate_values = [0, 0, 10, 10, 0, 0, 10, 10]
    stable_values = [20, 21, 20, 21, 20, 21, 20, 21]

    paths: list[Path] = []
    for index, (candidate, stable) in enumerate(
        zip(candidate_values, stable_values, strict=True)
    ):
        path = root / f"bias_{index:04d}.fit"
        data = np.array(
            [
                [candidate, stable],
                [stable, stable],
            ],
            dtype=np.uint16,
        )
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows: list[dict[str, object]] = []
    n_frames = len(paths)
    for frame_index, path in enumerate(paths):
        rows.append(
            {
                "dataset": "bias",
                "directory": str(root),
                "environment": "step04-v4.12-test",
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
    print("RTS Framework Step 04 candidate row serialization test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_row_") as temp_dir:
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

        candidate = step04.analyze_rts_pixel(
            plan,
            row=0,
            column=0,
            **thresholds,
        )
        stable = step04.analyze_rts_pixel(
            plan,
            row=0,
            column=1,
            **thresholds,
        )
        require(candidate.is_candidate, "candidate fixture was not accepted")
        require(not stable.is_candidate, "stable fixture was accepted")

        print("[1/4] Canonical flat column order is stable")
        row = step04.rts_candidate_to_row(candidate)
        require(list(row) == EXPECTED_COLUMNS, "column order changed")
        require(len(row) == 31, f"wrong column count: {len(row)}")
        require(
            all(not isinstance(value, (dict, list, tuple)) for value in row.values()),
            "row contains a nested value",
        )
        print("   Columns      : 31")
        print("   Structure    : flat")
        print("   Column order : canonical")
        print("   Result       : PASS")
        print()

        print("[2/4] Serialized values match the analysis result")
        require(row["dataset"] == "bias", "wrong dataset")
        require((row["row"], row["column"]) == (0, 0), "wrong coordinate")
        require(row["n_frames"] == 8, "wrong frame count")
        require(row["lower_state_center"] == candidate.score.lower_state_center,
                "wrong lower-state center")
        require(row["upper_state_center"] == candidate.score.upper_state_center,
                "wrong upper-state center")
        require(row["two_state_score"] == candidate.score.score,
                "wrong two-state score")
        require(row["transition_count"] == candidate.transitions.transition_count,
                "wrong transition count")
        require(row["is_candidate"] is True, "wrong final decision")

        numeric_values = [
            value
            for value in row.values()
            if isinstance(value, float)
        ]
        require(all(math.isfinite(value) for value in numeric_values),
                "row contains a non-finite float")
        print("   Metadata     : MATCH")
        print("   Statistics   : MATCH")
        print("   Two states   : MATCH")
        print("   Transitions  : MATCH")
        print("   Result       : PASS")
        print()

        print("[3/4] Conversion is deterministic and does not mutate the result")
        before = candidate.summary()
        first = step04.rts_candidate_to_row(candidate)
        second = step04.rts_candidate_to_row(candidate)
        require(first == second, "repeated conversion changed values")
        require(list(first) == list(second), "repeated conversion changed order")
        require(first is not second, "conversion reused the same dictionary")
        first["dataset"] = "changed"
        require(second["dataset"] == "bias", "returned dictionaries share state")
        require(candidate.summary() == before, "source result was modified")
        print("   Repeated values : identical")
        print("   New dictionary  : each call")
        print("   Source mutation : none")
        print("   Result          : PASS")
        print()

        print("[4/4] Invalid and non-candidate results are rejected")
        expect_step04_error(
            lambda: step04.rts_candidate_to_row(object()),
            "result must be an RTSPixelAnalysisResult",
        )
        expect_step04_error(
            lambda: step04.rts_candidate_to_row(stable),
            "final RTS candidate",
        )
        print("   Invalid type  : rejected")
        print("   Non-candidate : rejected")
        print("   File I/O      : none")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 candidate row serialization test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
