"""Integration test for Step 04 two-state transition analysis v4.5.0."""

from __future__ import annotations

import csv
import sys
import tempfile
from dataclasses import FrozenInstanceError, replace
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


def write_dataset(root: Path, values: list[int]) -> Path:
    paths: list[Path] = []
    for index, value in enumerate(values):
        path = root / f"bias_{index:04d}.fit"
        data = np.array(
            [
                [value, 100 + index],
                [200 + index, 300 + index],
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
                "environment": "step04-v4.5-test",
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


def readonly(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array.setflags(write=False)
    return array


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 transition analysis integration test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_transition_") as temp_dir:
        root = Path(temp_dir)
        values = [0, 1, 10, 11, 9, 2, 1, 10]
        manifest = write_dataset(root, values)
        bias_plan = step03.prepare_bias_analysis(manifest, "bias")
        plan = step04.prepare_rts_dictionary_analysis(bias_plan)
        series = step04.load_pixel_timeseries(plan, row=0, column=0)
        score = step04.compute_two_state_score(series)

        print("[1/4] Temporal order produces exact transition statistics")
        result = step04.analyze_two_state_transitions(series, score)
        expected_sequence = (
            "lower", "lower", "upper", "upper",
            "upper", "lower", "lower", "upper",
        )
        require(result.series is series, "source series was not retained")
        require(result.score_result is score, "score result was not retained")
        require(result.state_sequence == expected_sequence, "wrong state sequence")
        require(result.lower_state_count == 4, "wrong lower-state count")
        require(result.upper_state_count == 4, "wrong upper-state count")
        require(result.transition_count == 3, "wrong transition count")
        require(result.lower_to_upper_count == 2, "wrong lower->upper count")
        require(result.upper_to_lower_count == 1, "wrong upper->lower count")
        require(result.longest_lower_run == 2, "wrong longest lower run")
        require(result.longest_upper_run == 3, "wrong longest upper run")
        print("   Sequence         : LLUUULLU")
        print("   Transitions      : 3")
        print("   Lower -> upper   : 2")
        print("   Upper -> lower   : 1")
        print("   Longest runs     : lower=2, upper=3")
        print("   Result           : PASS")
        print()

        print("[2/4] Midpoint ties and no-transition series are deterministic")
        tied_series = replace(
            series,
            values=readonly([0.0, 5.0, 10.0]),
            n_frames=3,
        )
        tied_score = step04.compute_two_state_score(tied_series)
        tied_score = replace(
            tied_score,
            lower_state_center=0.0,
            upper_state_center=10.0,
            state_separation=10.0,
        )
        tied_result = step04.analyze_two_state_transitions(
            tied_series,
            tied_score,
        )
        require(
            tied_result.state_sequence[1] == "lower",
            "exact midpoint tie was not assigned to lower state",
        )

        constant_series = replace(
            series,
            values=readonly([5.0, 5.0, 5.0, 5.0]),
            n_frames=4,
        )
        constant_score = step04.compute_two_state_score(constant_series)
        constant_result = step04.analyze_two_state_transitions(
            constant_series,
            constant_score,
        )
        require(
            constant_result.state_sequence
            == ("lower", "lower", "lower", "lower"),
            "constant sequence was not assigned deterministically",
        )
        require(constant_result.transition_count == 0, "constant series transitioned")
        require(constant_result.longest_lower_run == 4, "wrong constant run length")
        require(constant_result.longest_upper_run == 0, "unused upper run not zero")
        print("   Midpoint tie     : lower")
        print("   Constant sequence: LLLL")
        print("   Constant changes : 0")
        print("   Result           : PASS")
        print()

        print("[3/4] Result and summary are immutable and deterministic")
        expected_summary = {
            "dataset": "bias",
            "row": 0,
            "column": 0,
            "n_frames": 8,
            "lower_state_count": 4,
            "upper_state_count": 4,
            "transition_count": 3,
            "lower_to_upper_count": 2,
            "upper_to_lower_count": 1,
            "longest_lower_run": 2,
            "longest_upper_run": 3,
            "state_sequence": list(expected_sequence),
        }
        require(result.summary() == expected_summary, "summary content changed")
        require(result.summary() == result.summary(), "summary is not deterministic")
        try:
            result.transition_count = 0
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "frozen transition result accepted modification")
        print("   Frozen        : YES")
        print("   Deterministic : YES")
        print("   JSON-ready    : YES")
        print("   Result        : PASS")
        print()

        print("[4/4] Invalid and mismatched inputs are rejected")
        expect_step04_error(
            lambda: step04.analyze_two_state_transitions(object(), score),
            "series must be a PixelTimeSeries",
        )
        expect_step04_error(
            lambda: step04.analyze_two_state_transitions(series, object()),
            "score_result must be a TwoStateScoreResult",
        )

        other_series = replace(series, values=readonly(values))
        expect_step04_error(
            lambda: step04.analyze_two_state_transitions(
                other_series,
                score,
            ),
            "same PixelTimeSeries object",
        )

        mismatched_score = replace(score, n_frames=7)
        expect_step04_error(
            lambda: step04.analyze_two_state_transitions(
                series,
                mismatched_score,
            ),
            "frame count does not match",
        )

        non_finite_series = replace(
            series,
            values=readonly([0.0, 1.0, np.nan, 11.0, 9.0, 2.0, 1.0, 10.0]),
        )
        non_finite_score = replace(score, series=non_finite_series)
        expect_step04_error(
            lambda: step04.analyze_two_state_transitions(
                non_finite_series,
                non_finite_score,
            ),
            "must all be finite",
        )
        print("   Invalid type       : rejected")
        print("   Different series   : rejected")
        print("   Frame-count mismatch: rejected")
        print("   NaN/Inf            : rejected")
        print("   Result             : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 transition analysis integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
