"""Integration test for Step 04 temporal RTS candidate classification v4.6.0."""

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
                "environment": "step04-v4.6-test",
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
    print("RTS Framework Step 04 temporal candidate integration test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_temporal_") as temp_dir:
        root = Path(temp_dir)
        manifest = write_dataset(root, [0, 1, 10, 11, 9, 2, 1, 10])
        bias_plan = step03.prepare_bias_analysis(manifest, "bias")
        plan = step04.prepare_rts_dictionary_analysis(bias_plan)
        series = step04.load_pixel_timeseries(plan, row=0, column=0)
        score = step04.compute_two_state_score(series)
        candidate = step04.classify_rts_candidate(
            score,
            minimum_score=0.9,
            minimum_state_count=2,
            minimum_separation=5.0,
        )
        transitions = step04.analyze_two_state_transitions(series, score)

        print("[1/4] Base and temporal conditions pass inclusively")
        result = step04.classify_temporal_rts_candidate(
            candidate,
            transitions,
            minimum_transition_count=3,
            minimum_lower_run=2,
            minimum_upper_run=3,
        )
        require(result.candidate_result is candidate, "candidate was not retained")
        require(
            result.transition_result is transitions,
            "transition result was not retained",
        )
        require(result.passes_base_candidate, "base candidate failed")
        require(result.passes_transition_count, "inclusive transition count failed")
        require(result.passes_lower_run, "inclusive lower run failed")
        require(result.passes_upper_run, "inclusive upper run failed")
        require(result.is_candidate, "all passing conditions were rejected")
        require(result.failed_conditions == (), "unexpected failed condition")
        print("   Base candidate      : PASS")
        print("   Transition threshold: PASS")
        print("   Lower-run threshold : PASS")
        print("   Upper-run threshold : PASS")
        print("   Temporal candidate  : YES")
        print("   Result              : PASS")
        print()

        print("[2/4] Each failed condition is recorded independently")
        base_failure_candidate = replace(candidate, is_candidate=False)
        base_failure = step04.classify_temporal_rts_candidate(
            base_failure_candidate,
            transitions,
            minimum_transition_count=3,
            minimum_lower_run=2,
            minimum_upper_run=3,
        )
        require(
            base_failure.failed_conditions == ("base_candidate",),
            "base-only failure was not isolated",
        )

        transition_failure = step04.classify_temporal_rts_candidate(
            candidate,
            transitions,
            minimum_transition_count=4,
            minimum_lower_run=2,
            minimum_upper_run=3,
        )
        require(
            transition_failure.failed_conditions == ("transition_count",),
            "transition-only failure was not isolated",
        )

        lower_failure = step04.classify_temporal_rts_candidate(
            candidate,
            transitions,
            minimum_transition_count=3,
            minimum_lower_run=3,
            minimum_upper_run=3,
        )
        require(
            lower_failure.failed_conditions == ("lower_run",),
            "lower-run-only failure was not isolated",
        )

        upper_failure = step04.classify_temporal_rts_candidate(
            candidate,
            transitions,
            minimum_transition_count=3,
            minimum_lower_run=2,
            minimum_upper_run=4,
        )
        require(
            upper_failure.failed_conditions == ("upper_run",),
            "upper-run-only failure was not isolated",
        )

        all_failure = step04.classify_temporal_rts_candidate(
            base_failure_candidate,
            transitions,
            minimum_transition_count=4,
            minimum_lower_run=3,
            minimum_upper_run=4,
        )
        require(
            all_failure.failed_conditions
            == (
                "base_candidate",
                "transition_count",
                "lower_run",
                "upper_run",
            ),
            "canonical failure order changed",
        )
        require(not all_failure.is_candidate, "failed result became candidate")
        print("   Base failure       : isolated")
        print("   Transition failure : isolated")
        print("   Lower-run failure  : isolated")
        print("   Upper-run failure  : isolated")
        print("   Combined order     : canonical")
        print("   Result             : PASS")
        print()

        print("[3/4] Result and summary are immutable and deterministic")
        expected_summary = {
            "dataset": "bias",
            "row": 0,
            "column": 0,
            "n_frames": 8,
            "base_candidate": True,
            "transition_count": 3,
            "longest_lower_run": 2,
            "longest_upper_run": 3,
            "minimum_transition_count": 3,
            "minimum_lower_run": 2,
            "minimum_upper_run": 3,
            "passes_base_candidate": True,
            "passes_transition_count": True,
            "passes_lower_run": True,
            "passes_upper_run": True,
            "is_candidate": True,
            "failed_conditions": [],
        }
        require(result.summary() == expected_summary, "summary content changed")
        require(result.summary() == result.summary(), "summary is not deterministic")
        try:
            result.is_candidate = False
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "frozen temporal result accepted modification")
        print("   Frozen        : YES")
        print("   Deterministic : YES")
        print("   JSON-ready    : YES")
        print("   Result        : PASS")
        print()

        print("[4/4] Invalid, mismatched, and bad threshold inputs are rejected")
        expect_step04_error(
            lambda: step04.classify_temporal_rts_candidate(
                object(),
                transitions,
                minimum_transition_count=0,
                minimum_lower_run=1,
                minimum_upper_run=1,
            ),
            "candidate_result must be an RTSCandidateResult",
        )
        expect_step04_error(
            lambda: step04.classify_temporal_rts_candidate(
                candidate,
                object(),
                minimum_transition_count=0,
                minimum_lower_run=1,
                minimum_upper_run=1,
            ),
            "transition_result must be a TwoStateTransitionResult",
        )

        different_candidate = replace(
            candidate,
            score_result=replace(score),
        )
        expect_step04_error(
            lambda: step04.classify_temporal_rts_candidate(
                different_candidate,
                transitions,
                minimum_transition_count=0,
                minimum_lower_run=1,
                minimum_upper_run=1,
            ),
            "same TwoStateScoreResult object",
        )

        for invalid_count in (-1, 1.5, True, "1"):
            expect_step04_error(
                lambda value=invalid_count: step04.classify_temporal_rts_candidate(
                    candidate,
                    transitions,
                    minimum_transition_count=value,
                    minimum_lower_run=1,
                    minimum_upper_run=1,
                ),
                "minimum_transition_count",
            )

        for name, invalid_value in (
            ("minimum_lower_run", 0),
            ("minimum_lower_run", 1.5),
            ("minimum_lower_run", True),
            ("minimum_upper_run", 0),
            ("minimum_upper_run", 1.5),
            ("minimum_upper_run", True),
        ):
            kwargs = {
                "minimum_transition_count": 0,
                "minimum_lower_run": 1,
                "minimum_upper_run": 1,
            }
            kwargs[name] = invalid_value
            expect_step04_error(
                lambda kwargs=kwargs, name=name: step04.classify_temporal_rts_candidate(
                    candidate,
                    transitions,
                    **kwargs,
                ),
                name,
            )

        print("   Invalid types       : rejected")
        print("   Mismatched results  : rejected")
        print("   Invalid transitions : rejected")
        print("   Invalid run lengths : rejected")
        print("   Result              : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 temporal candidate integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
