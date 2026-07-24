"""Integration test for Step 04 one-pixel orchestration v4.7.0."""

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
                "environment": "step04-v4.7-test",
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
    print("RTS Framework Step 04 one-pixel orchestration integration test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_pixel_") as temp_dir:
        root = Path(temp_dir)
        values = [0, 1, 10, 11, 9, 2, 1, 10]
        manifest = write_dataset(root, values)
        bias_plan = step03.prepare_bias_analysis(manifest, "bias")
        plan = step04.prepare_rts_dictionary_analysis(bias_plan)

        thresholds = {
            "minimum_score": 0.9,
            "minimum_state_count": 2,
            "minimum_separation": 5.0,
            "minimum_transition_count": 3,
            "minimum_lower_run": 2,
            "minimum_upper_run": 3,
        }

        print("[1/4] Orchestrator matches the explicit public-API pipeline")
        result = step04.analyze_rts_pixel(
            plan,
            row=0,
            column=0,
            **thresholds,
        )

        series = step04.load_pixel_timeseries(plan, row=0, column=0)
        statistics = step04.compute_pixel_timeseries_statistics(series)
        score = step04.compute_two_state_score(series)
        candidate = step04.classify_rts_candidate(
            score,
            minimum_score=thresholds["minimum_score"],
            minimum_state_count=thresholds["minimum_state_count"],
            minimum_separation=thresholds["minimum_separation"],
        )
        transitions = step04.analyze_two_state_transitions(series, score)
        temporal = step04.classify_temporal_rts_candidate(
            candidate,
            transitions,
            minimum_transition_count=thresholds["minimum_transition_count"],
            minimum_lower_run=thresholds["minimum_lower_run"],
            minimum_upper_run=thresholds["minimum_upper_run"],
        )

        require(result.series.values.tolist() == values, "wrong loaded series")
        require(result.statistics.summary() == statistics.summary(), "statistics differ")
        require(result.score.summary() == score.summary(), "score differs")
        require(result.candidate.summary() == candidate.summary(), "candidate differs")
        require(
            result.transitions.summary() == transitions.summary(),
            "transition analysis differs",
        )
        require(
            result.temporal_candidate.summary() == temporal.summary(),
            "temporal classification differs",
        )
        require(result.is_candidate, "expected final candidate")
        print("   Series             : MATCH")
        print("   Statistics         : MATCH")
        print("   Two-state score    : MATCH")
        print("   Base candidate     : MATCH")
        print("   Transition analysis: MATCH")
        print("   Temporal candidate : MATCH")
        print("   Result             : PASS")
        print()

        print("[2/4] Aggregate retains one internally consistent object graph")
        require(
            result.statistics.series is result.series,
            "statistics did not retain aggregate series",
        )
        require(result.score.series is result.series, "score did not retain series")
        require(
            result.candidate.score_result is result.score,
            "candidate did not retain aggregate score",
        )
        require(
            result.transitions.series is result.series,
            "transitions did not retain aggregate series",
        )
        require(
            result.transitions.score_result is result.score,
            "transitions did not retain aggregate score",
        )
        require(
            result.temporal_candidate.candidate_result is result.candidate,
            "temporal result did not retain aggregate candidate",
        )
        require(
            result.temporal_candidate.transition_result is result.transitions,
            "temporal result did not retain aggregate transitions",
        )
        print("   Series identity    : consistent")
        print("   Score identity     : consistent")
        print("   Candidate identity : consistent")
        print("   Transition identity: consistent")
        print("   Result             : PASS")
        print()

        print("[3/4] Aggregate and nested summary are immutable and deterministic")
        summary = result.summary()
        require(summary["dataset"] == "bias", "wrong summary dataset")
        require(summary["row"] == 0, "wrong summary row")
        require(summary["column"] == 0, "wrong summary column")
        require(summary["n_frames"] == 8, "wrong summary frame count")
        require(summary["statistics"] == result.statistics.summary(), "nested statistics changed")
        require(summary["score"] == result.score.summary(), "nested score changed")
        require(summary["candidate"] == result.candidate.summary(), "nested candidate changed")
        require(
            summary["transitions"] == result.transitions.summary(),
            "nested transitions changed",
        )
        require(
            summary["temporal_candidate"]
            == result.temporal_candidate.summary(),
            "nested temporal result changed",
        )
        require(summary["is_candidate"] is True, "wrong summary final decision")
        require(result.summary() == result.summary(), "summary is not deterministic")
        try:
            result.series = series
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "frozen pixel result accepted modification")
        print("   Frozen        : YES")
        print("   Deterministic : YES")
        print("   JSON-ready    : YES")
        print("   Result        : PASS")
        print()

        print("[4/4] Underlying validation errors propagate unchanged")
        expect_step04_error(
            lambda: step04.analyze_rts_pixel(
                plan,
                row=99,
                column=0,
                **thresholds,
            ),
            "row",
        )

        bad_thresholds = dict(thresholds)
        bad_thresholds["minimum_score"] = 2.0
        expect_step04_error(
            lambda: step04.analyze_rts_pixel(
                plan,
                row=0,
                column=0,
                **bad_thresholds,
            ),
            "minimum_score",
        )

        bad_thresholds = dict(thresholds)
        bad_thresholds["minimum_transition_count"] = -1
        expect_step04_error(
            lambda: step04.analyze_rts_pixel(
                plan,
                row=0,
                column=0,
                **bad_thresholds,
            ),
            "minimum_transition_count",
        )
        print("   Invalid coordinate : propagated")
        print("   Invalid base limit : propagated")
        print("   Invalid time limit : propagated")
        print("   Result             : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 one-pixel orchestration integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
