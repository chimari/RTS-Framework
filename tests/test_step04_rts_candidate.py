"""Integration test for Step 04 RTS-candidate classification v4.4.0."""

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
                "environment": "step04-v4.4-test",
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
    print("RTS Framework Step 04 candidate classification integration test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_classify_") as temp_dir:
        root = Path(temp_dir)
        manifest = write_dataset(root, [0, 1, 10, 11])
        bias_plan = step03.prepare_bias_analysis(manifest, "bias")
        plan = step04.prepare_rts_dictionary_analysis(bias_plan)
        series = step04.load_pixel_timeseries(plan, row=0, column=0)
        score = step04.compute_two_state_score(series)

        print("[1/4] Candidate passes all inclusive threshold conditions")
        result = step04.classify_rts_candidate(
            score,
            minimum_score=score.score,
            minimum_state_count=2,
            minimum_separation=10.0,
        )
        require(result.score_result is score, "score result was not retained")
        require(result.passes_score, "inclusive score threshold failed")
        require(
            result.passes_state_count,
            "inclusive occupancy threshold failed",
        )
        require(
            result.passes_separation,
            "inclusive separation threshold failed",
        )
        require(result.is_candidate, "all passing conditions were rejected")
        require(result.failed_conditions == (), "unexpected failed condition")
        print("   Score threshold      : PASS")
        print("   State-count threshold: PASS")
        print("   Separation threshold : PASS")
        print("   Candidate            : YES")
        print("   Result               : PASS")
        print()

        print("[2/4] Each failed condition is recorded independently")
        score_failure = step04.classify_rts_candidate(
            score,
            minimum_score=1.0,
            minimum_state_count=2,
            minimum_separation=10.0,
        )
        require(
            score_failure.failed_conditions == ("score",),
            "score-only failure was not isolated",
        )

        count_failure = step04.classify_rts_candidate(
            score,
            minimum_score=0.9,
            minimum_state_count=3,
            minimum_separation=10.0,
        )
        require(
            count_failure.failed_conditions == ("state_count",),
            "count-only failure was not isolated",
        )

        separation_failure = step04.classify_rts_candidate(
            score,
            minimum_score=0.9,
            minimum_state_count=2,
            minimum_separation=10.1,
        )
        require(
            separation_failure.failed_conditions == ("separation",),
            "separation-only failure was not isolated",
        )

        all_failure = step04.classify_rts_candidate(
            score,
            minimum_score=1.0,
            minimum_state_count=3,
            minimum_separation=10.1,
        )
        require(
            all_failure.failed_conditions
            == ("score", "state_count", "separation"),
            "canonical failure order changed",
        )
        require(not all_failure.is_candidate, "failed result became candidate")
        print("   Score failure      : isolated")
        print("   State-count failure: isolated")
        print("   Separation failure : isolated")
        print("   Combined order     : canonical")
        print("   Result             : PASS")
        print()

        print("[3/4] Result and summary are immutable and deterministic")
        expected_summary = {
            "dataset": "bias",
            "row": 0,
            "column": 0,
            "n_frames": 4,
            "score": score.score,
            "lower_state_count": 2,
            "upper_state_count": 2,
            "state_separation": 10.0,
            "minimum_score": score.score,
            "minimum_state_count": 2,
            "minimum_separation": 10.0,
            "passes_score": True,
            "passes_state_count": True,
            "passes_separation": True,
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
            require(False, "frozen candidate result accepted modification")
        print("   Frozen        : YES")
        print("   Deterministic : YES")
        print("   JSON-ready    : YES")
        print("   Result        : PASS")
        print()

        print("[4/4] Invalid result and threshold values are rejected")
        expect_step04_error(
            lambda: step04.classify_rts_candidate(
                object(),
                minimum_score=0.5,
                minimum_state_count=1,
                minimum_separation=0.0,
            ),
            "score_result must be a TwoStateScoreResult",
        )

        for invalid_score in (-0.1, 1.1, np.nan, np.inf, True, "0.5"):
            expect_step04_error(
                lambda value=invalid_score: step04.classify_rts_candidate(
                    score,
                    minimum_score=value,
                    minimum_state_count=1,
                    minimum_separation=0.0,
                ),
                "minimum_score",
            )

        for invalid_count in (0, -1, 1.5, True, "2"):
            expect_step04_error(
                lambda value=invalid_count: step04.classify_rts_candidate(
                    score,
                    minimum_score=0.5,
                    minimum_state_count=value,
                    minimum_separation=0.0,
                ),
                "minimum_state_count",
            )

        for invalid_separation in (-0.1, np.nan, np.inf, True, "1.0"):
            expect_step04_error(
                lambda value=invalid_separation: step04.classify_rts_candidate(
                    score,
                    minimum_score=0.5,
                    minimum_state_count=1,
                    minimum_separation=value,
                ),
                "minimum_separation",
            )
        print("   Invalid result      : rejected")
        print("   Invalid score       : rejected")
        print("   Invalid state count : rejected")
        print("   Invalid separation  : rejected")
        print("   Result              : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 candidate classification integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
