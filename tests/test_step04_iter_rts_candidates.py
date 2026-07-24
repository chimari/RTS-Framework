"""Integration test for Step 04 RTS candidate filtering v4.11.0."""

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
    # Pixel (0, 0): strong alternating two-state pattern.
    # Pixel (0, 1): nearly constant non-candidate pattern.
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
                "environment": "step04-v4.11-test",
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
    print("RTS Framework Step 04 candidate filtering test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_candidates_") as temp_dir:
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

        print("[1/4] Only final candidates are yielded in input order")
        results = [stable, candidate, stable, candidate]
        filtered = list(step04.iter_rts_candidates(results))
        require(filtered == [candidate, candidate], "candidate filtering changed")
        print("   Input results : 4")
        print("   Candidates    : 2")
        print("   Order         : preserved")
        print("   Result        : PASS")
        print()

        print("[2/4] Accepted objects are yielded unchanged")
        filtered = list(step04.iter_rts_candidates([candidate]))
        require(len(filtered) == 1, "candidate was not yielded")
        require(filtered[0] is candidate, "candidate object was copied or replaced")

        duplicate_input = [candidate, candidate]
        duplicate_output = list(step04.iter_rts_candidates(duplicate_input))
        require(len(duplicate_output) == 2, "duplicate references were removed")
        require(
            duplicate_output[0] is candidate and duplicate_output[1] is candidate,
            "duplicate object identity changed",
        )
        print("   Object identity : preserved")
        print("   Duplicate refs  : preserved")
        print("   Mutation        : none")
        print("   Result          : PASS")
        print()

        print("[3/4] One-shot input is consumed lazily")
        consumed: list[str] = []

        def one_shot():
            consumed.append("stable")
            yield stable
            consumed.append("candidate")
            yield candidate
            consumed.append("after")
            yield stable

        iterator = step04.iter_rts_candidates(one_shot())
        require(iter(iterator) is iterator, "result is not an iterator")
        require(consumed == [], "input was consumed eagerly")
        first = next(iterator)
        require(first is candidate, "wrong first candidate")
        require(
            consumed == ["stable", "candidate"],
            f"wrong lazy consumption state: {consumed}",
        )
        require(list(iterator) == [], "unexpected remaining candidates")
        require(
            consumed == ["stable", "candidate", "after"],
            "input was not fully consumed after exhaustion",
        )
        print("   Eager consumption : none")
        print("   First next()      : stops at first candidate")
        print("   One-shot source   : supported")
        print("   Result            : PASS")
        print()

        print("[4/4] Invalid inputs and items are rejected lazily")
        non_iterable = step04.iter_rts_candidates(123)
        expect_step04_error(
            lambda: next(non_iterable),
            "results must be an iterable",
        )

        mixed = step04.iter_rts_candidates([stable, "bad", candidate])
        expect_step04_error(
            lambda: next(mixed),
            "results item 1",
        )

        def delayed_bad_item():
            yield stable
            yield candidate
            yield object()

        delayed = step04.iter_rts_candidates(delayed_bad_item())
        require(next(delayed) is candidate, "candidate before bad item was lost")
        expect_step04_error(
            lambda: next(delayed),
            "results item 2",
        )

        print("   Non-iterable input : rejected")
        print("   Invalid item       : rejected")
        print("   Lazy propagation   : confirmed")
        print("   Result             : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 candidate filtering test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
