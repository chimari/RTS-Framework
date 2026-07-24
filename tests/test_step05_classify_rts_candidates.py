"""Integration test for Step 05 candidate classification v5.1.0."""

from __future__ import annotations

import csv
import json
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
from steps import step05_apply_rts_correction as step05


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def write_manifest(root: Path) -> Path:
    paths = []
    for index in range(8):
        data = np.array(
            [
                [0 if index % 4 < 2 else 10, 20, 30],
                [10 if index % 4 < 2 else 0, 5, 50],
            ],
            dtype=np.uint16,
        )
        path = root / f"bias_{index:04d}.fit"
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows = []
    for index, path in enumerate(paths):
        rows.append({
            "dataset": "bias",
            "directory": str(root),
            "environment": "step05-v5.1-test",
            "frame_index": index,
            "n_frames": len(paths),
            "temperature_C": -10.0,
            "temperature_start_C": -10.0,
            "temperature_end_C": -10.0,
            "temperature_fraction": index / (len(paths) - 1),
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


def write_target(
    path: Path,
    plan: step05.RTSCorrectionPlan,
    values: tuple[float, ...],
) -> None:
    data = np.full(plan.image_shape, 20.0, dtype=np.float32)
    for candidate, value in zip(plan.candidates, values, strict=True):
        data[candidate.row, candidate.column] = value
    fits.PrimaryHDU(data=data).writeto(path, overwrite=True)


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 05 candidate classification test")
    print("=" * 72)
    print(f"step05 version : {step05.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step05_classify_") as temp:
        root = Path(temp)
        manifest = write_manifest(root)
        analysis = step03.prepare_bias_analysis(manifest, "bias")
        plan04 = step04.prepare_rts_dictionary_analysis(analysis)
        built = step04.build_rts_dictionary_artifacts(
            plan04,
            root / "dictionary.csv",
            minimum_score=0.9,
            minimum_state_count=2,
            minimum_separation=5.0,
            minimum_transition_count=3,
            minimum_lower_run=2,
            minimum_upper_run=2,
        )
        step04.audit_rts_dictionary_input_files(built.metadata_path)

        target = root / "target.fit"
        fits.PrimaryHDU(
            data=np.zeros((2, 3), dtype=np.float32)
        ).writeto(target, overwrite=True)
        plan = step05.prepare_rts_correction(built.metadata_path, target)
        require(plan.candidate_count >= 2, "too few test candidates")

        print("[1/4] State centers are classified as LOWER and UPPER")
        values = tuple(
            candidate.lower_state_center if index % 2 == 0
            else candidate.upper_state_center
            for index, candidate in enumerate(plan.candidates)
        )
        write_target(target, plan, values)
        result = step05.classify_rts_correction_candidates(plan)
        expected = tuple(
            step05.RTSCandidateState.LOWER if index % 2 == 0
            else step05.RTSCandidateState.UPPER
            for index in range(plan.candidate_count)
        )
        require(
            tuple(item.state for item in result.classifications) == expected,
            "state-center classification changed",
        )
        print("   Lower centers : LOWER")
        print("   Upper centers : UPPER")
        print("   Result        : PASS")
        print()

        print("[2/4] MIDPOINT and OUTSIDE are distinguished")
        values = tuple(
            candidate.midpoint if index % 2 == 0
            else candidate.upper_state_center
            + 2.0 * candidate.state_separation
            for index, candidate in enumerate(plan.candidates)
        )
        write_target(target, plan, values)
        result = step05.classify_rts_correction_candidates(plan)
        expected = tuple(
            step05.RTSCandidateState.MIDPOINT if index % 2 == 0
            else step05.RTSCandidateState.OUTSIDE
            for index in range(plan.candidate_count)
        )
        require(
            tuple(item.state for item in result.classifications) == expected,
            "midpoint/outside classification changed",
        )
        print("   Between bands : MIDPOINT")
        print("   Beyond states : OUTSIDE")
        print("   Result        : PASS")
        print()

        print("[3/4] Invalid tolerance and changed FITS are rejected")
        for invalid in (-0.1, 0.5, float("inf"), True):
            try:
                step05.classify_rts_correction_candidates(
                    plan, state_tolerance_fraction=invalid
                )
            except step05.Step05Error:
                pass
            else:
                require(False, f"invalid tolerance accepted: {invalid!r}")

        fits.PrimaryHDU(
            data=np.zeros((3, 3), dtype=np.float32)
        ).writeto(target, overwrite=True)
        try:
            step05.classify_rts_correction_candidates(plan)
        except step05.Step05Error as exc:
            require(
                "image_shape changed" in str(exc),
                f"unexpected changed-FITS error: {exc}",
            )
        else:
            require(False, "changed FITS shape accepted")
        print("   Bad tolerance : rejected")
        print("   Changed shape : rejected")
        print("   Result        : PASS")
        print()

        print("[4/4] Results are immutable and deterministic")
        write_target(
            target,
            plan,
            tuple(candidate.midpoint for candidate in plan.candidates),
        )
        first = step05.classify_rts_correction_candidates(
            plan, state_tolerance_fraction=0.2
        )
        second = step05.classify_rts_correction_candidates(
            plan, state_tolerance_fraction=0.2
        )
        require(first.summary() == second.summary(), "summary changed")
        require(
            json.dumps(first.summary(), sort_keys=True)
            == json.dumps(second.summary(), sort_keys=True),
            "serialized summary changed",
        )
        try:
            first.state_tolerance_fraction = 0.1
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "result is mutable")
        try:
            first.classifications[0].pixel_value = 999.0
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "classification is mutable")
        print("   Result object : immutable")
        print("   Items         : immutable")
        print("   Summary       : deterministic")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 05 candidate classification test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
