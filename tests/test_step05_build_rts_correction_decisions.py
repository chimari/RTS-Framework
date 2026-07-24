"""Integration test for Step 05 correction decisions v5.2.0."""

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
            "environment": "step05-v5.2-test",
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


def prepare_plan(root: Path) -> step05.RTSCorrectionPlan:
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
    return step05.prepare_rts_correction(built.metadata_path, target)


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 05 correction-decision test")
    print("=" * 72)
    print(f"step05 version : {step05.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step05_decision_") as temp:
        root = Path(temp)
        plan = prepare_plan(root)
        require(plan.candidate_count >= 2, "too few test candidates")
        target = plan.input_path

        print("[1/4] LOWER and UPPER decisions produce correct targets")
        values = tuple(
            candidate.lower_state_center + 0.1 * candidate.state_separation
            if index % 2 == 0
            else candidate.upper_state_center - 0.1 * candidate.state_separation
            for index, candidate in enumerate(plan.candidates)
        )
        write_target(target, plan, values)
        classified = step05.classify_rts_correction_candidates(plan)
        decided = step05.build_rts_correction_decisions(classified)

        for decision in decided.decisions:
            require(decision.is_correctable, "state decision rejected")
            if decision.current_state is step05.RTSCandidateState.LOWER:
                expected_target = decision.candidate.lower_state_center
                expected_reason = (
                    step05.RTSCorrectionDecisionReason.LOWER_STATE
                )
            else:
                expected_target = decision.candidate.upper_state_center
                expected_reason = (
                    step05.RTSCorrectionDecisionReason.UPPER_STATE
                )
            require(
                decision.target_value == expected_target,
                "target value changed",
            )
            require(
                np.isclose(
                    decision.correction_value,
                    expected_target - decision.current_value,
                ),
                "correction value changed",
            )
            require(decision.reason is expected_reason, "reason changed")

        require(
            decided.correctable_count == plan.candidate_count,
            "correctable count changed",
        )
        print("   LOWER target  : lower_state_center")
        print("   UPPER target  : upper_state_center")
        print("   Delta         : target - current")
        print("   Result        : PASS")
        print()

        print("[2/4] MIDPOINT and OUTSIDE decisions are rejected")
        values = tuple(
            candidate.midpoint if index % 2 == 0
            else candidate.upper_state_center
            + 2.0 * candidate.state_separation
            for index, candidate in enumerate(plan.candidates)
        )
        write_target(target, plan, values)
        classified = step05.classify_rts_correction_candidates(plan)
        decided = step05.build_rts_correction_decisions(classified)

        for decision in decided.decisions:
            require(not decision.is_correctable, "unsafe decision accepted")
            require(decision.target_value is None, "rejected target exists")
            require(
                decision.correction_value is None,
                "rejected correction exists",
            )
            expected_reason = (
                step05.RTSCorrectionDecisionReason.MIDPOINT_UNCERTAIN
                if decision.current_state
                is step05.RTSCandidateState.MIDPOINT
                else step05.RTSCorrectionDecisionReason.OUTSIDE_DICTIONARY
            )
            require(decision.reason is expected_reason, "rejection reason changed")

        require(
            decided.rejected_count == plan.candidate_count,
            "rejected count changed",
        )
        print("   MIDPOINT      : rejected")
        print("   OUTSIDE       : rejected")
        print("   Target/delta  : None")
        print("   Result        : PASS")
        print()

        print("[3/4] Type boundary and summary counts are enforced")
        try:
            step05.build_rts_correction_decisions(plan)
        except step05.Step05Error:
            pass
        else:
            require(False, "non-classification result accepted")

        summary = decided.summary()
        require(
            summary["decision_count"] == plan.candidate_count,
            "decision summary count changed",
        )
        require(
            summary["correctable_count"] == 0,
            "summary correctable count changed",
        )
        require(
            summary["rejected_count"] == plan.candidate_count,
            "summary rejected count changed",
        )
        print("   Wrong input   : rejected")
        print("   Summary counts: consistent")
        print("   Result        : PASS")
        print()

        print("[4/4] Decisions are immutable and deterministic")
        first = step05.build_rts_correction_decisions(classified)
        second = step05.build_rts_correction_decisions(classified)
        require(first.summary() == second.summary(), "summary changed")
        require(
            json.dumps(first.summary(), sort_keys=True)
            == json.dumps(second.summary(), sort_keys=True),
            "serialized summary changed",
        )
        try:
            first.decisions = ()
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "decision result is mutable")
        try:
            first.decisions[0].is_correctable = True
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "decision is mutable")
        print("   Result object : immutable")
        print("   Decisions     : immutable")
        print("   Summary       : deterministic")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 05 correction-decision test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
