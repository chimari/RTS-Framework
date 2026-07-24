"""Integration test for Step 05 in-memory correction application v5.3.0."""

from __future__ import annotations

import csv
import hashlib
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            "environment": "step05-v5.3-test",
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


def prepare_plan(root: Path, dtype=np.float32) -> step05.RTSCorrectionPlan:
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
        data=np.zeros((2, 3), dtype=dtype)
    ).writeto(target, overwrite=True)
    return step05.prepare_rts_correction(built.metadata_path, target)


def write_target(
    path: Path,
    plan: step05.RTSCorrectionPlan,
    values: tuple[float, ...],
    dtype,
) -> np.ndarray:
    data = np.full(plan.image_shape, 20, dtype=dtype)
    for candidate, value in zip(plan.candidates, values, strict=True):
        data[candidate.row, candidate.column] = value
    fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
    return data


def build_result(plan: step05.RTSCorrectionPlan):
    classified = step05.classify_rts_correction_candidates(plan)
    return step05.build_rts_correction_decisions(classified)


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 05 in-memory correction application test")
    print("=" * 72)
    print(f"step05 version : {step05.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step05_apply_") as temp:
        root = Path(temp)
        plan = prepare_plan(root, np.float32)
        require(plan.candidate_count >= 2, "too few test candidates")
        target = plan.input_path

        print("[1/4] Correctable decisions modify only the output copy")
        values = tuple(
            candidate.lower_state_center + 0.1 * candidate.state_separation
            if index % 2 == 0
            else candidate.upper_state_center - 0.1 * candidate.state_separation
            for index, candidate in enumerate(plan.candidates)
        )
        original = write_target(target, plan, values, np.float32)
        before_hash = sha256(target)
        decisions = build_result(plan)
        applied = step05.apply_rts_correction_in_memory(decisions)
        after_hash = sha256(target)

        require(before_hash == after_hash, "input FITS was modified")
        with fits.open(target, memmap=False) as hdul:
            require(
                np.array_equal(hdul[0].data, original),
                "source FITS pixel values changed",
            )

        for decision in decisions.decisions:
            require(decision.is_correctable, "test decision was rejected")
            require(
                np.isclose(
                    applied.corrected_image[decision.coordinate],
                    decision.target_value,
                ),
                "corrected target value changed",
            )
        require(
            applied.applied_count == plan.candidate_count,
            "applied count changed",
        )
        require(applied.preserved_count == 0, "preserved count changed")
        print("   Input FITS    : unchanged")
        print("   Output array  : corrected")
        print("   Applied count : consistent")
        print("   Result        : PASS")
        print()

        print("[2/4] Rejected decisions preserve candidate values")
        values = tuple(
            candidate.midpoint if index % 2 == 0
            else candidate.upper_state_center
            + 2.0 * candidate.state_separation
            for index, candidate in enumerate(plan.candidates)
        )
        original = write_target(target, plan, values, np.float32)
        decisions = build_result(plan)
        applied = step05.apply_rts_correction_in_memory(decisions)

        require(
            np.array_equal(applied.corrected_image, original),
            "rejected candidate value changed",
        )
        require(applied.applied_count == 0, "rejected decision applied")
        require(
            applied.preserved_count == plan.candidate_count,
            "preserved count changed",
        )
        print("   MIDPOINT      : preserved")
        print("   OUTSIDE       : preserved")
        print("   Output array  : unchanged copy")
        print("   Result        : PASS")
        print()

        print("[3/4] Integer rounding and stale-value checks are enforced")
        integer_root = root / "integer"
        integer_root.mkdir()
        integer_plan = prepare_plan(integer_root, np.uint16)
        integer_values = tuple(
            candidate.lower_state_center
            for candidate in integer_plan.candidates
        )
        write_target(
            integer_plan.input_path,
            integer_plan,
            integer_values,
            np.uint16,
        )
        integer_decisions = build_result(integer_plan)
        integer_applied = step05.apply_rts_correction_in_memory(
            integer_decisions
        )
        require(
            integer_applied.corrected_image.dtype == np.dtype("uint16"),
            "integer dtype changed",
        )

        with fits.open(integer_plan.input_path, mode="update", memmap=False) as hdul:
            row, column = integer_plan.candidates[0].coordinate
            hdul[0].data[row, column] += 1
            hdul.flush()
        try:
            step05.apply_rts_correction_in_memory(integer_decisions)
        except step05.Step05Error as exc:
            require(
                "changed after classification" in str(exc),
                f"unexpected stale-value error: {exc}",
            )
        else:
            require(False, "stale candidate value accepted")
        print("   Integer dtype : preserved")
        print("   Target values : safely converted")
        print("   Stale input   : rejected")
        print("   Result        : PASS")
        print()

        print("[4/4] Result metadata and output array are immutable")
        write_target(target, plan, values, np.float32)
        decisions = build_result(plan)
        first = step05.apply_rts_correction_in_memory(decisions)
        second = step05.apply_rts_correction_in_memory(decisions)

        require(first.summary() == second.summary(), "summary changed")
        require(
            json.dumps(first.summary(), sort_keys=True)
            == json.dumps(second.summary(), sort_keys=True),
            "serialized summary changed",
        )
        require(
            np.array_equal(first.corrected_image, second.corrected_image),
            "corrected output is not deterministic",
        )
        require(
            not first.corrected_image.flags.writeable,
            "corrected image is writeable",
        )
        try:
            first.applied_count = 999
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "application result is mutable")
        try:
            first.corrected_image[0, 0] = 999
        except ValueError:
            pass
        else:
            require(False, "corrected image mutation succeeded")
        print("   Result object : immutable")
        print("   Output array  : read-only")
        print("   Summary       : deterministic")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 05 in-memory correction application test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
