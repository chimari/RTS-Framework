"""Integration test for Step 05 correction-plan preparation v5.0.0."""

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


def write_dataset(root: Path) -> Path:
    paths = []
    for index in range(8):
        data = np.array(
            [
                [0 if index % 4 < 2 else 10, 20, 30],
                [0 if index % 2 == 0 else 10, 5, 50],
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
            "environment": "step05-v5.0-test",
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


def build_kwargs() -> dict[str, object]:
    return {
        "minimum_score": 0.9,
        "minimum_state_count": 2,
        "minimum_separation": 5.0,
        "minimum_transition_count": 3,
        "minimum_lower_run": 2,
        "minimum_upper_run": 2,
    }


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 05 correction-plan preparation test")
    print("=" * 72)
    print(f"step05 version : {step05.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step05_prepare_") as temp:
        root = Path(temp)
        manifest = write_dataset(root)
        plan04 = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(manifest, "bias")
        )
        built = step04.build_rts_dictionary_artifacts(
            plan04,
            root / "dictionary.csv",
            **build_kwargs(),
        )
        audit = step04.audit_rts_dictionary_input_files(
            built.metadata_path
        )
        target = root / "target.fit"
        fits.PrimaryHDU(
            data=np.array([[2, 20, 30], [8, 5, 50]], dtype=np.uint16)
        ).writeto(target, overwrite=True)

        print("[1/4] Canonical Step 04 artifacts produce a correction plan")
        plan = step05.prepare_rts_correction(
            built.metadata_path,
            target,
        )
        require(plan.image_shape == (2, 3), "image shape changed")
        require(plan.pixel_dtype == "uint16", "pixel dtype changed")
        require(plan.dataset == "bias", "dataset changed")
        require(
            plan.candidate_count == built.build_result.candidate_count,
            "candidate count changed",
        )
        require(
            plan.coordinates
            == tuple(
                row.coordinate
                for row in plan.artifact_validation.artifacts.dictionary.rows
            ),
            "candidate ordering changed",
        )
        print("   Artifacts     : validated")
        print("   Target FITS   : validated")
        print("   Candidate plan: prepared")
        print("   Result        : PASS")
        print()

        print("[2/4] Input FITS constraints are enforced")
        missing = root / "missing.fit"
        try:
            step05.prepare_rts_correction(built.metadata_path, missing)
        except step05.Step05Error as exc:
            require(
                "input FITS does not exist" in str(exc),
                f"unexpected missing-file error: {exc}",
            )
        else:
            require(False, "missing FITS was accepted")

        wrong_shape = root / "wrong_shape.fit"
        fits.PrimaryHDU(
            data=np.zeros((3, 3), dtype=np.uint16)
        ).writeto(wrong_shape, overwrite=True)
        try:
            step05.prepare_rts_correction(
                built.metadata_path,
                wrong_shape,
            )
        except step05.Step05Error as exc:
            require(
                "image_shape does not match" in str(exc),
                f"unexpected shape error: {exc}",
            )
        else:
            require(False, "wrong image shape was accepted")

        cube = root / "cube.fit"
        fits.PrimaryHDU(
            data=np.zeros((2, 2, 3), dtype=np.uint16)
        ).writeto(cube, overwrite=True)
        try:
            step05.prepare_rts_correction(built.metadata_path, cube)
        except step05.Step05Error as exc:
            require(
                "must be two-dimensional" in str(exc),
                f"unexpected dimensionality error: {exc}",
            )
        else:
            require(False, "3-D FITS was accepted")
        print("   Missing FITS  : rejected")
        print("   Shape mismatch: rejected")
        print("   Non-2-D image : rejected")
        print("   Result        : PASS")
        print()

        print("[3/4] Corrupt Step 04 sidecars are rejected")
        comparison_path = audit.comparison_json_path
        original = comparison_path.read_bytes()
        comparison_path.write_text("{", encoding="utf-8")
        try:
            step05.prepare_rts_correction(
                built.metadata_path,
                target,
            )
        except step05.Step05Error as exc:
            require(
                "Could not validate Step 04 dictionary artifacts" in str(exc),
                f"unexpected artifact error: {exc}",
            )
        else:
            require(False, "corrupt Step 04 sidecar was accepted")
        comparison_path.write_bytes(original)
        print("   Sidecar damage: rejected")
        print("   Error boundary: Step05Error")
        print("   Result        : PASS")
        print()

        print("[4/4] Immutability and deterministic summaries are preserved")
        first = step05.prepare_rts_correction(
            built.metadata_path,
            target,
        )
        second = step05.prepare_rts_correction(
            built.metadata_path,
            target,
        )
        require(first.summary() == second.summary(),
                "plan summary is not deterministic")
        require(
            json.dumps(first.summary(), sort_keys=True)
            == json.dumps(second.summary(), sort_keys=True),
            "serialized summary is not deterministic",
        )
        try:
            first.dataset = "changed"
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "correction plan is mutable")
        if first.candidates:
            try:
                first.candidates[0].row = 99
            except (FrozenInstanceError, AttributeError):
                pass
            else:
                require(False, "correction candidate is mutable")
        print("   Plan          : immutable")
        print("   Candidates    : immutable")
        print("   Summary       : deterministic")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 05 correction-plan preparation test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
