"""Integration test for Step 05 batch correction v5.6.0."""

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
            "environment": "step05-v5.6-test",
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


def prepare_artifacts(root: Path) -> tuple[Path, tuple[Path, Path]]:
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

    inputs = []
    for image_index in range(2):
        source = root / f"target_{image_index}.fit"
        data = np.full((2, 3), 20 + image_index, dtype=np.float32)
        fits.PrimaryHDU(data=data).writeto(source, overwrite=True)
        plan = step05.prepare_rts_correction(built.metadata_path, source)
        for index, candidate in enumerate(plan.candidates):
            data[candidate.row, candidate.column] = (
                candidate.lower_state_center
                + 0.1 * candidate.state_separation
                if (index + image_index) % 2 == 0
                else candidate.upper_state_center
                - 0.1 * candidate.state_separation
            )
        fits.PrimaryHDU(data=data).writeto(source, overwrite=True)
        inputs.append(source)

    return built.metadata_path, tuple(inputs)


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 05 batch correction test")
    print("=" * 72)
    print(f"step05 version : {step05.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step05_batch_") as temp:
        root = Path(temp)
        metadata, inputs = prepare_artifacts(root)
        output_dir = root / "outputs"
        output_dir.mkdir()

        print("[1/4] Multiple FITS inputs are corrected deterministically")
        result = step05.run_rts_correction_batch(
            metadata,
            inputs,
            output_dir,
        )
        require(result.total_count == 2, "total_count changed")
        require(result.succeeded_count == 2, "succeeded_count changed")
        require(result.failed_count == 0, "failed_count changed")
        require(result.all_succeeded, "all_succeeded changed")
        for item in result.items:
            require(item.succeeded, "successful item marked failed")
            require(item.output_path.exists(), "batch output missing")
            require(item.output is not None, "output result missing")
            require(item.output.verified, "output not verified")
        print("   Inputs        : 2")
        print("   Outputs       : 2")
        print("   All verified  : True")
        print("   Result        : PASS")
        print()

        print("[2/4] Output naming, summary, and immutability are stable")
        expected_names = {
            "target_0_rts_corrected.fit",
            "target_1_rts_corrected.fit",
        }
        actual_names = {item.output_path.name for item in result.items}
        require(actual_names == expected_names, "output names changed")
        summary = result.summary()
        require(summary["total_count"] == 2, "summary total changed")
        require(summary["failed_count"] == 0, "summary failures changed")
        require(
            json.dumps(summary, sort_keys=True)
            == json.dumps(result.summary(), sort_keys=True),
            "summary is not deterministic",
        )
        try:
            result.overwrite = True
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "batch result is mutable")
        print("   Naming        : deterministic")
        print("   Summary       : deterministic")
        print("   Batch result  : immutable")
        print("   Result        : PASS")
        print()

        print("[3/4] continue_on_error records failure and continues")
        failure_dir = root / "failure_outputs"
        failure_dir.mkdir()
        invalid = root / "invalid.fit"
        fits.PrimaryHDU(
            data=np.zeros((3, 3), dtype=np.float32)
        ).writeto(invalid, overwrite=True)

        mixed = step05.run_rts_correction_batch(
            metadata,
            [inputs[0], invalid, inputs[1]],
            failure_dir,
            continue_on_error=True,
        )
        require(mixed.total_count == 3, "mixed total changed")
        require(mixed.succeeded_count == 2, "mixed success changed")
        require(mixed.failed_count == 1, "mixed failure changed")
        require(not mixed.all_succeeded, "mixed all_succeeded changed")
        require(mixed.items[1].succeeded is False, "failure not recorded")
        require(mixed.items[1].error is not None, "failure error missing")
        require(mixed.items[2].succeeded, "processing did not continue")
        print("   Successes     : 2")
        print("   Failures      : 1")
        print("   Continued     : True")
        print("   Result        : PASS")
        print()

        print("[4/4] Fail-fast and collision protections work")
        failfast_dir = root / "failfast_outputs"
        failfast_dir.mkdir()
        try:
            step05.run_rts_correction_batch(
                metadata,
                [invalid, inputs[0]],
                failfast_dir,
                continue_on_error=False,
            )
        except step05.Step05Error:
            pass
        else:
            require(False, "fail-fast batch did not raise")

        duplicate_dir = root / "duplicate_outputs"
        duplicate_dir.mkdir()
        try:
            step05.run_rts_correction_batch(
                metadata,
                [inputs[0], inputs[0]],
                duplicate_dir,
            )
        except step05.Step05Error as exc:
            require("duplicate" in str(exc), "unexpected duplicate error")
        else:
            require(False, "duplicate input was accepted")
        print("   Fail-fast     : raises Step05Error")
        print("   Duplicates    : rejected")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 05 batch correction test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
