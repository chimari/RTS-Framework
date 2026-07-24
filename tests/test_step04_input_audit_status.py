"""Integration test for Step 04 audit status policy v4.32.0."""

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


def write_dataset(root: Path) -> Path:
    paths = []
    for frame_index in range(8):
        data = np.array(
            [
                [0 if frame_index % 4 < 2 else 10, 20, 30],
                [0 if frame_index % 2 == 0 else 10, 5, 50],
            ],
            dtype=np.uint16,
        )
        path = root / f"bias_{frame_index:04d}.fit"
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows = []
    for frame_index, path in enumerate(paths):
        rows.append({
            "dataset": "bias",
            "directory": str(root),
            "environment": "step04-v4.32-test",
            "frame_index": frame_index,
            "n_frames": len(paths),
            "temperature_C": -10.0,
            "temperature_start_C": -10.0,
            "temperature_end_C": -10.0,
            "temperature_fraction": frame_index / (len(paths) - 1),
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


def make_comparison(
    baseline: step04.RTSInputFileFingerprintSet,
    *,
    changed: tuple[int, ...] = (),
    missing: tuple[int, ...] = (),
    additional: tuple[Path, ...] = (),
) -> step04.RTSInputFileFingerprintComparison:
    changes = []
    for index in changed:
        item = baseline.files[index]
        changes.append(
            step04.RTSInputFileFingerprintChange(
                index=index,
                path=item.path,
                expected_size_bytes=item.size_bytes,
                current_size_bytes=item.size_bytes + 1,
                expected_sha256=item.sha256,
                current_sha256="0" * 64,
                status="changed",
            )
        )
    for index in missing:
        item = baseline.files[index]
        changes.append(
            step04.RTSInputFileFingerprintChange(
                index=index,
                path=item.path,
                expected_size_bytes=item.size_bytes,
                current_size_bytes=None,
                expected_sha256=item.sha256,
                current_sha256=None,
                status="missing",
            )
        )
    changes.sort(key=lambda item: item.index)

    changed_indices = set(changed) | set(missing)
    unchanged_indices = tuple(
        index
        for index in range(baseline.file_count)
        if index not in changed_indices
    )

    return step04.RTSInputFileFingerprintComparison(
        metadata_path=baseline.metadata_path,
        algorithm=baseline.algorithm,
        expected_file_count=baseline.file_count,
        current_file_count=(
            baseline.file_count - len(missing) + len(additional)
        ),
        unchanged_indices=unchanged_indices,
        changes=tuple(changes),
        additional_paths=additional,
    )


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 input-audit status test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_status_") as temp:
        root = Path(temp)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(write_dataset(root), "bias")
        )
        built = step04.build_rts_dictionary_artifacts(
            plan, root / "dictionary.csv", **build_kwargs()
        )
        audit = step04.audit_rts_dictionary_input_files(
            built.metadata_path
        )
        baseline = audit.fingerprints

        print("[1/4] MATCH and ADDITIONAL_ONLY are successful")
        match = step04.evaluate_rts_input_audit(audit)
        require(
            match.name is step04.RTSAuditStatusName.MATCH,
            "match name changed",
        )
        require(match.ok, "match should be successful")
        require(match.exit_code == 0, "match exit code changed")

        additional_comparison = make_comparison(
            baseline,
            additional=(root / "extra.fit",),
        )
        additional = step04.evaluate_rts_input_audit(
            additional_comparison
        )
        require(
            additional.name is step04.RTSAuditStatusName.ADDITIONAL_ONLY,
            "additional-only name changed",
        )
        require(additional.ok, "additional-only should be successful")
        require(additional.exit_code == 0, "additional exit code changed")
        print("   MATCH           : exit 0")
        print("   ADDITIONAL_ONLY : exit 0")
        print("   Result          : PASS")
        print()

        print("[2/4] CHANGED is a deterministic failure")
        changed_comparison = make_comparison(
            baseline,
            changed=(0,),
        )
        changed = step04.evaluate_rts_input_audit(
            changed_comparison
        )
        require(
            changed.name is step04.RTSAuditStatusName.CHANGED,
            "changed name changed",
        )
        require(not changed.ok, "changed should fail")
        require(changed.exit_code == 1, "changed exit code changed")
        require(
            changed.message
            == "Input audit failed: 1 file changed.",
            "changed message changed",
        )
        print("   Name      : CHANGED")
        print("   Exit code : 1")
        print("   Message   : deterministic")
        print("   Result    : PASS")
        print()

        print("[3/4] MISSING and combined failures have distinct codes")
        missing = step04.evaluate_rts_input_audit(
            make_comparison(baseline, missing=(1,))
        )
        combined = step04.evaluate_rts_input_audit(
            make_comparison(
                baseline,
                changed=(0,),
                missing=(1,),
                additional=(root / "extra.fit",),
            )
        )
        require(
            missing.name is step04.RTSAuditStatusName.MISSING,
            "missing name changed",
        )
        require(missing.exit_code == 2, "missing exit code changed")
        require(
            combined.name
            is step04.RTSAuditStatusName.CHANGED_AND_MISSING,
            "combined name changed",
        )
        require(combined.exit_code == 3, "combined exit code changed")
        require(
            combined.additional_count == 1,
            "combined additional count changed",
        )
        print("   MISSING             : exit 2")
        print("   CHANGED_AND_MISSING : exit 3")
        print("   Additional context  : preserved")
        print("   Result              : PASS")
        print()

        print("[4/4] Summary, enum values, immutability, and errors")
        summary = combined.summary()
        require(
            summary["name"] == "CHANGED_AND_MISSING",
            "summary name changed",
        )
        require(
            step04.RTSAuditStatusName.CHANGED.value == "CHANGED",
            "enum value changed",
        )
        second = step04.evaluate_rts_input_audit(
            make_comparison(
                baseline,
                changed=(0,),
                missing=(1,),
                additional=(root / "extra.fit",),
            )
        )
        require(second == combined, "status is not deterministic")
        try:
            combined.exit_code = 99
        except (FrozenInstanceError, AttributeError, TypeError):
            pass
        else:
            require(False, "status is mutable")
        try:
            step04.evaluate_rts_input_audit(object())
        except step04.Step04Error:
            pass
        else:
            require(False, "invalid audit input was accepted")
        print("   Summary       : deterministic")
        print("   Enum values   : stable")
        print("   Status record : frozen")
        print("   Invalid input : rejected")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 input-audit status test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
