"""Integration test for Step 04 high-level input audit v4.31.0."""

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
            "environment": "step04-v4.31-test",
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


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 high-level input audit test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_audit_") as temp:
        root = Path(temp)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(write_dataset(root), "bias")
        )
        built = step04.build_rts_dictionary_artifacts(
            plan, root / "dictionary.csv", **build_kwargs()
        )

        print("[1/4] Metadata path runs the complete audit workflow")
        result = step04.audit_rts_dictionary_input_files(
            built.metadata_path
        )
        require(result.matches, "unchanged inputs should match")
        require(
            result.metadata_path == built.metadata_path,
            "metadata path changed",
        )
        require(
            result.fingerprint_json_path.is_file(),
            "fingerprint JSON missing",
        )
        require(
            result.comparison_json_path.is_file(),
            "comparison JSON missing",
        )
        print("   Fingerprint : created")
        print("   Comparison  : created")
        print("   Match       : exact")
        print("   Result      : PASS")
        print()

        print("[2/4] Metadata object and custom paths are supported")
        metadata = step04.load_rts_dictionary_metadata_json(
            built.metadata_path
        )
        fingerprint_path = root / "audit" / "fingerprints.json"
        comparison_path = root / "audit" / "comparison.json"
        custom = step04.audit_rts_dictionary_input_files(
            metadata,
            fingerprint_json_path=fingerprint_path,
            comparison_json_path=comparison_path,
        )
        require(
            custom.fingerprint_json_path == fingerprint_path,
            "custom fingerprint path changed",
        )
        require(
            custom.comparison_json_path == comparison_path,
            "custom comparison path changed",
        )
        require(fingerprint_path.is_file(), "custom fingerprint missing")
        require(comparison_path.is_file(), "custom comparison missing")
        print("   Metadata object : accepted")
        print("   Custom paths    : preserved")
        print("   Parent creation : complete")
        print("   Result          : PASS")
        print()

        print("[3/4] Audit captures modified and missing inputs")
        custom.fingerprints.files[0].path.write_bytes(
            custom.fingerprints.files[0].path.read_bytes() + b"\x00"
        )
        custom.fingerprints.files[1].path.unlink()
        changed = step04.audit_rts_dictionary_input_files(
            built.metadata_path,
            fingerprint_json_path=fingerprint_path,
            comparison_json_path=root / "changed-comparison.json",
        )
        require(not changed.matches, "changed inputs should not match")
        require(
            changed.comparison.changed_count == 1,
            "modified file not counted",
        )
        require(
            changed.comparison.missing_count == 1,
            "missing file not counted",
        )
        print("   Modified file : detected")
        print("   Missing file  : detected")
        print("   Match         : false")
        print("   Result        : PASS")
        print()

        print("[4/4] Summary, determinism, and immutability are preserved")
        summary = changed.summary()
        require(
            summary["matches"] is False,
            "summary matches value changed",
        )
        require(
            summary["fingerprint_file_count"]
            == changed.fingerprints.file_count,
            "summary file count changed",
        )
        require(
            summary["comparison"] == changed.comparison.summary(),
            "summary comparison changed",
        )
        second = step04.audit_rts_dictionary_input_files(
            built.metadata_path,
            fingerprint_json_path=fingerprint_path,
            comparison_json_path=root / "changed-comparison-2.json",
        )
        require(
            second.fingerprints == changed.fingerprints,
            "fingerprints are not deterministic",
        )
        require(
            second.comparison == changed.comparison,
            "comparison is not deterministic",
        )
        try:
            changed.metadata_path = root / "other.json"
        except (FrozenInstanceError, AttributeError, TypeError):
            pass
        else:
            require(False, "audit result is mutable")
        print("   Summary       : deterministic")
        print("   Fingerprints  : deterministic")
        print("   Comparison    : deterministic")
        print("   Audit result  : frozen")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 high-level input audit test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
