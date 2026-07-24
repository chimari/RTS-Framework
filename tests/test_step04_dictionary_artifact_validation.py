"""Integration test for complete Step 04 artifact validation v4.36.0."""

from __future__ import annotations

import csv
import json
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
            "environment": "step04-v4.36-test",
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


def expect_error(callback, expected_text: str) -> None:
    try:
        callback()
    except step04.Step04Error as exc:
        require(expected_text in str(exc), f"unexpected error: {exc}")
    else:
        require(False, f"expected Step04Error containing {expected_text!r}")


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 dictionary artifact validation test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_validation_") as temp:
        root = Path(temp)
        manifest = write_dataset(root)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(manifest, "bias")
        )
        built = step04.build_rts_dictionary_artifacts(
            plan,
            root / "dictionary.csv",
            **build_kwargs(),
        )
        audit = step04.audit_rts_dictionary_input_files(
            built.metadata_path
        )

        print("[1/4] Complete canonical artifact set validates")
        validation = step04.validate_rts_dictionary_artifacts(
            built.metadata_path
        )
        require(
            isinstance(
                validation,
                step04.RTSDictionaryArtifactValidation,
            ),
            "wrong validation result type",
        )
        require(
            validation.candidate_count
            == built.build_result.candidate_count,
            "candidate count does not match the build result",
        )
        require(validation.input_file_count == 8, "input count changed")
        require(validation.comparison.matches, "fresh comparison not MATCH")
        require(
            validation.summary()["dataset"] == "bias",
            "summary dataset changed",
        )
        print("   CSV + metadata : valid")
        print("   Fingerprints   : valid")
        print("   Comparison     : valid")
        print("   Cross-links    : consistent")
        print("   Result         : PASS")
        print()

        print("[2/4] Missing or malformed files are rejected")
        fingerprint_path = audit.fingerprint_json_path
        comparison_path = audit.comparison_json_path
        saved_fingerprint = fingerprint_path.read_bytes()
        fingerprint_path.unlink()
        expect_error(
            lambda: step04.validate_rts_dictionary_artifacts(
                built.metadata_path
            ),
            "fingerprint JSON does not exist",
        )
        fingerprint_path.write_bytes(saved_fingerprint)

        saved_comparison = comparison_path.read_text(encoding="utf-8")
        comparison_path.write_text("{", encoding="utf-8")
        expect_error(
            lambda: step04.validate_rts_dictionary_artifacts(
                built.metadata_path
            ),
            "Could not read input comparison JSON",
        )
        comparison_path.write_text(saved_comparison, encoding="utf-8")
        print("   Missing file   : rejected")
        print("   Malformed JSON : rejected")
        print("   Result         : PASS")
        print()

        print("[3/4] Cross-artifact metadata and counts are enforced")
        fingerprint_document = json.loads(
            fingerprint_path.read_text(encoding="utf-8")
        )
        original_metadata_path = fingerprint_document["metadata_path"]
        fingerprint_document["metadata_path"] = str(root / "other.metadata.json")
        fingerprint_path.write_text(
            json.dumps(fingerprint_document, indent=2) + "\n",
            encoding="utf-8",
        )
        expect_error(
            lambda: step04.validate_rts_dictionary_artifacts(
                built.metadata_path
            ),
            "fingerprint metadata_path does not match",
        )
        fingerprint_document["metadata_path"] = original_metadata_path
        fingerprint_path.write_text(
            json.dumps(fingerprint_document, indent=2) + "\n",
            encoding="utf-8",
        )

        comparison_document = json.loads(
            comparison_path.read_text(encoding="utf-8")
        )
        comparison_document["expected_file_count"] = 9
        comparison_document["missing_count"] = 1
        comparison_document["matches"] = False
        comparison_document["changes"].append(
            {
                "index": 8,
                "path": str(root / "missing_bias_0008.fit"),
                "expected_size_bytes": 0,
                "current_size_bytes": None,
                "expected_sha256": "0" * 64,
                "current_sha256": None,
                "status": "missing",
            }
        )
        comparison_path.write_text(
            json.dumps(comparison_document, indent=2) + "\n",
            encoding="utf-8",
        )
        expect_error(
            lambda: step04.validate_rts_dictionary_artifacts(
                built.metadata_path
            ),
            "expected_file_count does not match fingerprints",
        )
        comparison_document["expected_file_count"] = 8
        comparison_document["missing_count"] = 0
        comparison_document["matches"] = True
        comparison_document["changes"].pop()
        comparison_path.write_text(
            json.dumps(comparison_document, indent=2) + "\n",
            encoding="utf-8",
        )
        print("   Metadata link  : enforced")
        print("   Expected count : enforced")
        print("   Result         : PASS")
        print()

        print("[4/4] Explicit sidecar paths and determinism are preserved")
        explicit = step04.validate_rts_dictionary_artifacts(
            built.metadata_path,
            fingerprint_json_path=fingerprint_path,
            comparison_json_path=comparison_path,
        )
        require(
            explicit.summary() == validation.summary(),
            "explicit-path summary changed",
        )
        require(
            explicit.summary() == explicit.summary(),
            "summary is not deterministic",
        )
        require(
            explicit.fingerprint_json_path == fingerprint_path.resolve(),
            "fingerprint path was not normalized",
        )
        require(
            explicit.comparison_json_path == comparison_path.resolve(),
            "comparison path was not normalized",
        )
        print("   Explicit paths : supported")
        print("   Normalization  : stable")
        print("   Summary        : deterministic")
        print("   Result         : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 dictionary artifact validation test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
