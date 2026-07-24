"""Integration test for Step 04 fingerprint comparison v4.29.0."""

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


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def write_dataset(root: Path, n_frames: int = 8) -> Path:
    paths = []
    for frame_index in range(n_frames):
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
            "environment": "step04-v4.29-test",
            "frame_index": frame_index,
            "n_frames": len(paths),
            "temperature_C": -10.0,
            "temperature_start_C": -10.0,
            "temperature_end_C": -10.0,
            "temperature_fraction": frame_index / max(1, len(paths) - 1),
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


def kwargs() -> dict[str, object]:
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
    print("RTS Framework Step 04 input-file fingerprint comparison test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_compare_") as temp:
        root = Path(temp)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(write_dataset(root), "bias")
        )
        built = step04.build_rts_dictionary_artifacts(
            plan, root / "dictionary.csv", **kwargs()
        )
        fingerprints = step04.fingerprint_rts_dictionary_input_files(
            built.metadata_path
        )
        fingerprint_json = (
            step04.write_rts_input_file_fingerprints_json(fingerprints)
        )

        print("[1/4] Unchanged inputs produce an exact match")
        comparison = step04.compare_rts_input_file_fingerprints(
            fingerprint_json
        )
        require(
            isinstance(
                comparison, step04.RTSInputFileFingerprintComparison
            ),
            "wrong comparison type",
        )
        require(comparison.matches, "unchanged inputs did not match")
        require(comparison.unchanged_count == plan.n_frames,
                "wrong unchanged count")
        require(comparison.changed_count == 0, "unexpected changed files")
        require(comparison.missing_count == 0, "unexpected missing files")
        require(comparison.additional_count == 0,
                "unexpected additional files")
        print("   Match      : exact")
        print("   Unchanged  : all")
        print("   Differences: none")
        print("   Result     : PASS")
        print()

        print("[2/4] Modified and missing files are classified correctly")
        modified_path = fingerprints.files[0].path
        missing_path = fingerprints.files[1].path
        modified_path.write_bytes(modified_path.read_bytes() + b"\x00")
        missing_path.unlink()

        changed = step04.compare_rts_input_file_fingerprints(
            fingerprints
        )
        require(not changed.matches, "changed inputs still match")
        require(changed.changed_count == 1, "wrong changed count")
        require(changed.missing_count == 1, "wrong missing count")
        require(changed.unchanged_count == plan.n_frames - 2,
                "wrong unchanged count after changes")
        require(
            tuple(item.status for item in changed.changes)
            == ("changed", "missing"),
            "wrong change ordering or status",
        )
        require(
            changed.changes[0].current_size_bytes
            == changed.changes[0].expected_size_bytes + 1,
            "modified size was not reported",
        )
        require(
            changed.changes[1].current_size_bytes is None
            and changed.changes[1].current_sha256 is None,
            "missing file current values must be None",
        )
        print("   Modified file : changed")
        print("   Removed file  : missing")
        print("   Ordering      : deterministic")
        print("   Result        : PASS")
        print()

        print("[3/4] Additional metadata paths are reported")
        extra_path = root / "extra.fit"
        fits.PrimaryHDU(
            data=np.zeros((2, 3), dtype=np.uint16)
        ).writeto(extra_path, overwrite=True)

        metadata_doc = json.loads(
            built.metadata_path.read_text(encoding="utf-8")
        )
        metadata_doc["input"]["filepaths"].append(str(extra_path))
        metadata_doc["input"]["n_frames"] += 1
        additional_meta = root / "additional.metadata.json"
        additional_meta.write_text(
            json.dumps(metadata_doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        additional = step04.compare_rts_input_file_fingerprints(
            fingerprints, additional_meta
        )
        require(additional.additional_count == 1,
                "additional path was not reported")
        require(additional.additional_paths == (extra_path.resolve(),),
                "wrong additional path")
        require(
            additional.current_file_count
            == fingerprints.file_count + 1,
            "wrong current file count",
        )
        print("   Additional path : detected")
        print("   Current count   : updated")
        print("   Result          : PASS")
        print()

        print("[4/4] Summary and immutable records are deterministic")
        summary = changed.summary()
        repeated = step04.compare_rts_input_file_fingerprints(
            fingerprints
        )
        require(summary == repeated.summary(),
                "comparison summary changed")
        require(
            summary["changed_count"] == 1
            and summary["missing_count"] == 1,
            "summary counts are wrong",
        )
        try:
            changed.changes[0].status = "other"
        except (FrozenInstanceError, AttributeError, TypeError):
            pass
        else:
            require(False, "change record is mutable")
        try:
            changed.matches = True
        except (FrozenInstanceError, AttributeError, TypeError):
            pass
        else:
            require(False, "comparison result is mutable")
        print("   Summary    : deterministic")
        print("   Change     : frozen")
        print("   Comparison : frozen")
        print("   Result     : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 input-file fingerprint comparison test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
