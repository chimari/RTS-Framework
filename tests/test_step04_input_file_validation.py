"""Integration test for Step 04 input-file validation v4.26.0."""

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
            "environment": "step04-v4.26-test",
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


def kwargs() -> dict[str, object]:
    return {
        "minimum_score": 0.9,
        "minimum_state_count": 2,
        "minimum_separation": 5.0,
        "minimum_transition_count": 3,
        "minimum_lower_run": 2,
        "minimum_upper_run": 2,
    }


def write_json(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def expect_error(metadata, expected_text: str) -> None:
    try:
        step04.validate_rts_dictionary_input_files(metadata)
    except step04.Step04Error as exc:
        require(expected_text in str(exc), f"unexpected error: {exc}")
    else:
        require(False, f"expected failure containing {expected_text!r}")


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 input-file validation test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_inputs_") as temp:
        root = Path(temp)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(write_dataset(root), "bias")
        )
        built = step04.build_rts_dictionary_artifacts(
            plan, root / "dictionary.csv", **kwargs()
        )

        print("[1/4] Metadata path validates every recorded input file")
        result = step04.validate_rts_dictionary_input_files(
            built.metadata_path
        )
        require(isinstance(result, step04.RTSInputFileValidation),
                "wrong validation type")
        require(result.expected_file_count == plan.n_frames,
                "wrong expected count")
        require(result.validated_file_count == plan.n_frames,
                "wrong validated count")
        require(all(path.is_absolute()
                    for path in result.validated_filepaths),
                "validated paths are not normalized absolute paths")
        print("   Type       : RTSInputFileValidation")
        print("   File count : matched")
        print("   Paths      : normalized")
        print("   Result     : PASS")
        print()

        print("[2/4] Loaded metadata and immutable summary are supported")
        metadata = step04.load_rts_dictionary_metadata_json(
            built.metadata_path
        )
        from_object = step04.validate_rts_dictionary_input_files(metadata)
        require(result.summary() == from_object.summary(),
                "path and object validation differ")
        try:
            from_object.expected_file_count = 0
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "validation object is mutable")
        require(
            from_object.summary()["validated_file_count"] == plan.n_frames,
            "summary count changed",
        )
        print("   Metadata object : supported")
        print("   Summary         : deterministic")
        print("   Dataclass       : frozen")
        print("   Result          : PASS")
        print()

        base = json.loads(
            built.metadata_path.read_text(encoding="utf-8")
        )

        print("[3/4] Missing files and directory paths are rejected")
        missing_doc = json.loads(json.dumps(base))
        missing_doc["input"]["filepaths"][0] = str(root / "missing.fit")
        missing_meta = root / "missing-input.json"
        write_json(missing_meta, missing_doc)
        expect_error(missing_meta, "does not exist")

        directory_doc = json.loads(json.dumps(base))
        directory_doc["input"]["filepaths"][0] = str(root)
        directory_meta = root / "directory-input.json"
        write_json(directory_meta, directory_doc)
        expect_error(directory_meta, "is not a regular file")
        print("   Missing path : rejected")
        print("   Directory    : rejected")
        print("   Result       : PASS")
        print()

        print("[4/4] Duplicate normalized paths and count mismatches reject")
        duplicate_doc = json.loads(json.dumps(base))
        duplicate_doc["input"]["filepaths"][1] = (
            duplicate_doc["input"]["filepaths"][0]
        )
        duplicate_meta = root / "duplicate-input.json"
        write_json(duplicate_meta, duplicate_doc)
        expect_error(duplicate_meta, "duplicates another path")

        # The normal loader already rejects this malformed inventory.
        count_doc = json.loads(json.dumps(base))
        count_doc["input"]["filepaths"] = count_doc["input"]["filepaths"][:-1]
        count_meta = root / "count-input.json"
        write_json(count_meta, count_doc)
        expect_error(count_meta, "input.filepaths length must match input.n_frames")
        print("   Duplicate path : rejected")
        print("   Wrong count    : rejected")
        print("   Result         : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 input-file validation test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
