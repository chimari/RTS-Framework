"""Integration test for Step 04 fingerprint JSON v4.28.0."""

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
            "environment": "step04-v4.28-test",
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


def expect_error(path: Path, expected_text: str) -> None:
    try:
        step04.load_rts_input_file_fingerprints_json(path)
    except step04.Step04Error as exc:
        require(expected_text in str(exc), f"unexpected error: {exc}")
    else:
        require(False, f"expected failure containing {expected_text!r}")


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 input-file fingerprint JSON test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_hash_json_") as temp:
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

        print("[1/4] Fingerprints write atomically to canonical JSON")
        output_path = step04.write_rts_input_file_fingerprints_json(
            fingerprints
        )
        require(
            output_path
            == Path(str(built.metadata_path) + ".fingerprints.json"),
            "wrong default output path",
        )
        require(output_path.is_file(), "fingerprint JSON was not written")
        require(
            not output_path.with_name(output_path.name + ".tmp").exists(),
            "temporary file remains",
        )
        first_bytes = output_path.read_bytes()
        step04.write_rts_input_file_fingerprints_json(
            fingerprints, output_path
        )
        require(output_path.read_bytes() == first_bytes,
                "repeated write changed JSON bytes")
        print("   Sidecar path : canonical")
        print("   Atomic write : complete")
        print("   JSON bytes   : deterministic")
        print("   Result       : PASS")
        print()

        print("[2/4] Written JSON loads into the immutable fingerprint set")
        loaded = step04.load_rts_input_file_fingerprints_json(output_path)
        require(loaded == fingerprints, "loaded fingerprints changed")
        require(loaded.summary() == fingerprints.summary(),
                "loaded summary changed")
        try:
            loaded.algorithm = "other"
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "loaded fingerprint set is mutable")
        print("   Structure : complete")
        print("   Values    : preserved")
        print("   Dataclass : frozen")
        print("   Result    : PASS")
        print()

        document = json.loads(output_path.read_text(encoding="utf-8"))

        print("[3/4] Schema, digest, and aggregate violations are rejected")
        wrong_schema = root / "wrong-schema.json"
        changed = json.loads(json.dumps(document))
        changed["schema"] = "wrong"
        write_json(wrong_schema, changed)
        expect_error(wrong_schema, "unsupported input fingerprint schema")

        wrong_digest = root / "wrong-digest.json"
        changed = json.loads(json.dumps(document))
        changed["files"][0]["sha256"] = "XYZ"
        write_json(wrong_digest, changed)
        expect_error(wrong_digest, "must contain 64 hexadecimal")

        wrong_total = root / "wrong-total.json"
        changed = json.loads(json.dumps(document))
        changed["total_size_bytes"] += 1
        write_json(wrong_total, changed)
        expect_error(wrong_total, "must sum to total_size_bytes")
        print("   Wrong schema : rejected")
        print("   Wrong digest : rejected")
        print("   Wrong total  : rejected")
        print("   Result       : PASS")
        print()

        print("[4/4] Ordering, duplicates, invalid JSON, and missing files reject")
        wrong_order = root / "wrong-order.json"
        changed = json.loads(json.dumps(document))
        changed["files"][0]["index"] = 1
        write_json(wrong_order, changed)
        expect_error(wrong_order, "indices must be contiguous and ordered")

        duplicate = root / "duplicate.json"
        changed = json.loads(json.dumps(document))
        changed["files"][1]["path"] = changed["files"][0]["path"]
        write_json(duplicate, changed)
        expect_error(duplicate, "file paths must be unique")

        invalid = root / "invalid.json"
        invalid.write_text("{", encoding="utf-8")
        expect_error(invalid, "Could not read input fingerprint JSON")

        expect_error(root / "missing.json", "does not exist")
        print("   Wrong order    : rejected")
        print("   Duplicate path : rejected")
        print("   Invalid JSON   : rejected")
        print("   Missing file   : rejected")
        print("   Result         : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 input-file fingerprint JSON test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
