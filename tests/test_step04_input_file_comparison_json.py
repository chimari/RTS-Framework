"""Integration test for Step 04 comparison JSON v4.30.0."""

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
            "environment": "step04-v4.30-test",
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
        step04.load_rts_input_file_comparison_json(path)
    except step04.Step04Error as exc:
        require(expected_text in str(exc), f"unexpected error: {exc}")
    else:
        require(False, f"expected failure containing {expected_text!r}")


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 input-file comparison JSON test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_compare_json_") as temp:
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

        fingerprints.files[0].path.write_bytes(
            fingerprints.files[0].path.read_bytes() + b"\x00"
        )
        fingerprints.files[1].path.unlink()

        comparison = step04.compare_rts_input_file_fingerprints(
            fingerprints
        )

        print("[1/4] Comparison writes atomically to canonical JSON")
        output_path = step04.write_rts_input_file_comparison_json(
            comparison
        )
        require(
            output_path
            == Path(
                str(built.metadata_path)
                + ".fingerprint-comparison.json"
            ),
            "wrong default output path",
        )
        require(output_path.is_file(), "comparison JSON was not written")
        require(
            not output_path.with_name(output_path.name + ".tmp").exists(),
            "temporary file remains",
        )
        first_bytes = output_path.read_bytes()
        step04.write_rts_input_file_comparison_json(
            comparison, output_path
        )
        require(
            output_path.read_bytes() == first_bytes,
            "repeated write changed JSON bytes",
        )
        print("   Sidecar path : canonical")
        print("   Atomic write : complete")
        print("   JSON bytes   : deterministic")
        print("   Result       : PASS")
        print()

        print("[2/4] Written JSON loads into the immutable comparison")
        loaded = step04.load_rts_input_file_comparison_json(
            output_path
        )
        require(loaded == comparison, "loaded comparison changed")
        require(
            loaded.summary() == comparison.summary(),
            "loaded summary changed",
        )
        try:
            loaded.changes[0].status = "other"
        except (FrozenInstanceError, AttributeError, TypeError):
            pass
        else:
            require(False, "loaded comparison is mutable")
        print("   Structure : complete")
        print("   Values    : preserved")
        print("   Dataclass : frozen")
        print("   Result    : PASS")
        print()

        document = json.loads(output_path.read_text(encoding="utf-8"))

        print("[3/4] Schema, counts, and status violations are rejected")
        wrong_schema = root / "wrong-schema.json"
        changed = json.loads(json.dumps(document))
        changed["schema"] = "wrong"
        write_json(wrong_schema, changed)
        expect_error(wrong_schema, "unsupported input comparison schema")

        wrong_count = root / "wrong-count.json"
        changed = json.loads(json.dumps(document))
        changed["changed_count"] += 1
        write_json(wrong_count, changed)
        expect_error(wrong_count, "changed_count does not match changes")

        wrong_status = root / "wrong-status.json"
        changed = json.loads(json.dumps(document))
        changed["changes"][0]["status"] = "other"
        write_json(wrong_status, changed)
        expect_error(wrong_status, "status must be changed or missing")
        print("   Wrong schema : rejected")
        print("   Wrong count  : rejected")
        print("   Wrong status : rejected")
        print("   Result       : PASS")
        print()

        print("[4/4] Ordering, consistency, invalid JSON, and missing reject")
        wrong_order = root / "wrong-order.json"
        changed = json.loads(json.dumps(document))
        changed["changes"] = list(reversed(changed["changes"]))
        write_json(wrong_order, changed)
        expect_error(wrong_order, "ordered by ascending index")

        wrong_match = root / "wrong-match.json"
        changed = json.loads(json.dumps(document))
        changed["matches"] = True
        write_json(wrong_match, changed)
        expect_error(wrong_match, "matches is inconsistent")

        invalid = root / "invalid.json"
        invalid.write_text("{", encoding="utf-8")
        expect_error(invalid, "Could not read input comparison JSON")

        expect_error(root / "missing.json", "does not exist")
        print("   Wrong order  : rejected")
        print("   Wrong matches: rejected")
        print("   Invalid JSON : rejected")
        print("   Missing file : rejected")
        print("   Result       : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 input-file comparison JSON test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
