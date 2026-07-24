"""Integration test for Step 04 metadata loading v4.23.0."""

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
    values = [
        [0, 0, 10, 10, 0, 0, 10, 10],
        [20, 21, 20, 21, 20, 21, 20, 21],
        [30, 30, 45, 45, 30, 30, 45, 45],
        [0, 10, 0, 10, 0, 10, 0, 10],
        [5, 5, 15, 15, 5, 5, 15, 15],
        [50, 51, 50, 51, 50, 51, 50, 51],
    ]
    paths = []
    for frame_index in range(8):
        flat = [series[frame_index] for series in values]
        data = np.array([flat[:3], flat[3:]], dtype=np.uint16)
        path = root / f"bias_{frame_index:04d}.fit"
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows = []
    for frame_index, path in enumerate(paths):
        rows.append({
            "dataset": "bias",
            "directory": str(root),
            "environment": "step04-v4.23-test",
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


def expect_error(path: Path, expected_text: str) -> None:
    try:
        step04.load_rts_dictionary_metadata_json(path)
    except step04.Step04Error as exc:
        require(expected_text in str(exc), f"unexpected error: {exc}")
    else:
        require(False, f"expected failure containing {expected_text!r}")


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 dictionary metadata loading test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_load_metadata_") as temp:
        root = Path(temp)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(write_dataset(root), "bias")
        )
        artifacts = step04.build_rts_dictionary_artifacts(
            plan, root / "dictionary.csv", **kwargs()
        )

        print("[1/4] Written metadata loads into an immutable structured object")
        loaded = step04.load_rts_dictionary_metadata_json(
            artifacts.metadata_path
        )
        require(isinstance(loaded, step04.RTSDictionaryMetadata),
                "wrong metadata type")
        require(loaded.metadata_path == artifacts.metadata_path,
                "wrong metadata path")
        require(loaded.csv_path == artifacts.output_path, "wrong CSV path")
        require(loaded.dataset == "bias", "wrong dataset")
        require(loaded.n_frames == 8, "wrong frame count")
        require(loaded.image_shape == (2, 3), "wrong image shape")
        require(loaded.parameters.pixel_count == 6, "wrong pixel count")
        require(len(loaded.filepaths) == 8, "wrong filepath count")
        try:
            loaded.dataset = "changed"
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "metadata object is mutable")
        print("   Type       : RTSDictionaryMetadata")
        print("   Structure  : complete")
        print("   Dataclass  : frozen")
        print("   Result     : PASS")
        print()

        print("[2/4] Summary is deterministic and preserves normalized values")
        summary1 = loaded.summary()
        summary2 = step04.load_rts_dictionary_metadata_json(
            artifacts.metadata_path
        ).summary()
        require(summary1 == summary2, "summary is not deterministic")
        require(summary1["parameters"]["minimum_score"] == 0.9,
                "threshold missing")
        require(summary1["parameters"]["pixel_count"] == 6,
                "derived count missing")
        require(summary1["image_shape"] == (2, 3),
                "normalized shape missing")
        print("   Summary     : deterministic")
        print("   Thresholds  : preserved")
        print("   ROI counts  : preserved")
        print("   Result      : PASS")
        print()

        print("[3/4] Schema and required-field violations are rejected")
        base = json.loads(
            artifacts.metadata_path.read_text(encoding="utf-8")
        )

        wrong_schema = root / "wrong-schema.json"
        doc = json.loads(json.dumps(base))
        doc["schema"] = "other.schema"
        wrong_schema.write_text(json.dumps(doc), encoding="utf-8")
        expect_error(wrong_schema, "Unsupported metadata schema")

        missing_field = root / "missing-field.json"
        doc = json.loads(json.dumps(base))
        del doc["parameters"]["minimum_score"]
        missing_field.write_text(json.dumps(doc), encoding="utf-8")
        expect_error(missing_field, "missing required fields")

        mismatch = root / "mismatch.json"
        doc = json.loads(json.dumps(base))
        doc["dictionary"]["analyzed_pixel_count"] = 5
        mismatch.write_text(json.dumps(doc), encoding="utf-8")
        expect_error(mismatch, "parameters.pixel_count")
        print("   Wrong schema   : rejected")
        print("   Missing field  : rejected")
        print("   Count mismatch : rejected")
        print("   Result         : PASS")
        print()

        print("[4/4] Invalid JSON and missing files produce Step04Error")
        invalid = root / "invalid.json"
        invalid.write_text("{not valid json", encoding="utf-8")
        expect_error(invalid, "Could not read RTS dictionary metadata JSON")
        expect_error(root / "missing.json", "does not exist")
        print("   Invalid JSON : rejected")
        print("   Missing file : rejected")
        print("   Result       : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 metadata loading test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
