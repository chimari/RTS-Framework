"""Integration test for Step 04 joint artifact loading v4.25.0."""

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
            "environment": "step04-v4.25-test",
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


def expect_error(csv_path: Path, metadata_path: Path,
                 expected_text: str) -> None:
    try:
        step04.load_rts_dictionary_artifacts(csv_path, metadata_path)
    except step04.Step04Error as exc:
        require(expected_text in str(exc), f"unexpected error: {exc}")
    else:
        require(False, f"expected failure containing {expected_text!r}")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str],
              rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 joint artifact loading test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_artifacts_") as temp:
        root = Path(temp)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(write_dataset(root), "bias")
        )
        built = step04.build_rts_dictionary_artifacts(
            plan, root / "dictionary.csv", **kwargs()
        )

        print("[1/4] Canonical CSV/metadata pair loads as one frozen object")
        loaded = step04.load_rts_dictionary_artifacts(built.output_path)
        require(isinstance(loaded, step04.RTSDictionaryArtifacts),
                "wrong artifact type")
        require(loaded.dictionary.path == built.output_path,
                "wrong dictionary path")
        require(loaded.metadata.metadata_path == built.metadata_path,
                "wrong metadata path")
        require(loaded.candidate_count == built.build_result.candidate_count,
                "wrong candidate count")
        require(loaded.dataset == "bias", "wrong dataset")
        try:
            loaded.dictionary = None
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "artifact object is mutable")
        print("   Sidecar path : inferred")
        print("   Artifact type: RTSDictionaryArtifacts")
        print("   Dataclass    : frozen")
        print("   Result       : PASS")
        print()

        print("[2/4] Cross-validated summary is deterministic")
        explicit = step04.load_rts_dictionary_artifacts(
            built.output_path, built.metadata_path
        )
        require(loaded.summary() == explicit.summary(),
                "summary differs by loading mode")
        require(loaded.candidate_count == len(loaded.dictionary.rows),
                "candidate count mismatch")
        require(all(row.n_frames == loaded.metadata.n_frames
                    for row in loaded.dictionary.rows),
                "frame count was not preserved")
        require(all(
            row.minimum_score == loaded.metadata.parameters.minimum_score
            for row in loaded.dictionary.rows
        ), "threshold was not preserved")
        print("   Explicit path : supported")
        print("   Summary       : deterministic")
        print("   Frame counts  : matched")
        print("   Thresholds    : matched")
        print("   Result        : PASS")
        print()

        fields, rows = read_csv(built.output_path)
        base_metadata = json.loads(
            built.metadata_path.read_text(encoding="utf-8")
        )

        print("[3/4] Path, candidate-count, and dataset mismatches reject")
        wrong_path = root / "wrong-path.json"
        doc = json.loads(json.dumps(base_metadata))
        doc["dictionary"]["csv_path"] = str(root / "other.csv")
        write_json(wrong_path, doc)
        expect_error(built.output_path, wrong_path, "csv_path does not match")

        wrong_count = root / "wrong-count.json"
        doc = json.loads(json.dumps(base_metadata))
        doc["dictionary"]["candidate_count"] += 1
        write_json(wrong_count, doc)
        expect_error(
            built.output_path, wrong_count,
            "candidate_count does not match"
        )

        wrong_dataset_csv = root / "wrong-dataset.csv"
        changed = [dict(row) for row in rows]
        changed[0]["dataset"] = "other"
        write_csv(wrong_dataset_csv, fields, changed)
        doc = json.loads(json.dumps(base_metadata))
        doc["dictionary"]["csv_path"] = str(wrong_dataset_csv)
        wrong_dataset_meta = root / "wrong-dataset.json"
        write_json(wrong_dataset_meta, doc)
        expect_error(
            wrong_dataset_csv, wrong_dataset_meta,
            "dataset does not match"
        )
        print("   Wrong path      : rejected")
        print("   Wrong count     : rejected")
        print("   Wrong dataset   : rejected")
        print("   Result          : PASS")
        print()

        print("[4/4] ROI, n_frames, and threshold mismatches reject")
        outside_roi = root / "outside-roi.csv"
        changed = [dict(row) for row in rows]
        changed[0]["row"] = "2"
        write_csv(outside_roi, fields, changed)
        doc = json.loads(json.dumps(base_metadata))
        doc["dictionary"]["csv_path"] = str(outside_roi)
        outside_meta = root / "outside-roi.json"
        write_json(outside_meta, doc)
        expect_error(outside_roi, outside_meta, "outside metadata ROI")

        wrong_frames = root / "wrong-frames.csv"
        changed = [dict(row) for row in rows]
        changed[0]["n_frames"] = "9"
        changed[0]["lower_state_count"] = str(
            int(changed[0]["lower_state_count"]) + 1
        )
        write_csv(wrong_frames, fields, changed)
        doc = json.loads(json.dumps(base_metadata))
        doc["dictionary"]["csv_path"] = str(wrong_frames)
        frames_meta = root / "wrong-frames.json"
        write_json(frames_meta, doc)
        expect_error(wrong_frames, frames_meta, "n_frames does not match")

        wrong_threshold = root / "wrong-threshold.csv"
        changed = [dict(row) for row in rows]
        changed[0]["minimum_score"] = "0.8"
        write_csv(wrong_threshold, fields, changed)
        doc = json.loads(json.dumps(base_metadata))
        doc["dictionary"]["csv_path"] = str(wrong_threshold)
        threshold_meta = root / "wrong-threshold.json"
        write_json(threshold_meta, doc)
        expect_error(
            wrong_threshold, threshold_meta,
            "minimum_score does not match"
        )
        print("   Outside ROI      : rejected")
        print("   Wrong n_frames   : rejected")
        print("   Wrong threshold  : rejected")
        print("   Result           : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 joint artifact loading test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
