"""Integration test for Step 04 dictionary metadata v4.22.0."""

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
            "environment": "step04-v4.22-test",
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


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 dictionary metadata test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_metadata_") as temp:
        root = Path(temp)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(write_dataset(root), "bias")
        )

        print("[1/4] Artifact build writes CSV and canonical metadata sidecar")
        csv_path = root / "dictionary.csv"
        artifacts = step04.build_rts_dictionary_artifacts(
            plan, csv_path, **kwargs()
        )
        expected_metadata = root / "dictionary.csv.metadata.json"
        require(artifacts.output_path == csv_path, "wrong CSV path")
        require(artifacts.metadata_path == expected_metadata,
                "wrong metadata path")
        require(csv_path.is_file(), "CSV was not written")
        require(expected_metadata.is_file(), "metadata was not written")
        print("   CSV path      : canonical")
        print("   Metadata path : canonical sidecar")
        print("   Result        : PASS")
        print()

        print("[2/4] Metadata content is complete and deterministic")
        raw1 = expected_metadata.read_bytes()
        document = json.loads(raw1.decode("utf-8"))
        require(
            document["schema"]
            == "rts-framework.step04.dictionary-metadata",
            "wrong schema",
        )
        require(document["schema_version"] == 1, "wrong schema version")
        require(document["step04_version"] == step04.__version__,
                "wrong Step 04 version")
        require(document["dictionary"]["csv_path"] == str(csv_path),
                "wrong CSV path in metadata")
        require(document["dictionary"]["analyzed_pixel_count"] == 6,
                "wrong analyzed count")
        require(document["input"]["n_frames"] == 8, "wrong frame count")
        require(document["input"]["image_shape"] == [2, 3],
                "wrong image shape")
        require(len(document["input"]["filepaths"]) == 8,
                "file list missing")
        require(document["parameters"]["pixel_count"] == 6,
                "parameters missing")

        second = root / "metadata-copy.json"
        step04.write_rts_dictionary_metadata_json(
            second, plan, artifacts.build_result
        )
        require(second.read_bytes() == raw1,
                "metadata serialization is not deterministic")
        print("   Schema      : versioned")
        print("   Input files : recorded")
        print("   Parameters  : recorded")
        print("   JSON bytes  : deterministic")
        print("   Result      : PASS")
        print()

        print("[3/4] Explicit metadata path and atomic replacement work")
        custom = root / "custom" / "run.json"
        custom.parent.mkdir(parents=True, exist_ok=True)
        custom.write_text("old metadata\n", encoding="utf-8")
        custom_artifacts = step04.build_rts_dictionary_artifacts(
            plan,
            root / "custom.csv",
            metadata_path=custom,
            row_start=1,
            row_stop=2,
            column_start=1,
            column_stop=3,
            **kwargs(),
        )
        require(custom_artifacts.metadata_path == custom,
                "custom metadata path ignored")
        custom_doc = json.loads(custom.read_text(encoding="utf-8"))
        require(custom_doc["parameters"]["pixel_count"] == 2,
                "subregion metadata incorrect")
        require(not list(custom.parent.glob(".*.tmp")),
                "temporary metadata file remains")
        print("   Custom path : supported")
        print("   Replacement : atomic")
        print("   Temp files  : removed")
        print("   Result      : PASS")
        print()

        print("[4/4] Existing CSV APIs and cancellation remain unchanged")
        legacy = root / "legacy.csv"
        returned = step04.build_rts_dictionary_csv(
            plan, legacy, **kwargs()
        )
        require(returned == legacy, "legacy API changed")
        require(legacy.read_bytes() == csv_path.read_bytes(),
                "legacy CSV bytes changed")

        cancelled_csv = root / "cancelled.csv"
        cancelled_metadata = root / "cancelled.json"
        try:
            step04.build_rts_dictionary_artifacts(
                plan,
                cancelled_csv,
                metadata_path=cancelled_metadata,
                cancel_requested=lambda: True,
                **kwargs(),
            )
        except step04.Step04Cancelled:
            pass
        else:
            require(False, "cancellation was not raised")
        require(not cancelled_csv.exists(), "cancelled CSV exists")
        require(not cancelled_metadata.exists(), "cancelled metadata exists")
        print("   Legacy API   : unchanged")
        print("   CSV bytes    : identical")
        print("   Cancellation : no artifacts")
        print("   Result       : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 dictionary metadata test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
