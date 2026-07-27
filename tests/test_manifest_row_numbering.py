from __future__ import annotations

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.manifest import FrameManifest

FIELDNAMES = (
    "dataset",
    "directory",
    "environment",
    "frame_index",
    "n_frames",
    "temperature_C",
    "temperature_start_C",
    "temperature_end_C",
    "temperature_fraction",
    "exposure_s",
    "filename",
    "filepath",
)


def _write_manifest(path: Path) -> None:
    rows = [
        {
            "dataset": "cold",
            "directory": "data",
            "environment": "cold",
            "frame_index": 0,
            "n_frames": 2,
            "temperature_C": -12.0,
            "temperature_start_C": -12.0,
            "temperature_end_C": -12.0,
            "temperature_fraction": 0.0,
            "exposure_s": 0.0,
            "filename": "cold_0000.fit",
            "filepath": "data/cold_0000.fit",
        },
        {
            "dataset": "room",
            "directory": "data",
            "environment": "room",
            "frame_index": 0,
            "n_frames": 1,
            "temperature_C": 20.0,
            "temperature_start_C": 20.0,
            "temperature_end_C": 20.0,
            "temperature_fraction": 0.0,
            "exposure_s": 0.0,
            "filename": "room_0000.fit",
            "filepath": "data/room_0000.fit",
        },
        {
            "dataset": "cold",
            "directory": "data",
            "environment": "cold",
            "frame_index": 1,
            "n_frames": 2,
            "temperature_C": -12.0,
            "temperature_start_C": -12.0,
            "temperature_end_C": -12.0,
            "temperature_fraction": 1.0,
            "exposure_s": 0.0,
            "filename": "cold_0001.fit",
            "filepath": "data/cold_0001.fit",
        },
    ]

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_from_csv_preserves_physical_csv_line_numbers(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path)

    manifest = FrameManifest.from_csv(manifest_path)

    assert [frame.manifest_row for frame in manifest.frames] == [2, 3, 4]


def test_dataset_grouping_does_not_change_manifest_row_numbers(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path)

    manifest = FrameManifest.from_csv(manifest_path)

    cold = manifest.get_dataset("cold")
    room = manifest.get_dataset("room")

    assert [frame.manifest_row for frame in cold.frames] == [2, 4]
    assert [frame.manifest_row for frame in room.frames] == [3]
