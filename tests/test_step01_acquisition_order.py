"""Regression test for normalized-manifest acquisition order."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steps.step01_prepare_dataset import (
    NORMALIZED_MANIFEST_COLUMNS,
    prepare_dataset,
    write_normalized_manifest,
)

from astropy.io import fits

from steps.step01_prepare_dataset import (
    NORMALIZED_MANIFEST_COLUMNS,
    prepare_dataset,
    write_normalized_manifest,
)


def _write_fits(path: Path, value: int) -> None:
    """Write a small valid FITS test image."""
    image = np.full((4, 5), value, dtype=np.uint16)
    fits.writeto(path, image, overwrite=True)


def _manifest_row(
    path: Path,
    *,
    dataset: str,
    environment: str,
    frame_index: int,
    n_frames: int,
) -> dict[str, object]:
    """Build one valid manifest CSV row."""
    return {
        "dataset": dataset,
        "directory": str(path.parent),
        "environment": environment,
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": -12.0,
        "temperature_start_C": -12.0,
        "temperature_end_C": -12.0,
        "temperature_fraction": (
            0.0 if n_frames == 1 else frame_index / (n_frames - 1)
        ),
        "exposure_s": 0.0,
        "filename": path.name,
        "filepath": str(path),
        "image_width": 5,
        "image_height": 4,
        "pixel_dtype": "uint16",
        "byte_order": "not-applicable",
    }


def test_normalized_manifest_preserves_acquisition_order(
    tmp_path: Path,
) -> None:
    """Output rows must retain the acquisition order of the source CSV."""
    cold_0 = tmp_path / "cold_0000.fits"
    room_0 = tmp_path / "room_0000.fits"
    cold_1 = tmp_path / "cold_0001.fits"

    _write_fits(cold_0, 10)
    _write_fits(room_0, 20)
    _write_fits(cold_1, 30)

    # Deliberately interleave two datasets.
    #
    # Acquisition order:
    #   cold frame 0
    #   room frame 0
    #   cold frame 1
    #
    # Sorting by dataset would incorrectly move cold frame 1 before
    # room frame 0.
    source_rows = [
        _manifest_row(
            cold_0,
            dataset="bias-cold",
            environment="cold",
            frame_index=0,
            n_frames=2,
        ),
        _manifest_row(
            room_0,
            dataset="bias-room",
            environment="room",
            frame_index=0,
            n_frames=1,
        ),
        _manifest_row(
            cold_1,
            dataset="bias-cold",
            environment="cold",
            frame_index=1,
            n_frames=2,
        ),
    ]

    source_manifest = tmp_path / "source_manifest.csv"
    output_manifest = tmp_path / "normalized_manifest.csv"

    with source_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=NORMALIZED_MANIFEST_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(source_rows)

    result = prepare_dataset(
        source_manifest,
        validation_mode="shape",
    )

    assert result.valid, result.summary()

    write_normalized_manifest(result, output_manifest)

    with output_manifest.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        output_rows = list(csv.DictReader(stream))

    assert [row["filename"] for row in output_rows] == [
        "cold_0000.fits",
        "room_0000.fits",
        "cold_0001.fits",
    ]
