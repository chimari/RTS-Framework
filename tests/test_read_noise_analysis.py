"""Small integration test for optional Step 02 read-noise output."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from astropy.io import fits

from common.read_noise_analysis import ReadNoiseConfig, analyze_read_noise_dataset
from steps.step02_prepare_frame_groups import prepare_frame_groups


def test_read_noise_outputs(tmp_path: Path) -> None:
    rows = []
    for index in range(4):
        path = tmp_path / f"bias_{index:02d}.fit"
        rng = np.random.default_rng(index)
        data = 1000 + rng.normal(0, 2, size=(32, 40))
        fits.PrimaryHDU(data.astype(np.float32)).writeto(path)
        rows.append({
            "dataset": "bias",
            "directory": str(tmp_path),
            "environment": "test",
            "frame_index": index,
            "n_frames": 4,
            "temperature_C": -10.0,
            "temperature_start_C": -10.0,
            "temperature_end_C": -10.0,
            "temperature_fraction": index / 3,
            "exposure_s": 0.0,
            "filename": path.name,
            "filepath": str(path),
            "image_width": 40,
            "image_height": 32,
            "pixel_dtype": "float32",
            "byte_order": "not_applicable",
        })

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    group = prepare_frame_groups(manifest).get_group("bias")
    output = tmp_path / "science"
    result = analyze_read_noise_dataset(
        group,
        ReadNoiseConfig(output_dir=output),
    )

    assert result.status == "PASSED"
    assert result.pairs == 2
    assert (output / "read_noise_summary.csv").is_file()
    assert (output / "temporal_noise_map.fits").is_file()
    assert (output / "pair_difference_histogram_full_range_log.png").is_file()
