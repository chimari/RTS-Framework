"""Integration test for deterministic Step 02 statistics CSV."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steps import step02_prepare_frame_groups as step02


EXPECTED_COLUMNS = [
    "dataset",
    "frame_index",
    "filepath",
    "temperature_C",
    "exposure_s",
    "finite_pixels",
    "total_pixels",
    "minimum",
    "maximum",
    "mean",
    "median",
    "stddev",
]


def write_fits(path: Path, data: np.ndarray) -> None:
    fits.PrimaryHDU(data=data).writeto(path, overwrite=True)


def make_row(
    image: Path,
    *,
    frame_index: int,
    n_frames: int,
    temperature_C: float,
    width: int,
    height: int,
) -> dict[str, object]:
    return {
        "dataset": "bias",
        "directory": str(image.parent),
        "environment": "test",
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": temperature_C,
        "temperature_start_C": -12.0,
        "temperature_end_C": -11.5,
        "temperature_fraction": frame_index / (n_frames - 1),
        "exposure_s": 0.0,
        "filename": image.name,
        "filepath": str(image),
        "image_width": width,
        "image_height": height,
        "pixel_dtype": "uint16",
        "byte_order": "not-applicable",
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 02 statistics CSV integration test")
    print("=" * 72)
    print(f"step02 version : {step02.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step02_csv_") as temp_dir:
        root = Path(temp_dir)
        height, width = 2, 3

        frame0 = root / "frame_0000.fit"
        frame1 = root / "frame_0001.fit"
        write_fits(
            frame0,
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16),
        )
        write_fits(
            frame1,
            np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint16),
        )

        manifest = root / "manifest.csv"
        # Intentionally reverse the input row order.
        write_manifest(
            manifest,
            [
                make_row(
                    frame1,
                    frame_index=1,
                    n_frames=2,
                    temperature_C=-11.5,
                    width=width,
                    height=height,
                ),
                make_row(
                    frame0,
                    frame_index=0,
                    n_frames=2,
                    temperature_C=-12.0,
                    width=width,
                    height=height,
                ),
            ],
        )

        group = step02.prepare_frame_groups(manifest).get_group("bias")
        statistics = step02.compute_dataset_statistics(group)

        output = root / "output" / "statistics.csv"

        print("[1/4] Statistics CSV is created")
        written = step02.write_statistics_csv(statistics, output)
        require(written == output, "writer returned wrong path")
        require(output.is_file(), "CSV was not created")
        print("   CSV    : created")
        print("   Result : PASS")
        print()

        print("[2/4] Column order and frame order are canonical")
        columns, rows = read_csv(output)
        require(columns == EXPECTED_COLUMNS, "column order is wrong")
        require(len(rows) == 2, "wrong row count")
        require(
            [row["frame_index"] for row in rows] == ["0", "1"],
            "frame rows are not ordered",
        )
        require(
            all(Path(row["filepath"]).is_absolute() for row in rows),
            "filepaths are not absolute",
        )
        print("   Columns     : canonical")
        print("   Frame order : 0, 1")
        print("   Paths       : absolute")
        print("   Result      : PASS")
        print()

        print("[3/4] Numeric values are preserved")
        first = rows[0]
        second = rows[1]
        require(first["dataset"] == "bias", "wrong dataset")
        require(first["finite_pixels"] == "6", "wrong finite count")
        require(first["total_pixels"] == "6", "wrong total count")
        require(float(first["minimum"]) == 1.0, "wrong minimum")
        require(float(first["maximum"]) == 6.0, "wrong maximum")
        require(float(first["mean"]) == 3.5, "wrong mean")
        require(float(first["median"]) == 3.5, "wrong median")
        require(float(second["mean"]) == 35.0, "wrong second mean")
        print("   Frame 0 mean : 3.5")
        print("   Frame 1 mean : 35")
        print("   Result       : PASS")
        print()

        print("[4/4] Repeated output is byte-for-byte identical")
        repeated = root / "output" / "statistics.repeat.csv"
        step02.write_statistics_csv(statistics, repeated)
        require(
            output.read_bytes() == repeated.read_bytes(),
            "CSV output is not deterministic",
        )
        require(b"\r\n" not in output.read_bytes(), "CSV does not use LF")
        print("   Deterministic : YES")
        print("   Line endings  : LF")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 02 statistics CSV integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
