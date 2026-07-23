"""Smoke test for Step 01 using one real image supplied on the command line."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steps import step01_prepare_dataset as step01


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    image = args.image
    if not image.is_file():
        print(f"FAIL: image does not exist: {image}")
        return 1

    # The existing FITS smoke test has established this geometry and dtype.
    width = 9576
    height = 6388
    dtype = "uint16"

    print("=" * 72)
    print("RTS Framework Step 01 smoke test")
    print("=" * 72)
    print(f"step01 version : {step01.__version__}")
    print(f"image          : {image}")
    print()

    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_path = Path(temp_dir) / "manifest.csv"
        columns = (
            "dataset", "directory", "environment", "frame_index", "n_frames",
            "temperature_C", "temperature_start_C", "temperature_end_C",
            "temperature_fraction", "exposure_s", "filename", "filepath",
            "image_width", "image_height", "pixel_dtype", "byte_order",
        )
        row = {
            "dataset": "smoke-test",
            "directory": str(image.parent),
            "environment": "test",
            "frame_index": 0,
            "n_frames": 1,
            "temperature_C": -12.1,
            "temperature_start_C": -12.1,
            "temperature_end_C": -12.1,
            "temperature_fraction": 0.0,
            "exposure_s": 0.0,
            "filename": image.name,
            "filepath": str(image),
            "image_width": width,
            "image_height": height,
            "pixel_dtype": dtype,
            "byte_order": "not-applicable",
        }
        with manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerow(row)

        progress_calls: list[tuple[int, int, str]] = []

        def show_progress(current, total, frame):
            progress_calls.append((current, total, frame.filename))
            print(f"Checking image {current}/{total}: {frame.filename}")

        mode = "full" if args.full else "shape"
        result = step01.prepare_dataset(
            manifest_path,
            validation_mode=mode,
            progress=show_progress,
        )

        print()
        print(result.summary())
        print()

        if not result.valid:
            print("Result: FAIL")
            return 1
        if result.manifest.n_frames != 1 or result.manifest.n_datasets != 1:
            print("Result: FAIL (unexpected manifest counts)")
            return 1
        if progress_calls != [(1, 1, image.name)]:
            print(f"Result: FAIL (unexpected progress calls: {progress_calls})")
            return 1

    print("Result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
