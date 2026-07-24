"""Integration test for Step 02 per-frame statistics."""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steps import step02_prepare_frame_groups as step02


def write_fits(path: Path, data: np.ndarray) -> None:
    fits.PrimaryHDU(data=data).writeto(path, overwrite=True)


def make_row(
    image: Path,
    *,
    frame_index: int,
    n_frames: int,
    width: int,
    height: int,
    dtype: str,
) -> dict[str, object]:
    return {
        "dataset": "stats",
        "directory": str(image.parent),
        "environment": "test",
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": -10.0 + frame_index,
        "temperature_start_C": -10.0,
        "temperature_end_C": -9.0,
        "temperature_fraction": (
            0.0 if n_frames == 1 else frame_index / (n_frames - 1)
        ),
        "exposure_s": 0.0,
        "filename": image.name,
        "filepath": str(image),
        "image_width": width,
        "image_height": height,
        "pixel_dtype": dtype,
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


def close(actual: float, expected: float, tol: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=tol, abs_tol=tol)


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 02 frame-statistics integration test")
    print("=" * 72)
    print(f"step02 version : {step02.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step02_stats_") as temp_dir:
        root = Path(temp_dir)
        height, width = 2, 3

        frame0 = root / "frame_0000.fit"
        frame1 = root / "frame_0001.fit"

        data0 = np.array(
            [[1, 2, 3], [4, 5, 6]],
            dtype=np.uint16,
        )
        data1 = np.array(
            [[10, 20, 30], [40, 50, 60]],
            dtype=np.uint16,
        )
        write_fits(frame0, data0)
        write_fits(frame1, data1)

        manifest = root / "valid.csv"
        write_manifest(
            manifest,
            [
                make_row(
                    frame0,
                    frame_index=0,
                    n_frames=2,
                    width=width,
                    height=height,
                    dtype="uint16",
                ),
                make_row(
                    frame1,
                    frame_index=1,
                    n_frames=2,
                    width=width,
                    height=height,
                    dtype="uint16",
                ),
            ],
        )

        group = step02.prepare_frame_groups(manifest).get_group("stats")

        print("[1/4] Integer frame statistics are correct")
        result = step02.compute_dataset_statistics(group)
        require(result.dataset == "stats", "wrong dataset name")
        require(result.n_frames == 2, "wrong frame count")

        first = result.frames[0]
        require(first.frame_index == 0, "wrong first frame index")
        require(first.total_pixels == 6, "wrong total pixel count")
        require(first.finite_pixels == 6, "wrong finite pixel count")
        require(close(first.minimum, 1.0), "wrong minimum")
        require(close(first.maximum, 6.0), "wrong maximum")
        require(close(first.mean, 3.5), "wrong mean")
        require(close(first.median, 3.5), "wrong median")
        require(close(first.stddev, float(np.std(data0))), "wrong stddev")

        second = result.frames[1]
        require(close(second.mean, 35.0), "wrong second mean")
        require(close(second.median, 35.0), "wrong second median")
        print("   Frame 0 mean   : 3.5")
        print("   Frame 0 median : 3.5")
        print("   Frame 1 mean   : 35")
        print("   Result         : PASS")
        print()

        print("[2/4] Dataset ranges are derived from frame statistics")
        require(close(result.frame_mean_min, 3.5), "wrong mean minimum")
        require(close(result.frame_mean_max, 35.0), "wrong mean maximum")
        require(close(result.frame_median_min, 3.5), "wrong median minimum")
        require(close(result.frame_median_max, 35.0), "wrong median maximum")
        print("   Mean range   : 3.5 .. 35")
        print("   Median range : 3.5 .. 35")
        print("   Result       : PASS")
        print()

        print("[3/4] Progress callback is forwarded through lazy loading")
        events: list[tuple[int, int, int]] = []

        def progress(current, total, frame):
            events.append((current, total, frame.frame_index))

        progressed = step02.compute_dataset_statistics(
            group,
            progress=progress,
        )
        require(progressed.n_frames == 2, "progressed result is wrong")
        require(events == [(1, 2, 0), (2, 2, 1)], "progress events are wrong")
        print("   Events : 1/2, 2/2")
        print("   Result : PASS")
        print()

        print("[4/4] Non-finite floating pixels are excluded")
        float_frame = root / "float.fit"
        float_data = np.array(
            [[1.0, np.nan], [3.0, np.inf]],
            dtype=np.float32,
        )
        write_fits(float_frame, float_data)

        float_manifest = root / "float.csv"
        write_manifest(
            float_manifest,
            [
                make_row(
                    float_frame,
                    frame_index=0,
                    n_frames=1,
                    width=2,
                    height=2,
                    dtype="float32",
                )
            ],
        )
        float_group = step02.prepare_frame_groups(
            float_manifest
        ).get_group("stats")
        float_result = step02.compute_dataset_statistics(float_group)
        record = float_result.frames[0]

        require(record.total_pixels == 4, "wrong float total count")
        require(record.finite_pixels == 2, "wrong float finite count")
        require(close(record.minimum, 1.0), "wrong float minimum")
        require(close(record.maximum, 3.0), "wrong float maximum")
        require(close(record.mean, 2.0), "wrong float mean")
        require(close(record.median, 2.0), "wrong float median")
        require(close(record.stddev, 1.0), "wrong float stddev")
        print("   Total pixels  : 4")
        print("   Finite pixels : 2")
        print("   Mean          : 2")
        print("   Result        : PASS")
        print()

        print(result.summary_line())
        print()

    print("=" * 72)
    print("FINISHED: Step 02 frame-statistics integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
