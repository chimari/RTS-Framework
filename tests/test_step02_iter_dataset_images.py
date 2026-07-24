"""Integration test for lazy Step 02 dataset image iteration."""

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


def write_fits(path: Path, data: np.ndarray) -> None:
    fits.PrimaryHDU(data=data).writeto(path, overwrite=True)


def make_row(
    image: Path,
    *,
    dataset: str,
    frame_index: int,
    n_frames: int,
    width: int,
    height: int,
    dtype: str = "uint16",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "directory": str(image.parent),
        "environment": "test",
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": -12.0 + frame_index * 0.1,
        "temperature_start_C": -12.0,
        "temperature_end_C": -11.9,
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


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 02 lazy image iterator test")
    print("=" * 72)
    print(f"step02 version : {step02.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step02_iter_") as temp_dir:
        root = Path(temp_dir)
        height, width = 4, 6

        frame0 = root / "frame_0000.fit"
        frame1 = root / "frame_0001.fit"
        write_fits(frame0, np.full((height, width), 10, dtype=np.uint16))
        write_fits(frame1, np.full((height, width), 20, dtype=np.uint16))

        manifest = root / "valid.csv"
        write_manifest(
            manifest,
            [
                make_row(
                    frame0,
                    dataset="bias",
                    frame_index=0,
                    n_frames=2,
                    width=width,
                    height=height,
                ),
                make_row(
                    frame1,
                    dataset="bias",
                    frame_index=1,
                    n_frames=2,
                    width=width,
                    height=height,
                ),
            ],
        )

        group = step02.prepare_frame_groups(manifest).get_group("bias")

        print("[1/4] Images are yielded in frame order")
        yielded = list(step02.iter_dataset_images(group))
        require(len(yielded) == 2, "expected two yielded images")
        require(
            [frame.frame_index for frame, _ in yielded] == [0, 1],
            "frames were yielded out of order",
        )
        require(
            [int(image[0, 0]) for _, image in yielded] == [10, 20],
            "unexpected image contents",
        )
        require(
            all(image.dtype == np.dtype("uint16") for _, image in yielded),
            "unexpected image dtype",
        )
        print("   Frame order : 0, 1")
        print("   Pixel values: 10, 20")
        print("   Result      : PASS")
        print()

        print("[2/4] Progress callback receives one-based positions")
        progress_events: list[tuple[int, int, int, str]] = []

        def progress(current, total, frame):
            progress_events.append(
                (current, total, frame.frame_index, frame.filepath.name)
            )

        iterator = step02.iter_dataset_images(group, progress=progress)
        first_frame, first_image = next(iterator)
        require(
            progress_events == [(1, 2, 0, "frame_0000.fit")],
            "progress callback was not lazy",
        )
        require(first_frame.frame_index == 0, "wrong first frame")
        require(int(first_image[0, 0]) == 10, "wrong first image")
        second_frame, second_image = next(iterator)
        require(
            progress_events
            == [
                (1, 2, 0, "frame_0000.fit"),
                (2, 2, 1, "frame_0001.fit"),
            ],
            "progress callback positions are wrong",
        )
        require(second_frame.frame_index == 1, "wrong second frame")
        require(int(second_image[0, 0]) == 20, "wrong second image")
        try:
            next(iterator)
        except StopIteration:
            stopped = True
        else:
            stopped = False
        require(stopped, "iterator did not stop")
        print("   Lazy callback : YES")
        print("   Positions     : 1/2, 2/2")
        print("   Result        : PASS")
        print()

        print("[3/4] Read errors include dataset and frame context")
        frame1.unlink()
        try:
            list(step02.iter_dataset_images(group))
        except step02.Step02Error as exc:
            message = str(exc)
        else:
            print("FAIL: missing image was accepted")
            return 1

        require("dataset='bias'" in message, "dataset missing from error")
        require("frame_index=1" in message, "frame index missing from error")
        require("frame_0001.fit" in message, "filepath missing from error")
        print("   Missing file  : rejected")
        print("   Dataset       : reported")
        print("   Frame index   : reported")
        print("   Result        : PASS")
        print()

        print("[4/4] Changed image geometry is detected at read time")
        write_fits(frame1, np.zeros((height + 1, width), dtype=np.uint16))
        try:
            list(step02.iter_dataset_images(group))
        except step02.Step02Error as exc:
            shape_message = str(exc)
        else:
            print("FAIL: changed image shape was accepted")
            return 1

        require(
            "shape does not match dataset metadata" in shape_message,
            "unexpected shape error",
        )
        require(
            f"expected=({height}, {width})" in shape_message,
            "expected shape missing from error",
        )
        require(
            f"actual=({height + 1}, {width})" in shape_message,
            "actual shape missing from error",
        )
        print("   Changed shape : rejected")
        print("   Expected      : reported")
        print("   Actual        : reported")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 02 lazy image iterator test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
