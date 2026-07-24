"""Integration test for Step 03 lazy bias-frame iteration v3.1.0."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
from types import GeneratorType

import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steps import step03_prepare_bias_analysis as step03


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def write_fits(path: Path, data: np.ndarray) -> None:
    fits.PrimaryHDU(data=data).writeto(path, overwrite=True)


def make_row(
    image: Path,
    *,
    frame_index: int,
    n_frames: int,
) -> dict[str, object]:
    return {
        "dataset": "bias",
        "directory": str(image.parent),
        "environment": "step03-v3.1-test",
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": -10.0 + 0.1 * frame_index,
        "temperature_start_C": -10.0,
        "temperature_end_C": -9.8,
        "temperature_fraction": frame_index / (n_frames - 1),
        "exposure_s": 0.0,
        "filename": image.name,
        "filepath": str(image),
        "image_width": 3,
        "image_height": 2,
        "pixel_dtype": "uint16",
        "byte_order": "not-applicable",
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def expect_step03_error(callable_, contains: str) -> None:
    try:
        callable_()
    except step03.Step03Error as exc:
        require(contains in str(exc), f"wrong error message: {exc}")
    else:
        require(False, "Step03Error was not raised")


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 03 lazy bias-frame integration test")
    print("=" * 72)
    print(f"step03 version : {step03.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step03_frames_") as temp_dir:
        root = Path(temp_dir)
        paths = [root / f"bias_{index:04d}.fit" for index in range(3)]
        source_arrays = [
            np.arange(0, 6, dtype=np.uint16).reshape(2, 3),
            np.arange(10, 16, dtype=np.uint16).reshape(2, 3),
            np.arange(20, 26, dtype=np.uint16).reshape(2, 3),
        ]
        for path, data in zip(paths, source_arrays, strict=True):
            write_fits(path, data)

        # Deliberately reverse manifest row order. Step 02 must restore
        # canonical frame_index order before Step 03 yields images.
        manifest = root / "manifest.normalized.csv"
        write_manifest(
            manifest,
            [
                make_row(paths[2], frame_index=2, n_frames=3),
                make_row(paths[0], frame_index=0, n_frames=3),
                make_row(paths[1], frame_index=1, n_frames=3),
            ],
        )
        plan = step03.prepare_bias_analysis(manifest, "bias")

        print("[1/4] Frames are yielded lazily in canonical order")
        iterator = step03.iter_bias_frames(plan)
        require(isinstance(iterator, GeneratorType), "result is not a generator")
        first = next(iterator)
        second = next(iterator)
        third = next(iterator)
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            require(False, "iterator yielded too many frames")
        require(np.array_equal(first, source_arrays[0]), "wrong first frame")
        require(np.array_equal(second, source_arrays[1]), "wrong second frame")
        require(np.array_equal(third, source_arrays[2]), "wrong third frame")
        print("   Iterator : generator")
        print("   Order    : 0, 1, 2")
        print("   Result   : PASS")
        print()

        print("[2/4] Every frame is independent float64 and C-contiguous")
        frames = list(step03.iter_bias_frames(plan))
        for index, frame in enumerate(frames):
            require(frame.dtype == np.dtype(np.float64), "dtype is not float64")
            require(frame.shape == (2, 3), "shape was not preserved")
            require(frame.flags.c_contiguous, "frame is not C-contiguous")
            require(frame.flags.owndata, "frame does not own its memory")
            require(
                not np.shares_memory(frame, source_arrays[index]),
                "frame shares source memory",
            )
        require(
            not np.shares_memory(frames[0], frames[1]),
            "yielded frames share memory",
        )
        print("   dtype       : float64")
        print("   C-contiguous: YES")
        print("   Own data    : YES")
        print("   Result      : PASS")
        print()

        print("[3/4] Yielded arrays are read-only")
        require(not frames[0].flags.writeable, "frame is writable")
        try:
            frames[0][0, 0] = 999.0
        except ValueError:
            pass
        else:
            require(False, "read-only frame accepted modification")
        print("   Writable : NO")
        print("   Result   : PASS")
        print()

        print("[4/4] Invalid input and read failures become Step03Error")
        expect_step03_error(
            lambda: next(step03.iter_bias_frames(object())),
            "plan must be a BiasAnalysisPlan",
        )

        paths[1].unlink()
        expect_step03_error(
            lambda: list(step03.iter_bias_frames(plan)),
            "Unable to iterate bias dataset 'bias'",
        )
        print("   Invalid plan : rejected")
        print("   Missing FITS : translated")
        print("   Result       : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 03 lazy bias-frame integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
