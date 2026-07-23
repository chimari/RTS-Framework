#!/usr/bin/env python3
"""
Smoke test for common.image_io FITS reading.

Usage
-----
python tests/test_image_io.py IMAGE_PATH --expect fits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common import image_io  # noqa: E402
from common.manifest import FrameRecord  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test image normalization, format detection, reading, and validation."
    )
    parser.add_argument("image_path", type=Path, help="Image file to inspect.")
    parser.add_argument(
        "--expect",
        choices=("fits", "raw"),
        default=None,
        help="Optional expected format.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.image_path

    print("=" * 72)
    print("RTS Framework image_io smoke test")
    print("=" * 72)
    print(f"image_io.py file    : {image_io.__file__}")
    print(f"image_io.py version : {getattr(image_io, '__version__', '(not defined)')}")
    print()

    print("[1/10] Source normalization")
    normalized = image_io._normalize_source(source)
    print(f"   Input type : {type(source).__name__}")
    print(f"   Path       : {normalized}")
    print("   Result     : PASS")
    print()

    print("[2/10] Format detection")
    detected = image_io.detect_format(source)
    print(f"   Format     : {detected.name}")
    print(f"   Value      : {detected.value}")

    if args.expect is not None and detected.value != args.expect:
        print(
            f"   Result     : FAIL "
            f"(expected {args.expect!r}, detected {detected.value!r})"
        )
        return 1

    print("   Result     : PASS")
    print()

    print("[3/10] Image reading from Path")
    try:
        image = image_io.read_image(source)

    except Exception as exc:
        import traceback

        print(f"\nException: {type(exc).__name__}")
        print(f"Message  : {exc}")
        print("\nFull traceback:")
        traceback.print_exc()

        return 1

    if not isinstance(image, np.ndarray):
        print(f"   Result     : FAIL (returned {type(image).__name__}, not ndarray)")
        return 1

    if image.ndim != 2:
        print(f"   Result     : FAIL (ndim={image.ndim}, shape={image.shape})")
        return 1

    print(f"   Type       : {type(image).__name__}")
    print(f"   Shape      : {image.shape}")
    print(f"   dtype      : {image.dtype}")
    print(f"   Min        : {np.min(image)}")
    print(f"   Max        : {np.max(image)}")
    print("   Result     : PASS")
    print()

    print("[4/10] Image reading from FrameRecord")
    frame = FrameRecord(
        manifest_row=0,
        dataset="smoke-test",
        directory=str(source.parent),
        environment="test",
        frame_index=0,
        n_frames=1,
        temperature_C=0.0,
        temperature_start_C=0.0,
        temperature_end_C=0.0,
        temperature_fraction=0.0,
        exposure_s=1.0,
        filename=source.name,
        filepath=source,
        image_width=image.shape[1],
        image_height=image.shape[0],
        pixel_dtype=str(image.dtype),
        byte_order="not-applicable",
    )

    frame_image = image_io.read_image(frame)

    if not np.array_equal(frame_image, image):
        print("   Result     : FAIL (FrameRecord result differs from Path result)")
        return 1

    print(f"   Source     : {type(frame).__name__}")
    print(f"   Shape      : {frame_image.shape}")
    print(f"   dtype      : {frame_image.dtype}")
    print("   Match Path : YES")
    print("   Result     : PASS")
    print()

    print("[5/10] Image validation from Path")
    image_io.validate_image(source)
    print("   Result     : PASS")
    print()

    print("[6/10] Image validation from FrameRecord")
    image_io.validate_image(frame)
    print(f"   Geometry   : {frame.image_width} x {frame.image_height}")
    print(f"   dtype      : {frame.pixel_dtype}")
    print("   Metadata   : MATCH")
    print("   Result     : PASS")
    print()

    def expect_image_io_error(label: str, callback) -> bool:
        try:
            callback()
        except image_io.ImageIOError as exc:
            print(f"   Error      : {exc}")
            print("   Result     : PASS")
            print()
            return True
        except Exception as exc:
            print(
                "   Result     : FAIL "
                f"(expected ImageIOError, got {type(exc).__name__}: {exc})"
            )
            print()
            return False

        print("   Result     : FAIL (ImageIOError was not raised)")
        print()
        return False

    print("[7/10] Missing file rejection")
    missing_path = source.with_name(f"{source.name}.missing")
    if not expect_image_io_error(
        "missing file",
        lambda: image_io.validate_image(missing_path),
    ):
        return 1

    print("[8/10] Shape mismatch rejection")
    wrong_shape_frame = FrameRecord(
        manifest_row=1,
        dataset="smoke-test",
        directory=str(source.parent),
        environment="test",
        frame_index=0,
        n_frames=1,
        temperature_C=0.0,
        temperature_start_C=0.0,
        temperature_end_C=0.0,
        temperature_fraction=0.0,
        exposure_s=1.0,
        filename=source.name,
        filepath=source,
        image_width=image.shape[1] + 1,
        image_height=image.shape[0],
        pixel_dtype=str(image.dtype),
        byte_order="not-applicable",
    )
    if not expect_image_io_error(
        "shape mismatch",
        lambda: image_io.validate_image(wrong_shape_frame),
    ):
        return 1

    print("[9/10] dtype mismatch rejection")
    wrong_dtype = "float32" if image.dtype != np.dtype("float32") else "uint16"
    wrong_dtype_frame = FrameRecord(
        manifest_row=2,
        dataset="smoke-test",
        directory=str(source.parent),
        environment="test",
        frame_index=0,
        n_frames=1,
        temperature_C=0.0,
        temperature_start_C=0.0,
        temperature_end_C=0.0,
        temperature_fraction=0.0,
        exposure_s=1.0,
        filename=source.name,
        filepath=source,
        image_width=image.shape[1],
        image_height=image.shape[0],
        pixel_dtype=wrong_dtype,
        byte_order="not-applicable",
    )
    if not expect_image_io_error(
        "dtype mismatch",
        lambda: image_io.validate_image(wrong_dtype_frame),
    ):
        return 1

    print("[10/10] Incomplete geometry rejection")
    incomplete_geometry_frame = FrameRecord(
        manifest_row=3,
        dataset="smoke-test",
        directory=str(source.parent),
        environment="test",
        frame_index=0,
        n_frames=1,
        temperature_C=0.0,
        temperature_start_C=0.0,
        temperature_end_C=0.0,
        temperature_fraction=0.0,
        exposure_s=1.0,
        filename=source.name,
        filepath=source,
        image_width=image.shape[1],
        image_height=None,
        pixel_dtype=str(image.dtype),
        byte_order="not-applicable",
    )
    if not expect_image_io_error(
        "incomplete geometry",
        lambda: image_io.validate_image(incomplete_geometry_frame),
    ):
        return 1

    print("=" * 72)
    print("FINISHED: image_io validation tests passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
