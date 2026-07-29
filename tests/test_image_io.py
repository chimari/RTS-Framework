#!/usr/bin/env python3
"""
Image input/output helpers for the RTS Framework.

This implementation provides source normalization, image-format detection,
two-dimensional FITS and headerless RAW image reading, image-shape inspection,
and image validation.

Headerless RAW input requires image geometry and pixel-layout metadata from a
FrameRecord.

Public API
----------
- ImageFormat
- ImageSource
- detect_format
- read_image
- get_image_shape
- validate_image
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
        description=(
            "Test image normalization, format detection, reading, "
            "and validation."
        )
    )

    parser.add_argument(
        "image_path",
        type=Path,
        help="Image file to inspect.",
    )

    parser.add_argument(
        "--expect",
        choices=("fits", "raw"),
        default=None,
        help="Optional expected format.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="RAW image width in pixels.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="RAW image height in pixels.",
    )

    parser.add_argument(
        "--dtype",
        default=None,
        help="RAW NumPy pixel dtype, for example uint16.",
    )

    parser.add_argument(
        "--byte-order",
        choices=("native", "little", "big"),
        default=None,
        help="RAW byte order.",
    )

    args = parser.parse_args()

    raw_metadata = (
        args.width,
        args.height,
        args.dtype,
        args.byte_order,
    )

    if args.expect == "raw":
        missing = [
            name
            for name, value in (
                ("--width", args.width),
                ("--height", args.height),
                ("--dtype", args.dtype),
                ("--byte-order", args.byte_order),
            )
            if value is None
        ]

        if missing:
            parser.error(
                "RAW input requires: " + ", ".join(missing)
            )

    elif any(value is not None for value in raw_metadata):
        parser.error(
            "--width, --height, --dtype, and --byte-order "
            "may only be used with --expect raw"
        )

    if args.width is not None and args.width <= 0:
        parser.error("--width must be a positive integer")

    if args.height is not None and args.height <= 0:
        parser.error("--height must be a positive integer")

    if args.dtype is not None:
        try:
            np.dtype(args.dtype)
        except TypeError:
            parser.error(
                f"--dtype is not NumPy-compatible: {args.dtype!r}"
            )

    return args


def make_frame_record(
    source: Path,
    image: np.ndarray | None,
    args: argparse.Namespace,
) -> FrameRecord:
    """Create a FrameRecord for either FITS or RAW testing."""

    if args.expect == "fits":
        assert image is not None

        width = image.shape[1]
        height = image.shape[0]
        dtype = str(image.dtype)
        byte_order = "not-applicable"

    else:
        width = args.width
        height = args.height
        dtype = args.dtype
        byte_order = args.byte_order

    return FrameRecord(
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
        image_width=width,
        image_height=height,
        pixel_dtype=dtype,
        byte_order=byte_order,
    )


def main() -> int:
    args = parse_args()
    source = args.image_path

    print("=" * 72)
    print("RTS Framework image_io smoke test")
    print("=" * 72)
    print(f"image_io.py file    : {image_io.__file__}")
    print(f"image_io.py version : {getattr(image_io, '__version__', '(not defined)')}")
    print()

    print("[1/13] Source normalization")
    normalized = image_io._normalize_source(source)
    print(f"   Input type : {type(source).__name__}")
    print(f"   Path       : {normalized}")
    print("   Result     : PASS")
    print()

    print("[2/13] Format detection")
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

    primary_label = (
        "Path"
        if args.expect == "fits"
        else "FrameRecord"
    )
        
    print(f"[3/13] Image reading from {primary_label}")
    frame = None

    if args.expect == "raw":
        frame = make_frame_record(
            source=source,
            image=None,
            args=args,
        )
        
    try:
        if args.expect == "fits":
            image = image_io.read_image(source)
        else:
            image = image_io.read_image(frame)

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

    print("[4/13] Image reading from FrameRecord")
    if args.expect == "fits":
        frame = make_frame_record(
            source=source,
            image=image,
            args=args,
        )    

    frame_image = image_io.read_image(frame)

    if not np.array_equal(frame_image, image):
        print(
            "   Result     : FAIL "
            "(FrameRecord result differs from first read)"
        )
        return 1

    print(f"   Source     : {type(frame).__name__}")
    print(f"   Shape      : {frame_image.shape}")
    print(f"   dtype      : {frame_image.dtype}")
    print("   Match first read : YES")
    print("   Result     : PASS")
    print()

    print(f"[5/13] Image shape from {primary_label}")
    
    if args.expect == "fits":
        primary_shape = image_io.get_image_shape(source)
    else:
        primary_shape = image_io.get_image_shape(frame)
        
    print(f"   Shape      : {primary_shape}")
    print(f"   Match read : {primary_shape == image.shape}")
    
    if primary_shape != image.shape:
        print("   Result     : FAIL")
        return 1
    
    print("   Result     : PASS")
    print()

    print("[6/13] Image shape from FrameRecord")
    frame_shape = image_io.get_image_shape(frame)
    print(f"   Source     : {type(frame).__name__}")
    print(f"   Shape      : {frame_shape}")
    print(f"   Match first shape : {frame_shape == primary_shape}")
    if frame_shape != primary_shape:
        print("   Result     : FAIL")
        return 1
    print("   Result     : PASS")
    print()

    print(f"[7/13] Image validation from {primary_label}")
    if args.expect == "fits":
        image_io.validate_image(source)
    else:
        image_io.validate_image(frame)

    print("   Result     : PASS")
    print()

    print("[8/13] Image validation from FrameRecord")
    image_io.validate_image(frame)
    print(f"   Geometry   : {frame.image_width} x {frame.image_height}")
    print(f"   dtype      : {frame.pixel_dtype}")
    print("   Metadata   : MATCH")
    print("   Result     : PASS")
    print()

    def expect_image_io_error(primary_label: str, callback) -> bool:
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

    print("[9/13] Missing file rejection")
    missing_path = source.with_name(f"{source.name}.missing")
    if not expect_image_io_error(
        "missing file",
        lambda: image_io.validate_image(missing_path),
    ):
        return 1

    print("[10/13] Shape mismatch rejection")
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
        byte_order = (
            "not-applicable"
            if args.expect == "fits"
            else args.byte_order
        )        
    )
    if not expect_image_io_error(
        "shape mismatch",
        lambda: image_io.validate_image(wrong_shape_frame),
    ):
        return 1

    print("[11/13] dtype mismatch rejection")
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
        byte_order = (
            "not-applicable"
            if args.expect == "fits"
            else args.byte_order
        )        
    )
    if not expect_image_io_error(
        "dtype mismatch",
        lambda: image_io.validate_image(wrong_dtype_frame),
    ):
        return 1

    print("[12/13] Incomplete geometry rejection")
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
        byte_order = (
            "not-applicable"
            if args.expect == "fits"
            else args.byte_order
        )
    )
    if not expect_image_io_error(
        "incomplete geometry",
        lambda: image_io.validate_image(incomplete_geometry_frame),
    ):
        return 1

    print("[13/13] Missing-file shape rejection")
    missing_shape_path = source.with_name(f"{source.name}.shape-missing")
    if not expect_image_io_error(
        "missing shape file",
        lambda: image_io.get_image_shape(missing_shape_path),
    ):
        return 1

    print("=" * 72)
    print("FINISHED: image_io FITS/RAW smoke tests passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
