#!/usr/bin/env python3
"""
Smoke test for common.image_io format detection.

Usage
-----
python tests/test_image_io.py IMAGE_PATH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Allow direct execution from the repository's tests/ directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common import image_io  # noqa: E402

print(image_io.__file__)
print(dir(image_io))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test image-source normalization and format detection."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="FITS or RAW image file to inspect.",
    )
    parser.add_argument(
        "--expect",
        choices=("fits", "raw"),
        default=None,
        help="Optional expected format. The test fails if it does not match.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.image_path

    print("=" * 72)
    print("RTS Framework image_io smoke test")
    print("=" * 72)
    print(f"image_io.py version : {getattr(image_io, '__version__', '(not defined)')}")
    print()

    print("[1/2] Source normalization")
    normalized = image_io._normalize_source(source)
    print(f"   Input type : {type(source).__name__}")
    print(f"   Path       : {normalized}")
    print("   Result     : PASS")
    print()

    print("[2/2] Format detection")
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
    print("=" * 72)
    print("FINISHED: image_io smoke test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
