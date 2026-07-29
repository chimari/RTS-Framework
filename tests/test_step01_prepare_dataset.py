"""Integration test for Step 01 using one real FITS image."""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.image_io import get_image_metadata
from steps import step01_prepare_dataset as step01


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Step 01 multi-dataset integration tests."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full image validation instead of shape-only validation.",
    )

    parser.add_argument(
        "--width",
        type=int,
        help="RAW image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="RAW image height in pixels.",
    )
    parser.add_argument(
        "--dtype",
        help="RAW pixel dtype, for example uint16.",
    )
    parser.add_argument(
        "--byte-order",
        choices=("little", "big", "native", "not-applicable"),
        help="RAW byte order.",
    )

    return parser


def make_row(
    image: Path,
    *,
    dataset: str,
    environment: str,
    frame_index: int,
    n_frames: int,
    filepath: Path | None = None,
    raw_layout: dict[str, object] | None = None,
) -> dict[str, object]:
    selected_path = image if filepath is None else filepath

    suffix = image.suffix.lower()

    if suffix in {".fit", ".fits", ".fts"}:
        metadata = get_image_metadata(image)

        image_width = metadata.width
        image_height = metadata.height
        pixel_dtype = metadata.pixel_dtype
        byte_order = metadata.byte_order

    else:
        if raw_layout is None:
            raise RuntimeError(
                "RAW test requires --width, --height, "
                "--dtype, and --byte-order."
            )

        image_width = raw_layout["image_width"]
        image_height = raw_layout["image_height"]
        pixel_dtype = raw_layout["pixel_dtype"]
        byte_order = raw_layout["byte_order"]

    return {
        "dataset": dataset,
        "directory": str(selected_path.parent),
        "environment": environment,
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": -12.1,
        "temperature_start_C": -12.1,
        "temperature_end_C": -12.1,
        "temperature_fraction": (
            0.0 if n_frames <= 1 else frame_index / (n_frames - 1)
        ),
        "exposure_s": 0.0,
        "filename": selected_path.name,
        "filepath": str(selected_path),
        "image_width": image_width,
        "image_height": image_height,
        "pixel_dtype": pixel_dtype,
        "byte_order": byte_order,
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def valid_rows(
    images: list[Path],
    raw_layout: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    return [
        make_row(
            images[0],
            dataset="bias-cold",
            environment="cold",
            frame_index=0,
            n_frames=2,
            raw_layout=raw_layout,
        ),
        make_row(
            images[1],
            dataset="bias-cold",
            environment="cold",
            frame_index=1,
            n_frames=2,
            raw_layout=raw_layout,
        ),
        make_row(
            images[2],
            dataset="bias-room",
            environment="room",
            frame_index=0,
            n_frames=1,
            raw_layout=raw_layout,
        ),
    ]

def invalid_rows(
    images: list[Path],
    raw_layout: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    missing = images[4].with_name(
        f"missing_frame{images[4].suffix}"
    )

    bad_shape = make_row(
        images[3],
        dataset="broken-shape",
        environment="test",
        frame_index=0,
        n_frames=1,
        raw_layout=raw_layout,
    )

    bad_shape["image_width"] = int(bad_shape["image_width"]) + 1

    return valid_rows(images, raw_layout) + [
        bad_shape,
        make_row(
            images[4],
            dataset="broken-missing",
            environment="test",
            frame_index=0,
            n_frames=1,
            filepath=missing,
            raw_layout=raw_layout,
        ),
    ]

def run_test(
    image: Path,
    *,
    full: bool,
    raw_layout: dict[str, object] | None = None,
) -> int:
    if not image.is_file():
        print(f"FAIL: image does not exist: {image}")
        return 1

    mode = "full" if full else "shape"

    print("=" * 72)
    print("RTS Framework Step 01 multi-dataset integration test")
    print("=" * 72)
    print(f"step01 version : {step01.__version__}")
    print(f"image          : {image}")
    print(f"mode           : {mode}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step01_") as temp_dir:
        temp_root = Path(temp_dir)
        source_image = image.resolve()
        
        test_images = []
        for index in range(5):
            linked_image = temp_root / f"frame_{index:04d}{image.suffix}"
            linked_image.symlink_to(source_image)
            test_images.append(linked_image)
        
        print("[1/2] Valid multi-dataset manifest")
        valid_manifest = temp_root / "valid_manifest.csv"
        write_manifest(
            valid_manifest,
            valid_rows(test_images, raw_layout),
        )

        valid_progress: list[tuple[int, int, str]] = []

        def record_valid_progress(current, total, frame):
            valid_progress.append((current, total, frame.dataset))
            print(
                f"Checking image {current}/{total}: "
                f"{frame.dataset} / {frame.filename}"
            )

        valid_result = step01.prepare_dataset(
            valid_manifest,
            validation_mode=mode,
            progress=record_valid_progress,
        )

        print()
        print(valid_result.summary())
        print()

        if not valid_result.valid:
            print("Result: FAIL (valid manifest was rejected)")
            return 1
        if valid_result.n_checked_images != 3:
            print(
                "Result: FAIL "
                f"(expected 3 checked images, "
                f"got {valid_result.n_checked_images})"
            )
            return 1
        if valid_result.manifest.n_datasets != 2:
            print(
                "Result: FAIL "
                f"(expected 2 datasets, "
                f"got {valid_result.manifest.n_datasets})"
            )
            return 1
        if len(valid_progress) != 3:
            print(
                "Result: FAIL "
                f"(expected 3 progress calls, got {len(valid_progress)})"
            )
            return 1

        print("   Frames checked : 3 / 3")
        print("   Dataset count  : 2 / 2")
        print("   Progress calls : 3 / 3")
        print("   Result         : PASS")
        print()

        print("[2/2] Invalid manifest with multiple image errors")
        invalid_manifest = temp_root / "invalid_manifest.csv"
        write_manifest(
            invalid_manifest,
            invalid_rows(test_images, raw_layout),
        )

        invalid_progress: list[tuple[int, int, str]] = []

        def record_invalid_progress(current, total, frame):
            invalid_progress.append((current, total, frame.dataset))
            print(
                f"Checking image {current}/{total}: "
                f"{frame.dataset} / {frame.filename}"
            )

        invalid_result = step01.prepare_dataset(
            invalid_manifest,
            validation_mode=mode,
            progress=record_invalid_progress,
        )

        print()
        print(invalid_result.summary())
        print()

        if invalid_result.valid:
            print("Result: FAIL (invalid manifest unexpectedly passed)")
            return 1
        if invalid_result.n_checked_images != 5:
            print(
                "Result: FAIL "
                f"(expected all 5 images checked, "
                f"got {invalid_result.n_checked_images})"
            )
            return 1
        if len(invalid_progress) != 5:
            print(
                "Result: FAIL "
                f"(expected 5 progress calls, got {len(invalid_progress)})"
            )
            return 1
        if len(invalid_result.image_issues) != 2:
            print(
                "Result: FAIL "
                f"(expected 2 image issues, "
                f"got {len(invalid_result.image_issues)})"
            )
            return 1

        details = [issue.detail for issue in invalid_result.image_issues]
        has_shape_error = any(
            (
                "does not match manifest metadata" in detail
                or (
                    "RAW file size" in detail
                    and "does not match" in detail
                )
            )
            for detail in details
        )
        has_missing_error = any(
            "Image file does not exist" in detail for detail in details
        )

        if not has_shape_error or not has_missing_error:
            print("Result: FAIL (expected error details were not found)")
            for detail in details:
                print(f"   Detail: {detail}")
            return 1

        print("   Frames checked : 5 / 5")
        print("   Progress calls : 5 / 5")
        print("   Image issues   : 2 / 2")
        print(f"   Shape mismatch : {'YES' if has_shape_error else 'NO'}")
        print(f"   Missing file   : {'YES' if has_missing_error else 'NO'}")
        print("   Continued scan : YES")
        print("   Result         : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 01 multi-dataset integration test passed")
    print("=" * 72)
    return 0


def main() -> int:
    args = build_parser().parse_args()

    fits_suffixes = {".fit", ".fits", ".fts"}
    is_fits = args.image.suffix.lower() in fits_suffixes

    raw_layout: dict[str, object] | None = None

    if not is_fits:
        raw_arguments = {
            "--width": args.width,
            "--height": args.height,
            "--dtype": args.dtype,
            "--byte-order": args.byte_order,
        }

        missing_arguments = [
            name
            for name, value in raw_arguments.items()
            if value is None
        ]

        if missing_arguments:
            print(
                "FAIL: Headerless RAW input requires: "
                + ", ".join(missing_arguments)
            )
            return 2

        if args.width <= 0 or args.height <= 0:
            print("FAIL: --width and --height must be positive integers.")
            return 2

        raw_layout = {
            "image_width": args.width,
            "image_height": args.height,
            "pixel_dtype": args.dtype,
            "byte_order": args.byte_order,
        }

    return run_test(
        args.image,
        full=args.full,
        raw_layout=raw_layout,
    )

if __name__ == "__main__":
    raise SystemExit(main())
