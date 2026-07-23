"""Smoke test for the Step 01 JSON report writer."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steps import step01_prepare_dataset as step01


WIDTH = 9576
HEIGHT = 6388
DTYPE = "uint16"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test Step 01 machine-readable JSON reports."
    )
    parser.add_argument("image", type=Path)
    return parser


def make_row(
    image: Path,
    *,
    dataset: str,
    image_width: int = WIDTH,
) -> dict[str, object]:
    return {
        "dataset": dataset,
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
        "image_width": image_width,
        "image_height": HEIGHT,
        "pixel_dtype": DTYPE,
        "byte_order": "not-applicable",
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def run_test(image: Path) -> int:
    if not image.is_file():
        print(f"FAIL: image does not exist: {image}")
        return 1

    print("=" * 72)
    print("RTS Framework Step 01 JSON report test")
    print("=" * 72)
    print(f"step01 version : {step01.__version__}")
    print(f"image          : {image}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step01_report_") as temp_dir:
        root = Path(temp_dir)
        source = image.resolve()

        valid_image = root / f"valid{image.suffix}"
        invalid_image = root / f"invalid{image.suffix}"
        valid_image.symlink_to(source)
        invalid_image.symlink_to(source)

        print("[1/3] Passing report")
        valid_manifest = root / "valid.csv"
        write_manifest(
            valid_manifest,
            [make_row(valid_image, dataset="bias-valid")],
        )
        valid_result = step01.prepare_dataset(valid_manifest)
        valid_report = step01.write_report(
            valid_result,
            root / "reports" / "valid.json",
        )
        valid_payload = load_json(valid_report)

        checks = [
            valid_report.is_file(),
            valid_payload["schema_version"] == "1.0",
            valid_payload["status"] == "passed",
            valid_payload["validation_mode"] == "shape",
            valid_payload["manifest"]["frame_count"] == 1,
            valid_payload["manifest"]["dataset_count"] == 1,
            valid_payload["counts"]["frames_checked"] == 1,
            valid_payload["counts"]["manifest_errors"] == 0,
            valid_payload["counts"]["image_errors"] == 0,
            valid_payload["manifest_issues"] == [],
            valid_payload["image_issues"] == [],
        ]
        if not all(checks):
            print("   Result : FAIL")
            print(json.dumps(valid_payload, indent=2, ensure_ascii=False))
            return 1

        print(f"   Report : {valid_report}")
        print("   Status : passed")
        print("   Result : PASS")
        print()

        print("[2/3] Failing report with image issue")
        invalid_manifest = root / "invalid.csv"
        write_manifest(
            invalid_manifest,
            [
                make_row(
                    invalid_image,
                    dataset="bias-invalid",
                    image_width=WIDTH + 1,
                )
            ],
        )
        invalid_result = step01.prepare_dataset(invalid_manifest)
        invalid_report = step01.write_report(
            invalid_result,
            root / "reports" / "invalid.json",
        )
        invalid_payload = load_json(invalid_report)

        checks = [
            invalid_payload["status"] == "failed",
            invalid_payload["counts"]["frames_checked"] == 1,
            invalid_payload["counts"]["image_errors"] == 1,
            len(invalid_payload["image_issues"]) == 1,
            invalid_payload["image_issues"][0]["dataset"] == "bias-invalid",
            "Image shape does not match"
            in invalid_payload["image_issues"][0]["detail"],
        ]
        if not all(checks):
            print("   Result : FAIL")
            print(json.dumps(invalid_payload, indent=2, ensure_ascii=False))
            return 1

        print(f"   Report       : {invalid_report}")
        print("   Status       : failed")
        print("   Image issues : 1")
        print("   Result       : PASS")
        print()

        print("[3/3] Deterministic output")
        repeat_report = step01.write_report(
            valid_result,
            root / "reports" / "valid_repeat.json",
        )
        if valid_report.read_bytes() != repeat_report.read_bytes():
            print("   Result : FAIL (identical results produced different JSON)")
            return 1

        print("   Timestamp omitted : YES")
        print("   Byte-for-byte same: YES")
        print("   Result            : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 01 JSON report test passed")
    print("=" * 72)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return run_test(args.image)


if __name__ == "__main__":
    raise SystemExit(main())
