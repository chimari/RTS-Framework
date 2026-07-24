"""Integration test for Step 03 stack-memory estimation v3.4.0."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

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


def write_manifest(path: Path, image_paths: list[Path]) -> None:
    rows: list[dict[str, object]] = []
    n_frames = len(image_paths)
    for frame_index, image in enumerate(image_paths):
        rows.append(
            {
                "dataset": "bias",
                "directory": str(image.parent),
                "environment": "step03-v3.4-test",
                "frame_index": frame_index,
                "n_frames": n_frames,
                "temperature_C": -10.0,
                "temperature_start_C": -10.0,
                "temperature_end_C": -10.0,
                "temperature_fraction": frame_index / (n_frames - 1),
                "exposure_s": 0.0,
                "filename": image.name,
                "filepath": str(image),
                "image_width": 5,
                "image_height": 4,
                "pixel_dtype": "uint16",
                "byte_order": "not-applicable",
            }
        )

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
    print("RTS Framework Step 03 stack-memory estimation integration test")
    print("=" * 72)
    print(f"step03 version : {step03.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step03_memory_") as temp_dir:
        root = Path(temp_dir)
        image_paths = [root / f"bias_{index:04d}.fit" for index in range(3)]
        for index, path in enumerate(image_paths):
            data = np.full((4, 5), index, dtype=np.uint16)
            fits.PrimaryHDU(data=data).writeto(path, overwrite=True)

        manifest = root / "manifest.normalized.csv"
        write_manifest(manifest, image_paths)
        plan = step03.prepare_bias_analysis(manifest, "bias")

        print("[1/4] Default float64 estimate has the exact byte count")
        estimate = step03.estimate_image_stack_memory(plan)
        expected_pixels = 3 * 4 * 5
        expected_bytes = expected_pixels * 8
        require(estimate.n_frames == 3, "wrong frame count")
        require(estimate.image_shape == (4, 5), "wrong image shape")
        require(estimate.dtype == "float64", "wrong dtype name")
        require(estimate.bytes_per_pixel == 8, "wrong item size")
        require(estimate.pixel_count == expected_pixels, "wrong pixel count")
        require(estimate.total_bytes == expected_bytes, "wrong total bytes")
        print(f"   Pixels      : {estimate.pixel_count}")
        print(f"   Bytes/pixel : {estimate.bytes_per_pixel}")
        print(f"   Total bytes : {estimate.total_bytes}")
        print("   Result      : PASS")
        print()

        print("[2/4] Binary units and alternate numeric dtypes are correct")
        float32_estimate = step03.estimate_image_stack_memory(
            plan,
            dtype=np.float32,
        )
        require(float32_estimate.dtype == "float32", "wrong float32 name")
        require(float32_estimate.bytes_per_pixel == 4, "wrong float32 size")
        require(
            float32_estimate.total_bytes == expected_pixels * 4,
            "wrong float32 bytes",
        )
        require(
            estimate.kibibytes == estimate.total_bytes / 1024.0,
            "wrong KiB value",
        )
        require(
            estimate.mebibytes == estimate.total_bytes / (1024.0 ** 2),
            "wrong MiB value",
        )
        require(
            estimate.gibibytes == estimate.total_bytes / (1024.0 ** 3),
            "wrong GiB value",
        )
        print("   float64 : 8 bytes/pixel")
        print("   float32 : 4 bytes/pixel")
        print("   Units   : KiB, MiB, GiB")
        print("   Result  : PASS")
        print()

        print("[3/4] Summary is canonical and JSON serializable")
        expected_summary = {
            "n_frames": 3,
            "image_shape": [4, 5],
            "dtype": "float64",
            "bytes_per_pixel": 8,
            "pixel_count": 60,
            "total_bytes": 480,
            "kibibytes": 480 / 1024.0,
            "mebibytes": 480 / (1024.0 ** 2),
            "gibibytes": 480 / (1024.0 ** 3),
        }
        summary = estimate.summary()
        require(summary == expected_summary, "summary content changed")
        encoded = json.dumps(
            summary,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        require(
            encoded.startswith('{"n_frames":3,"image_shape":[4,5],'),
            "summary JSON order is not canonical",
        )
        print("   JSON serializable : YES")
        print("   image_shape       : list")
        print("   Deterministic     : YES")
        print("   Result            : PASS")
        print()

        print("[4/4] Estimation allocates no stack and validates inputs")
        with patch.object(
            step03.np,
            "empty",
            side_effect=AssertionError("np.empty must not be called"),
        ), patch.object(
            step03,
            "iter_bias_frames",
            side_effect=AssertionError("frames must not be read"),
        ):
            repeated = step03.estimate_image_stack_memory(plan)
        require(repeated == estimate, "repeated estimate changed")

        expect_step03_error(
            lambda: step03.estimate_image_stack_memory(object()),
            "plan must be a BiasAnalysisPlan",
        )
        expect_step03_error(
            lambda: step03.estimate_image_stack_memory(plan, dtype=object),
            "must not contain objects",
        )
        expect_step03_error(
            lambda: step03.estimate_image_stack_memory(plan, dtype="U10"),
            "must be numeric",
        )
        structured = np.dtype([("value", np.float32)])
        expect_step03_error(
            lambda: step03.estimate_image_stack_memory(
                plan,
                dtype=structured,
            ),
            "must not be structured",
        )
        print("   Stack allocation : NO")
        print("   Frame reads      : NO")
        print("   Invalid dtypes   : rejected")
        print("   Result           : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 03 stack-memory estimation integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
