"""Integration test for Step 04 RTS dictionary planning v4.0.0."""

from __future__ import annotations

import csv
import sys
import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steps import step03_prepare_bias_analysis as step03
from steps import step04_prepare_rts_dictionary_analysis as step04


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def write_dataset(root: Path, n_frames: int) -> Path:
    image_paths = [root / f"bias_{index:04d}.fit" for index in range(n_frames)]
    for index, path in enumerate(image_paths):
        data = np.full((4, 5), index, dtype=np.uint16)
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)

    rows: list[dict[str, object]] = []
    for frame_index, image in enumerate(image_paths):
        fraction = 0.0 if n_frames == 1 else frame_index / (n_frames - 1)
        rows.append(
            {
                "dataset": "bias",
                "directory": str(root),
                "environment": "step04-v4.0-test",
                "frame_index": frame_index,
                "n_frames": n_frames,
                "temperature_C": -10.0 + frame_index * 0.1,
                "temperature_start_C": -10.0,
                "temperature_end_C": -10.0 + (n_frames - 1) * 0.1,
                "temperature_fraction": fraction,
                "exposure_s": 0.0,
                "filename": image.name,
                "filepath": str(image),
                "image_width": 5,
                "image_height": 4,
                "pixel_dtype": "uint16",
                "byte_order": "not-applicable",
            }
        )

    manifest = root / "manifest.normalized.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest


def expect_step04_error(callable_, contains: str) -> None:
    try:
        callable_()
    except step04.Step04Error as exc:
        require(contains in str(exc), f"wrong error message: {exc}")
    else:
        require(False, "Step04Error was not raised")


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 RTS dictionary planning integration test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_plan_") as temp_dir:
        root = Path(temp_dir)
        valid_root = root / "valid"
        valid_root.mkdir()
        valid_manifest = write_dataset(valid_root, 4)
        bias_plan = step03.prepare_bias_analysis(valid_manifest, "bias")

        print("[1/4] A valid BiasAnalysisPlan produces the canonical RTS plan")
        plan = step04.prepare_rts_dictionary_analysis(bias_plan)
        require(plan.bias_plan is bias_plan, "source bias plan was not retained")
        require(plan.dataset == "bias", "wrong dataset")
        require(plan.n_frames == 4, "wrong frame count")
        require(plan.image_shape == (4, 5), "wrong image shape")
        require(plan.minimum_frames == 3, "wrong default minimum")
        print("   Dataset        : bias")
        print("   Frames         : 4")
        print("   Minimum frames : 3")
        print("   Shape          : 4x5")
        print("   Result         : PASS")
        print()

        print("[2/4] The plan and summary are immutable and deterministic")
        expected_summary = "\n".join(
            [
                "RTS Framework Step 04",
                "=====================",
                "Status         : READY",
                "Dataset        : bias",
                "Frames         : 4",
                "Minimum frames : 3",
                "Shape          : 4x5",
            ]
        )
        require(plan.summary() == expected_summary, "summary content changed")
        require(plan.summary() == plan.summary(), "summary is not deterministic")
        try:
            plan.n_frames = 99
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "frozen plan accepted modification")
        print("   Frozen        : YES")
        print("   Deterministic : YES")
        print("   Result        : PASS")
        print()

        print("[3/4] Preparation is metadata-only and performs no image reads")
        with patch.object(
            step03,
            "iter_bias_frames",
            side_effect=AssertionError("image frames must not be read"),
        ), patch.object(
            step03.np,
            "empty",
            side_effect=AssertionError("image arrays must not be allocated"),
        ):
            repeated = step04.prepare_rts_dictionary_analysis(
                bias_plan,
                min_frames=4,
            )
        require(repeated.minimum_frames == 4, "custom minimum was not retained")
        require(repeated.bias_plan is bias_plan, "bias plan identity changed")
        print("   FITS reads       : NO")
        print("   Image allocation : NO")
        print("   Custom minimum   : 4")
        print("   Result           : PASS")
        print()

        print("[4/4] Invalid inputs and insufficient frame counts are rejected")
        short_root = root / "short"
        short_root.mkdir()
        short_manifest = write_dataset(short_root, 2)
        short_bias_plan = step03.prepare_bias_analysis(
            short_manifest,
            "bias",
            min_frames=2,
        )

        expect_step04_error(
            lambda: step04.prepare_rts_dictionary_analysis(object()),
            "bias_plan must be a BiasAnalysisPlan",
        )
        for invalid in (True, 2, 3.0, "3"):
            expect_step04_error(
                lambda invalid=invalid: step04.prepare_rts_dictionary_analysis(
                    bias_plan,
                    min_frames=invalid,
                ),
                "min_frames must be an integer of at least 3",
            )
        expect_step04_error(
            lambda: step04.prepare_rts_dictionary_analysis(short_bias_plan),
            "requires at least 3",
        )
        expect_step04_error(
            lambda: step04.prepare_rts_dictionary_analysis(
                bias_plan,
                min_frames=5,
            ),
            "requires at least 5",
        )
        print("   Invalid plan     : rejected")
        print("   Invalid minimum  : rejected")
        print("   Too few frames   : rejected")
        print("   Result           : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 RTS dictionary planning integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
