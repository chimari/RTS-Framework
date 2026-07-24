"""Integration test for Step 03 bias-analysis preparation v3.0.0."""

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
from steps import step03_prepare_bias_analysis as step03


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def write_fits(path: Path, value: int) -> None:
    fits.PrimaryHDU(
        data=np.full((2, 3), value, dtype=np.uint16)
    ).writeto(path, overwrite=True)


def make_row(
    dataset: str,
    image: Path,
    *,
    frame_index: int,
    n_frames: int,
    temperature_C: float,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "directory": str(image.parent),
        "environment": "step03-test",
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": temperature_C,
        "temperature_start_C": -12.0,
        "temperature_end_C": -11.5,
        "temperature_fraction": (
            0.0 if n_frames == 1 else frame_index / (n_frames - 1)
        ),
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
    print("RTS Framework Step 03 bias-analysis preparation integration test")
    print("=" * 72)
    print(f"step03 version : {step03.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step03_prepare_") as temp_dir:
        root = Path(temp_dir)
        paths = [root / f"frame_{index:04d}.fit" for index in range(4)]
        for index, path in enumerate(paths):
            write_fits(path, index)

        manifest = root / "manifest.normalized.csv"
        write_manifest(
            manifest,
            [
                make_row(
                    "room",
                    paths[3],
                    frame_index=0,
                    n_frames=1,
                    temperature_C=20.0,
                ),
                make_row(
                    "bias",
                    paths[2],
                    frame_index=2,
                    n_frames=3,
                    temperature_C=-11.5,
                ),
                make_row(
                    "bias",
                    paths[0],
                    frame_index=0,
                    n_frames=3,
                    temperature_C=-12.0,
                ),
                make_row(
                    "bias",
                    paths[1],
                    frame_index=1,
                    n_frames=3,
                    temperature_C=-11.8,
                ),
            ],
        )

        print("[1/4] Manifest source produces a canonical immutable plan")
        plan = step03.prepare_bias_analysis(manifest, "bias")
        require(plan.dataset == "bias", "wrong dataset")
        require(plan.n_frames == 3, "wrong frame count")
        require(plan.image_shape == (2, 3), "wrong image shape")
        require(plan.pixel_dtype == "uint16", "wrong pixel dtype")
        require(plan.exposure_s == 0.0, "wrong exposure")
        require(plan.temperature_min_C == -12.0, "wrong minimum temperature")
        require(plan.temperature_max_C == -11.5, "wrong maximum temperature")
        require(
            [path.name for path in plan.filepaths]
            == ["frame_0000.fit", "frame_0001.fit", "frame_0002.fit"],
            "frame paths are not in canonical order",
        )
        print("   Dataset : bias")
        print("   Frames  : 3")
        print("   Shape   : 2x3")
        print("   Result  : PASS")
        print()

        print("[2/4] Existing Step02Result is accepted without regrouping")
        result = step02.prepare_frame_groups(manifest)
        second = step03.prepare_bias_analysis(result, "bias", min_frames=3)
        require(second.group is result.get_group("bias"), "group was not reused")
        require(second == plan, "plans differ")
        expect_step03_error(
            lambda: step03.prepare_bias_analysis(
                result,
                "bias",
                frame_root=root,
            ),
            "frame_root cannot be used",
        )
        print("   Step02Result : reused")
        print("   Result       : PASS")
        print()

        print("[3/4] Dataset selection and frame threshold are strict")
        expect_step03_error(
            lambda: step03.prepare_bias_analysis(manifest, "missing"),
            "Available datasets: bias, room",
        )
        expect_step03_error(
            lambda: step03.prepare_bias_analysis(manifest, "room"),
            "at least 2 are required",
        )
        expect_step03_error(
            lambda: step03.prepare_bias_analysis(manifest, " bias"),
            "exact name",
        )
        expect_step03_error(
            lambda: step03.prepare_bias_analysis(
                manifest,
                "bias",
                min_frames=4,
            ),
            "at least 4 are required",
        )
        print("   Missing dataset : rejected")
        print("   One frame       : rejected")
        print("   Inexact name    : rejected")
        print("   Result          : PASS")
        print()

        print("[4/4] Summary is deterministic and image pixels are not retained")
        expected = "\n".join(
            [
                "RTS Framework Step 03",
                "=====================",
                "Status      : READY",
                "Dataset     : bias",
                "Frames      : 3",
                "Shape       : 2x3",
                "Pixel dtype : uint16",
                "Exposure    : 0 s",
                "Temperature : -12..-11.5 C",
            ]
        )
        require(plan.summary() == expected, "summary text is not canonical")
        require(
            not any(
                isinstance(value, np.ndarray)
                for value in (
                    plan.group,
                    plan.dataset,
                    plan.n_frames,
                    plan.image_shape,
                    plan.pixel_dtype,
                    plan.exposure_s,
                )
            ),
            "plan unexpectedly retains image pixels",
        )
        print("   Summary    : canonical")
        print("   Pixel data : not retained")
        print("   Result     : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 03 bias-analysis preparation test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
