"""Integration test for Step 02 statistics summary JSON."""

from __future__ import annotations

import csv
import json
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
    frame_index: int,
    n_frames: int,
    temperature_C: float,
    exposure_s: float,
) -> dict[str, object]:
    return {
        "dataset": "bias",
        "directory": str(image.parent),
        "environment": "test",
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": temperature_C,
        "temperature_start_C": -12.0,
        "temperature_end_C": -11.5,
        "temperature_fraction": frame_index / (n_frames - 1),
        "exposure_s": exposure_s,
        "filename": image.name,
        "filepath": str(image),
        "image_width": 3,
        "image_height": 2,
        "pixel_dtype": "uint16",
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
    print("RTS Framework Step 02 statistics-summary JSON integration test")
    print("=" * 72)
    print(f"step02 version : {step02.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step02_summary_") as temp_dir:
        root = Path(temp_dir)

        frame0 = root / "frame_0000.fit"
        frame1 = root / "frame_0001.fit"
        write_fits(
            frame0,
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16),
        )
        write_fits(
            frame1,
            np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint16),
        )

        manifest = root / "manifest.csv"
        write_manifest(
            manifest,
            [
                make_row(
                    frame0,
                    frame_index=0,
                    n_frames=2,
                    temperature_C=-12.0,
                    exposure_s=0.0,
                ),
                make_row(
                    frame1,
                    frame_index=1,
                    n_frames=2,
                    temperature_C=-11.5,
                    exposure_s=0.0,
                ),
            ],
        )

        group = step02.prepare_frame_groups(manifest).get_group("bias")
        statistics = step02.compute_dataset_statistics(group)

        print("[1/4] Canonical summary payload is correct")
        payload = step02.build_statistics_summary(group, statistics)
        require(
            payload["schema"] == "rts-framework.step02.statistics-summary",
            "wrong schema",
        )
        require(payload["schema_version"] == 1, "wrong schema version")
        require(payload["dataset"] == "bias", "wrong dataset")
        require(payload["n_frames"] == 2, "wrong frame count")
        require(payload["image"]["shape"] == [2, 3], "wrong shape")
        require(payload["image"]["pixel_dtype"] == "uint16", "wrong dtype")
        print("   Dataset : bias")
        print("   Shape   : [2, 3]")
        print("   Frames  : 2")
        print("   Result  : PASS")
        print()

        print("[2/4] Ranges are derived correctly")
        require(
            payload["temperature_C"] == {
                "minimum": -12.0,
                "maximum": -11.5,
            },
            "wrong temperature range",
        )
        require(
            payload["exposure_s"] == {
                "minimum": 0.0,
                "maximum": 0.0,
            },
            "wrong exposure range",
        )
        require(
            payload["frame_statistics"]["mean"] == {
                "minimum": 3.5,
                "maximum": 35.0,
            },
            "wrong mean range",
        )
        require(
            payload["frame_statistics"]["median"] == {
                "minimum": 3.5,
                "maximum": 35.0,
            },
            "wrong median range",
        )
        print("   Temperature : -12 .. -11.5 C")
        print("   Exposure    : 0 .. 0 s")
        print("   Mean        : 3.5 .. 35")
        print("   Result      : PASS")
        print()

        print("[3/4] JSON output is canonical and deterministic")
        output = root / "summary" / "bias.json"
        repeated = root / "summary" / "bias.repeat.json"
        step02.write_statistics_summary_json(group, statistics, output)
        step02.write_statistics_summary_json(group, statistics, repeated)

        require(output.is_file(), "summary JSON missing")
        require(
            output.read_bytes() == repeated.read_bytes(),
            "JSON output is not deterministic",
        )
        raw = output.read_bytes()
        require(b"\r\n" not in raw, "JSON does not use LF")
        require(raw.endswith(b"\n"), "JSON lacks final newline")
        loaded = json.loads(output.read_text(encoding="utf-8"))
        require(loaded == payload, "written JSON differs from payload")
        print("   Deterministic : YES")
        print("   Line endings  : LF")
        print("   Final newline : YES")
        print("   Result        : PASS")
        print()

        print("[4/4] Mismatched group and statistics are rejected")
        mismatched = step02.DatasetStatistics(
            dataset="other",
            frames=statistics.frames,
        )
        try:
            step02.build_statistics_summary(group, mismatched)
        except step02.Step02Error as exc:
            require(
                "names do not match" in str(exc),
                "wrong mismatch error",
            )
        else:
            require(False, "mismatch was not rejected")
        print("   Name mismatch : rejected")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 02 statistics-summary JSON integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
