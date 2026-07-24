"""Integration test for Step 02 dataset grouping."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steps import step02_prepare_frame_groups as step02


WIDTH = 9576
HEIGHT = 6388


def make_row(
    image: Path,
    *,
    dataset: str,
    frame_index: int,
    n_frames: int,
    temperature_C: float,
    environment: str,
    exposure_s: float = 0.0,
    image_width: int = WIDTH,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "directory": str(image.parent),
        "environment": environment,
        "frame_index": frame_index,
        "n_frames": n_frames,
        "temperature_C": temperature_C,
        "temperature_start_C": temperature_C,
        "temperature_end_C": temperature_C,
        "temperature_fraction": (
            0.0 if n_frames == 1 else frame_index / (n_frames - 1)
        ),
        "exposure_s": exposure_s,
        "filename": image.name,
        "filepath": str(image),
        "image_width": image_width,
        "image_height": HEIGHT,
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
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} IMAGE")
        return 2

    source = Path(sys.argv[1]).resolve()
    require(source.is_file(), f"image does not exist: {source}")

    print("=" * 72)
    print("RTS Framework Step 02 frame-group integration test")
    print("=" * 72)
    print(f"step02 version : {step02.__version__}")
    print(f"image          : {source}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step02_") as temp_dir:
        root = Path(temp_dir)
        images = []
        for index in range(5):
            image = root / f"frame_{index:04d}{source.suffix}"
            image.symlink_to(source)
            images.append(image)

        print("[1/4] Group unordered rows by dataset and frame_index")
        valid_manifest = root / "valid.csv"
        write_manifest(
            valid_manifest,
            [
                make_row(
                    images[2],
                    dataset="room",
                    frame_index=0,
                    n_frames=1,
                    temperature_C=20.0,
                    environment="room",
                ),
                make_row(
                    images[1],
                    dataset="cold",
                    frame_index=1,
                    n_frames=2,
                    temperature_C=-11.8,
                    environment="cold",
                ),
                make_row(
                    images[0],
                    dataset="cold",
                    frame_index=0,
                    n_frames=2,
                    temperature_C=-12.2,
                    environment="cold",
                ),
            ],
        )

        result = step02.prepare_frame_groups(valid_manifest)

        require(result.n_datasets == 2, "expected 2 datasets")
        require(result.n_frames == 3, "expected 3 frames")
        require(
            [group.name for group in result.groups] == ["cold", "room"],
            "dataset order is not stable",
        )

        cold = result.get_group("cold")
        room = result.get_group("room")
        require(
            [frame.frame_index for frame in cold.frames] == [0, 1],
            "cold frames are not ordered",
        )
        require(cold.n_frames == 2, "cold frame count is wrong")
        require(cold.image_shape == (HEIGHT, WIDTH), "cold shape is wrong")
        require(cold.pixel_dtype == "uint16", "cold dtype is wrong")
        require(cold.exposure_s == 0.0, "cold exposure is wrong")
        require(cold.temperature_min_C == -12.2, "cold minimum is wrong")
        require(cold.temperature_max_C == -11.8, "cold maximum is wrong")
        require(room.n_frames == 1, "room frame count is wrong")

        print("   Dataset order : cold, room")
        print("   Cold frames   : 0, 1")
        print("   Frame count   : 3")
        print("   Result        : PASS")
        print()

        print("[2/4] Result iteration and exact group lookup")
        require(
            [group.name for group in result] == ["cold", "room"],
            "result iteration is wrong",
        )
        try:
            result.get_group("missing")
        except KeyError:
            missing_raised = True
        else:
            missing_raised = False
        require(missing_raised, "missing group did not raise KeyError")

        print("   Iterable       : YES")
        print("   get_group      : YES")
        print("   Missing lookup : KeyError")
        print("   Result         : PASS")
        print()

        print("[3/4] Inconsistent dataset metadata is rejected")
        inconsistent_manifest = root / "inconsistent.csv"
        write_manifest(
            inconsistent_manifest,
            [
                make_row(
                    images[3],
                    dataset="broken",
                    frame_index=0,
                    n_frames=2,
                    temperature_C=0.0,
                    environment="test",
                ),
                make_row(
                    images[4],
                    dataset="broken",
                    frame_index=1,
                    n_frames=2,
                    temperature_C=0.1,
                    environment="test",
                    image_width=WIDTH + 1,
                ),
            ],
        )

        try:
            step02.prepare_frame_groups(inconsistent_manifest)
        except step02.Step02Error as exc:
            inconsistent_message = str(exc)
        else:
            print("FAIL: inconsistent shape was accepted")
            return 1

        require(
            "inconsistent image_width" in inconsistent_message,
            "unexpected inconsistency error",
        )
        print("   Inconsistent shape : rejected")
        print("   Result             : PASS")
        print()

        print("[4/4] Missing frame_index sequence is rejected")
        missing_index_manifest = root / "missing_index.csv"
        write_manifest(
            missing_index_manifest,
            [
                make_row(
                    images[3],
                    dataset="gap",
                    frame_index=0,
                    n_frames=2,
                    temperature_C=0.0,
                    environment="test",
                ),
            ],
        )

        try:
            step02.prepare_frame_groups(missing_index_manifest)
        except step02.Step02Error as exc:
            gap_message = str(exc)
        else:
            print("FAIL: incomplete dataset was accepted")
            return 1

        require(
            "declares n_frames=2" in gap_message,
            "unexpected missing-frame error",
        )
        print("   Declared frames : 2")
        print("   Actual rows     : 1")
        print("   Result          : PASS")
        print()

        print(result.summary())
        print()

    print("=" * 72)
    print("FINISHED: Step 02 frame-group integration test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
