"""Integration test for Step 04 high-level RTS dictionary build v4.14.0."""

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

from steps import step03_prepare_bias_analysis as step03
from steps import step04_prepare_rts_dictionary_analysis as step04


def require(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


def write_dataset(root: Path) -> Path:
    candidate_a = [0, 0, 10, 10, 0, 0, 10, 10]
    stable = [20, 21, 20, 21, 20, 21, 20, 21]
    candidate_b = [30, 30, 45, 45, 30, 30, 45, 45]
    rejected_temporal = [0, 10, 0, 10, 0, 10, 0, 10]

    paths: list[Path] = []
    for index, values in enumerate(
        zip(candidate_a, stable, candidate_b, rejected_temporal, strict=True)
    ):
        path = root / f"bias_{index:04d}.fit"
        a, s, b, t = values
        data = np.array(
            [
                [a, s],
                [b, t],
            ],
            dtype=np.uint16,
        )
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows: list[dict[str, object]] = []
    n_frames = len(paths)
    for frame_index, path in enumerate(paths):
        rows.append(
            {
                "dataset": "bias",
                "directory": str(root),
                "environment": "step04-v4.14-test",
                "frame_index": frame_index,
                "n_frames": n_frames,
                "temperature_C": -10.0,
                "temperature_start_C": -10.0,
                "temperature_end_C": -10.0,
                "temperature_fraction": frame_index / (n_frames - 1),
                "exposure_s": 0.0,
                "filename": path.name,
                "filepath": str(path),
                "image_width": 2,
                "image_height": 2,
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


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def expect_step04_error(callable_, contains: str) -> None:
    try:
        callable_()
    except step04.Step04Error as exc:
        require(contains in str(exc), f"wrong error message: {exc}")
    else:
        require(False, "Step04Error was not raised")


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 high-level dictionary build test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_build_") as temp_dir:
        root = Path(temp_dir)
        manifest = write_dataset(root)
        bias_plan = step03.prepare_bias_analysis(manifest, "bias")
        plan = step04.prepare_rts_dictionary_analysis(bias_plan)

        thresholds = {
            "minimum_score": 0.9,
            "minimum_state_count": 2,
            "minimum_separation": 5.0,
            "minimum_transition_count": 3,
            "minimum_lower_run": 2,
            "minimum_upper_run": 2,
        }

        print("[1/4] Full-image build matches the explicit pipeline")
        high_level = root / "high_level.csv"
        explicit = root / "explicit.csv"

        returned = step04.build_rts_dictionary_csv(
            plan,
            high_level,
            **thresholds,
        )
        analyses = step04.iter_image_rts_analyses(plan, **thresholds)
        candidates = step04.iter_rts_candidates(analyses)
        step04.write_rts_dictionary_csv(explicit, candidates)

        require(returned == high_level, "wrong returned path")
        require(
            high_level.read_bytes() == explicit.read_bytes(),
            "high-level output differs from explicit composition",
        )

        fields, rows = read_rows(high_level)
        require(fields == list(step04.RTS_DICTIONARY_COLUMNS),
                "header order changed")
        require(
            [(row["row"], row["column"]) for row in rows]
            == [("0", "0"), ("1", "0")],
            "wrong candidate coordinates or row-major order",
        )
        print("   Explicit pipeline : identical")
        print("   Candidate rows    : 2")
        print("   Row-major order   : preserved")
        print("   Result            : PASS")
        print()

        print("[2/4] ROI limits the analyzed coordinates")
        roi_output = root / "roi.csv"
        step04.build_rts_dictionary_csv(
            plan,
            roi_output,
            row_start=1,
            row_stop=2,
            column_start=0,
            column_stop=2,
            **thresholds,
        )
        _, roi_rows = read_rows(roi_output)
        require(
            [(row["row"], row["column"]) for row in roi_rows]
            == [("1", "0")],
            "ROI output contains wrong coordinates",
        )

        empty_roi = root / "empty_roi.csv"
        step04.build_rts_dictionary_csv(
            plan,
            empty_roi,
            row_start=1,
            row_stop=1,
            **thresholds,
        )
        _, empty_rows = read_rows(empty_roi)
        require(empty_rows == [], "empty ROI produced candidate rows")
        print("   Selected ROI : respected")
        print("   Empty ROI    : header only")
        print("   Result       : PASS")
        print()

        print("[3/4] Thresholds pass through unchanged")
        permissive = root / "permissive.csv"
        step04.build_rts_dictionary_csv(
            plan,
            permissive,
            minimum_score=0.9,
            minimum_state_count=2,
            minimum_separation=5.0,
            minimum_transition_count=7,
            minimum_lower_run=1,
            minimum_upper_run=1,
        )
        _, permissive_rows = read_rows(permissive)
        require(
            [(row["row"], row["column"]) for row in permissive_rows]
            == [("1", "1")],
            "temporal thresholds were not passed through correctly",
        )
        print("   Two-state thresholds : forwarded")
        print("   Temporal thresholds  : forwarded")
        print("   Result               : PASS")
        print()

        print("[4/4] Validation errors preserve an existing destination")
        protected = root / "protected.csv"
        original = "ORIGINAL\n"
        protected.write_text(original, encoding="utf-8")

        expect_step04_error(
            lambda: step04.build_rts_dictionary_csv(
                plan,
                protected,
                row_start=-1,
                **thresholds,
            ),
            "row_start",
        )
        require(
            protected.read_text(encoding="utf-8") == original,
            "destination changed after ROI validation failure",
        )

        expect_step04_error(
            lambda: step04.build_rts_dictionary_csv(
                object(),
                protected,
                **thresholds,
            ),
            "plan",
        )
        require(
            protected.read_text(encoding="utf-8") == original,
            "destination changed after plan validation failure",
        )
        print("   Invalid ROI  : rejected")
        print("   Invalid plan : rejected")
        print("   Destination  : preserved")
        print("   Result       : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 high-level dictionary build test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
