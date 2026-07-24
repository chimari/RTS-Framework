"""Integration test for Step 04 RTS dictionary CSV writing v4.13.0."""

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
    candidate_b = [30, 30, 45, 45, 30, 30, 45, 45]
    stable = [20, 21, 20, 21, 20, 21, 20, 21]

    paths: list[Path] = []
    for index, values in enumerate(zip(candidate_a, candidate_b, stable, strict=True)):
        path = root / f"bias_{index:04d}.fit"
        a, b, c = values
        data = np.array(
            [
                [a, b],
                [c, c],
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
                "environment": "step04-v4.13-test",
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


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
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
    print("RTS Framework Step 04 RTS dictionary CSV writing test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_csv_") as temp_dir:
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

        candidate_a = step04.analyze_rts_pixel(
            plan, row=0, column=0, **thresholds
        )
        candidate_b = step04.analyze_rts_pixel(
            plan, row=0, column=1, **thresholds
        )
        stable = step04.analyze_rts_pixel(
            plan, row=1, column=0, **thresholds
        )

        require(candidate_a.is_candidate, "candidate A fixture was rejected")
        require(candidate_b.is_candidate, "candidate B fixture was rejected")
        require(not stable.is_candidate, "stable fixture was accepted")

        print("[1/4] Normal candidate CSV is deterministic")
        output = root / "output" / "rts_dictionary.csv"
        returned = step04.write_rts_dictionary_csv(
            output,
            [candidate_a, candidate_b],
        )
        require(returned == output, "wrong returned path")
        require(output.exists(), "CSV was not created")

        fieldnames, rows = read_csv(output)
        require(
            fieldnames == list(step04.RTS_DICTIONARY_COLUMNS),
            "CSV header order changed",
        )
        require(len(rows) == 2, "wrong CSV row count")
        require(
            [(row["row"], row["column"]) for row in rows]
            == [("0", "0"), ("0", "1")],
            "CSV coordinate order changed",
        )
        raw = output.read_bytes()
        require(b"\r\n" not in raw, "CSV uses CRLF instead of LF")
        require(raw.endswith(b"\n"), "CSV does not end with LF")
        print("   Encoding     : UTF-8")
        print("   Header       : canonical")
        print("   Line ending  : LF")
        print("   Rows         : 2")
        print("   Result       : PASS")
        print()

        print("[2/4] Empty input writes a header-only CSV")
        empty_output = root / "empty.csv"
        step04.write_rts_dictionary_csv(empty_output, [])
        empty_fields, empty_rows = read_csv(empty_output)
        require(
            empty_fields == list(step04.RTS_DICTIONARY_COLUMNS),
            "empty CSV header order changed",
        )
        require(empty_rows == [], "empty CSV contains data rows")
        require(
            empty_output.read_text(encoding="utf-8").count("\n") == 1,
            "empty CSV is not header-only",
        )
        print("   File exists : yes")
        print("   Header      : present")
        print("   Data rows   : 0")
        print("   Result      : PASS")
        print()

        print("[3/4] Input order and duplicate references are preserved")
        duplicate_output = root / "duplicates.csv"

        def one_shot():
            yield candidate_b
            yield candidate_a
            yield candidate_b

        step04.write_rts_dictionary_csv(duplicate_output, one_shot())
        _, duplicate_rows = read_csv(duplicate_output)
        require(
            [(row["row"], row["column"]) for row in duplicate_rows]
            == [("0", "1"), ("0", "0"), ("0", "1")],
            "order or duplicates changed",
        )
        print("   One-shot source : supported")
        print("   Input order     : preserved")
        print("   Duplicate rows  : preserved")
        print("   Result          : PASS")
        print()

        print("[4/4] Mid-stream errors do not replace the destination")
        protected = root / "protected.csv"
        original = "ORIGINAL\n"
        protected.write_text(original, encoding="utf-8")

        def invalid_stream():
            yield candidate_a
            yield stable
            yield candidate_b

        expect_step04_error(
            lambda: step04.write_rts_dictionary_csv(protected, invalid_stream()),
            "candidates item 1",
        )
        require(
            protected.read_text(encoding="utf-8") == original,
            "destination was modified after a failed write",
        )
        leftovers = list(root.glob(f".{protected.name}.*.tmp"))
        require(leftovers == [], f"temporary files remain: {leftovers}")

        expect_step04_error(
            lambda: step04.write_rts_dictionary_csv(root, [candidate_a]),
            "must not be a directory",
        )
        expect_step04_error(
            lambda: step04.write_rts_dictionary_csv(root / "bad.csv", 123),
            "candidates must be an iterable",
        )
        print("   Existing file : preserved")
        print("   Temporary file: removed")
        print("   Invalid path  : rejected")
        print("   Invalid source: rejected")
        print("   Result        : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 RTS dictionary CSV writing test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
