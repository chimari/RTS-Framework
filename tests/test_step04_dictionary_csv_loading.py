"""Integration test for Step 04 dictionary CSV loading v4.24.0."""

from __future__ import annotations

import csv
import sys
import tempfile
from dataclasses import FrozenInstanceError
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
    values = [
        [0, 0, 10, 10, 0, 0, 10, 10],
        [20, 21, 20, 21, 20, 21, 20, 21],
        [30, 30, 45, 45, 30, 30, 45, 45],
        [0, 10, 0, 10, 0, 10, 0, 10],
        [5, 5, 15, 15, 5, 5, 15, 15],
        [50, 51, 50, 51, 50, 51, 50, 51],
    ]
    paths = []
    for frame_index in range(8):
        flat = [series[frame_index] for series in values]
        data = np.array([flat[:3], flat[3:]], dtype=np.uint16)
        path = root / f"bias_{frame_index:04d}.fit"
        fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
        paths.append(path)

    rows = []
    for frame_index, path in enumerate(paths):
        rows.append({
            "dataset": "bias",
            "directory": str(root),
            "environment": "step04-v4.24-test",
            "frame_index": frame_index,
            "n_frames": len(paths),
            "temperature_C": -10.0,
            "temperature_start_C": -10.0,
            "temperature_end_C": -10.0,
            "temperature_fraction": frame_index / (len(paths) - 1),
            "exposure_s": 0.0,
            "filename": path.name,
            "filepath": str(path),
            "image_width": 3,
            "image_height": 2,
            "pixel_dtype": "uint16",
            "byte_order": "not-applicable",
        })

    manifest = root / "manifest.normalized.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def kwargs() -> dict[str, object]:
    return {
        "minimum_score": 0.9,
        "minimum_state_count": 2,
        "minimum_separation": 5.0,
        "minimum_transition_count": 3,
        "minimum_lower_run": 2,
        "minimum_upper_run": 2,
    }


def expect_error(path: Path, expected_text: str) -> None:
    try:
        step04.load_rts_dictionary_csv(path)
    except step04.Step04Error as exc:
        require(expected_text in str(exc), f"unexpected error: {exc}")
    else:
        require(False, f"expected failure containing {expected_text!r}")


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fields: list[str],
               rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    print("=" * 72)
    print("RTS Framework Step 04 dictionary CSV loading test")
    print("=" * 72)
    print(f"step04 version : {step04.__version__}")
    print()

    with tempfile.TemporaryDirectory(prefix="rts_step04_load_csv_") as temp:
        root = Path(temp)
        plan = step04.prepare_rts_dictionary_analysis(
            step03.prepare_bias_analysis(write_dataset(root), "bias")
        )
        csv_path = root / "dictionary.csv"
        step04.build_rts_dictionary_csv(plan, csv_path, **kwargs())

        print("[1/4] Written CSV loads into immutable structured rows")
        loaded = step04.load_rts_dictionary_csv(csv_path)
        require(isinstance(loaded, step04.RTSDictionaryCSV),
                "wrong file type")
        require(loaded.path == csv_path, "wrong path")
        require(loaded.candidate_count == len(loaded.rows),
                "wrong candidate count")
        require(loaded.candidate_count > 0, "test produced no candidates")
        require(loaded.datasets == ("bias",), "wrong datasets")
        require(all(isinstance(row, step04.RTSDictionaryRow)
                    for row in loaded.rows), "wrong row type")
        try:
            loaded.rows[0].dataset = "changed"
        except (FrozenInstanceError, AttributeError):
            pass
        else:
            require(False, "row object is mutable")
        print("   File type  : RTSDictionaryCSV")
        print("   Row type   : RTSDictionaryRow")
        print("   Dataclass  : frozen")
        print("   Result     : PASS")
        print()

        print("[2/4] Values and canonical summaries are preserved")
        fields, raw_rows = read_rows(csv_path)
        first = loaded.rows[0]
        require(tuple(first.summary()) == step04.RTS_DICTIONARY_COLUMNS,
                "summary column order changed")
        require(first.dataset == raw_rows[0]["dataset"],
                "dataset changed")
        require(first.row == int(raw_rows[0]["row"]), "row changed")
        require(first.is_candidate is True, "candidate flag changed")
        require(loaded.summary()["candidate_count"] == len(raw_rows),
                "summary count changed")
        require(fields == list(step04.RTS_DICTIONARY_COLUMNS),
                "source columns changed")
        print("   Column order : canonical")
        print("   Numeric types: normalized")
        print("   Candidate    : preserved")
        print("   Result       : PASS")
        print()

        print("[3/4] Header, type, and scientific consistency errors reject")
        bad_header = root / "bad-header.csv"
        write_rows(bad_header, fields[:-1], [
            {key: value for key, value in raw_rows[0].items()
             if key != fields[-1]}
        ])
        expect_error(bad_header, "header does not match")

        bad_type = root / "bad-type.csv"
        rows = [dict(raw_rows[0])]
        rows[0]["row"] = "1.5"
        write_rows(bad_type, fields, rows)
        expect_error(bad_type, "field row must be an integer")

        bad_consistency = root / "bad-consistency.csv"
        rows = [dict(raw_rows[0])]
        rows[0]["transition_count"] = str(
            int(rows[0]["transition_count"]) + 1
        )
        write_rows(bad_consistency, fields, rows)
        expect_error(bad_consistency, "transition counts are inconsistent")
        print("   Wrong header : rejected")
        print("   Wrong type   : rejected")
        print("   Inconsistency: rejected")
        print("   Result       : PASS")
        print()

        print("[4/4] Duplicate coordinates, invalid flags, and missing files reject")
        duplicate = root / "duplicate.csv"
        write_rows(duplicate, fields, [raw_rows[0], raw_rows[0]])
        expect_error(duplicate, "duplicates a dataset coordinate")

        false_candidate = root / "false-candidate.csv"
        rows = [dict(raw_rows[0])]
        rows[0]["is_candidate"] = "False"
        write_rows(false_candidate, fields, rows)
        expect_error(false_candidate, "not a final RTS candidate")

        expect_error(root / "missing.csv", "does not exist")
        print("   Duplicate coordinate : rejected")
        print("   False candidate      : rejected")
        print("   Missing file         : rejected")
        print("   Result               : PASS")
        print()

    print("=" * 72)
    print("FINISHED: Step 04 dictionary CSV loading test passed")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
