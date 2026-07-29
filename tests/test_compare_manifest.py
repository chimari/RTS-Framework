"""Tests for tools/compare_manifest.py."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.compare_manifest import (
    ComparisonError,
    compare_manifests,
    load_manifest,
)


COLUMNS = (
    "dataset",
    "frame_index",
    "filepath",
    "temperature_C",
    "exposure_s",
)


def write_manifest(
    path: Path,
    rows: list[dict[str, object]],
    *,
    columns: tuple[str, ...] = COLUMNS,
) -> Path:
    """Write a small normalized-manifest fixture."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


@pytest.fixture
def sample_rows() -> list[dict[str, object]]:
    return [
        {
            "dataset": "cold",
            "frame_index": 1,
            "filepath": "/data/cold_0001.fits",
            "temperature_C": -12.0,
            "exposure_s": 0.0,
        },
        {
            "dataset": "cold",
            "frame_index": 2,
            "filepath": "/data/cold_0002.fits",
            "temperature_C": -12.0,
            "exposure_s": 0.0,
        },
        {
            "dataset": "room",
            "frame_index": 1,
            "filepath": "/data/room_0001.fits",
            "temperature_C": 24.0,
            "exposure_s": 0.0,
        },
    ]


def test_identical_manifests_have_no_differences(
    tmp_path: Path,
    sample_rows: list[dict[str, object]],
) -> None:
    reference_path = write_manifest(
        tmp_path / "reference.csv",
        sample_rows,
    )
    candidate_path = write_manifest(
        tmp_path / "candidate.csv",
        sample_rows,
    )

    reference = load_manifest(reference_path)
    candidate = load_manifest(candidate_path)

    assert compare_manifests(reference, candidate) == []


def test_value_difference_reports_csv_line_and_column(
    tmp_path: Path,
    sample_rows: list[dict[str, object]],
) -> None:
    candidate_rows = [row.copy() for row in sample_rows]
    candidate_rows[1]["temperature_C"] = -11.5

    reference = load_manifest(
        write_manifest(tmp_path / "reference.csv", sample_rows)
    )
    candidate = load_manifest(
        write_manifest(tmp_path / "candidate.csv", candidate_rows)
    )

    differences = compare_manifests(reference, candidate)
    formatted = [difference.format() for difference in differences]

    assert any(
        "[value]" in message
        and "CSV line 3" in message
        and "'temperature_C'" in message
        for message in formatted
    )


def test_row_count_difference_is_reported(
    tmp_path: Path,
    sample_rows: list[dict[str, object]],
) -> None:
    reference = load_manifest(
        write_manifest(tmp_path / "reference.csv", sample_rows)
    )
    candidate = load_manifest(
        write_manifest(tmp_path / "candidate.csv", sample_rows[:-1])
    )

    categories = {
        difference.category
        for difference in compare_manifests(reference, candidate)
    }

    assert "row-count" in categories


def test_column_order_difference_is_reported(
    tmp_path: Path,
    sample_rows: list[dict[str, object]],
) -> None:
    candidate_columns = (
        "frame_index",
        "dataset",
        "filepath",
        "temperature_C",
        "exposure_s",
    )

    reference = load_manifest(
        write_manifest(tmp_path / "reference.csv", sample_rows)
    )
    candidate = load_manifest(
        write_manifest(
            tmp_path / "candidate.csv",
            sample_rows,
            columns=candidate_columns,
        )
    )

    categories = {
        difference.category
        for difference in compare_manifests(reference, candidate)
    }

    assert "columns" in categories


def test_dataset_first_appearance_order_difference_is_reported(
    tmp_path: Path,
    sample_rows: list[dict[str, object]],
) -> None:
    candidate_rows = [
        sample_rows[2],
        sample_rows[0],
        sample_rows[1],
    ]

    reference = load_manifest(
        write_manifest(tmp_path / "reference.csv", sample_rows)
    )
    candidate = load_manifest(
        write_manifest(tmp_path / "candidate.csv", candidate_rows)
    )

    categories = {
        difference.category
        for difference in compare_manifests(reference, candidate)
    }

    assert "dataset-order" in categories


def test_duplicate_column_names_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-columns.csv"
    path.write_text(
        "dataset,dataset,filepath\n"
        "cold,cold,/data/frame.fits\n",
        encoding="utf-8",
    )

    with pytest.raises(ComparisonError, match="duplicate column"):
        load_manifest(path)


def test_cli_returns_zero_for_identical_manifests(
    tmp_path: Path,
    sample_rows: list[dict[str, object]],
) -> None:
    reference_path = write_manifest(
        tmp_path / "reference.csv",
        sample_rows,
    )
    candidate_path = write_manifest(
        tmp_path / "candidate.csv",
        sample_rows,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "tools/compare_manifest.py",
            str(reference_path),
            str(candidate_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Status             : PASS" in completed.stdout
    assert "Differences        : 0" in completed.stdout


def test_cli_returns_one_for_different_manifests(
    tmp_path: Path,
    sample_rows: list[dict[str, object]],
) -> None:
    candidate_rows = [row.copy() for row in sample_rows]
    candidate_rows[0]["dataset"] = "changed"

    reference_path = write_manifest(
        tmp_path / "reference.csv",
        sample_rows,
    )
    candidate_path = write_manifest(
        tmp_path / "candidate.csv",
        candidate_rows,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "tools/compare_manifest.py",
            str(reference_path),
            str(candidate_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "Status             : FAIL" in completed.stdout
    assert "[value]" in completed.stdout
