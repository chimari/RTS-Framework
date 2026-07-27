#!/usr/bin/env python3
"""Compare two normalized RTS Framework manifests.

This tool performs an order-sensitive, field-by-field comparison.
It is intended for parity and regression validation between pipeline implementations.

Exit codes
----------
0
    The manifests are identical.
1
    One or more differences were found.
2
    The comparison could not be performed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ManifestTable:
    """One normalized manifest loaded from CSV."""

    path: Path
    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def datasets(self) -> tuple[str, ...]:
        """Return dataset names in first-appearance order."""
        seen: set[str] = set()
        ordered: list[str] = []

        for row in self.rows:
            dataset = row.get("dataset", "")
            if dataset not in seen:
                seen.add(dataset)
                ordered.append(dataset)

        return tuple(ordered)


@dataclass(frozen=True, slots=True)
class Difference:
    """One detected difference."""

    category: str
    detail: str

    def format(self) -> str:
        return f"[{self.category}] {self.detail}"


class ComparisonError(Exception):
    """Raised when a manifest cannot be compared."""


def load_manifest(path: str | Path) -> ManifestTable:
    """Load one normalized manifest without changing row order or values."""
    manifest_path = Path(path)

    try:
        with manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream)

            if reader.fieldnames is None:
                raise ComparisonError(
                    f"CSV has no header: {manifest_path}"
                )

            columns = tuple(reader.fieldnames)

            if len(columns) != len(set(columns)):
                raise ComparisonError(
                    f"CSV contains duplicate column names: {manifest_path}"
                )

            rows = tuple(
                {
                    column: row.get(column, "")
                    for column in columns
                }
                for row in reader
            )

    except ComparisonError:
        raise
    except (OSError, csv.Error, UnicodeError) as exc:
        raise ComparisonError(
            f"Unable to read manifest {manifest_path}: {exc}"
        ) from exc

    return ManifestTable(
        path=manifest_path,
        columns=columns,
        rows=rows,
    )


def compare_manifests(
    reference: ManifestTable,
    candidate: ManifestTable,
) -> list[Difference]:
    """Compare two normalized manifests exactly and in acquisition order."""
    differences: list[Difference] = []

    if reference.columns != candidate.columns:
        differences.append(
            Difference(
                "columns",
                (
                    "column order differs: "
                    f"reference={list(reference.columns)!r}, "
                    f"candidate={list(candidate.columns)!r}"
                ),
            )
        )

    if reference.n_rows != candidate.n_rows:
        differences.append(
            Difference(
                "row-count",
                (
                    f"reference={reference.n_rows}, "
                    f"candidate={candidate.n_rows}"
                ),
            )
        )

    if reference.datasets != candidate.datasets:
        differences.append(
            Difference(
                "dataset-order",
                (
                    f"reference={list(reference.datasets)!r}, "
                    f"candidate={list(candidate.datasets)!r}"
                ),
            )
        )

    common_columns = tuple(
        column
        for column in reference.columns
        if column in candidate.columns
    )

    common_row_count = min(reference.n_rows, candidate.n_rows)

    for row_index in range(common_row_count):
        reference_row = reference.rows[row_index]
        candidate_row = candidate.rows[row_index]

        for column in common_columns:
            reference_value = reference_row.get(column, "")
            candidate_value = candidate_row.get(column, "")

            if reference_value != candidate_value:
                # CSV line 1 is the header, so row_index 0 is line 2.
                csv_line = row_index + 2
                differences.append(
                    Difference(
                        "value",
                        (
                            f"CSV line {csv_line}, column {column!r}: "
                            f"reference={reference_value!r}, "
                            f"candidate={candidate_value!r}"
                        ),
                    )
                )

    return differences


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two normalized manifests exactly, "
            "including acquisition order."
        )
    )
    parser.add_argument(
        "reference",
        type=Path,
        help="Normalized manifest produced by the reference implementation.",
    )
    parser.add_argument(
        "candidate",
        type=Path,
        help="Normalized manifest produced by the candidate implementation.",
    )
    parser.add_argument(
        "--max-differences",
        type=int,
        default=20,
        help=(
            "Maximum number of differences to print. "
            "Use -1 to print all differences. Default: 20."
        ),
    )
    return parser


def print_summary(
    reference: ManifestTable,
    candidate: ManifestTable,
    differences: Sequence[Difference],
    *,
    max_differences: int | None,
) -> None:
    status = "PASS" if not differences else "FAIL"

    print("RTS Framework manifest comparison")
    print("=================================")
    print(f"Status             : {status}")
    print(f"Reference          : {reference.path}")
    print(f"Candidate          : {candidate.path}")
    print(f"Reference rows     : {reference.n_rows}")
    print(f"Candidate rows     : {candidate.n_rows}")
    print(f"Reference datasets : {len(reference.datasets)}")
    print(f"Candidate datasets : {len(candidate.datasets)}")
    print(f"Differences        : {len(differences)}")

    if not differences:
        print()
        print("The normalized manifests are identical.")
        return

    shown = (
        list(differences)
        if max_differences is None
        else list(differences[:max_differences])
    )

    print()
    print("Differences")
    print("-----------")

    for difference in shown:
        print(difference.format())

    omitted = len(differences) - len(shown)
    if omitted > 0:
        print(f"... {omitted} additional difference(s) omitted")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.max_differences < -1:
        parser.error("--max-differences must be -1 or zero or greater")

    try:
        reference = load_manifest(args.reference)
        candidate = load_manifest(args.candidate)
    except ComparisonError as exc:
        print(f"Comparison error: {exc}", file=sys.stderr)
        return 2

    differences = compare_manifests(reference, candidate)

    max_differences = (
        None if args.max_differences == -1
        else args.max_differences
    )

    print_summary(
        reference,
        candidate,
        differences,
        max_differences=max_differences,
    )

    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
