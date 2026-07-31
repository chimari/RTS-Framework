#!/usr/bin/env python3
"""
Create a smaller RTS Framework input manifest for testing.

Example
-------
python tools/make_subset_manifest.py \
    input.csv \
    subset.csv \
    --frames 30

python tools/make_subset_manifest.py \
    input.csv \
    subset.csv \
    --frames 30 \
    --datasets bias_-12dec,bias_room
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Create a subset RTS Framework input manifest."
    )
    p.add_argument("input_manifest", type=Path)
    p.add_argument("output_manifest", type=Path)
    p.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Maximum number of frames to keep for each dataset.",
    )
    p.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names to include. "
             "Default: include all datasets.",
    )
    return p.parse_args()


def main():

    args = parse_args()

    selected = None
    if args.datasets:
        selected = {
            x.strip()
            for x in args.datasets.split(",")
            if x.strip()
        }

    groups = defaultdict(list)

    with args.input_manifest.open(newline="") as fp:
        reader = csv.DictReader(fp)

        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError("Missing CSV header.")

        for row in reader:

            dataset = row["dataset"]

            if selected is not None and dataset not in selected:
                continue

            if len(groups[dataset]) < args.frames:
                groups[dataset].append(row)

    with args.output_manifest.open(
        "w",
        newline="",
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for dataset in sorted(groups):

            rows = groups[dataset]
            n = len(rows)

            for i, row in enumerate(rows):

                row = dict(row)
                row["frame_index"] = str(i)
                row["n_frames"] = str(n)

                writer.writerow(row)

    print(
        f"Wrote {args.output_manifest} "
        f"({sum(len(v) for v in groups.values())} frames, "
        f"{len(groups)} datasets)"
    )


if __name__ == "__main__":
    main()
