#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_measure_state_transition_statistics.py

Step08 v8.0.2 of the IMX811 RTS pipeline.

Purpose
-------
Calculate objective state-transition and dwell-time statistics from the
frame-by-frame state assignments produced by Step07.

This step does NOT decide whether a pixel is RTS and does NOT build the RTS
dictionary. Threshold-based RTS selection belongs to Step09.

Inputs
------
From Step07:
  frame_state_uint8.npy
  assign_quality_uint8.npy
  temperature_bin_index.csv
  candidate_catalog.csv
  summary.json

From Step05:
  dataset_index.csv

Main outputs
------------
  transition_count_uint32.npy
      shape = (temperature_bin, candidate, 3, 3)
      Includes self-transitions.

  transition_probability_float32.npy
      Row-normalized transition matrix. NaN when a row has no outgoing pair.

  reliable_transition_count_uint32.npy
      Consecutive-pair transition counts for which both endpoint frames have
      quality excellent or normal. Pairs touching near-valley/invalid samples
      are excluded; gaps are never bridged.

  switch_count_uint32.npy
  reliable_switch_count_uint32.npy
  switching_frequency_float32.npy
  reliable_switching_frequency_float32.npy
  valid_pair_count_uint32.npy
  reliable_pair_count_uint32.npy

  state_visit_count_uint32.npy
  mean_dwell_frame_float32.npy
  median_dwell_frame_float32.npy
  max_dwell_frame_uint32.npy
  min_dwell_frame_uint32.npy

  observed_state_count_uint8.npy
  bidirectional_pair_count_uint8.npy
      Number of state pairs with transitions observed in both directions:
      (1,2), (1,3), and (2,3). Range 0..3.

  direct_1_3_switch_count_uint32.npy
      Number of direct 1<->3 switches.

Optional output
---------------
  run_length_table.csv.gz
      Enabled with --save-run-table. Contains one row per contiguous run,
      normally only for Step06 multistate candidates. This can become large.

Quality handling
----------------
Raw transition and dwell statistics use all valid Step07 assignments.

"Reliable" transition statistics use only consecutive frame pairs for which
both endpoint quality codes are 1 (excellent) or 2 (normal). A pair involving
quality 0 or 3 is excluded. The algorithm never joins samples across an
excluded pair.

Dwell statistics are intentionally not filtered by quality because deleting a
near-boundary sample would split or merge runs and change their physical
meaning. Step09 can combine dwell statistics with reliable transition counts.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from common.cli import add_common_arguments, validate_common_arguments
from common.io import prepare_output_dir, sha256_file, write_json
from common.version import PIPELINE_VERSION

SCRIPT_VERSION = "8.0.2"
MAX_STATES = 3

Q_INVALID = 0
Q_EXCELLENT = 1
Q_NORMAL = 2
Q_NEAR_VALLEY = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure transition, switching, visit, and dwell statistics "
            "without making an RTS decision."
        )
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("07_frame_state_assignment"),
        help="Step07 output directory.",
    )
    parser.add_argument(
        "--centered-dir",
        type=Path,
        default=Path("05_dataset_centered_timeseries"),
        help="Step05 output directory containing dataset_index.csv.",
    )
    parser.add_argument(
        "--candidate-block",
        type=int,
        default=1024,
        help="Number of candidate columns processed per block.",
    )
    parser.add_argument(
        "--save-run-table",
        action="store_true",
        help="Write run_length_table.csv.gz.",
    )
    parser.add_argument(
        "--run-table-all-candidates",
        action="store_true",
        help=(
            "With --save-run-table, include all candidates. By default only "
            "candidate/bin combinations having Step06 state_count >= 2 are "
            "written."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume array calculation from checkpoint. The optional run table "
            "is not resumable and cannot be combined with --resume."
        ),
    )
    parser.add_argument("--hash-inputs", action="store_true")
    parser.add_argument("--flush-every-blocks", type=int, default=5)
    add_common_arguments(parser, output_default="08_state_transition_statistics")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_file(
    directory: Path,
    summary: dict,
    keys: tuple[str, ...],
    fallback: str,
) -> Path:
    outputs = summary.get("outputs", {})
    for key in keys:
        value = outputs.get(key) if isinstance(outputs, dict) else None
        if not value:
            value = summary.get(key)
        if value:
            path = Path(str(value))
            if path.is_file():
                return path
            local = directory / path.name
            if local.is_file():
                return local
    path = directory / fallback
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_indices(value: object) -> list[int]:
    text = str(value).strip()
    if not text:
        return []
    return [int(item) for item in text.split(";") if item.strip()]


def build_bin_frames(
    bin_table: pd.DataFrame,
    dataset_table: pd.DataFrame,
    n_frames: int,
) -> list[np.ndarray]:
    required_bin_columns = {
        "temperature_bin_index",
        "dataset_indices",
    }
    required_dataset_columns = {
        "dataset_index",
        "frame_start",
        "frame_stop_exclusive",
    }
    missing_bins = required_bin_columns - set(bin_table.columns)
    missing_datasets = required_dataset_columns - set(dataset_table.columns)
    if missing_bins:
        raise KeyError(
            f"temperature_bin_index.csv missing columns: {sorted(missing_bins)}"
        )
    if missing_datasets:
        raise KeyError(
            f"dataset_index.csv missing columns: {sorted(missing_datasets)}"
        )

    bins = bin_table.sort_values(
        "temperature_bin_index"
    ).reset_index(drop=True)
    bin_numbers = pd.to_numeric(
        bins["temperature_bin_index"], errors="raise"
    ).to_numpy(np.int64)
    if not np.array_equal(bin_numbers, np.arange(len(bins))):
        raise ValueError("temperature_bin_index must be 0..n_bins-1")

    datasets = dataset_table.copy()
    datasets["dataset_index"] = pd.to_numeric(
        datasets["dataset_index"], errors="raise"
    ).astype(np.int64)
    datasets = datasets.set_index("dataset_index")

    owner = np.full(n_frames, -1, dtype=np.int32)
    result: list[np.ndarray] = []

    for row in bins.itertuples(index=False):
        bin_number = int(row.temperature_bin_index)
        pieces: list[np.ndarray] = []
        for dataset_number in parse_indices(row.dataset_indices):
            if dataset_number not in datasets.index:
                raise KeyError(f"Unknown dataset_index {dataset_number}")
            dataset = datasets.loc[dataset_number]
            start = int(dataset.frame_start)
            stop = int(dataset.frame_stop_exclusive)
            if start < 0 or stop > n_frames or stop <= start:
                raise ValueError(
                    f"Invalid frame range [{start}, {stop}) "
                    f"for dataset {dataset_number}"
                )
            pieces.append(np.arange(start, stop, dtype=np.int64))

        if not pieces:
            raise ValueError(f"Temperature bin {bin_number} contains no frames")

        frames = np.concatenate(pieces)
        if np.any(owner[frames] >= 0):
            raise ValueError("A frame belongs to more than one temperature bin")
        owner[frames] = bin_number
        result.append(frames)

    if np.any(owner < 0):
        missing = int(np.count_nonzero(owner < 0))
        raise ValueError(f"{missing} frames do not belong to a temperature bin")
    return result


def run_boundaries(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return run starts and exclusive stops for a one-dimensional state sequence.
    """
    if states.ndim != 1 or states.size == 0:
        raise ValueError("run_boundaries requires a non-empty 1-D array")
    changes = np.flatnonzero(states[1:] != states[:-1]) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [states.size]))
    return starts.astype(np.int64), stops.astype(np.int64)


def calculate_candidate_statistics(
    states: np.ndarray,
    quality: np.ndarray,
) -> dict[str, np.ndarray | int | float]:
    """
    Calculate statistics for one candidate within one temperature bin.
    """
    if states.ndim != 1 or quality.ndim != 1:
        raise ValueError("states and quality must be 1-D")
    if states.size != quality.size or states.size == 0:
        raise ValueError("states and quality length mismatch")
    if np.any((states < 1) | (states > MAX_STATES)):
        raise ValueError("Invalid state code")
    if np.any((quality < Q_INVALID) | (quality > Q_NEAR_VALLEY)):
        raise ValueError("Invalid quality code")

    n_frames = int(states.size)
    transition = np.zeros((MAX_STATES, MAX_STATES), dtype=np.uint32)
    reliable_transition = np.zeros_like(transition)

    if n_frames >= 2:
        source = states[:-1].astype(np.int64) - 1
        target = states[1:].astype(np.int64) - 1
        flat = source * MAX_STATES + target
        transition[:] = np.bincount(
            flat, minlength=MAX_STATES * MAX_STATES
        ).reshape(MAX_STATES, MAX_STATES)

        reliable_pair_mask = (
            ((quality[:-1] == Q_EXCELLENT) | (quality[:-1] == Q_NORMAL))
            & ((quality[1:] == Q_EXCELLENT) | (quality[1:] == Q_NORMAL))
        )
        if np.any(reliable_pair_mask):
            reliable_flat = flat[reliable_pair_mask]
            reliable_transition[:] = np.bincount(
                reliable_flat, minlength=MAX_STATES * MAX_STATES
            ).reshape(MAX_STATES, MAX_STATES)
        reliable_pair_count = int(np.count_nonzero(reliable_pair_mask))
    else:
        reliable_pair_count = 0

    valid_pair_count = max(0, n_frames - 1)
    switch_count = int(transition.sum() - np.trace(transition))
    reliable_switch_count = int(
        reliable_transition.sum() - np.trace(reliable_transition)
    )

    transition_probability = np.full(
        (MAX_STATES, MAX_STATES), np.nan, dtype=np.float32
    )
    row_totals = transition.sum(axis=1)
    nonzero_rows = row_totals > 0
    if np.any(nonzero_rows):
        transition_probability[nonzero_rows] = (
            transition[nonzero_rows]
            / row_totals[nonzero_rows, None]
        ).astype(np.float32)

    starts, stops = run_boundaries(states)
    run_states = states[starts].astype(np.int64)
    run_lengths = (stops - starts).astype(np.uint32)

    visit_count = np.zeros(MAX_STATES, dtype=np.uint32)
    mean_dwell = np.full(MAX_STATES, np.nan, dtype=np.float32)
    median_dwell = np.full(MAX_STATES, np.nan, dtype=np.float32)
    max_dwell = np.zeros(MAX_STATES, dtype=np.uint32)
    min_dwell = np.zeros(MAX_STATES, dtype=np.uint32)

    for state_number in range(1, MAX_STATES + 1):
        lengths = run_lengths[run_states == state_number]
        if lengths.size:
            index = state_number - 1
            visit_count[index] = lengths.size
            mean_dwell[index] = float(np.mean(lengths))
            median_dwell[index] = float(np.median(lengths))
            max_dwell[index] = int(np.max(lengths))
            min_dwell[index] = int(np.min(lengths))

    observed_state_count = int(np.unique(states).size)

    bidirectional_pairs = 0
    for left, right in ((0, 1), (0, 2), (1, 2)):
        if transition[left, right] > 0 and transition[right, left] > 0:
            bidirectional_pairs += 1

    direct_1_3_switch_count = int(
        transition[0, 2] + transition[2, 0]
    )

    return {
        "transition": transition,
        "reliable_transition": reliable_transition,
        "transition_probability": transition_probability,
        "switch_count": switch_count,
        "reliable_switch_count": reliable_switch_count,
        "valid_pair_count": valid_pair_count,
        "reliable_pair_count": reliable_pair_count,
        "switching_frequency": (
            switch_count / valid_pair_count
            if valid_pair_count else math.nan
        ),
        "reliable_switching_frequency": (
            reliable_switch_count / reliable_pair_count
            if reliable_pair_count else math.nan
        ),
        "visit_count": visit_count,
        "mean_dwell": mean_dwell,
        "median_dwell": median_dwell,
        "max_dwell": max_dwell,
        "min_dwell": min_dwell,
        "observed_state_count": observed_state_count,
        "bidirectional_pair_count": bidirectional_pairs,
        "direct_1_3_switch_count": direct_1_3_switch_count,
        "run_starts": starts,
        "run_stops": stops,
        "run_states": run_states,
        "run_lengths": run_lengths,
    }


def checkpoint_payload(
    shape: tuple[int, int],
    n_bins: int,
    completed_bin: int,
    completed_candidate_stop: int,
) -> dict:
    return {
        "step": "08_measure_state_transition_statistics",
        "script_version": SCRIPT_VERSION,
        "shape": list(shape),
        "temperature_bin_count": n_bins,
        "completed_temperature_bin_index": completed_bin,
        "completed_candidate_stop": completed_candidate_stop,
    }


def main() -> int:
    args = parse_args()
    validate_common_arguments(args)

    if args.candidate_block <= 0:
        raise ValueError("--candidate-block must be positive")
    if args.flush_every_blocks <= 0:
        raise ValueError("--flush-every-blocks must be positive")
    if args.resume and args.save_run_table:
        raise ValueError("--resume cannot be combined with --save-run-table")
    if args.run_table_all_candidates and not args.save_run_table:
        raise ValueError(
            "--run-table-all-candidates requires --save-run-table"
        )

    started = time.perf_counter()
    state_directory = args.state_dir.expanduser().resolve()
    centered_directory = args.centered_dir.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()

    step07_summary_path = state_directory / "summary.json"
    step07_summary = load_json(step07_summary_path)
    if not step07_summary.get("validation_passed", False):
        raise ValueError("Step07 validation_passed is not true")
    if not str(step07_summary.get("script_version", "")).startswith("7.2"):
        raise ValueError("Step08 v8.0 requires Step07 v7.2 or later")

    state_path = resolve_file(
        state_directory,
        step07_summary,
        ("state", "frame_state_output"),
        "frame_state_uint8.npy",
    )
    quality_path = resolve_file(
        state_directory,
        step07_summary,
        ("quality", "assignment_quality_output"),
        "assign_quality_uint8.npy",
    )
    step06_state_count_path = state_directory.parent / "06_histogram_state_detection" / "state_count_uint8.npy"
    # Prefer a directly copied or nearby file when available, but do not require
    # a fixed Step06 directory name.
    direct_state_count = state_directory / "state_count_uint8.npy"
    if direct_state_count.is_file():
        step06_state_count_path = direct_state_count
    else:
        # The Step07 summary may not record this path. Search only sibling
        # directories for a unique candidate.
        sibling_matches = sorted(
            state_directory.parent.glob("06*/state_count_uint8.npy")
        )
        if len(sibling_matches) == 1:
            step06_state_count_path = sibling_matches[0]

    temperature_bin_path = state_directory / "temperature_bin_index.csv"
    candidate_catalog_path = state_directory / "candidate_catalog.csv"
    frame_metadata_path = state_directory / "frame_metadata.csv"
    dataset_index_path = centered_directory / "dataset_index.csv"

    for path in (
        state_path,
        quality_path,
        temperature_bin_path,
        candidate_catalog_path,
        frame_metadata_path,
        dataset_index_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    frame_state = np.load(state_path, mmap_mode="r")
    assignment_quality = np.load(quality_path, mmap_mode="r")
    if frame_state.ndim != 2 or frame_state.dtype.kind != "u":
        raise ValueError("frame_state must be a two-dimensional unsigned array")
    if assignment_quality.shape != frame_state.shape:
        raise ValueError("assign_quality shape mismatch")

    n_frames, n_candidates = map(int, frame_state.shape)
    if np.any((frame_state < 1) | (frame_state > MAX_STATES)):
        raise ValueError("frame_state contains invalid values")
    if np.any(
        (assignment_quality < Q_INVALID)
        | (assignment_quality > Q_NEAR_VALLEY)
    ):
        raise ValueError("assign_quality contains invalid values")

    bin_table = pd.read_csv(temperature_bin_path)
    dataset_table = pd.read_csv(dataset_index_path)
    candidate_catalog = pd.read_csv(candidate_catalog_path)
    frame_metadata = pd.read_csv(frame_metadata_path)

    if len(candidate_catalog) != n_candidates:
        raise ValueError("candidate_catalog row count mismatch")
    if len(frame_metadata) != n_frames:
        raise ValueError("frame_metadata row count mismatch")

    bin_frames = build_bin_frames(bin_table, dataset_table, n_frames)
    n_bins = len(bin_frames)

    step06_state_count = None
    if step06_state_count_path.is_file():
        candidate_state_count = np.load(
            step06_state_count_path, mmap_mode="r"
        )
        if candidate_state_count.shape == (n_bins, n_candidates):
            step06_state_count = candidate_state_count
        else:
            print(
                "WARNING: Step06 state_count shape mismatch; "
                "run table will use observed states.",
                file=sys.stderr,
            )

    shape2 = (n_bins, n_candidates)
    shape3 = (n_bins, n_candidates, MAX_STATES)
    shape4 = (n_bins, n_candidates, MAX_STATES, MAX_STATES)

    names = {
        "transition_count": "transition_count_uint32.npy",
        "transition_probability": "transition_probability_float32.npy",
        "reliable_transition_count": "reliable_transition_count_uint32.npy",
        "switch_count": "switch_count_uint32.npy",
        "reliable_switch_count": "reliable_switch_count_uint32.npy",
        "switching_frequency": "switching_frequency_float32.npy",
        "reliable_switching_frequency": (
            "reliable_switching_frequency_float32.npy"
        ),
        "valid_pair_count": "valid_pair_count_uint32.npy",
        "reliable_pair_count": "reliable_pair_count_uint32.npy",
        "state_visit_count": "state_visit_count_uint32.npy",
        "mean_dwell": "mean_dwell_frame_float32.npy",
        "median_dwell": "median_dwell_frame_float32.npy",
        "max_dwell": "max_dwell_frame_uint32.npy",
        "min_dwell": "min_dwell_frame_uint32.npy",
        "observed_state_count": "observed_state_count_uint8.npy",
        "bidirectional_pair_count": "bidirectional_pair_count_uint8.npy",
        "direct_1_3_switch_count": "direct_1_3_switch_count_uint32.npy",
    }

    if args.resume:
        saved_checkpoint = load_json(output_directory / "checkpoint.json")
        if (
            saved_checkpoint.get("shape") != list((n_frames, n_candidates))
            or int(
                saved_checkpoint.get("temperature_bin_count", -1)
            ) != n_bins
        ):
            raise ValueError("Checkpoint does not match current inputs")
        resume_bin = int(
            saved_checkpoint.get("completed_temperature_bin_index", 0)
        )
        resume_candidate = int(
            saved_checkpoint.get("completed_candidate_stop", 0)
        )
        mode = "r+"
    else:
        prepare_output_dir(output_directory, overwrite=args.overwrite)
        resume_bin = 0
        resume_candidate = 0
        mode = "w+"

    transition_count = open_memmap(
        output_directory / names["transition_count"],
        mode=mode, dtype=np.uint32, shape=shape4,
    )
    transition_probability = open_memmap(
        output_directory / names["transition_probability"],
        mode=mode, dtype=np.float32, shape=shape4,
    )
    reliable_transition_count = open_memmap(
        output_directory / names["reliable_transition_count"],
        mode=mode, dtype=np.uint32, shape=shape4,
    )
    switch_count = open_memmap(
        output_directory / names["switch_count"],
        mode=mode, dtype=np.uint32, shape=shape2,
    )
    reliable_switch_count = open_memmap(
        output_directory / names["reliable_switch_count"],
        mode=mode, dtype=np.uint32, shape=shape2,
    )
    switching_frequency = open_memmap(
        output_directory / names["switching_frequency"],
        mode=mode, dtype=np.float32, shape=shape2,
    )
    reliable_switching_frequency = open_memmap(
        output_directory / names["reliable_switching_frequency"],
        mode=mode, dtype=np.float32, shape=shape2,
    )
    valid_pair_count = open_memmap(
        output_directory / names["valid_pair_count"],
        mode=mode, dtype=np.uint32, shape=shape2,
    )
    reliable_pair_count = open_memmap(
        output_directory / names["reliable_pair_count"],
        mode=mode, dtype=np.uint32, shape=shape2,
    )
    state_visit_count = open_memmap(
        output_directory / names["state_visit_count"],
        mode=mode, dtype=np.uint32, shape=shape3,
    )
    mean_dwell = open_memmap(
        output_directory / names["mean_dwell"],
        mode=mode, dtype=np.float32, shape=shape3,
    )
    median_dwell = open_memmap(
        output_directory / names["median_dwell"],
        mode=mode, dtype=np.float32, shape=shape3,
    )
    max_dwell = open_memmap(
        output_directory / names["max_dwell"],
        mode=mode, dtype=np.uint32, shape=shape3,
    )
    min_dwell = open_memmap(
        output_directory / names["min_dwell"],
        mode=mode, dtype=np.uint32, shape=shape3,
    )
    observed_state_count = open_memmap(
        output_directory / names["observed_state_count"],
        mode=mode, dtype=np.uint8, shape=shape2,
    )
    bidirectional_pair_count = open_memmap(
        output_directory / names["bidirectional_pair_count"],
        mode=mode, dtype=np.uint8, shape=shape2,
    )
    direct_1_3_switch_count = open_memmap(
        output_directory / names["direct_1_3_switch_count"],
        mode=mode, dtype=np.uint32, shape=shape2,
    )

    arrays = (
        transition_count,
        transition_probability,
        reliable_transition_count,
        switch_count,
        reliable_switch_count,
        switching_frequency,
        reliable_switching_frequency,
        valid_pair_count,
        reliable_pair_count,
        state_visit_count,
        mean_dwell,
        median_dwell,
        max_dwell,
        min_dwell,
        observed_state_count,
        bidirectional_pair_count,
        direct_1_3_switch_count,
    )

    if not args.resume:
        for array in arrays:
            if array.dtype.kind == "f":
                array[:] = np.nan
            else:
                array[:] = 0

        shutil.copy2(
            candidate_catalog_path,
            output_directory / "candidate_catalog.csv",
        )
        shutil.copy2(
            frame_metadata_path,
            output_directory / "frame_metadata.csv",
        )
        shutil.copy2(
            temperature_bin_path,
            output_directory / "temperature_bin_index.csv",
        )

    run_file = None
    run_writer = None
    run_table_path = output_directory / "run_length_table.csv.gz"
    if args.save_run_table:
        run_file = gzip.open(
            run_table_path, "wt", encoding="utf-8", newline=""
        )
        run_writer = csv.writer(run_file)
        run_writer.writerow([
            "temperature_bin_index",
            "candidate_index",
            "state",
            "start_frame_global",
            "stop_frame_global_exclusive",
            "start_position_within_bin",
            "stop_position_within_bin_exclusive",
            "run_length_frames",
            "start_quality",
            "stop_quality",
            "contains_near_valley",
        ])

    blocks_per_bin = math.ceil(n_candidates / args.candidate_block)
    total_blocks = n_bins * blocks_per_bin
    completed_blocks = (
        resume_bin * blocks_per_bin
        + resume_candidate // args.candidate_block
    )
    blocks_since_flush = 0

    print(
        f"Step08 v8.0: {n_frames:,} frames x "
        f"{n_candidates:,} candidates; {n_bins} temperature bins"
    )
    print(
        "RTS classification is intentionally deferred to Step09."
    )

    try:
        for bin_number, rows in enumerate(bin_frames):
            if bin_number < resume_bin:
                continue
            candidate_start = (
                resume_candidate if bin_number == resume_bin else 0
            )
            label = bin_table.iloc[bin_number].get(
                "temperature_bin_label", f"bin_{bin_number}"
            )
            print(
                f"Temperature bin {bin_number + 1}/{n_bins}: "
                f"{label}; {len(rows)} frames"
            )

            for c0 in range(
                candidate_start,
                n_candidates,
                args.candidate_block,
            ):
                c1 = min(c0 + args.candidate_block, n_candidates)
                columns = np.arange(c0, c1, dtype=np.int64)
                state_block = np.asarray(frame_state[np.ix_(rows, columns)])
                quality_block = np.asarray(
                    assignment_quality[np.ix_(rows, columns)]
                )

                for local_candidate in range(c1 - c0):
                    candidate = c0 + local_candidate
                    result = calculate_candidate_statistics(
                        state_block[:, local_candidate],
                        quality_block[:, local_candidate],
                    )

                    transition_count[bin_number, candidate] = result[
                        "transition"
                    ]
                    reliable_transition_count[
                        bin_number, candidate
                    ] = result["reliable_transition"]
                    transition_probability[
                        bin_number, candidate
                    ] = result["transition_probability"]
                    switch_count[bin_number, candidate] = result[
                        "switch_count"
                    ]
                    reliable_switch_count[
                        bin_number, candidate
                    ] = result["reliable_switch_count"]
                    valid_pair_count[bin_number, candidate] = result[
                        "valid_pair_count"
                    ]
                    reliable_pair_count[
                        bin_number, candidate
                    ] = result["reliable_pair_count"]
                    switching_frequency[
                        bin_number, candidate
                    ] = result["switching_frequency"]
                    reliable_switching_frequency[
                        bin_number, candidate
                    ] = result["reliable_switching_frequency"]
                    state_visit_count[bin_number, candidate] = result[
                        "visit_count"
                    ]
                    mean_dwell[bin_number, candidate] = result[
                        "mean_dwell"
                    ]
                    median_dwell[bin_number, candidate] = result[
                        "median_dwell"
                    ]
                    max_dwell[bin_number, candidate] = result[
                        "max_dwell"
                    ]
                    min_dwell[bin_number, candidate] = result[
                        "min_dwell"
                    ]
                    observed_state_count[
                        bin_number, candidate
                    ] = result["observed_state_count"]
                    bidirectional_pair_count[
                        bin_number, candidate
                    ] = result["bidirectional_pair_count"]
                    direct_1_3_switch_count[
                        bin_number, candidate
                    ] = result["direct_1_3_switch_count"]

                    if run_writer is not None:
                        include_runs = args.run_table_all_candidates
                        if not include_runs:
                            if step06_state_count is not None:
                                include_runs = (
                                    int(
                                        step06_state_count[
                                            bin_number, candidate
                                        ]
                                    ) >= 2
                                )
                            else:
                                include_runs = (
                                    int(result["observed_state_count"]) >= 2
                                )

                        if include_runs:
                            run_starts = result["run_starts"]
                            run_stops = result["run_stops"]
                            run_states = result["run_states"]
                            run_lengths = result["run_lengths"]
                            quality_vector = quality_block[
                                :, local_candidate
                            ]
                            for start, stop, state, length in zip(
                                run_starts,
                                run_stops,
                                run_states,
                                run_lengths,
                            ):
                                run_quality = quality_vector[start:stop]
                                run_writer.writerow([
                                    bin_number,
                                    candidate,
                                    int(state),
                                    int(rows[start]),
                                    int(rows[stop - 1]) + 1,
                                    int(start),
                                    int(stop),
                                    int(length),
                                    int(run_quality[0]),
                                    int(run_quality[-1]),
                                    int(
                                        np.any(
                                            run_quality == Q_NEAR_VALLEY
                                        )
                                    ),
                                ])

                completed_blocks += 1
                blocks_since_flush += 1

                if (
                    blocks_since_flush >= args.flush_every_blocks
                    or c1 == n_candidates
                ):
                    for array in arrays:
                        array.flush()
                    if run_file is not None:
                        run_file.flush()

                    next_bin, next_candidate = (
                        (bin_number + 1, 0)
                        if c1 == n_candidates
                        else (bin_number, c1)
                    )
                    write_json(
                        output_directory / "checkpoint.json",
                        checkpoint_payload(
                            (n_frames, n_candidates),
                            n_bins,
                            next_bin,
                            next_candidate,
                        ),
                    )
                    blocks_since_flush = 0

                if (
                    args.progress_every
                    and (
                        completed_blocks % args.progress_every == 0
                        or c1 == n_candidates
                    )
                ):
                    elapsed = time.perf_counter() - started
                    fraction = completed_blocks / total_blocks
                    eta = (
                        elapsed * (1.0 - fraction) / fraction
                        if fraction > 0
                        else math.nan
                    )
                    print(
                        f"  candidates {c1:,}/{n_candidates:,}; "
                        f"blocks {completed_blocks}/{total_blocks}; "
                        f"elapsed {elapsed:.1f}s; ETA {eta:.1f}s"
                    )
            resume_candidate = 0
    finally:
        if run_file is not None:
            run_file.close()

    for array in arrays:
        array.flush()

    transition_array = np.asarray(transition_count)
    reliable_transition_array = np.asarray(reliable_transition_count)
    switch_array = np.asarray(switch_count)
    reliable_switch_array = np.asarray(reliable_switch_count)
    pair_array = np.asarray(valid_pair_count)
    reliable_pair_array = np.asarray(reliable_pair_count)
    visit_array = np.asarray(state_visit_count)
    observed_array = np.asarray(observed_state_count)
    bidirectional_array = np.asarray(bidirectional_pair_count)

    # Core consistency checks.
    expected_transition_shape = (
        n_bins, n_candidates, MAX_STATES, MAX_STATES
    )
    if transition_array.shape != expected_transition_shape:
        raise RuntimeError(
            "Unexpected transition_count shape: "
            f"{transition_array.shape}; expected {expected_transition_shape}"
        )
    if reliable_transition_array.shape != expected_transition_shape:
        raise RuntimeError(
            "Unexpected reliable_transition_count shape: "
            f"{reliable_transition_array.shape}; "
            f"expected {expected_transition_shape}"
        )
    expected_pair_counts = np.array(
        [max(0, len(rows) - 1) for rows in bin_frames],
        dtype=np.uint32,
    )[:, None]
    pair_count_ok = bool(
        np.all(pair_array == expected_pair_counts)
    )
    transition_sum_ok = bool(
        np.array_equal(
            transition_array.sum(axis=(2, 3)),
            pair_array.astype(np.uint64),
        )
    )
    transition_total = transition_array.sum(axis=(2, 3), dtype=np.uint64)
    transition_self = (
        transition_array[:, :, 0, 0].astype(np.uint64)
        + transition_array[:, :, 1, 1].astype(np.uint64)
        + transition_array[:, :, 2, 2].astype(np.uint64)
    )
    transition_switch = transition_total - transition_self

    reliable_transition_total = reliable_transition_array.sum(
        axis=(2, 3), dtype=np.uint64
    )
    reliable_transition_self = (
        reliable_transition_array[:, :, 0, 0].astype(np.uint64)
        + reliable_transition_array[:, :, 1, 1].astype(np.uint64)
        + reliable_transition_array[:, :, 2, 2].astype(np.uint64)
    )
    reliable_transition_switch = (
        reliable_transition_total - reliable_transition_self
    )

    switch_count_ok = bool(
        np.array_equal(
            switch_array.astype(np.uint64),
            transition_switch,
        )
    )
    reliable_transition_sum_ok = bool(
        np.array_equal(
            reliable_transition_total,
            reliable_pair_array.astype(np.uint64),
        )
    )
    reliable_switch_count_ok = bool(
        np.array_equal(
            reliable_switch_array.astype(np.uint64),
            reliable_transition_switch,
        )
    )
    reliable_switch_not_greater_ok = bool(
        np.all(reliable_switch_array <= switch_array)
    )
    visit_count_ok = bool(
        np.all(visit_array.sum(axis=2) == switch_array + 1)
    )

    validation_passed = bool(
        pair_count_ok
        and transition_sum_ok
        and switch_count_ok
        and reliable_transition_sum_ok
        and reliable_switch_count_ok
        and reliable_switch_not_greater_ok
        and visit_count_ok
        and np.all((observed_array >= 1) & (observed_array <= MAX_STATES))
    )

    # Summaries by temperature bin.
    bin_summary_rows: list[dict] = []
    for bin_number, rows in enumerate(bin_frames):
        switches = switch_array[bin_number]
        reliable_switches = reliable_switch_array[bin_number]
        bidirectional = bidirectional_array[bin_number]
        row = bin_table.iloc[bin_number].to_dict()
        row.update({
            "frame_count": int(len(rows)),
            "candidate_count": n_candidates,
            "candidate_with_any_switch_count": int(
                np.count_nonzero(switches > 0)
            ),
            "candidate_with_any_switch_fraction": float(
                np.mean(switches > 0)
            ),
            "candidate_with_5plus_switch_count": int(
                np.count_nonzero(switches >= 5)
            ),
            "candidate_with_10plus_switch_count": int(
                np.count_nonzero(switches >= 10)
            ),
            "candidate_with_any_reliable_switch_count": int(
                np.count_nonzero(reliable_switches > 0)
            ),
            "candidate_with_bidirectional_pair_count": int(
                np.count_nonzero(bidirectional > 0)
            ),
            "candidate_with_two_or_more_bidirectional_pairs_count": int(
                np.count_nonzero(bidirectional >= 2)
            ),
            "median_switch_count": float(np.median(switches)),
            "maximum_switch_count": int(np.max(switches)),
            "mean_switching_frequency": float(
                np.nanmean(
                    np.asarray(switching_frequency[bin_number])
                )
            ),
            "mean_reliable_switching_frequency": float(
                np.nanmean(
                    np.asarray(
                        reliable_switching_frequency[bin_number]
                    )
                )
            ),
        })
        bin_summary_rows.append(row)

    pd.DataFrame(bin_summary_rows).to_csv(
        output_directory / "temperature_bin_transition_summary.csv",
        index=False,
    )

    candidate_summary = candidate_catalog.copy()
    candidate_summary["max_observed_state_count"] = observed_array.max(axis=0)
    candidate_summary["temperature_bin_with_any_switch_count"] = (
        np.count_nonzero(switch_array > 0, axis=0)
    )
    candidate_summary["temperature_bin_with_5plus_switch_count"] = (
        np.count_nonzero(switch_array >= 5, axis=0)
    )
    candidate_summary["temperature_bin_with_bidirectional_pair_count"] = (
        np.count_nonzero(bidirectional_array > 0, axis=0)
    )
    candidate_summary["total_switch_count"] = switch_array.sum(axis=0)
    candidate_summary["total_reliable_switch_count"] = (
        reliable_switch_array.sum(axis=0)
    )
    candidate_summary["max_switch_count_in_one_bin"] = switch_array.max(axis=0)
    candidate_summary["max_bidirectional_pair_count"] = (
        bidirectional_array.max(axis=0)
    )
    candidate_summary["total_direct_1_3_switch_count"] = np.asarray(
        direct_1_3_switch_count
    ).sum(axis=0)
    candidate_summary.to_csv(
        output_directory / "candidate_transition_summary.csv",
        index=False,
    )

    any_switch_candidate = np.any(switch_array > 0, axis=0)
    any_reliable_switch_candidate = np.any(
        reliable_switch_array > 0, axis=0
    )
    any_5plus_switch_candidate = np.any(switch_array >= 5, axis=0)
    any_10plus_switch_candidate = np.any(switch_array >= 10, axis=0)
    any_bidirectional_candidate = np.any(
        bidirectional_array > 0, axis=0
    )
    any_two_bidirectional_candidate = np.any(
        bidirectional_array >= 2, axis=0
    )


    any_3plus_switch_candidate = np.any(switch_array >= 3, axis=0)
    direct_13_array = np.asarray(direct_1_3_switch_count)
    any_direct_13_candidate = np.any(direct_13_array > 0, axis=0)

    # Exact global run-length summary, independent of --save-run-table.
    maximum_bin_frame_count = max(len(rows) for rows in bin_frames)
    run_histogram = np.zeros(maximum_bin_frame_count + 1, dtype=np.uint64)
    total_run_length = 0
    for bin_number, rows in enumerate(bin_frames):
        for c0 in range(0, n_candidates, args.candidate_block):
            c1 = min(c0 + args.candidate_block, n_candidates)
            columns = np.arange(c0, c1, dtype=np.int64)
            state_block = np.asarray(frame_state[np.ix_(rows, columns)])
            for local_candidate in range(c1 - c0):
                starts, stops = run_boundaries(state_block[:, local_candidate])
                lengths = stops - starts
                run_histogram += np.bincount(
                    lengths, minlength=maximum_bin_frame_count + 1
                ).astype(np.uint64)
                total_run_length += int(lengths.sum())

    total_runs = int(run_histogram.sum())
    cumulative_runs = np.cumsum(run_histogram)

    def _run_percentile(probability: float) -> int:
        target = max(1, int(math.ceil(probability * total_runs)))
        return int(np.searchsorted(cumulative_runs, target, side="left"))

    run_length_summary = {
        "total_runs": total_runs,
        "mean_run_length_frames": float(total_run_length / total_runs),
        "median_run_length_frames": _run_percentile(0.50),
        "p95_run_length_frames": _run_percentile(0.95),
        "p99_run_length_frames": _run_percentile(0.99),
        "maximum_run_length_frames": int(np.flatnonzero(run_histogram)[-1]),
    }

    # Diagnostic score for inspection only; this is not an RTS decision.
    reliable_fraction = np.divide(
        reliable_pair_array,
        pair_array,
        out=np.zeros_like(reliable_pair_array, dtype=np.float64),
        where=pair_array > 0,
    )
    diagnostic_score = (
        np.log1p(reliable_switch_array.astype(np.float64))
        * (1.0 + bidirectional_array.astype(np.float64))
        * observed_array.astype(np.float64)
        * np.sqrt(reliable_fraction)
    )

    candidate_indices = np.arange(n_candidates, dtype=np.int64)
    best_bin_index = np.argmax(diagnostic_score, axis=0)
    best_score = diagnostic_score[best_bin_index, candidate_indices]
    score_table = pd.DataFrame({
        "candidate_index": candidate_indices,
        "diagnostic_ranking_score": best_score,
        "best_temperature_bin_index": best_bin_index,
        "best_bin_switch_count": switch_array[
            best_bin_index, candidate_indices
        ],
        "best_bin_reliable_switch_count": reliable_switch_array[
            best_bin_index, candidate_indices
        ],
        "best_bin_observed_state_count": observed_array[
            best_bin_index, candidate_indices
        ],
        "best_bin_bidirectional_pair_count": bidirectional_array[
            best_bin_index, candidate_indices
        ],
        "best_bin_direct_1_3_switch_count": direct_13_array[
            best_bin_index, candidate_indices
        ],
    })
    for column in candidate_catalog.columns:
        if column not in score_table.columns:
            score_table[column] = candidate_catalog[column].to_numpy()
    score_table.sort_values(
        ["diagnostic_ranking_score", "best_bin_reliable_switch_count"],
        ascending=[False, False],
    ).to_csv(output_directory / "candidate_score.csv", index=False)

    flat_score = diagnostic_score.ravel()
    top_n = min(100, flat_score.size)
    top_flat = np.argpartition(flat_score, -top_n)[-top_n:]
    top_flat = top_flat[np.argsort(flat_score[top_flat])[::-1]]
    top_bins, top_candidates = np.unravel_index(
        top_flat, diagnostic_score.shape
    )

    top_candidate_rows = []
    top_run_rows = []
    for rank, (bin_number, candidate) in enumerate(
        zip(top_bins, top_candidates), start=1
    ):
        bin_number = int(bin_number)
        candidate = int(candidate)
        record = {
            "rank": rank,
            "candidate_index": candidate,
            "temperature_bin_index": bin_number,
            "diagnostic_ranking_score": float(
                diagnostic_score[bin_number, candidate]
            ),
            "switch_count": int(switch_array[bin_number, candidate]),
            "reliable_switch_count": int(
                reliable_switch_array[bin_number, candidate]
            ),
            "observed_state_count": int(
                observed_array[bin_number, candidate]
            ),
            "bidirectional_pair_count": int(
                bidirectional_array[bin_number, candidate]
            ),
            "direct_1_3_switch_count": int(
                direct_13_array[bin_number, candidate]
            ),
        }
        for state_index in range(MAX_STATES):
            state_number = state_index + 1
            record[f"state_{state_number}_visit_count"] = int(
                visit_array[bin_number, candidate, state_index]
            )
            record[f"state_{state_number}_mean_dwell_frames"] = float(
                mean_dwell[bin_number, candidate, state_index]
            )
            record[f"state_{state_number}_median_dwell_frames"] = float(
                median_dwell[bin_number, candidate, state_index]
            )
            record[f"state_{state_number}_max_dwell_frames"] = int(
                max_dwell[bin_number, candidate, state_index]
            )
        for column in candidate_catalog.columns:
            if column not in record:
                record[column] = candidate_catalog.iloc[candidate][column]
        for column in bin_table.columns:
            if column not in record:
                record[column] = bin_table.iloc[bin_number][column]
        top_candidate_rows.append(record)

        rows = bin_frames[bin_number]
        states = np.asarray(frame_state[rows, candidate])
        qualities = np.asarray(assignment_quality[rows, candidate])
        starts, stops = run_boundaries(states)
        for run_index, (run_start, run_stop) in enumerate(
            zip(starts, stops), start=1
        ):
            run_quality = qualities[run_start:run_stop]
            top_run_rows.append({
                "rank": rank,
                "candidate_index": candidate,
                "temperature_bin_index": bin_number,
                "run_index": run_index,
                "state": int(states[run_start]),
                "start_frame_global": int(rows[run_start]),
                "stop_frame_global_exclusive": int(rows[run_stop - 1]) + 1,
                "run_length_frames": int(run_stop - run_start),
                "contains_near_valley": int(
                    np.any(run_quality == Q_NEAR_VALLEY)
                ),
            })

    pd.DataFrame(top_candidate_rows).to_csv(
        output_directory / "top100_switching_candidates.csv",
        index=False,
    )
    pd.DataFrame(top_run_rows).to_csv(
        output_directory / "top100_runs.csv",
        index=False,
    )

    elapsed = time.perf_counter() - started
    summary = {
        "step": "08_measure_state_transition_statistics",
        "script_version": SCRIPT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "validation_passed": validation_passed,
        "responsibility": (
            "objective transition and dwell statistics only; "
            "no RTS classification and no dictionary generation"
        ),
        "state_dir": str(state_directory),
        "centered_dir": str(centered_directory),
        "array_shape": [n_frames, n_candidates],
        "frame_count": n_frames,
        "candidate_count": n_candidates,
        "temperature_bin_count": n_bins,
        "maximum_supported_state_count": MAX_STATES,
        "quality_policy": {
            "raw_statistics": "all valid Step07 state assignments",
            "reliable_transition_statistics": (
                "only consecutive pairs whose two endpoint qualities are "
                "excellent or normal; excluded pairs are not bridged"
            ),
            "dwell_statistics": (
                "all state assignments; quality filtering is not used because "
                "it would split or merge physical runs"
            ),
        },
        "candidate_counts_any_temperature_bin": {
            "any_switch": int(np.count_nonzero(any_switch_candidate)),
            "any_reliable_switch": int(
                np.count_nonzero(any_reliable_switch_candidate)
            ),
            "three_or_more_switches": int(
                np.count_nonzero(any_3plus_switch_candidate)
            ),
            "five_or_more_switches": int(
                np.count_nonzero(any_5plus_switch_candidate)
            ),
            "ten_or_more_switches": int(
                np.count_nonzero(any_10plus_switch_candidate)
            ),
            "at_least_one_bidirectional_state_pair": int(
                np.count_nonzero(any_bidirectional_candidate)
            ),
            "at_least_two_bidirectional_state_pairs": int(
                np.count_nonzero(any_two_bidirectional_candidate)
            ),
            "direct_1_3_switch": int(
                np.count_nonzero(any_direct_13_candidate)
            ),
        },
        "run_length_summary": run_length_summary,
        "diagnostic_ranking_score": {
            "purpose": "inspection/ranking only; not RTS classification",
            "formula": (
                "log1p(reliable_switch_count) * "
                "(1 + bidirectional_pair_count) * observed_state_count * "
                "sqrt(reliable_pair_count / valid_pair_count)"
            ),
        },
        "validation": {
            "pair_count_validation_passed": pair_count_ok,
            "transition_sum_validation_passed": transition_sum_ok,
            "switch_count_validation_passed": switch_count_ok,
            "reliable_transition_sum_validation_passed": (
                reliable_transition_sum_ok
            ),
            "reliable_switch_count_validation_passed": (
                reliable_switch_count_ok
            ),
            "reliable_switch_not_greater_validation_passed": (
                reliable_switch_not_greater_ok
            ),
            "visit_count_equals_switch_plus_one_validation_passed": (
                visit_count_ok
            ),
        },
        "run_table_saved": bool(args.save_run_table),
        "run_table_scope": (
            "all candidates"
            if args.run_table_all_candidates
            else "Step06 multistate candidate/bin combinations only"
        ) if args.save_run_table else None,
        "outputs": {
            **{
                key: str(output_directory / filename)
                for key, filename in names.items()
            },
            "candidate_transition_summary": str(
                output_directory / "candidate_transition_summary.csv"
            ),
            "temperature_bin_transition_summary": str(
                output_directory / "temperature_bin_transition_summary.csv"
            ),
            "candidate_score": str(
                output_directory / "candidate_score.csv"
            ),
            "top100_switching_candidates": str(
                output_directory / "top100_switching_candidates.csv"
            ),
            "top100_runs": str(
                output_directory / "top100_runs.csv"
            ),
            "run_length_table": (
                str(run_table_path) if args.save_run_table else None
            ),
        },
        "elapsed_seconds": elapsed,
    }
    write_json(output_directory / "summary.json", summary)

    manifest = {
        "inputs": {
            "frame_state": str(state_path),
            "assign_quality": str(quality_path),
            "step07_summary": str(step07_summary_path),
            "temperature_bin_index": str(temperature_bin_path),
            "candidate_catalog": str(candidate_catalog_path),
            "frame_metadata": str(frame_metadata_path),
            "dataset_index": str(dataset_index_path),
        },
        "outputs": [
            *names.values(),
            "candidate_catalog.csv",
            "frame_metadata.csv",
            "temperature_bin_index.csv",
            "candidate_transition_summary.csv",
            "temperature_bin_transition_summary.csv",
            "candidate_score.csv",
            "top100_switching_candidates.csv",
            "top100_runs.csv",
            "checkpoint.json",
            "summary.json",
            "manifest.json",
        ] + (
            ["run_length_table.csv.gz"] if args.save_run_table else []
        ),
    }
    if step06_state_count is not None:
        manifest["inputs"]["step06_state_count"] = str(
            step06_state_count_path
        )
    if args.hash_inputs:
        manifest["input_sha256"] = {
            key: sha256_file(Path(value))
            for key, value in manifest["inputs"].items()
        }
    write_json(output_directory / "manifest.json", manifest)

    print(f"PASS: Step08 v8.0 completed in {elapsed:.1f} s")
    print(
        "  candidates with any switch in at least one bin: "
        f"{summary['candidate_counts_any_temperature_bin']['any_switch']:,}"
    )
    print(
        "  candidates with >=5 switches in at least one bin: "
        f"{summary['candidate_counts_any_temperature_bin']['five_or_more_switches']:,}"
    )
    print(
        "  candidates with bidirectional switching in at least one bin: "
        f"{summary['candidate_counts_any_temperature_bin']['at_least_one_bidirectional_state_pair']:,}"
    )
    return 0 if validation_passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exception:
        print(f"ERROR: {exception}", file=sys.stderr)
        raise
