#!/usr/bin/env python3
"""
Step09 v9.0.0
RTS candidate classification and temperature-bin dictionary generation.

Responsibilities
----------------
- Use Step07 state assignments and Step08 transition/dwell statistics.
- Measure temporal plateau structure and state-separation significance.
- Classify candidate/bin combinations as reject, review, or accepted RTS.
- Aggregate accepted bins to candidate-level decisions.
- Generate a temperature-dependent RTS dictionary for Step10.

This script does not correct image data.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "9.1.0"
MAX_STATES = 3

DECISION_REJECT = 0
DECISION_REVIEW = 1
DECISION_ACCEPT = 2
DECISION_STRONG_ACCEPT = 3

Q_INVALID = 0
Q_EXCELLENT = 1
Q_NORMAL = 2
Q_NEAR_VALLEY = 3

REASON_NAMES = {
    0: "not_multistate",
    1: "too_few_reliable_switches",
    2: "no_bidirectional_transition",
    3: "low_reliable_pair_fraction",
    4: "low_minimum_state_occupancy",
    5: "too_few_visits_in_active_state",
    6: "short_maximum_plateau",
    7: "short_median_plateau",
    8: "high_singleton_run_fraction",
    9: "switch_fraction_too_high",
    10: "insufficient_state_separation",
    11: "missing_or_invalid_metric",
    12: "high_near_valley_fraction",
    13: "too_many_direct_1_3_switches",
}


@dataclass(frozen=True)
class Thresholds:
    min_reliable_switches: int = 5
    min_reliable_pair_fraction_2state: float = 0.90
    min_reliable_pair_fraction_3state: float = 0.95
    min_state_occupancy_2state: float = 0.05
    min_state_occupancy_3state: float = 0.02
    min_visits_per_active_state: int = 3
    min_max_plateau_frames: int = 5
    min_median_plateau_frames: float = 2.0
    max_singleton_run_fraction: float = 0.75
    max_switch_fraction: float = 0.50
    min_separation_sigma: float = 3.0
    min_separation_adu: float = 1.0
    max_near_valley_fraction: float = 0.10
    min_bidirectional_pairs_2state: int = 1
    min_bidirectional_pairs_3state: int = 2
    max_direct_1_3_fraction_3state: float = 0.20
    strong_min_separation_sigma: float = 4.0
    strong_min_state_occupancy_2state: float = 0.08
    strong_min_state_occupancy_3state: float = 0.05
    strong_min_max_plateau_frames: int = 10
    strong_max_switch_fraction: float = 0.40
    strong_max_near_valley_fraction: float = 0.05
    review_min_reliable_switches: int = 3
    review_min_separation_sigma: float = 2.0
    candidate_min_accepted_bins: int = 1
    candidate_min_review_bins: int = 1

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Classify RTS candidates and generate a temperature-dependent "
            "dictionary from Step07 and Step08 outputs."
        )
    )
    p.add_argument("--state-dir", type=Path, required=True,
                   help="Step07 output directory.")
    p.add_argument("--transition-dir", type=Path, required=True,
                   help="Step08 v8.0.2 output directory.")
    p.add_argument("--centered-dir", type=Path, required=True,
                   help="Step05 output directory.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("09_rts_dictionary_v9_0_0"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--chunk-size", type=int, default=1024,
                   help="Candidate chunk size for plateau/separation analysis.")

    # Classification thresholds
    p.add_argument("--min-reliable-switches", type=int, default=5)
    p.add_argument("--min-reliable-pair-fraction-2state", type=float, default=0.90)
    p.add_argument("--min-reliable-pair-fraction-3state", type=float, default=0.95)
    p.add_argument("--min-state-occupancy-2state", type=float, default=0.05)
    p.add_argument("--min-state-occupancy-3state", type=float, default=0.02)
    p.add_argument("--min-visits-per-active-state", type=int, default=3)
    p.add_argument("--min-max-plateau-frames", type=int, default=5)
    p.add_argument("--min-median-plateau-frames", type=float, default=2.0)
    p.add_argument("--max-singleton-run-fraction", type=float, default=0.75)
    p.add_argument("--max-switch-fraction", type=float, default=0.50)
    p.add_argument("--min-separation-sigma", type=float, default=3.0)
    p.add_argument("--min-separation-adu", type=float, default=1.0)
    p.add_argument("--max-near-valley-fraction", type=float, default=0.10)
    p.add_argument("--min-bidirectional-pairs-2state", type=int, default=1)
    p.add_argument("--min-bidirectional-pairs-3state", type=int, default=2)
    p.add_argument("--max-direct-1-3-fraction-3state", type=float, default=0.20)
    p.add_argument("--strong-min-separation-sigma", type=float, default=4.0)
    p.add_argument("--strong-min-state-occupancy-2state", type=float, default=0.08)
    p.add_argument("--strong-min-state-occupancy-3state", type=float, default=0.05)
    p.add_argument("--strong-min-max-plateau-frames", type=int, default=10)
    p.add_argument("--strong-max-switch-fraction", type=float, default=0.40)
    p.add_argument("--strong-max-near-valley-fraction", type=float, default=0.05)
    p.add_argument("--review-min-reliable-switches", type=int, default=3)
    p.add_argument("--review-min-separation-sigma", type=float, default=2.0)
    p.add_argument("--candidate-min-accepted-bins", type=int, default=1)
    p.add_argument("--candidate-min-review-bins", type=int, default=1)

    p.add_argument(
        "--save-rejected-bin-table",
        action="store_true",
        help="Include rejected candidate/bin rows in the long table.",
    )
    return p.parse_args()


def thresholds_from_args(a: argparse.Namespace) -> Thresholds:
    return Thresholds(
        min_reliable_switches=a.min_reliable_switches,
        min_reliable_pair_fraction_2state=a.min_reliable_pair_fraction_2state,
        min_reliable_pair_fraction_3state=a.min_reliable_pair_fraction_3state,
        min_state_occupancy_2state=a.min_state_occupancy_2state,
        min_state_occupancy_3state=a.min_state_occupancy_3state,
        min_visits_per_active_state=a.min_visits_per_active_state,
        min_max_plateau_frames=a.min_max_plateau_frames,
        min_median_plateau_frames=a.min_median_plateau_frames,
        max_singleton_run_fraction=a.max_singleton_run_fraction,
        max_switch_fraction=a.max_switch_fraction,
        min_separation_sigma=a.min_separation_sigma,
        min_separation_adu=a.min_separation_adu,
        max_near_valley_fraction=a.max_near_valley_fraction,
        min_bidirectional_pairs_2state=a.min_bidirectional_pairs_2state,
        min_bidirectional_pairs_3state=a.min_bidirectional_pairs_3state,
        max_direct_1_3_fraction_3state=a.max_direct_1_3_fraction_3state,
        strong_min_separation_sigma=a.strong_min_separation_sigma,
        strong_min_state_occupancy_2state=a.strong_min_state_occupancy_2state,
        strong_min_state_occupancy_3state=a.strong_min_state_occupancy_3state,
        strong_min_max_plateau_frames=a.strong_min_max_plateau_frames,
        strong_max_switch_fraction=a.strong_max_switch_fraction,
        strong_max_near_valley_fraction=a.strong_max_near_valley_fraction,
        review_min_reliable_switches=a.review_min_reliable_switches,
        review_min_separation_sigma=a.review_min_separation_sigma,
        candidate_min_accepted_bins=a.candidate_min_accepted_bins,
        candidate_min_review_bins=a.candidate_min_review_bins,
    )

def validate_thresholds(t: Thresholds) -> None:
    if t.min_reliable_switches < 1:
        raise ValueError("min_reliable_switches must be >= 1")
    for name in (
        "min_reliable_pair_fraction_2state",
        "min_reliable_pair_fraction_3state",
        "min_state_occupancy_2state",
        "min_state_occupancy_3state",
        "max_singleton_run_fraction",
        "max_switch_fraction",
        "max_near_valley_fraction",
        "max_direct_1_3_fraction_3state",
        "strong_min_state_occupancy_2state",
        "strong_min_state_occupancy_3state",
        "strong_max_switch_fraction",
        "strong_max_near_valley_fraction",
    ):
        if not 0 <= getattr(t, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if t.min_visits_per_active_state < 1:
        raise ValueError("min_visits_per_active_state must be >= 1")
    if t.min_max_plateau_frames < 1:
        raise ValueError("min_max_plateau_frames must be >= 1")
    if t.min_median_plateau_frames < 1:
        raise ValueError("min_median_plateau_frames must be >= 1")
    if t.min_bidirectional_pairs_2state < 1 or t.min_bidirectional_pairs_3state < 1:
        raise ValueError("bidirectional-pair thresholds must be >= 1")
    if t.min_separation_sigma <= 0 or t.min_separation_adu <= 0:
        raise ValueError("separation thresholds must be positive")
    if t.strong_min_separation_sigma < t.min_separation_sigma:
        raise ValueError("strong separation threshold must be >= accept threshold")
    if t.strong_min_max_plateau_frames < t.min_max_plateau_frames:
        raise ValueError("strong plateau threshold must be >= accept threshold")

def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def resolve_file(
    directory: Path,
    summary: dict,
    keys: Iterable[str],
    fallbacks: Iterable[str],
) -> Path:
    outputs = summary.get("outputs", {})
    for key in keys:
        values = []
        if isinstance(outputs, dict):
            values.append(outputs.get(key))
        values.append(summary.get(key))
        for value in values:
            if not value:
                continue
            p = Path(str(value))
            if p.is_file():
                return p
            local = directory / p.name
            if local.is_file():
                return local
    for name in fallbacks:
        p = directory / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Could not resolve {list(keys)} / {list(fallbacks)} in {directory}"
    )


def parse_indices(value: object) -> list[int]:
    text = str(value).strip()
    if not text:
        return []
    return [int(x) for x in text.split(";") if x.strip()]


def build_bin_frames(
    bin_table: pd.DataFrame,
    dataset_table: pd.DataFrame,
    frame_count: int,
) -> list[np.ndarray]:
    bins = bin_table.sort_values("temperature_bin_index").reset_index(drop=True)
    numbers = pd.to_numeric(
        bins["temperature_bin_index"], errors="raise"
    ).to_numpy(np.int64)
    if not np.array_equal(numbers, np.arange(len(bins))):
        raise ValueError("temperature_bin_index must be 0..N-1")

    required = {"dataset_index", "frame_start", "frame_stop_exclusive"}
    if not required.issubset(dataset_table.columns):
        raise KeyError(f"dataset_index.csv requires {sorted(required)}")
    datasets = dataset_table.copy()
    datasets["dataset_index"] = pd.to_numeric(
        datasets["dataset_index"], errors="raise"
    ).astype(np.int64)
    datasets = datasets.set_index("dataset_index")

    owner = np.full(frame_count, -1, np.int32)
    result: list[np.ndarray] = []
    for row in bins.itertuples(index=False):
        pieces = []
        for dataset_index in parse_indices(row.dataset_indices):
            if dataset_index not in datasets.index:
                raise KeyError(f"Unknown dataset_index {dataset_index}")
            d = datasets.loc[dataset_index]
            start = int(d.frame_start)
            stop = int(d.frame_stop_exclusive)
            if start < 0 or stop > frame_count or stop <= start:
                raise ValueError(f"Invalid frame interval [{start},{stop})")
            pieces.append(np.arange(start, stop, dtype=np.int64))
        if not pieces:
            raise ValueError(
                f"Temperature bin {row.temperature_bin_index} has no frames"
            )
        frames = np.concatenate(pieces)
        if np.any(owner[frames] >= 0):
            raise ValueError("A frame belongs to multiple bins")
        owner[frames] = int(row.temperature_bin_index)
        result.append(frames)
    if np.any(owner < 0):
        raise ValueError("Some frames are not assigned to a temperature bin")
    return result


def temperature_label(row: pd.Series, b: int) -> str:
    for col in (
        "temperature_bin_label", "bin_label", "label", "temperature_label"
    ):
        if col in row and pd.notna(row[col]):
            return str(row[col])
    for col in (
        "temperature_center_C", "temperature_median_C",
        "temperature_mean_C", "temperature_C"
    ):
        if col in row and pd.notna(row[col]):
            return f"{float(row[col]):+.2f}C"
    return f"Tbin_{b:02d}"


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad
    if sigma <= 0:
        # Quantized data can have MAD=0. Fall back to pairwise spread.
        q16, q84 = np.percentile(x, [16, 84])
        sigma = 0.5 * (q84 - q16)
    return float(sigma)


def run_lengths(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return run state labels and run lengths for valid states (>0)."""
    s = np.asarray(states, dtype=np.int16)
    valid = s > 0
    if not np.any(valid):
        return np.empty(0, np.int16), np.empty(0, np.int32)

    # Invalid samples split runs and are not bridged.
    boundaries = np.empty(len(s), dtype=bool)
    boundaries[0] = True
    boundaries[1:] = (
        (s[1:] != s[:-1])
        | (~valid[1:])
        | (~valid[:-1])
    )
    starts = np.flatnonzero(boundaries & valid)
    if starts.size == 0:
        return np.empty(0, np.int16), np.empty(0, np.int32)

    # Each valid start ends at the next boundary or array end.
    all_boundaries = np.flatnonzero(boundaries)
    next_boundary = np.r_[all_boundaries[1:], len(s)]
    boundary_to_end = dict(zip(all_boundaries.tolist(), next_boundary.tolist()))
    lengths = np.array(
        [boundary_to_end[int(start)] - int(start) for start in starts],
        dtype=np.int32,
    )
    return s[starts], lengths


def reason_text(mask: int) -> str:
    names = [name for bit, name in REASON_NAMES.items() if mask & (1 << bit)]
    return ";".join(names)


def add_reason(mask: int, bit: int, condition_failed: bool) -> int:
    return mask | (1 << bit) if condition_failed else mask


def classify_bin(
    *,
    observed_state_count: int,
    reliable_switch_count: int,
    bidirectional_pair_count: int,
    reliable_pair_fraction: float,
    minimum_state_occupancy: float,
    minimum_state_visits: int,
    maximum_plateau: int,
    median_plateau: float,
    singleton_fraction: float,
    switch_fraction: float,
    minimum_separation_sigma: float,
    minimum_separation_adu: float,
    near_valley_fraction: float,
    direct_1_3_fraction: float,
    t: Thresholds,
) -> tuple[int, int]:
    values = (
        reliable_pair_fraction, minimum_state_occupancy, median_plateau,
        singleton_fraction, switch_fraction, minimum_separation_sigma,
        minimum_separation_adu, near_valley_fraction,
    )
    if any(not np.isfinite(v) for v in values):
        return DECISION_REJECT, 1 << 11

    if observed_state_count == 2:
        min_reliable_pair_fraction = t.min_reliable_pair_fraction_2state
        min_state_occupancy = t.min_state_occupancy_2state
        min_bidirectional_pairs = t.min_bidirectional_pairs_2state
        strong_min_occupancy = t.strong_min_state_occupancy_2state
    elif observed_state_count == 3:
        min_reliable_pair_fraction = t.min_reliable_pair_fraction_3state
        min_state_occupancy = t.min_state_occupancy_3state
        min_bidirectional_pairs = t.min_bidirectional_pairs_3state
        strong_min_occupancy = t.strong_min_state_occupancy_3state
    else:
        return DECISION_REJECT, 1 << 0

    mask = 0
    mask = add_reason(mask, 1, reliable_switch_count < t.min_reliable_switches)
    mask = add_reason(mask, 2, bidirectional_pair_count < min_bidirectional_pairs)
    mask = add_reason(mask, 3, reliable_pair_fraction < min_reliable_pair_fraction)
    mask = add_reason(mask, 4, minimum_state_occupancy < min_state_occupancy)
    mask = add_reason(mask, 5, minimum_state_visits < t.min_visits_per_active_state)
    mask = add_reason(mask, 6, maximum_plateau < t.min_max_plateau_frames)
    mask = add_reason(mask, 7, median_plateau < t.min_median_plateau_frames)
    mask = add_reason(mask, 8, singleton_fraction > t.max_singleton_run_fraction)
    mask = add_reason(mask, 9, switch_fraction > t.max_switch_fraction)
    mask = add_reason(
        mask, 10,
        minimum_separation_sigma < t.min_separation_sigma
        or minimum_separation_adu < t.min_separation_adu,
    )
    mask = add_reason(mask, 12, near_valley_fraction > t.max_near_valley_fraction)
    if observed_state_count == 3:
        if not np.isfinite(direct_1_3_fraction):
            mask |= 1 << 11
        else:
            mask = add_reason(
                mask, 13,
                direct_1_3_fraction > t.max_direct_1_3_fraction_3state,
            )

    if mask == 0:
        strong_ok = (
            minimum_separation_sigma >= t.strong_min_separation_sigma
            and minimum_state_occupancy >= strong_min_occupancy
            and maximum_plateau >= t.strong_min_max_plateau_frames
            and switch_fraction <= t.strong_max_switch_fraction
            and near_valley_fraction <= t.strong_max_near_valley_fraction
        )
        return (DECISION_STRONG_ACCEPT if strong_ok else DECISION_ACCEPT), 0

    review_ok = (
        reliable_switch_count >= t.review_min_reliable_switches
        and bidirectional_pair_count >= 1
        and minimum_separation_sigma >= t.review_min_separation_sigma
        and maximum_plateau >= 2
    )
    return (DECISION_REVIEW if review_ok else DECISION_REJECT), mask

def main() -> int:
    args = parse_args()
    thresholds = thresholds_from_args(args)
    validate_thresholds(thresholds)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    started = time.perf_counter()
    state_dir = args.state_dir.expanduser().resolve()
    transition_dir = args.transition_dir.expanduser().resolve()
    centered_dir = args.centered_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output(output_dir, args.overwrite)

    s7 = read_json(state_dir / "summary.json")
    s8 = read_json(transition_dir / "summary.json")
    s5 = read_json(centered_dir / "summary.json")
    if not s7.get("validation_passed", False):
        raise ValueError("Step07 validation did not pass")
    if not s8.get("validation_passed", False):
        raise ValueError("Step08 validation did not pass")

    state_path = resolve_file(
        state_dir, s7, ("state", "frame_state_output"),
        ("frame_state_uint8.npy",)
    )
    quality_path = resolve_file(
        state_dir, s7, ("quality", "assignment_quality_output"),
        ("assign_quality_uint8.npy",)
    )
    center_path = resolve_file(
        state_dir, s7, ("center", "state_center"),
        ("state_center_refined_ADU_float32.npy",)
    )
    occupancy_path = resolve_file(
        state_dir, s7, ("occupancy", "state_occupancy"),
        ("state_occupancy_refined_float32.npy",)
    )
    centered_path = resolve_file(
        centered_dir, s5,
        ("centered_residual", "centered_residual_output"),
        (
            "centered_residual_x4_int16.npy",
            "centered_residual_float32.npy",
        )
    )

    switch_path = resolve_file(
        transition_dir, s8, ("switch_count",),
        ("switch_count_uint32.npy",)
    )
    reliable_switch_path = resolve_file(
        transition_dir, s8, ("reliable_switch_count",),
        ("reliable_switch_count_uint32.npy",)
    )
    valid_pair_path = resolve_file(
        transition_dir, s8, ("valid_pair_count",),
        ("valid_pair_count_uint32.npy",)
    )
    reliable_pair_path = resolve_file(
        transition_dir, s8, ("reliable_pair_count",),
        ("reliable_pair_count_uint32.npy",)
    )
    visit_path = resolve_file(
        transition_dir, s8, ("state_visit_count",),
        ("state_visit_count_uint32.npy",)
    )
    observed_path = resolve_file(
        transition_dir, s8, ("observed_state_count",),
        ("observed_state_count_uint8.npy",)
    )
    bidirectional_path = resolve_file(
        transition_dir, s8, ("bidirectional_pair_count",),
        ("bidirectional_pair_count_uint8.npy",)
    )
    direct13_path = resolve_file(
        transition_dir, s8, ("direct_1_3_switch_count",),
        ("direct_1_3_switch_count_uint32.npy",)
    )

    bin_table_path = state_dir / "temperature_bin_index.csv"
    catalog_path = state_dir / "candidate_catalog.csv"
    dataset_path = centered_dir / "dataset_index.csv"
    for p in (bin_table_path, catalog_path, dataset_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    frame_state = np.load(state_path, mmap_mode="r")
    quality = np.load(quality_path, mmap_mode="r")
    state_center = np.load(center_path, mmap_mode="r")
    occupancy = np.load(occupancy_path, mmap_mode="r")
    centered_raw = np.load(centered_path, mmap_mode="r")

    switch = np.load(switch_path, mmap_mode="r")
    reliable_switch = np.load(reliable_switch_path, mmap_mode="r")
    valid_pair = np.load(valid_pair_path, mmap_mode="r")
    reliable_pair = np.load(reliable_pair_path, mmap_mode="r")
    visits = np.load(visit_path, mmap_mode="r")
    observed = np.load(observed_path, mmap_mode="r")
    bidirectional = np.load(bidirectional_path, mmap_mode="r")
    direct13 = np.load(direct13_path, mmap_mode="r")

    if frame_state.ndim != 2:
        raise ValueError("frame_state must be 2D")
    n_frames, n_candidates = map(int, frame_state.shape)

    bin_table = pd.read_csv(bin_table_path).sort_values(
        "temperature_bin_index"
    ).reset_index(drop=True)
    catalog = pd.read_csv(catalog_path)
    dataset_table = pd.read_csv(dataset_path)
    bin_frames = build_bin_frames(bin_table, dataset_table, n_frames)
    n_bins = len(bin_frames)

    expected2 = (n_bins, n_candidates)
    expected3 = (n_bins, n_candidates, MAX_STATES)
    for name, arr, shape in (
        ("quality", quality, frame_state.shape),
        ("centered", centered_raw, frame_state.shape),
        ("state_center", state_center, expected3),
        ("occupancy", occupancy, expected3),
        ("switch", switch, expected2),
        ("reliable_switch", reliable_switch, expected2),
        ("valid_pair", valid_pair, expected2),
        ("reliable_pair", reliable_pair, expected2),
        ("visits", visits, expected3),
        ("observed", observed, expected2),
        ("bidirectional", bidirectional, expected2),
        ("direct13", direct13, expected2),
    ):
        if arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape} != {shape}")
    if len(catalog) != n_candidates:
        raise ValueError("candidate_catalog row count mismatch")

    centered_scale = 4.0 if "_x4_" in centered_path.name else 1.0

    decision = np.zeros(expected2, np.uint8)
    reason_mask = np.zeros(expected2, np.uint16)
    reliable_pair_fraction = np.full(expected2, np.nan, np.float32)
    switch_fraction = np.full(expected2, np.nan, np.float32)
    min_active_occupancy = np.full(expected2, np.nan, np.float32)
    min_active_visits = np.zeros(expected2, np.uint32)

    run_count = np.zeros(expected2, np.uint32)
    singleton_run_fraction = np.full(expected2, np.nan, np.float32)
    median_plateau = np.full(expected2, np.nan, np.float32)
    max_plateau = np.zeros(expected2, np.uint32)

    state_sigma = np.full(expected3, np.nan, np.float32)
    min_separation_adu = np.full(expected2, np.nan, np.float32)
    min_separation_sigma = np.full(expected2, np.nan, np.float32)
    near_valley_fraction = np.full(expected2, np.nan, np.float32)
    direct_1_3_fraction = np.full(expected2, np.nan, np.float32)

    labels = [
        temperature_label(bin_table.iloc[b], b) for b in range(n_bins)
    ]

    for b, frames in enumerate(bin_frames):
        print(f"Temperature bin {b+1}/{n_bins}: {labels[b]}", flush=True)
        n_bin_frames = len(frames)
        for c0 in range(0, n_candidates, args.chunk_size):
            c1 = min(n_candidates, c0 + args.chunk_size)
            obs_chunk = np.asarray(observed[b, c0:c1])
            active_local = np.flatnonzero(obs_chunk >= 2)
            if active_local.size == 0:
                continue

            for local in active_local:
                c = c0 + int(local)
                active_states = int(observed[b, c])
                states = np.asarray(frame_state[frames, c], dtype=np.int16)
                qualities = np.asarray(quality[frames, c], dtype=np.int16)
                values = (
                    np.asarray(centered_raw[frames, c], dtype=np.float64)
                    / centered_scale
                )

                active_slice = slice(0, active_states)
                occ = np.asarray(occupancy[b, c, active_slice], float)
                vis = np.asarray(visits[b, c, active_slice], np.uint64)
                min_active_occupancy[b, c] = (
                    float(np.min(occ)) if occ.size else np.nan
                )
                min_active_visits[b, c] = (
                    int(np.min(vis)) if vis.size else 0
                )

                vp = int(valid_pair[b, c])
                rp = int(reliable_pair[b, c])
                reliable_pair_fraction[b, c] = rp / vp if vp > 0 else np.nan
                switch_fraction[b, c] = (
                    int(reliable_switch[b, c]) / rp if rp > 0 else np.nan
                )

                valid_quality_count = int(np.count_nonzero(qualities != Q_INVALID))
                near_valley_fraction[b, c] = (
                    float(np.count_nonzero(qualities == Q_NEAR_VALLEY))
                    / valid_quality_count
                    if valid_quality_count > 0 else np.nan
                )
                rsw = int(reliable_switch[b, c])
                direct_1_3_fraction[b, c] = (
                    int(direct13[b, c]) / rsw
                    if active_states == 3 and rsw > 0
                    else 0.0
                )

                run_states, lengths = run_lengths(states)
                if lengths.size:
                    run_count[b, c] = lengths.size
                    singleton_run_fraction[b, c] = np.mean(lengths == 1)
                    median_plateau[b, c] = np.median(lengths)
                    max_plateau[b, c] = int(np.max(lengths))

                centers = np.asarray(
                    state_center[b, c, active_slice], dtype=float
                )
                sigmas = []
                for state_number in range(1, active_states + 1):
                    state_mask = (
                        (states == state_number)
                        & (qualities != Q_INVALID)
                    )
                    sigma = robust_sigma(values[state_mask])
                    state_sigma[b, c, state_number - 1] = sigma
                    sigmas.append(sigma)

                sep_adu_values = []
                sep_sigma_values = []
                for s in range(active_states - 1):
                    c1v, c2v = centers[s], centers[s + 1]
                    sig1, sig2 = sigmas[s], sigmas[s + 1]
                    if not np.isfinite(c1v) or not np.isfinite(c2v):
                        continue
                    sep = abs(c2v - c1v)
                    sep_adu_values.append(sep)
                    denom = math.sqrt(
                        max(sig1, 0.0) ** 2 + max(sig2, 0.0) ** 2
                    )
                    if np.isfinite(denom) and denom > 0:
                        sep_sigma_values.append(sep / denom)

                if sep_adu_values:
                    min_separation_adu[b, c] = min(sep_adu_values)
                if sep_sigma_values:
                    min_separation_sigma[b, c] = min(sep_sigma_values)

                dec, reasons = classify_bin(
                    observed_state_count=active_states,
                    reliable_switch_count=int(reliable_switch[b, c]),
                    bidirectional_pair_count=int(bidirectional[b, c]),
                    reliable_pair_fraction=float(
                        reliable_pair_fraction[b, c]
                    ),
                    minimum_state_occupancy=float(
                        min_active_occupancy[b, c]
                    ),
                    minimum_state_visits=int(min_active_visits[b, c]),
                    maximum_plateau=int(max_plateau[b, c]),
                    median_plateau=float(median_plateau[b, c]),
                    singleton_fraction=float(
                        singleton_run_fraction[b, c]
                    ),
                    switch_fraction=float(switch_fraction[b, c]),
                    minimum_separation_sigma=float(
                        min_separation_sigma[b, c]
                    ),
                    minimum_separation_adu=float(
                        min_separation_adu[b, c]
                    ),
                    near_valley_fraction=float(near_valley_fraction[b, c]),
                    direct_1_3_fraction=float(direct_1_3_fraction[b, c]),
                    t=thresholds,
                )
                decision[b, c] = dec
                reason_mask[b, c] = reasons

    # Non-multistate entries get an explicit reason.
    reason_mask[np.asarray(observed) < 2] |= np.uint16(1 << 0)

    # Candidate-level aggregation.
    accepted_bin_count = np.count_nonzero(
        (decision == DECISION_ACCEPT)
        | (decision == DECISION_STRONG_ACCEPT),
        axis=0,
    ).astype(np.uint8)
    strong_accepted_bin_count = np.count_nonzero(
        decision == DECISION_STRONG_ACCEPT, axis=0
    ).astype(np.uint8)
    review_bin_count = np.count_nonzero(
        decision == DECISION_REVIEW, axis=0
    ).astype(np.uint8)
    multistate_bin_count = np.count_nonzero(
        np.asarray(observed) >= 2, axis=0
    ).astype(np.uint8)

    candidate_decision = np.zeros(n_candidates, np.uint8)
    candidate_decision[
        review_bin_count >= thresholds.candidate_min_review_bins
    ] = DECISION_REVIEW
    candidate_decision[
        accepted_bin_count >= thresholds.candidate_min_accepted_bins
    ] = DECISION_ACCEPT
    candidate_decision[
        strong_accepted_bin_count >= thresholds.candidate_min_accepted_bins
    ] = DECISION_STRONG_ACCEPT

    # Save arrays.
    np.save(output_dir / "rts_bin_decision_uint8.npy", decision)
    np.save(output_dir / "rts_bin_reason_mask_uint16.npy", reason_mask)
    np.save(
        output_dir / "candidate_rts_decision_uint8.npy",
        candidate_decision,
    )
    np.save(
        output_dir / "reliable_pair_fraction_float32.npy",
        reliable_pair_fraction,
    )
    np.save(
        output_dir / "switch_fraction_float32.npy",
        switch_fraction,
    )
    np.save(
        output_dir / "minimum_active_state_occupancy_float32.npy",
        min_active_occupancy,
    )
    np.save(
        output_dir / "minimum_active_state_visits_uint32.npy",
        min_active_visits,
    )
    np.save(output_dir / "run_count_uint32.npy", run_count)
    np.save(
        output_dir / "singleton_run_fraction_float32.npy",
        singleton_run_fraction,
    )
    np.save(
        output_dir / "median_plateau_frame_float32.npy",
        median_plateau,
    )
    np.save(
        output_dir / "maximum_plateau_frame_uint32.npy",
        max_plateau,
    )
    np.save(
        output_dir / "within_state_sigma_ADU_float32.npy",
        state_sigma,
    )
    np.save(
        output_dir / "minimum_state_separation_ADU_float32.npy",
        min_separation_adu,
    )
    np.save(
        output_dir / "minimum_state_separation_sigma_float32.npy",
        min_separation_sigma,
    )
    np.save(output_dir / "near_valley_fraction_float32.npy", near_valley_fraction)
    np.save(output_dir / "direct_1_3_fraction_float32.npy", direct_1_3_fraction)

    # Candidate summary table.
    candidate_summary = catalog.copy()
    candidate_summary.insert(0, "candidate_index", np.arange(n_candidates))
    candidate_summary["decision_code"] = candidate_decision
    candidate_summary["decision"] = np.choose(
        candidate_decision,
        ["reject", "review", "accept", "strong_accept"],
    )
    candidate_summary["multistate_bin_count"] = multistate_bin_count
    candidate_summary["review_bin_count"] = review_bin_count
    candidate_summary["accepted_bin_count"] = accepted_bin_count
    candidate_summary["strong_accepted_bin_count"] = strong_accepted_bin_count
    candidate_summary["acceptance_fraction_of_multistate_bins"] = np.divide(
        accepted_bin_count,
        multistate_bin_count,
        out=np.zeros(n_candidates, dtype=np.float32),
        where=multistate_bin_count > 0,
    )
    candidate_summary["maximum_reliable_switch_count"] = np.max(
        reliable_switch, axis=0
    )
    candidate_summary["maximum_plateau_frames"] = np.max(
        max_plateau, axis=0
    )
    candidate_summary["best_separation_sigma"] = np.nanmax(
        min_separation_sigma, axis=0
    )
    candidate_summary.to_csv(
        output_dir / "candidate_rts_decision.csv", index=False
    )

    # Long dictionary and diagnostic table.
    long_rows = []
    dictionary_rows = []
    for b in range(n_bins):
        for c in range(n_candidates):
            dec = int(decision[b, c])
            if dec == DECISION_REJECT and not args.save_rejected_bin_table:
                continue
            n_state = int(observed[b, c])
            row = {
                "candidate_index": c,
                "temperature_bin_index": b,
                "temperature_bin_label": labels[b],
                "decision_code": dec,
                "decision": ("reject", "review", "accept", "strong_accept")[dec],
                "reason_mask": int(reason_mask[b, c]),
                "reason_text": reason_text(int(reason_mask[b, c])),
                "observed_state_count": n_state,
                "reliable_switch_count": int(reliable_switch[b, c]),
                "bidirectional_pair_count": int(bidirectional[b, c]),
                "direct_1_3_switch_count": int(direct13[b, c]),
                "reliable_pair_fraction": float(
                    reliable_pair_fraction[b, c]
                ),
                "switch_fraction": float(switch_fraction[b, c]),
                "minimum_active_state_occupancy": float(
                    min_active_occupancy[b, c]
                ),
                "minimum_active_state_visits": int(
                    min_active_visits[b, c]
                ),
                "run_count": int(run_count[b, c]),
                "singleton_run_fraction": float(
                    singleton_run_fraction[b, c]
                ),
                "median_plateau_frames": float(median_plateau[b, c]),
                "maximum_plateau_frames": int(max_plateau[b, c]),
                "minimum_state_separation_ADU": float(
                    min_separation_adu[b, c]
                ),
                "minimum_state_separation_sigma": float(
                    min_separation_sigma[b, c]
                ),
                "near_valley_fraction": float(near_valley_fraction[b, c]),
                "direct_1_3_fraction": float(direct_1_3_fraction[b, c]),
            }
            for s in range(MAX_STATES):
                row[f"state_{s+1}_center_ADU"] = float(
                    state_center[b, c, s]
                )
                row[f"state_{s+1}_occupancy"] = float(
                    occupancy[b, c, s]
                )
                row[f"state_{s+1}_visit_count"] = int(
                    visits[b, c, s]
                )
                row[f"state_{s+1}_sigma_ADU"] = float(
                    state_sigma[b, c, s]
                )
            long_rows.append(row)

            if dec in (DECISION_ACCEPT, DECISION_STRONG_ACCEPT):
                drow = dict(row)
                # Add detector coordinates where available.
                for col in catalog.columns:
                    drow[col] = catalog.iloc[c][col]
                dictionary_rows.append(drow)

    pd.DataFrame(long_rows).to_csv(
        output_dir / "candidate_bin_diagnostics.csv", index=False
    )
    pd.DataFrame(dictionary_rows).to_csv(
        output_dir / "rts_temperature_dictionary.csv", index=False
    )

    # Failure-reason counts among multistate entries.
    reason_counts = {}
    multistate_mask = np.asarray(observed) >= 2
    for bit, name in REASON_NAMES.items():
        reason_counts[name] = int(
            np.count_nonzero(
                multistate_mask & ((reason_mask & (1 << bit)) != 0)
            )
        )

    validation = {
        "all_accepted_are_multistate": bool(
            np.all(np.asarray(observed)[(decision == DECISION_ACCEPT) | (decision == DECISION_STRONG_ACCEPT)] >= 2)
        ),
        "accepted_reason_mask_is_zero": bool(
            np.all(reason_mask[(decision == DECISION_ACCEPT) | (decision == DECISION_STRONG_ACCEPT)] == 0)
        ),
        "candidate_accept_count_matches": bool(
            np.count_nonzero(
                (candidate_decision == DECISION_ACCEPT)
                | (candidate_decision == DECISION_STRONG_ACCEPT)
            )
            == np.count_nonzero(
                accepted_bin_count
                >= thresholds.candidate_min_accepted_bins
            )
        ),
        "decision_values_valid": bool(
            np.all(np.isin(decision, [0, 1, 2, 3]))
        ),
    }
    validation_passed = all(validation.values())

    elapsed = time.perf_counter() - started
    summary = {
        "step": "09_classify_rts_and_generate_dictionary",
        "script_version": SCRIPT_VERSION,
        "validation_passed": validation_passed,
        "responsibility": (
            "RTS classification and temperature-dependent dictionary "
            "generation; no image correction"
        ),
        "thresholds": asdict(thresholds),
        "frame_count": n_frames,
        "candidate_count": n_candidates,
        "temperature_bin_count": n_bins,
        "candidate_counts": {
            "reject": int(np.count_nonzero(
                candidate_decision == DECISION_REJECT
            )),
            "review": int(np.count_nonzero(
                candidate_decision == DECISION_REVIEW
            )),
            "accept": int(np.count_nonzero(
                candidate_decision == DECISION_ACCEPT
            )),
            "strong_accept": int(np.count_nonzero(
                candidate_decision == DECISION_STRONG_ACCEPT
            )),
        },
        "candidate_bin_counts": {
            "reject": int(np.count_nonzero(decision == DECISION_REJECT)),
            "review": int(np.count_nonzero(decision == DECISION_REVIEW)),
            "accept": int(np.count_nonzero(decision == DECISION_ACCEPT)),
            "strong_accept": int(np.count_nonzero(
                decision == DECISION_STRONG_ACCEPT
            )),
        },
        "failure_reason_counts_among_multistate_bins": reason_counts,
        "validation": validation,
        "outputs": {
            "candidate_decision": str(
                output_dir / "candidate_rts_decision.csv"
            ),
            "candidate_bin_diagnostics": str(
                output_dir / "candidate_bin_diagnostics.csv"
            ),
            "temperature_dictionary": str(
                output_dir / "rts_temperature_dictionary.csv"
            ),
            "bin_decision_array": str(
                output_dir / "rts_bin_decision_uint8.npy"
            ),
            "bin_reason_mask_array": str(
                output_dir / "rts_bin_reason_mask_uint16.npy"
            ),
        },
        "elapsed_seconds": elapsed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary["candidate_counts"], indent=2))
    print(json.dumps(summary["candidate_bin_counts"], indent=2))
    print(f"validation_passed = {validation_passed}")
    print(f"elapsed = {elapsed:.1f} s")
    return 0 if validation_passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
