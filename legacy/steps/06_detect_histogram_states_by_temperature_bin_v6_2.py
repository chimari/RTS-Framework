#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_detect_histogram_states_by_temperature_bin.py

Step06 v6.2 of the IMX811 RTS pipeline.

For every candidate pixel and temperature bin:
  centered residual -> exact x4 histogram -> Gaussian smoothing -> peak
  detection/filtering -> final state count and state boundaries.

This step does not assign individual frames and does not calculate transitions.

New in v6.2
-----------
- Saves valley_boundary_ADU_float32.npy.
- The saved boundary is the first minimum of the smoothed histogram between
  each adjacent pair of final retained peaks.
- Step07 can therefore use exactly the same boundaries without rebuilding
  histograms.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy.ndimage import gaussian_filter1d

from common.cli import add_common_arguments, validate_common_arguments
from common.io import prepare_output_dir, sha256_file, write_json
from common.version import PIPELINE_VERSION

SCRIPT_VERSION = "6.2.0"


@dataclass(frozen=True)
class Peak:
    index: int
    height: float
    height_fraction: float
    prominence: float
    prominence_fraction: float
    occupancy: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect histogram states and save their valley boundaries."
    )
    p.add_argument("--centered-dir", type=Path,
                   default=Path("05_dataset_centered_timeseries"))
    p.add_argument("--temperature-bin-tolerance", type=float, default=1.5)
    p.add_argument("--candidate-block", type=int, default=512)
    p.add_argument(
        "--max-states", type=int, default=3,
        help="Maximum reported state count. Default: 3."
    )
    p.add_argument("--smooth-sigma-ADU", type=float, default=0.50)
    p.add_argument("--min-peak-separation-ADU", type=float, default=1.50)
    p.add_argument("--min-peak-height-fraction", type=float, default=0.010)
    p.add_argument("--min-prominence-fraction", type=float, default=0.010)
    p.add_argument("--min-state-occupancy", type=float, default=0.04)
    p.add_argument("--max-valley-ratio", type=float, default=0.82)
    p.add_argument("--min-frames", type=int, default=100)
    p.add_argument("--histogram-margin-ADU", type=float, default=1.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--hash-inputs", action="store_true")
    p.add_argument("--flush-every-blocks", type=int, default=5)
    add_common_arguments(p, output_default="06_histogram_state_detection")
    return p.parse_args()


def find_centered_residual(directory: Path, summary: dict) -> Path:
    for key in ("centered_residual", "centered_residual_output"):
        value = summary.get(key)
        if value:
            p = Path(str(value))
            if p.is_file():
                return p
            local = directory / p.name
            if local.is_file():
                return local
    found = sorted(directory.glob("centered_residual_x4_*.npy"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected one centered_residual_x4_*.npy in {directory}; "
            f"found {len(found)}"
        )
    return found[0]


def build_temperature_bins(
    dataset_index: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, list[list[int]]]:
    required = {
        "dataset_index", "dataset", "frame_start", "frame_stop_exclusive",
        "frame_count", "temperature_mean_C",
    }
    missing = required - set(dataset_index.columns)
    if missing:
        raise KeyError(f"dataset_index.csv missing columns: {sorted(missing)}")

    work = dataset_index.copy()
    work["temperature_mean_C"] = pd.to_numeric(
        work["temperature_mean_C"], errors="raise"
    )
    work = work.sort_values(
        "temperature_mean_C", ascending=False
    ).reset_index(drop=True)

    groups: list[list[int]] = []
    current: list[int] = []
    tmin = tmax = math.nan
    for row in work.itertuples(index=False):
        d = int(row.dataset_index)
        t = float(row.temperature_mean_C)
        if not current:
            current, tmin, tmax = [d], t, t
        elif max(tmax, t) - min(tmin, t) <= tolerance:
            current.append(d)
            tmin, tmax = min(tmin, t), max(tmax, t)
        else:
            groups.append(current)
            current, tmin, tmax = [d], t, t
    if current:
        groups.append(current)

    by_index = dataset_index.set_index("dataset_index")
    rows: list[dict] = []
    frame_lists: list[list[int]] = []

    for b, datasets in enumerate(groups):
        names, temperatures, counts, frames = [], [], [], []
        for d in datasets:
            row = by_index.loc[d]
            names.append(str(row["dataset"]))
            temperatures.append(float(row["temperature_mean_C"]))
            counts.append(int(row["frame_count"]))
            frames.extend(
                range(
                    int(row["frame_start"]),
                    int(row["frame_stop_exclusive"]),
                )
            )
        weights = np.asarray(counts, dtype=float)
        mean_temperature = float(
            np.average(temperatures, weights=weights)
        )
        frame_lists.append(frames)
        rows.append({
            "temperature_bin_index": b,
            "temperature_bin_label": f"Tbin_{b:02d}_{mean_temperature:+.2f}C",
            "temperature_mean_C": mean_temperature,
            "temperature_min_C": float(min(temperatures)),
            "temperature_max_C": float(max(temperatures)),
            "dataset_count": len(datasets),
            "frame_count": len(frames),
            "dataset_indices": ";".join(map(str, datasets)),
            "datasets": ";".join(names),
        })
    return pd.DataFrame(rows), frame_lists


def local_maxima(values: np.ndarray) -> np.ndarray:
    n = values.size
    if n == 1:
        return np.array([0], dtype=np.int64)
    maxima: list[int] = []
    if values[0] > values[1]:
        maxima.append(0)
    for i in range(1, n - 1):
        if values[i] >= values[i - 1] and values[i] > values[i + 1]:
            maxima.append(i)
    if values[-1] >= values[-2]:
        maxima.append(n - 1)
    if not maxima:
        maxima = [int(np.argmax(values))]
    return np.asarray(maxima, dtype=np.int64)


def peak_prominence_simple(values: np.ndarray, peak: int) -> float:
    left_min = float(np.min(values[:peak + 1]))
    right_min = float(np.min(values[peak:]))
    return max(0.0, float(values[peak]) - max(left_min, right_min))


def valley_indices(values: np.ndarray, peaks_sorted: list[int]) -> list[int]:
    """Return first minimum between each adjacent peak."""
    result: list[int] = []
    for left, right in zip(peaks_sorted[:-1], peaks_sorted[1:]):
        if right <= left:
            raise ValueError("Peak indices must be strictly increasing")
        result.append(
            left + int(np.argmin(values[left:right + 1]))
        )
    return result


def basin_boundaries(
    values: np.ndarray,
    peaks_sorted: list[int],
) -> list[tuple[int, int]]:
    if len(peaks_sorted) == 1:
        return [(0, values.size)]
    cuts = valley_indices(values, peaks_sorted)
    bounds: list[tuple[int, int]] = []
    start = 0
    for cut in cuts:
        stop = max(start + 1, cut + 1)
        bounds.append((start, stop))
        start = stop
    bounds.append((start, values.size))
    return bounds


def enforce_separation(
    peaks: list[Peak],
    minimum_separation_bins: int,
) -> list[Peak]:
    chosen: list[Peak] = []
    for peak in sorted(
        peaks,
        key=lambda p: (p.prominence, p.height),
        reverse=True,
    ):
        if all(
            abs(peak.index - other.index) >= minimum_separation_bins
            for other in chosen
        ):
            chosen.append(peak)
    return sorted(chosen, key=lambda p: p.index)


def adjacent_valley_metrics(
    smoothed: np.ndarray,
    peaks: list[Peak],
) -> tuple[list[int], list[float]]:
    if len(peaks) <= 1:
        return [], []
    positions = valley_indices(smoothed, [p.index for p in peaks])
    ratios: list[float] = []
    for position, left, right in zip(
        positions, peaks[:-1], peaks[1:]
    ):
        denominator = max(min(left.height, right.height), 1e-12)
        ratios.append(float(smoothed[position]) / denominator)
    return positions, ratios


def remove_shallow_splits(
    smoothed: np.ndarray,
    peaks: list[Peak],
    maximum_valley_ratio: float,
) -> list[Peak]:
    peaks = sorted(peaks, key=lambda p: p.index)
    while len(peaks) > 1:
        _, ratios = adjacent_valley_metrics(smoothed, peaks)
        worst = int(np.argmax(ratios))
        if ratios[worst] <= maximum_valley_ratio:
            break
        left, right = peaks[worst], peaks[worst + 1]
        drop = (
            worst
            if (left.prominence, left.height)
            < (right.prominence, right.height)
            else worst + 1
        )
        peaks.pop(drop)
    return peaks


def analyze_histogram(
    values_x4: np.ndarray,
    histogram_min_x4: int,
    histogram_max_x4: int,
    smooth_sigma_bins: float,
    minimum_separation_bins: int,
    minimum_height_fraction: float,
    minimum_prominence_fraction: float,
    minimum_occupancy: float,
    maximum_valley_ratio: float,
    maximum_states: int,
) -> tuple[
    int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    n = int(values_x4.size)
    n_histogram = histogram_max_x4 - histogram_min_x4 + 1
    indices = values_x4.astype(np.int64) - histogram_min_x4
    if np.any(indices < 0) or np.any(indices >= n_histogram):
        raise ValueError("Residual lies outside global histogram range")

    counts = np.bincount(
        indices, minlength=n_histogram
    ).astype(np.float64)
    smoothed = (
        gaussian_filter1d(
            counts, smooth_sigma_bins, mode="nearest"
        )
        if smooth_sigma_bins > 0
        else counts
    )

    raw_peaks: list[Peak] = []
    for p in local_maxima(smoothed):
        height = float(smoothed[p])
        prominence = peak_prominence_simple(smoothed, int(p))
        raw_peaks.append(Peak(
            index=int(p),
            height=height,
            height_fraction=height / n,
            prominence=prominence,
            prominence_fraction=prominence / n,
            occupancy=0.0,
        ))

    filtered = [
        p for p in raw_peaks
        if p.height_fraction >= minimum_height_fraction
        and p.prominence_fraction >= minimum_prominence_fraction
    ]
    if not filtered:
        p = int(np.argmax(smoothed))
        prominence = peak_prominence_simple(smoothed, p)
        filtered = [Peak(
            p, float(smoothed[p]), float(smoothed[p] / n),
            prominence, float(prominence / n), 1.0,
        )]

    filtered = enforce_separation(
        filtered, minimum_separation_bins
    )
    if len(filtered) > maximum_states * 3:
        filtered = sorted(
            filtered,
            key=lambda p: (p.prominence, p.height),
            reverse=True,
        )[:maximum_states * 3]
        filtered.sort(key=lambda p: p.index)

    bounds = basin_boundaries(
        smoothed, [p.index for p in filtered]
    )
    occupied: list[Peak] = []
    for peak, (low, high) in zip(filtered, bounds):
        occupancy = float(counts[low:high].sum() / n)
        if occupancy >= minimum_occupancy:
            occupied.append(Peak(
                peak.index, peak.height, peak.height_fraction,
                peak.prominence, peak.prominence_fraction, occupancy,
            ))

    if not occupied:
        strongest = max(
            filtered, key=lambda p: (p.prominence, p.height)
        )
        occupied = [Peak(
            strongest.index, strongest.height,
            strongest.height_fraction, strongest.prominence,
            strongest.prominence_fraction, 1.0,
        )]

    occupied = remove_shallow_splits(
        smoothed, occupied, maximum_valley_ratio
    )

    if len(occupied) > maximum_states:
        occupied = sorted(
            occupied,
            key=lambda p: (p.prominence, p.height),
            reverse=True,
        )[:maximum_states]
        occupied.sort(key=lambda p: p.index)
        occupied = remove_shallow_splits(
            smoothed, occupied, maximum_valley_ratio
        )

    final_bounds = basin_boundaries(
        smoothed, [p.index for p in occupied]
    )
    final: list[Peak] = []
    for peak, (low, high) in zip(occupied, final_bounds):
        final.append(Peak(
            peak.index, peak.height, peak.height_fraction,
            peak.prominence, peak.prominence_fraction,
            float(counts[low:high].sum() / n),
        ))

    final = remove_shallow_splits(
        smoothed, final, maximum_valley_ratio
    )
    final_valley_indices, final_valley_ratios = (
        adjacent_valley_metrics(smoothed, final)
    )

    centers = np.full(maximum_states, np.nan, dtype=np.float32)
    occupancies = np.full(maximum_states, np.nan, dtype=np.float32)
    prominences = np.full(maximum_states, np.nan, dtype=np.float32)
    heights = np.full(maximum_states, np.nan, dtype=np.float32)
    valley_ratios = np.full(
        max(0, maximum_states - 1), np.nan, dtype=np.float32
    )
    valley_boundaries = np.full(
        max(0, maximum_states - 1), np.nan, dtype=np.float32
    )

    for j, peak in enumerate(final):
        centers[j] = (histogram_min_x4 + peak.index) / 4.0
        occupancies[j] = peak.occupancy
        prominences[j] = peak.prominence_fraction
        heights[j] = peak.height_fraction

    for j, (position, ratio) in enumerate(
        zip(final_valley_indices, final_valley_ratios)
    ):
        valley_boundaries[j] = (
            histogram_min_x4 + position
        ) / 4.0
        valley_ratios[j] = ratio

    return (
        len(final), centers, occupancies, prominences,
        heights, valley_ratios, valley_boundaries,
    )


def checkpoint_payload(
    shape: tuple[int, int],
    maximum_states: int,
    completed_bin: int,
    completed_stop: int,
) -> dict:
    return {
        "step": "06_detect_histogram_states_by_temperature_bin",
        "script_version": SCRIPT_VERSION,
        "shape": list(shape),
        "max_states": maximum_states,
        "completed_temperature_bin_index": completed_bin,
        "completed_candidate_stop": completed_stop,
    }


def main() -> int:
    args = parse_args()
    validate_common_arguments(args)

    if args.temperature_bin_tolerance < 0:
        raise ValueError("--temperature-bin-tolerance must be non-negative")
    if args.candidate_block <= 0 or args.max_states < 1:
        raise ValueError("--candidate-block and --max-states must be positive")
    if args.max_states > 3:
        raise ValueError("Step06 v6.2 supports at most 3 states")
    if args.min_frames < 1 or args.flush_every_blocks < 1:
        raise ValueError("--min-frames and --flush-every-blocks must be positive")
    if args.smooth_sigma_ADU < 0 or args.min_peak_separation_ADU <= 0:
        raise ValueError("Invalid smoothing or peak-separation parameter")
    for name in (
        "min_peak_height_fraction",
        "min_prominence_fraction",
        "min_state_occupancy",
        "max_valley_ratio",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")

    started = time.perf_counter()
    input_directory = args.centered_dir.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()

    step05_summary_path = input_directory / "summary.json"
    if not step05_summary_path.is_file():
        raise FileNotFoundError(step05_summary_path)
    step05_summary = json.loads(
        step05_summary_path.read_text(encoding="utf-8")
    )
    if not step05_summary.get("validation_passed", False):
        raise ValueError("Step05 validation_passed is not true")

    residual_path = find_centered_residual(
        input_directory, step05_summary
    )
    dataset_index_path = input_directory / "dataset_index.csv"
    candidate_catalog_path = input_directory / "candidate_catalog.csv"
    frame_metadata_path = input_directory / "frame_metadata.csv"

    for path in (
        residual_path, dataset_index_path,
        candidate_catalog_path, frame_metadata_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    residual = np.load(residual_path, mmap_mode="r")
    if residual.ndim != 2 or residual.dtype.kind != "i":
        raise ValueError("Centered residual must be a signed 2-D array")
    n_frames, n_candidates = map(int, residual.shape)

    candidate_catalog = pd.read_csv(candidate_catalog_path)
    dataset_index = pd.read_csv(dataset_index_path)
    frame_metadata = pd.read_csv(frame_metadata_path)
    if len(candidate_catalog) != n_candidates:
        raise ValueError("Candidate catalog row count mismatch")
    if len(frame_metadata) != n_frames:
        raise ValueError("Frame metadata row count mismatch")

    bin_index, frame_lists = build_temperature_bins(
        dataset_index, args.temperature_bin_tolerance
    )
    n_bins = len(bin_index)
    short_bins = [
        i for i, rows in enumerate(frame_lists)
        if len(rows) < args.min_frames
    ]
    if short_bins:
        raise ValueError(
            f"Temperature bins below --min-frames: {short_bins}"
        )

    observed_min = int(np.min(residual))
    observed_max = int(np.max(residual))
    margin_x4 = int(math.ceil(args.histogram_margin_ADU * 4.0))
    histogram_min = observed_min - margin_x4
    histogram_max = observed_max + margin_x4
    smooth_sigma_bins = args.smooth_sigma_ADU * 4.0
    minimum_separation_bins = max(
        1, int(math.ceil(args.min_peak_separation_ADU * 4.0))
    )

    shape2 = (n_bins, n_candidates)
    shape3 = (n_bins, n_candidates, args.max_states)
    valley_shape = (
        n_bins, n_candidates, max(0, args.max_states - 1)
    )

    names = {
        "state": "state_count_uint8.npy",
        "center": "peak_center_ADU_float32.npy",
        "occupancy": "peak_occupancy_float32.npy",
        "prominence": "peak_prominence_fraction_float32.npy",
        "height": "peak_height_fraction_float32.npy",
        "valley_ratio": "adjacent_valley_ratio_float32.npy",
        "valley_boundary": "valley_boundary_ADU_float32.npy",
    }

    if args.resume:
        checkpoint = json.loads(
            (output_directory / "checkpoint.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            checkpoint.get("shape") != list(shape2)
            or int(checkpoint.get("max_states", -1)) != args.max_states
        ):
            raise ValueError("Checkpoint does not match current inputs")
        resume_bin = int(
            checkpoint.get("completed_temperature_bin_index", 0)
        )
        resume_stop = int(
            checkpoint.get("completed_candidate_stop", 0)
        )
        mode = "r+"
    else:
        prepare_output_dir(
            output_directory, overwrite=args.overwrite
        )
        resume_bin = resume_stop = 0
        mode = "w+"

    state_count = open_memmap(
        output_directory / names["state"],
        mode=mode, dtype=np.uint8, shape=shape2,
    )
    peak_center = open_memmap(
        output_directory / names["center"],
        mode=mode, dtype=np.float32, shape=shape3,
    )
    peak_occupancy = open_memmap(
        output_directory / names["occupancy"],
        mode=mode, dtype=np.float32, shape=shape3,
    )
    peak_prominence = open_memmap(
        output_directory / names["prominence"],
        mode=mode, dtype=np.float32, shape=shape3,
    )
    peak_height = open_memmap(
        output_directory / names["height"],
        mode=mode, dtype=np.float32, shape=shape3,
    )
    valley_ratio = open_memmap(
        output_directory / names["valley_ratio"],
        mode=mode, dtype=np.float32, shape=valley_shape,
    )
    valley_boundary = open_memmap(
        output_directory / names["valley_boundary"],
        mode=mode, dtype=np.float32, shape=valley_shape,
    )
    arrays = (
        state_count, peak_center, peak_occupancy,
        peak_prominence, peak_height,
        valley_ratio, valley_boundary,
    )

    if not args.resume:
        state_count[:] = 0
        for array in arrays[1:]:
            array[:] = np.nan
        shutil.copy2(
            candidate_catalog_path,
            output_directory / "candidate_catalog.csv",
        )
        bin_index.to_csv(
            output_directory / "temperature_bin_index.csv",
            index=False,
        )

    blocks_per_bin = math.ceil(
        n_candidates / args.candidate_block
    )
    total_blocks = n_bins * blocks_per_bin
    completed_blocks = (
        resume_bin * blocks_per_bin
        + resume_stop // args.candidate_block
    )
    blocks_since_flush = 0

    print(
        f"Step06 v6.2: {n_candidates:,} candidates x "
        f"{n_bins} temperature bins"
    )
    print(
        f"Histogram x4 range [{histogram_min}, {histogram_max}] "
        f"({histogram_min/4:.2f} to {histogram_max/4:.2f} ADU)"
    )

    for b, row_list in enumerate(frame_lists):
        if b < resume_bin:
            continue
        candidate_start = resume_stop if b == resume_bin else 0
        rows = np.asarray(row_list, dtype=np.int64)
        label = str(bin_index.iloc[b]["temperature_bin_label"])
        print(
            f"Temperature bin {b+1}/{n_bins}: "
            f"{label}, {len(rows)} frames"
        )

        for c0 in range(
            candidate_start, n_candidates, args.candidate_block
        ):
            c1 = min(c0 + args.candidate_block, n_candidates)
            columns = np.arange(c0, c1, dtype=np.int64)
            block = np.asarray(residual[np.ix_(rows, columns)])

            for j in range(block.shape[1]):
                result = analyze_histogram(
                    block[:, j],
                    histogram_min,
                    histogram_max,
                    smooth_sigma_bins,
                    minimum_separation_bins,
                    args.min_peak_height_fraction,
                    args.min_prominence_fraction,
                    args.min_state_occupancy,
                    args.max_valley_ratio,
                    args.max_states,
                )
                candidate = c0 + j
                state_count[b, candidate] = result[0]
                peak_center[b, candidate, :] = result[1]
                peak_occupancy[b, candidate, :] = result[2]
                peak_prominence[b, candidate, :] = result[3]
                peak_height[b, candidate, :] = result[4]
                if args.max_states > 1:
                    valley_ratio[b, candidate, :] = result[5]
                    valley_boundary[b, candidate, :] = result[6]

            completed_blocks += 1
            blocks_since_flush += 1
            if (
                blocks_since_flush >= args.flush_every_blocks
                or c1 == n_candidates
            ):
                for array in arrays:
                    array.flush()
                next_bin, next_stop = (
                    (b + 1, 0)
                    if c1 == n_candidates
                    else (b, c1)
                )
                write_json(
                    output_directory / "checkpoint.json",
                    checkpoint_payload(
                        shape2, args.max_states,
                        next_bin, next_stop,
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
                    if fraction > 0 else math.nan
                )
                print(
                    f"  candidates {c1:,}/{n_candidates:,}; "
                    f"blocks {completed_blocks}/{total_blocks}; "
                    f"elapsed {elapsed:.1f}s; ETA {eta:.1f}s"
                )
        resume_stop = 0

    for array in arrays:
        array.flush()

    state_array = np.asarray(state_count)
    bin_rows: list[dict] = []
    for b in range(n_bins):
        row = bin_index.iloc[b].to_dict()
        counts = np.bincount(
            state_array[b].astype(np.int64),
            minlength=args.max_states + 1,
        )
        for k in range(args.max_states + 1):
            row[f"state_{k}_count"] = int(counts[k])
            row[f"state_{k}_fraction"] = float(
                counts[k] / n_candidates
            )
        row["multistate_count"] = int(
            np.count_nonzero(state_array[b] >= 2)
        )
        row["multistate_fraction"] = float(
            np.mean(state_array[b] >= 2)
        )
        bin_rows.append(row)
    pd.DataFrame(bin_rows).to_csv(
        output_directory / "temperature_bin_state_summary.csv",
        index=False,
    )

    candidate_summary = candidate_catalog.copy()
    candidate_summary["max_state_count"] = state_array.max(axis=0)
    candidate_summary["multistate_temperature_bin_count"] = (
        np.count_nonzero(state_array >= 2, axis=0)
    )
    candidate_summary["three_state_temperature_bin_count"] = (
        np.count_nonzero(state_array == 3, axis=0)
    )
    candidate_summary["best_temperature_bin_index"] = np.argmax(
        state_array, axis=0
    )
    candidate_summary.to_csv(
        output_directory / "candidate_state_summary.csv",
        index=False,
    )

    # Boundaries must be finite and ordered only where they are required.
    boundary_array = np.asarray(valley_boundary)
    centers_array = np.asarray(peak_center)
    boundary_ok = True
    for k in range(2, args.max_states + 1):
        mask = state_array == k
        if np.any(mask):
            required_boundaries = boundary_array[:, :, :k - 1][mask]
            required_centers = centers_array[:, :, :k][mask]
            boundary_ok = boundary_ok and bool(
                np.all(np.isfinite(required_boundaries))
            )
            boundary_ok = boundary_ok and bool(
                np.all(np.diff(required_boundaries, axis=1) > 0)
            )
            boundary_ok = boundary_ok and bool(
                np.all(required_boundaries > required_centers[:, :-1])
                and np.all(required_boundaries < required_centers[:, 1:])
            )

    validation_passed = bool(
        state_array.shape == shape2
        and np.all(
            (state_array >= 1)
            & (state_array <= args.max_states)
        )
        and np.all(np.isfinite(centers_array[:, :, 0]))
        and boundary_ok
    )

    elapsed = time.perf_counter() - started
    summary = {
        "step": "06_detect_histogram_states_by_temperature_bin",
        "script_version": SCRIPT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "validation_passed": validation_passed,
        "centered_dir": str(input_directory),
        "centered_residual": str(residual_path),
        "array_shape": [n_frames, n_candidates],
        "candidate_count": n_candidates,
        "temperature_bin_count": n_bins,
        "temperature_bin_tolerance_C": args.temperature_bin_tolerance,
        "max_states": args.max_states,
        "histogram_physical_bin_width_ADU": 0.25,
        "histogram_x4_min": histogram_min,
        "histogram_x4_max": histogram_max,
        "observed_residual_x4_min": observed_min,
        "observed_residual_x4_max": observed_max,
        "smooth_sigma_ADU": args.smooth_sigma_ADU,
        "min_peak_separation_ADU": args.min_peak_separation_ADU,
        "min_peak_height_fraction": args.min_peak_height_fraction,
        "min_prominence_fraction": args.min_prominence_fraction,
        "min_state_occupancy": args.min_state_occupancy,
        "max_valley_ratio": args.max_valley_ratio,
        "min_frames": args.min_frames,
        "algorithm": (
            "exact x4 histogram -> Gaussian smoothing -> local maxima -> "
            "prominence/height/occupancy filters -> minimum separation -> "
            "adjacent-valley merge -> save final adjacent valley positions"
        ),
        "state_count_output": str(
            output_directory / names["state"]
        ),
        "peak_center_output": str(
            output_directory / names["center"]
        ),
        "peak_occupancy_output": str(
            output_directory / names["occupancy"]
        ),
        "adjacent_valley_ratio_output": str(
            output_directory / names["valley_ratio"]
        ),
        "valley_boundary_output": str(
            output_directory / names["valley_boundary"]
        ),
        "valley_boundary_definition": (
            "first minimum of the Step06 smoothed histogram between "
            "each adjacent pair of final retained peaks"
        ),
        "valley_boundary_validation_passed": bool(boundary_ok),
        "multistate_candidate_count_any_temperature": int(
            np.count_nonzero(np.any(state_array >= 2, axis=0))
        ),
        "three_state_candidate_count_any_temperature": int(
            np.count_nonzero(np.any(state_array == 3, axis=0))
        ),
        "elapsed_seconds": elapsed,
    }
    write_json(output_directory / "summary.json", summary)

    manifest = {
        "inputs": {
            "centered_residual": str(residual_path),
            "dataset_index": str(dataset_index_path),
            "candidate_catalog": str(candidate_catalog_path),
            "frame_metadata": str(frame_metadata_path),
            "step05_summary": str(step05_summary_path),
        },
        "outputs": [
            *names.values(),
            "temperature_bin_index.csv",
            "candidate_catalog.csv",
            "candidate_state_summary.csv",
            "temperature_bin_state_summary.csv",
            "checkpoint.json",
            "summary.json",
            "manifest.json",
        ],
    }
    if args.hash_inputs:
        manifest["input_sha256"] = {
            key: sha256_file(Path(value))
            for key, value in manifest["inputs"].items()
        }
    write_json(output_directory / "manifest.json", manifest)

    print(f"PASS: Step06 v6.2 completed in {elapsed:.1f} s")
    print(
        "  any-temperature multistate candidates: "
        f"{summary['multistate_candidate_count_any_temperature']:,}"
    )
    print(
        "  any-temperature 3-state candidates: "
        f"{summary['three_state_candidate_count_any_temperature']:,}"
    )
    return 0 if validation_passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exception:
        print(f"ERROR: {exception}", file=sys.stderr)
        raise
