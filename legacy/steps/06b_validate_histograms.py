#!/usr/bin/env python3
"""
06b_validate_histograms.py

Visual validation for Step06 histogram state detection.

For each selected candidate pixel, create one diagnostic figure containing:

  * top: all-temperature combined histogram
  * middle: one histogram for every temperature bin (normally 3 x 3)
  * bottom: centered-residual time series, segmented by temperature bin

All histogram panels use the same ADU x-axis and the same normalized y-axis.
The temperature-bin panels overlay the exact Step06 peak centers, occupancies,
prominences, and valley positions.  The combined panel is re-analysed with the
same parameters as Step06 and is intended only as a visual summary; it does not
alter the Step06 results.

Selection examples
------------------
Single extraction-order candidate column (zero based):

  python3 06b_validate_histograms.py \
      --centered-dir 05_test_roi --state-dir 06_test_roi \
      --candidate 12345 --output-dir 06b_validation

By global detector coordinate:

  python3 06b_validate_histograms.py \
      --centered-dir 05_test_roi --state-dir 06_test_roi \
      --x 7123 --y 4510 --output-dir 06b_validation

Random examples from a Step06 state class:

  python3 06b_validate_histograms.py \
      --centered-dir 05_test_roi --state-dir 06_test_roi \
      --random-state 3 --count 50 --seed 1 \
      --output-dir 06b_validation_state3

For --random-state 1, candidates whose maximum state count over all temperature
bins is exactly one are selected.  For states 2--4, candidates having that state
count in at least one temperature bin are selected.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

SCRIPT_VERSION = "6b.1.0"


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
        description="Create visual validation figures for Step06 histogram state detection."
    )
    p.add_argument("--centered-dir", type=Path, required=True,
                   help="Step05 output directory.")
    p.add_argument("--state-dir", type=Path, required=True,
                   help="Step06 output directory.")
    p.add_argument("--output-dir", type=Path, default=Path("06b_histogram_validation"))

    sel = p.add_argument_group("candidate selection")
    sel.add_argument("--candidate", type=int, action="append", default=[],
                     help="Zero-based candidate column in extraction order. Repeatable.")
    sel.add_argument("--x", type=int, help="Global detector x coordinate; use with --y.")
    sel.add_argument("--y", type=int, help="Global detector y coordinate; use with --x.")
    sel.add_argument("--random-state", type=int, choices=(1, 2, 3),
                     help="Randomly select candidates from this Step06 state class.")
    sel.add_argument("--count", type=int, default=1,
                     help="Number of random candidates to draw. Default: 1.")
    sel.add_argument("--seed", type=int, default=0,
                     help="Random seed. Default: 0.")

    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--format", choices=("png", "pdf"), default="png")
    p.add_argument("--x-min-ADU", type=float, default=None,
                   help="Optional common histogram x-axis minimum.")
    p.add_argument("--x-max-ADU", type=float, default=None,
                   help="Optional common histogram x-axis maximum.")
    p.add_argument("--hist-alpha", type=float, default=0.42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def find_one(directory: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"Multiple files match {pattern!r} in {directory}")
    raise FileNotFoundError(f"No file matching {patterns} in {directory}")


def local_maxima(y: np.ndarray) -> np.ndarray:
    n = y.size
    if n == 1:
        return np.array([0], dtype=np.int64)
    maxima: list[int] = []
    if y[0] > y[1]:
        maxima.append(0)
    for i in range(1, n - 1):
        if y[i] >= y[i - 1] and y[i] > y[i + 1]:
            maxima.append(i)
    if y[-1] >= y[-2]:
        maxima.append(n - 1)
    if not maxima:
        maxima = [int(np.argmax(y))]
    return np.asarray(maxima, dtype=np.int64)


def peak_prominence_simple(y: np.ndarray, p: int) -> float:
    left_min = float(np.min(y[: p + 1]))
    right_min = float(np.min(y[p:]))
    return max(0.0, float(y[p]) - max(left_min, right_min))


def basin_boundaries(y: np.ndarray, peaks_sorted: list[int]) -> list[tuple[int, int]]:
    if len(peaks_sorted) == 1:
        return [(0, y.size)]
    cuts: list[int] = []
    for a, b in zip(peaks_sorted[:-1], peaks_sorted[1:]):
        if b <= a + 1:
            cuts.append(a + 1)
        else:
            cuts.append(a + int(np.argmin(y[a:b + 1])))
    bounds: list[tuple[int, int]] = []
    start = 0
    for cut in cuts:
        stop = max(start + 1, cut + 1)
        bounds.append((start, stop))
        start = stop
    bounds.append((start, y.size))
    return bounds


def enforce_separation(peaks: list[Peak], min_sep_bins: int) -> list[Peak]:
    chosen: list[Peak] = []
    for pk in sorted(peaks, key=lambda q: (q.prominence, q.height), reverse=True):
        if all(abs(pk.index - q.index) >= min_sep_bins for q in chosen):
            chosen.append(pk)
    return sorted(chosen, key=lambda q: q.index)


def remove_shallow_splits(y: np.ndarray, peaks: list[Peak],
                          max_valley_ratio: float) -> tuple[list[Peak], list[float]]:
    peaks = sorted(peaks, key=lambda q: q.index)
    while len(peaks) > 1:
        ratios: list[float] = []
        for a, b in zip(peaks[:-1], peaks[1:]):
            valley = float(np.min(y[a.index:b.index + 1]))
            denom = max(min(a.height, b.height), 1e-12)
            ratios.append(valley / denom)
        worst = int(np.argmax(ratios))
        if ratios[worst] <= max_valley_ratio:
            return peaks, ratios
        a, b = peaks[worst], peaks[worst + 1]
        drop = worst if (a.prominence, a.height) < (b.prominence, b.height) else worst + 1
        peaks.pop(drop)
    return peaks, []


def analyze_histogram(values_x4: np.ndarray, hist_min: int, hist_max: int,
                      smooth_sigma_bins: float, min_sep_bins: int,
                      min_height_fraction: float, min_prominence_fraction: float,
                      min_occupancy: float, max_valley_ratio: float,
                      max_states: int, three_state_min_middle_occupancy: float = 0.06,
                      three_state_max_valley_ratio: float = 0.78,
                      three_state_min_adjacent_sep_bins: int = 6) -> tuple[np.ndarray, np.ndarray, list[Peak], list[Peak], list[int], list[float]]:
    """Reproduce Step06 analysis and return raw and accepted peak details."""
    n = int(values_x4.size)
    n_hist = hist_max - hist_min + 1
    counts = np.bincount(values_x4.astype(np.int64) - hist_min,
                         minlength=n_hist).astype(np.float64)
    smoothed = gaussian_filter1d(counts, smooth_sigma_bins, mode="nearest") \
        if smooth_sigma_bins > 0 else counts.copy()

    raw_peaks: list[Peak] = []
    for p in local_maxima(smoothed):
        h = float(smoothed[p])
        prom = peak_prominence_simple(smoothed, int(p))
        raw_peaks.append(Peak(int(p), h, h / n, prom, prom / n, 0.0))

    filtered = [p for p in raw_peaks
                if p.height_fraction >= min_height_fraction
                and p.prominence_fraction >= min_prominence_fraction]
    if not filtered:
        p = int(np.argmax(smoothed))
        prom = peak_prominence_simple(smoothed, p)
        filtered = [Peak(p, float(smoothed[p]), float(smoothed[p] / n),
                         float(prom), float(prom / n), 1.0)]

    filtered = enforce_separation(filtered, min_sep_bins)
    if len(filtered) > max_states * 3:
        filtered = sorted(filtered, key=lambda q: (q.prominence, q.height),
                          reverse=True)[:max_states * 3]
        filtered.sort(key=lambda q: q.index)

    bounds = basin_boundaries(smoothed, [p.index for p in filtered])
    with_occ: list[Peak] = []
    for pk, (lo, hi) in zip(filtered, bounds):
        occ = float(counts[lo:hi].sum() / n)
        if occ >= min_occupancy:
            with_occ.append(Peak(pk.index, pk.height, pk.height_fraction,
                                 pk.prominence, pk.prominence_fraction, occ))
    if not with_occ:
        strongest = max(filtered, key=lambda q: (q.prominence, q.height))
        with_occ = [Peak(strongest.index, strongest.height,
                         strongest.height_fraction, strongest.prominence,
                         strongest.prominence_fraction, 1.0)]

    with_occ, _ = remove_shallow_splits(smoothed, with_occ, max_valley_ratio)
    if len(with_occ) > max_states:
        with_occ = sorted(with_occ,
                          key=lambda q: (q.prominence, q.height),
                          reverse=True)[:max_states]
        with_occ.sort(key=lambda q: q.index)
        with_occ, _ = remove_shallow_splits(smoothed, with_occ, max_valley_ratio)

    final_bounds = basin_boundaries(smoothed, [p.index for p in with_occ])
    final: list[Peak] = []
    for pk, (lo, hi) in zip(with_occ, final_bounds):
        final.append(Peak(pk.index, pk.height, pk.height_fraction,
                          pk.prominence, pk.prominence_fraction,
                          float(counts[lo:hi].sum() / n)))
    final, ratios = remove_shallow_splits(smoothed, final, max_valley_ratio)

    if len(final) == 3:
        if final[1].occupancy < three_state_min_middle_occupancy:
            final = [final[0], final[2]]
        else:
            seps = [final[1].index - final[0].index, final[2].index - final[1].index]
            ratios3 = []
            for a, b in zip(final[:-1], final[1:]):
                valley = float(np.min(smoothed[a.index:b.index + 1]))
                ratios3.append(valley / max(min(a.height, b.height), 1e-12))
            bad_pair = None
            if min(seps) < three_state_min_adjacent_sep_bins:
                bad_pair = int(np.argmin(seps))
            elif max(ratios3) > three_state_max_valley_ratio:
                bad_pair = int(np.argmax(ratios3))
            if bad_pair is not None:
                a, b = final[bad_pair], final[bad_pair + 1]
                drop = bad_pair if (a.prominence, a.height) < (b.prominence, b.height) else bad_pair + 1
                final = [pk for j, pk in enumerate(final) if j != drop]
        bounds2 = basin_boundaries(smoothed, [pk.index for pk in final])
        final = [Peak(pk.index, pk.height, pk.height_fraction, pk.prominence,
                      pk.prominence_fraction, float(counts[lo:hi].sum() / n))
                 for pk, (lo, hi) in zip(final, bounds2)]
        final, ratios = remove_shallow_splits(smoothed, final, max_valley_ratio)

    valley_idx: list[int] = []
    for a, b in zip(final[:-1], final[1:]):
        valley_idx.append(a.index + int(np.argmin(smoothed[a.index:b.index + 1])))
    return counts, smoothed, raw_peaks, final, valley_idx, ratios


def build_frame_lists(bin_index: pd.DataFrame,
                      dataset_index: pd.DataFrame) -> list[np.ndarray]:
    required_bin = {"temperature_bin_index", "dataset_indices"}
    required_ds = {"dataset_index", "frame_start", "frame_stop_exclusive"}
    if required_bin - set(bin_index.columns):
        raise KeyError(f"temperature_bin_index.csv missing {required_bin - set(bin_index.columns)}")
    if required_ds - set(dataset_index.columns):
        raise KeyError(f"dataset_index.csv missing {required_ds - set(dataset_index.columns)}")
    by_idx = dataset_index.set_index("dataset_index")
    frame_lists: list[np.ndarray] = []
    bins = bin_index.sort_values("temperature_bin_index")
    for row in bins.itertuples(index=False):
        tokens = [t for t in str(row.dataset_indices).split(";") if t != ""]
        frames: list[int] = []
        for token in tokens:
            r = by_idx.loc[int(token)]
            frames.extend(range(int(r.frame_start), int(r.frame_stop_exclusive)))
        frame_lists.append(np.asarray(frames, dtype=np.int64))
    return frame_lists


def catalog_xy_columns(catalog: pd.DataFrame) -> tuple[str, str]:
    candidates = [("x", "y"), ("global_x", "global_y"),
                  ("x_global", "y_global"), ("X", "Y")]
    for xcol, ycol in candidates:
        if xcol in catalog.columns and ycol in catalog.columns:
            return xcol, ycol
    raise KeyError("candidate_catalog.csv has no recognized global x/y columns")


def select_candidates(args: argparse.Namespace, catalog: pd.DataFrame,
                      state_count: np.ndarray, xcol: str, ycol: str) -> list[int]:
    selected: list[int] = []
    n_candidates = len(catalog)
    for c in args.candidate:
        if c < 0 or c >= n_candidates:
            raise IndexError(f"--candidate {c} outside [0,{n_candidates - 1}]")
        selected.append(int(c))

    if (args.x is None) ^ (args.y is None):
        raise ValueError("--x and --y must be supplied together")
    if args.x is not None:
        hit = np.flatnonzero((catalog[xcol].to_numpy() == args.x) &
                             (catalog[ycol].to_numpy() == args.y))
        if hit.size == 0:
            raise KeyError(f"No candidate at global coordinate ({args.x},{args.y})")
        selected.extend(hit.astype(int).tolist())

    if args.random_state is not None:
        if args.count <= 0:
            raise ValueError("--count must be positive")
        sc = np.asarray(state_count)
        if args.random_state == 1:
            pool = np.flatnonzero(np.max(sc, axis=0) == 1)
        else:
            pool = np.flatnonzero(np.any(sc == args.random_state, axis=0))
        if pool.size == 0:
            raise RuntimeError(f"No candidates match random state {args.random_state}")
        rng = np.random.default_rng(args.seed)
        take = min(args.count, int(pool.size))
        selected.extend(rng.choice(pool, size=take, replace=False).astype(int).tolist())

    if not selected:
        raise ValueError("Select candidates with --candidate, --x/--y, or --random-state")
    return list(dict.fromkeys(selected))


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)


def draw_hist_panel(ax: plt.Axes, values_x4: np.ndarray, hist_min: int,
                    hist_max: int, sigma_bins: float, peaks_adu: np.ndarray,
                    occupancies: np.ndarray, prominences: np.ndarray,
                    state_count: int, title: str, common_ylim: tuple[float, float],
                    hist_alpha: float, xlim: tuple[float, float]) -> None:
    n_hist = hist_max - hist_min + 1
    counts = np.bincount(values_x4.astype(np.int64) - hist_min,
                         minlength=n_hist).astype(float)
    density = counts / max(1, values_x4.size)
    smooth = gaussian_filter1d(counts, sigma_bins, mode="nearest") / max(1, values_x4.size) \
        if sigma_bins > 0 else density
    xgrid = np.arange(hist_min, hist_max + 1, dtype=float) / 4.0

    ax.bar(xgrid, density, width=0.25, alpha=hist_alpha,
           edgecolor="none", label="Histogram")
    ax.plot(xgrid, smooth, linewidth=1.4, label="Smoothed")

    finite_peaks = np.asarray(peaks_adu)[np.isfinite(peaks_adu)]
    finite_occ = np.asarray(occupancies)[np.isfinite(peaks_adu)]
    finite_prom = np.asarray(prominences)[np.isfinite(peaks_adu)]
    for j, center in enumerate(finite_peaks):
        idx = int(np.clip(round(center * 4.0) - hist_min, 0, n_hist - 1))
        ax.plot(center, smooth[idx], marker="^", markersize=5)
        ax.axvline(center, linewidth=0.7, alpha=0.45)
        if j < finite_occ.size:
            ax.annotate(f"S{j+1}\n{finite_occ[j]*100:.0f}%",
                        (center, smooth[idx]), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8)

    for a, b in zip(finite_peaks[:-1], finite_peaks[1:]):
        ia = int(np.clip(round(a * 4.0) - hist_min, 0, n_hist - 1))
        ib = int(np.clip(round(b * 4.0) - hist_min, 0, n_hist - 1))
        if ib >= ia:
            iv = ia + int(np.argmin(smooth[ia:ib + 1]))
            ax.plot(xgrid[iv], smooth[iv], marker="v", markersize=4)

    ax.set_title(f"{title}\n{values_x4.size} frames | Step06: {state_count} state(s)", fontsize=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*common_ylim)
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=7)


def make_figure(candidate: int, catalog: pd.DataFrame, xcol: str, ycol: str,
                residual: np.ndarray, frame_lists: list[np.ndarray],
                bin_index: pd.DataFrame, state_count: np.ndarray,
                peak_center: np.ndarray, peak_occupancy: np.ndarray,
                peak_prominence: np.ndarray, summary06: dict,
                output_path: Path, args: argparse.Namespace) -> None:
    values_all_x4 = np.asarray(residual[:, candidate], dtype=np.int16)
    hist_min = int(summary06["histogram_x4_min"])
    hist_max = int(summary06["histogram_x4_max"])
    sigma_bins = float(summary06["smooth_sigma_ADU"]) * 4.0
    min_sep_bins = max(1, int(math.ceil(float(summary06["min_peak_separation_ADU"]) * 4.0)))
    max_states = int(summary06["max_states"])

    all_counts, all_smooth, all_raw_peaks, all_peaks, all_valleys, all_ratios = analyze_histogram(
        values_all_x4, hist_min, hist_max, sigma_bins, min_sep_bins,
        float(summary06["min_peak_height_fraction"]),
        float(summary06["min_prominence_fraction"]),
        float(summary06["min_state_occupancy"]),
        float(summary06["max_valley_ratio"]), max_states,
        float(summary06.get("three_state_min_middle_occupancy", 0.06)),
        float(summary06.get("three_state_max_valley_ratio", 0.78)),
        max(1, int(math.ceil(float(summary06.get("three_state_min_adjacent_separation_ADU", 1.5)) * 4.0))),
    )

    xgrid = np.arange(hist_min, hist_max + 1, dtype=float) / 4.0
    xlo = args.x_min_ADU if args.x_min_ADU is not None else hist_min / 4.0
    xhi = args.x_max_ADU if args.x_max_ADU is not None else hist_max / 4.0
    if xlo >= xhi:
        raise ValueError("Histogram x-axis minimum must be below maximum")

    max_density = float(np.max(all_counts / values_all_x4.size))
    for frames in frame_lists:
        vals = np.asarray(residual[frames, candidate], dtype=np.int16)
        counts = np.bincount(vals.astype(np.int64) - hist_min,
                             minlength=hist_max - hist_min + 1)
        max_density = max(max_density, float(np.max(counts / max(1, vals.size))))
    common_ylim = (0.0, max_density * 1.28 if max_density > 0 else 1.0)

    n_bins = len(frame_lists)
    ncols = 3
    nrows_hist = int(math.ceil(n_bins / ncols))
    fig = plt.figure(figsize=(18, 5.2 + 3.8 * nrows_hist), constrained_layout=True)
    gs = fig.add_gridspec(2 + nrows_hist, ncols,
                          height_ratios=[1.55] + [1.20] * nrows_hist + [0.78])

    bins_sorted = bin_index.sort_values("temperature_bin_index").reset_index(drop=True)
    ax_all = fig.add_subplot(gs[0, :])
    all_density = all_counts / values_all_x4.size
    all_smooth_density = all_smooth / values_all_x4.size
    ax_all.bar(xgrid, all_density, width=0.25, alpha=args.hist_alpha,
               edgecolor="none", label="Histogram")
    ax_all.plot(xgrid, all_smooth_density, linewidth=1.8, label="Smoothed")
    accepted_idx = {pk.index for pk in all_peaks}
    for pk in all_raw_peaks:
        if pk.index not in accepted_idx:
            ax_all.plot(xgrid[pk.index], all_smooth_density[pk.index], marker="^",
                        markersize=4, alpha=0.35, markerfacecolor="none",
                        markeredgecolor="0.4")
    for j, pk in enumerate(all_peaks):
        center = (hist_min + pk.index) / 4.0
        ax_all.plot(center, all_smooth_density[pk.index], marker="^", markersize=6)
        ax_all.axvline(center, linewidth=0.8, alpha=0.45)
        ax_all.annotate(f"S{j+1}: {center:+.2f} ADU\nocc={pk.occupancy*100:.1f}%",
                        (center, all_smooth_density[pk.index]), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8)
    for iv in all_valleys:
        ax_all.plot(xgrid[iv], all_smooth_density[iv], marker="v", markersize=5)
    ax_all.set_xlim(xlo, xhi)
    ax_all.set_ylim(*common_ylim)
    ax_all.set_ylabel("Fraction per 0.25 ADU bin")
    ax_all.set_title(
        f"ALL TEMPERATURE BINS COMBINED | visual re-analysis: {len(all_peaks)} state(s)",
        fontsize=12,
    )
    ax_all.grid(alpha=0.18)
    state_vector = state_count[:, candidate].astype(int).tolist()
    state_lines = [
        f"{float(bins_sorted.iloc[b]['temperature_mean_C']):+5.1f}°C : {state_vector[b]} state"
        for b in range(n_bins)
    ] if 'bins_sorted' in locals() else []
    peak_lines = [f"S{j+1}: {(hist_min+pk.index)/4.0:+.2f} ADU  occ={pk.occupancy*100:.1f}%"
                  for j, pk in enumerate(all_peaks)]
    ax_all.text(0.995, 0.96, "\n".join(peak_lines), transform=ax_all.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.72, edgecolor="0.7"))
    ax_all.legend(loc="upper left", fontsize=9)

    for b in range(nrows_hist * ncols):
        ax = fig.add_subplot(gs[1 + b // ncols, b % ncols], sharex=ax_all)
        if b >= n_bins:
            ax.axis("off")
            continue
        frames = frame_lists[b]
        vals = np.asarray(residual[frames, candidate], dtype=np.int16)
        row = bins_sorted.iloc[b]
        label = f"{float(row['temperature_mean_C']):+.2f} °C"
        draw_hist_panel(
            ax, vals, hist_min, hist_max, sigma_bins,
            np.asarray(peak_center[b, candidate]),
            np.asarray(peak_occupancy[b, candidate]),
            np.asarray(peak_prominence[b, candidate]),
            int(state_count[b, candidate]), label, common_ylim,
            args.hist_alpha, (xlo, xhi),
        )
        if b % ncols == 0:
            ax.set_ylabel("Fraction/bin", fontsize=8)
        if b // ncols == nrows_hist - 1:
            ax.set_xlabel("Centered local residual (ADU)", fontsize=10)

    ax_ts = fig.add_subplot(gs[-1, :])
    cursor = 0
    tick_pos: list[float] = []
    tick_labels: list[str] = []
    for b, frames in enumerate(frame_lists):
        vals = np.asarray(residual[frames, candidate], dtype=float) / 4.0
        xx = np.arange(cursor, cursor + vals.size)
        ax_ts.plot(xx, vals, linewidth=0.70, alpha=0.82,
                   label=f"{float(bins_sorted.iloc[b]['temperature_mean_C']):+.1f}°C")
        for center in np.asarray(peak_center[b, candidate]):
            if np.isfinite(center):
                ax_ts.hlines(float(center), cursor, cursor + vals.size - 1,
                             linewidth=0.75, alpha=0.45)
        tick_pos.append(cursor + max(0, vals.size - 1) / 2.0)
        tick_labels.append(f"{float(bins_sorted.iloc[b]['temperature_mean_C']):+.1f}°C")
        cursor += vals.size
        if b < n_bins - 1:
            ax_ts.axvline(cursor - 0.5, linewidth=0.65, alpha=0.4)
    ax_ts.axhline(0.0, linewidth=0.7, alpha=0.5)
    ax_ts.set_xlim(0, max(1, cursor - 1))
    ax_ts.set_ylabel("Residual (ADU)")
    ax_ts.set_xlabel("Frames grouped by temperature bin")
    ax_ts.set_xticks(tick_pos, tick_labels, fontsize=8)
    ax_ts.grid(alpha=0.18)
    ax_ts.set_title("Centered-residual time series (temperature-bin segments)", fontsize=9)

    x = int(catalog.iloc[candidate][xcol])
    y = int(catalog.iloc[candidate][ycol])
    max_state = int(np.max(state_count[:, candidate]))
    multibins = int(np.count_nonzero(state_count[:, candidate] >= 2))
    fig.suptitle(
        f"Step06 validation | candidate column {candidate} | global (x,y)=({x},{y}) | "
        f"max state={max_state} | multistate bins={multibins}/{n_bins}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    centered_dir = args.centered_dir.resolve()
    state_dir = args.state_dir.resolve()
    output_dir = args.output_dir.resolve()

    summary05 = load_json(centered_dir / "summary.json")
    summary06 = load_json(state_dir / "summary.json")
    if not summary05.get("validation_passed", False):
        raise ValueError("Step05 validation_passed is not true")
    if not summary06.get("validation_passed", False):
        raise ValueError("Step06 validation_passed is not true")

    residual_path = find_one(centered_dir, ["centered_residual_x4_*.npy",
                                             "centered_residual_x4_int16.npy"])
    residual = np.load(residual_path, mmap_mode="r")
    catalog = pd.read_csv(state_dir / "candidate_catalog.csv")
    dataset_index = pd.read_csv(centered_dir / "dataset_index.csv")
    bin_index = pd.read_csv(state_dir / "temperature_bin_index.csv")

    state_count = np.load(state_dir / "state_count_uint8.npy", mmap_mode="r")
    peak_center = np.load(state_dir / "peak_center_ADU_float32.npy", mmap_mode="r")
    peak_occupancy = np.load(state_dir / "peak_occupancy_float32.npy", mmap_mode="r")
    peak_prominence = np.load(state_dir / "peak_prominence_fraction_float32.npy", mmap_mode="r")

    if residual.ndim != 2:
        raise ValueError("centered residual array must be 2-D")
    if state_count.shape != (len(bin_index), residual.shape[1]):
        raise ValueError("Step06 state-count shape is inconsistent with Step05/temperature bins")
    if len(catalog) != residual.shape[1]:
        raise ValueError("candidate catalog length does not match residual columns")

    frame_lists = build_frame_lists(bin_index, dataset_index)
    xcol, ycol = catalog_xy_columns(catalog)
    selected = select_candidates(args, catalog, state_count, xcol, ycol)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for seq, candidate in enumerate(selected, start=1):
        x = int(catalog.iloc[candidate][xcol])
        y = int(catalog.iloc[candidate][ycol])
        max_state = int(np.max(state_count[:, candidate]))
        filename = safe_filename(
            f"{seq:04d}_candidate{candidate:06d}_x{x}_y{y}_maxstate{max_state}.{args.format}"
        )
        output_path = output_dir / filename
        if args.verbose:
            print(f"[{seq}/{len(selected)}] {output_path.name}", flush=True)
        make_figure(candidate, catalog, xcol, ycol, residual, frame_lists,
                    bin_index, state_count, peak_center, peak_occupancy,
                    peak_prominence, summary06, output_path, args)
        rows.append({
            "sequence": seq,
            "candidate_column": candidate,
            "x": x,
            "y": y,
            "max_state_count": max_state,
            "multistate_temperature_bin_count": int(np.count_nonzero(state_count[:, candidate] >= 2)),
            "output_file": filename,
        })

    pd.DataFrame(rows).to_csv(output_dir / "validation_index.csv", index=False)
    report = {
        "step": "06b_validate_histograms",
        "script_version": SCRIPT_VERSION,
        "validation_figure_count": len(rows),
        "centered_dir": str(centered_dir),
        "state_dir": str(state_dir),
        "output_dir": str(output_dir),
        "selection": {
            "candidate": args.candidate,
            "x": args.x,
            "y": args.y,
            "random_state": args.random_state,
            "count": args.count,
            "seed": args.seed,
        },
        "histogram_common_x_axis_ADU": [
            args.x_min_ADU if args.x_min_ADU is not None else summary06["histogram_x4_min"] / 4.0,
            args.x_max_ADU if args.x_max_ADU is not None else summary06["histogram_x4_max"] / 4.0,
        ],
        "outputs": [r["output_file"] for r in rows],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {len(rows)} validation figure(s) in {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
