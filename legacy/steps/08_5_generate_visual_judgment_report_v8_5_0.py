#!/usr/bin/env python3
"""
Step08.5: Generate a visualization and judgment report before RTS classification.

This step does not classify pixels as RTS and does not generate a dictionary.
It visualizes the objective outputs from Steps 05, 07, and 08 so that Step09
thresholds can be chosen from the observed distributions and representative
time series.

Main outputs
------------
overview_report.pdf
01_switch_count_histogram.png
02_reliable_switch_count_histogram.png
03_bidirectional_pair_histogram.png
04_run_length_distribution.png
05_score_distribution.png
06_switch_vs_occupancy.png
07_temperature_dependence.png
08_state_count_composition.png
top_candidate_gallery.pdf
top_candidate_gallery_index.csv
report_summary.json
README_report.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


SCRIPT_VERSION = "8.5.0"
MAX_STATES = 3
Q_EXCELLENT = 1
Q_NORMAL = 2
Q_NEAR_VALLEY = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate diagnostic plots and candidate galleries from Step08 "
            "without making an RTS classification."
        )
    )
    p.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="Step07 output directory.",
    )
    p.add_argument(
        "--transition-dir",
        type=Path,
        required=True,
        help="Step08 v8.0.2 output directory.",
    )
    p.add_argument(
        "--centered-dir",
        type=Path,
        required=True,
        help="Step05 output directory.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("08_5_visual_judgment_report"),
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of ranked candidate/bin combinations in the gallery.",
    )
    p.add_argument(
        "--gallery-per-page",
        type=int,
        default=4,
        choices=(1, 2, 4),
        help="Number of candidate panels per PDF page.",
    )
    p.add_argument(
        "--max-scatter-points",
        type=int,
        default=100000,
        help="Maximum points used in scatter plots.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=12345,
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return p.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_from_summary(
    directory: Path,
    summary: dict,
    keys: tuple[str, ...],
    fallbacks: tuple[str, ...],
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
            path = Path(str(value))
            if path.is_file():
                return path
            local = directory / path.name
            if local.is_file():
                return local

    for name in fallbacks:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not resolve any of {keys} or {fallbacks} in {directory}"
    )


def prepare_output(directory: Path, overwrite: bool) -> None:
    if directory.exists():
        if not overwrite:
            raise FileExistsError(
                f"{directory} already exists; use --overwrite"
            )
        import shutil
        shutil.rmtree(directory)
    directory.mkdir(parents=True)


def parse_indices(value: object) -> list[int]:
    text = str(value).strip()
    if not text:
        return []
    return [int(v) for v in text.split(";") if v.strip()]


def build_bin_frames(
    bin_table: pd.DataFrame,
    dataset_table: pd.DataFrame,
    frame_count: int,
) -> list[np.ndarray]:
    required_bins = {"temperature_bin_index", "dataset_indices"}
    required_data = {
        "dataset_index", "frame_start", "frame_stop_exclusive"
    }
    if not required_bins.issubset(bin_table.columns):
        raise KeyError(
            f"temperature_bin_index.csv requires {sorted(required_bins)}"
        )
    if not required_data.issubset(dataset_table.columns):
        raise KeyError(
            f"dataset_index.csv requires {sorted(required_data)}"
        )

    bins = bin_table.sort_values(
        "temperature_bin_index"
    ).reset_index(drop=True)
    bin_numbers = pd.to_numeric(
        bins["temperature_bin_index"], errors="raise"
    ).to_numpy(np.int64)
    if not np.array_equal(bin_numbers, np.arange(len(bins))):
        raise ValueError("temperature_bin_index must be 0..N-1")

    datasets = dataset_table.copy()
    datasets["dataset_index"] = pd.to_numeric(
        datasets["dataset_index"], errors="raise"
    ).astype(np.int64)
    datasets = datasets.set_index("dataset_index")

    owner = np.full(frame_count, -1, dtype=np.int32)
    result = []
    for row in bins.itertuples(index=False):
        pieces = []
        for dataset_index in parse_indices(row.dataset_indices):
            if dataset_index not in datasets.index:
                raise KeyError(f"Unknown dataset_index {dataset_index}")
            dataset = datasets.loc[dataset_index]
            start = int(dataset.frame_start)
            stop = int(dataset.frame_stop_exclusive)
            if start < 0 or stop > frame_count or stop <= start:
                raise ValueError(
                    f"Invalid frame range [{start}, {stop})"
                )
            pieces.append(np.arange(start, stop, dtype=np.int64))
        if not pieces:
            raise ValueError(
                f"Temperature bin {row.temperature_bin_index} has no frames"
            )
        frames = np.concatenate(pieces)
        if np.any(owner[frames] >= 0):
            raise ValueError("A frame belongs to multiple temperature bins")
        owner[frames] = int(row.temperature_bin_index)
        result.append(frames)

    if np.any(owner < 0):
        raise ValueError(
            f"{np.count_nonzero(owner < 0)} frames are not assigned to a bin"
        )
    return result


def resolve_centered_residual(
    centered_dir: Path,
    step05_summary: dict,
    state_summary: dict,
) -> tuple[Path, float]:
    candidates = []
    for summary in (state_summary, step05_summary):
        outputs = summary.get("outputs", {})
        for key in (
            "centered_residual",
            "centered_residual_output",
        ):
            if isinstance(outputs, dict):
                candidates.append(outputs.get(key))
            candidates.append(summary.get(key))

    for value in candidates:
        if not value:
            continue
        path = Path(str(value))
        if not path.is_file():
            path = centered_dir / path.name
        if path.is_file():
            return path, residual_scale(path)

    found = sorted(centered_dir.glob("centered_residual*.npy"))
    if len(found) == 1:
        return found[0], residual_scale(found[0])
    raise FileNotFoundError(
        "Could not uniquely locate centered_residual*.npy in "
        f"{centered_dir}; found {len(found)}"
    )


def residual_scale(path: Path) -> float:
    name = path.name.lower()
    for factor in (2, 4, 8, 16):
        if f"_x{factor}_" in name or f"_x{factor}." in name:
            return float(factor)
    return 1.0


def bin_label(row: pd.Series, index: int) -> str:
    for column in (
        "temperature_bin_label",
        "bin_label",
        "label",
        "temperature_label",
    ):
        if column in row and pd.notna(row[column]):
            return str(row[column])
    for column in (
        "temperature_center_C",
        "temperature_median_C",
        "temperature_mean_C",
        "temperature_C",
    ):
        if column in row and pd.notna(row[column]):
            return f"{float(row[column]):.2f} °C"
    return f"Bin {index}"


def positive_log_bins(values: np.ndarray, max_bins: int = 80) -> np.ndarray:
    values = np.asarray(values)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return np.array([0.5, 1.5])
    vmax = float(values.max())
    if vmax <= 1:
        return np.array([0.5, 1.5])
    n = min(max_bins, max(10, int(math.ceil(math.log2(vmax))) * 4))
    edges = np.unique(
        np.rint(np.geomspace(1, vmax + 1, n)).astype(np.int64)
    )
    edges = np.r_[0.5, edges + 0.5]
    return np.unique(edges)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_overview_page(
    pdf: PdfPages,
    summary08: dict,
    n_candidates: int,
    n_bins: int,
) -> None:
    counts = summary08.get("candidate_counts_any_temperature_bin", {})
    run = summary08.get("run_length_summary", {})
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(
        "Step08.5 RTS visual judgment report",
        fontsize=18,
        y=0.96,
    )
    lines = [
        f"Step08 script version: {summary08.get('script_version', 'unknown')}",
        f"Candidates: {n_candidates:,}",
        f"Temperature bins: {n_bins}",
        f"Candidates with any switch: {counts.get('any_switch', 'n/a')}",
        (
            "Candidates with ≥1 bidirectional pair: "
            f"{counts.get('at_least_one_bidirectional_state_pair', 'n/a')}"
        ),
        (
            "Candidates with ≥2 bidirectional pairs: "
            f"{counts.get('at_least_two_bidirectional_state_pairs', 'n/a')}"
        ),
        f"Total runs: {run.get('total_runs', 'n/a')}",
        f"Median run length: {run.get('median_run_length_frames', 'n/a')} frames",
        f"95th percentile run length: {run.get('p95_run_length_frames', 'n/a')} frames",
        "",
        "This report is diagnostic only.",
        "No RTS classification or dictionary generation is performed.",
    ]
    fig.text(
        0.08, 0.84, "\n".join(lines),
        va="top", ha="left", fontsize=13, linespacing=1.6,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.max_scatter_points <= 0:
        raise ValueError("--max-scatter-points must be positive")

    started = time.perf_counter()
    state_dir = args.state_dir.expanduser().resolve()
    transition_dir = args.transition_dir.expanduser().resolve()
    centered_dir = args.centered_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output(output_dir, args.overwrite)

    state_summary = read_json(state_dir / "summary.json")
    transition_summary = read_json(transition_dir / "summary.json")
    step05_summary = read_json(centered_dir / "summary.json")

    if not state_summary.get("validation_passed", False):
        raise ValueError("Step07 validation_passed is not true")
    if not transition_summary.get("validation_passed", False):
        raise ValueError("Step08 validation_passed is not true")

    state_path = resolve_from_summary(
        state_dir, state_summary,
        ("state", "frame_state_output"),
        ("frame_state_uint8.npy",),
    )
    quality_path = resolve_from_summary(
        state_dir, state_summary,
        ("quality", "assignment_quality_output"),
        ("assign_quality_uint8.npy",),
    )
    distance_path = resolve_from_summary(
        state_dir, state_summary,
        ("distance", "frame_distance"),
        ("frame_distance_ADU_float32.npy",),
    )
    center_path = resolve_from_summary(
        state_dir, state_summary,
        ("center", "state_center"),
        ("state_center_refined_ADU_float32.npy",),
    )
    occupancy_path = resolve_from_summary(
        state_dir, state_summary,
        ("occupancy", "state_occupancy"),
        ("state_occupancy_refined_float32.npy",),
    )
    bin_table_path = state_dir / "temperature_bin_index.csv"
    catalog_path = state_dir / "candidate_catalog.csv"
    frame_metadata_path = state_dir / "frame_metadata.csv"
    dataset_index_path = centered_dir / "dataset_index.csv"

    centered_path, centered_scale = resolve_centered_residual(
        centered_dir, step05_summary, state_summary
    )

    switch_path = resolve_from_summary(
        transition_dir, transition_summary,
        ("switch_count",),
        ("switch_count_uint32.npy",),
    )
    reliable_switch_path = resolve_from_summary(
        transition_dir, transition_summary,
        ("reliable_switch_count",),
        ("reliable_switch_count_uint32.npy",),
    )
    valid_pair_path = resolve_from_summary(
        transition_dir, transition_summary,
        ("valid_pair_count",),
        ("valid_pair_count_uint32.npy",),
    )
    reliable_pair_path = resolve_from_summary(
        transition_dir, transition_summary,
        ("reliable_pair_count",),
        ("reliable_pair_count_uint32.npy",),
    )
    visit_path = resolve_from_summary(
        transition_dir, transition_summary,
        ("state_visit_count",),
        ("state_visit_count_uint32.npy",),
    )
    mean_dwell_path = resolve_from_summary(
        transition_dir, transition_summary,
        ("mean_dwell",),
        ("mean_dwell_frame_float32.npy",),
    )
    median_dwell_path = resolve_from_summary(
        transition_dir, transition_summary,
        ("median_dwell",),
        ("median_dwell_frame_float32.npy",),
    )
    observed_path = resolve_from_summary(
        transition_dir, transition_summary,
        ("observed_state_count",),
        ("observed_state_count_uint8.npy",),
    )
    bidirectional_path = resolve_from_summary(
        transition_dir, transition_summary,
        ("bidirectional_pair_count",),
        ("bidirectional_pair_count_uint8.npy",),
    )
    score_path = transition_dir / "candidate_score.csv"
    top_path = transition_dir / "top100_switching_candidates.csv"

    for path in (
        bin_table_path, catalog_path, frame_metadata_path,
        dataset_index_path, score_path, top_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    frame_state = np.load(state_path, mmap_mode="r")
    quality = np.load(quality_path, mmap_mode="r")
    distance = np.load(distance_path, mmap_mode="r")
    state_center = np.load(center_path, mmap_mode="r")
    occupancy = np.load(occupancy_path, mmap_mode="r")
    centered_raw = np.load(centered_path, mmap_mode="r")

    switch = np.load(switch_path, mmap_mode="r")
    reliable_switch = np.load(reliable_switch_path, mmap_mode="r")
    valid_pair = np.load(valid_pair_path, mmap_mode="r")
    reliable_pair = np.load(reliable_pair_path, mmap_mode="r")
    visits = np.load(visit_path, mmap_mode="r")
    mean_dwell = np.load(mean_dwell_path, mmap_mode="r")
    median_dwell = np.load(median_dwell_path, mmap_mode="r")
    observed = np.load(observed_path, mmap_mode="r")
    bidirectional = np.load(bidirectional_path, mmap_mode="r")

    if frame_state.ndim != 2:
        raise ValueError("frame_state must be two-dimensional")
    n_frames, n_candidates = map(int, frame_state.shape)

    bin_table = pd.read_csv(bin_table_path).sort_values(
        "temperature_bin_index"
    ).reset_index(drop=True)
    catalog = pd.read_csv(catalog_path)
    frame_metadata = pd.read_csv(frame_metadata_path)
    dataset_index = pd.read_csv(dataset_index_path)
    score_table = pd.read_csv(score_path)
    top_table = pd.read_csv(top_path)

    bin_frames = build_bin_frames(bin_table, dataset_index, n_frames)
    n_bins = len(bin_frames)

    expected2 = (n_bins, n_candidates)
    expected3 = (n_bins, n_candidates, MAX_STATES)
    if switch.shape != expected2:
        raise ValueError(f"switch shape {switch.shape} != {expected2}")
    if occupancy.shape != expected3:
        raise ValueError(f"occupancy shape {occupancy.shape} != {expected3}")
    if state_center.shape != expected3:
        raise ValueError(f"state_center shape {state_center.shape} != {expected3}")
    if centered_raw.shape != frame_state.shape:
        raise ValueError(
            f"centered residual shape {centered_raw.shape} "
            f"!= frame state shape {frame_state.shape}"
        )
    if len(catalog) != n_candidates:
        raise ValueError("candidate_catalog row count mismatch")

    labels = [
        bin_label(bin_table.iloc[i], i)
        for i in range(n_bins)
    ]
    rng = np.random.default_rng(args.random_seed)

    overview_pdf_path = output_dir / "overview_report.pdf"
    with PdfPages(overview_pdf_path) as pdf:
        add_overview_page(pdf, transition_summary, n_candidates, n_bins)

        # 1. Raw switch distribution among switched combinations.
        raw_switch = np.asarray(switch).ravel()
        raw_positive = raw_switch[raw_switch > 0]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.hist(
            raw_positive,
            bins=positive_log_bins(raw_positive),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Switch count per candidate / temperature bin")
        ax.set_ylabel("Number of combinations")
        ax.set_title("Raw switch-count distribution")
        ax.grid(True, which="both", alpha=0.25)
        save_path = output_dir / "01_switch_count_histogram.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 2. Reliable switch distribution.
        reliable_values = np.asarray(reliable_switch).ravel()
        reliable_positive = reliable_values[reliable_values > 0]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.hist(
            reliable_positive,
            bins=positive_log_bins(reliable_positive),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(
            "Reliable switch count per candidate / temperature bin"
        )
        ax.set_ylabel("Number of combinations")
        ax.set_title("Reliable switch-count distribution")
        ax.grid(True, which="both", alpha=0.25)
        save_path = output_dir / "02_reliable_switch_count_histogram.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 3. Bidirectional state-pair distribution.
        bidir_values = np.asarray(bidirectional).ravel()
        values, counts = np.unique(bidir_values, return_counts=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(values.astype(str), counts)
        ax.set_yscale("log")
        ax.set_xlabel("Bidirectional state-pair count")
        ax.set_ylabel("Candidate / temperature-bin combinations")
        ax.set_title("Bidirectional transition evidence")
        ax.grid(True, axis="y", alpha=0.25)
        save_path = output_dir / "03_bidirectional_pair_histogram.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 4. Dwell/run length proxy distribution from per-state medians.
        dwell_values = np.asarray(median_dwell).ravel()
        dwell_values = dwell_values[
            np.isfinite(dwell_values) & (dwell_values > 0)
        ]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.hist(
            dwell_values,
            bins=positive_log_bins(dwell_values),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Median dwell length [frames]")
        ax.set_ylabel("Observed state entries")
        ax.set_title("Distribution of per-state median dwell lengths")
        ax.grid(True, which="both", alpha=0.25)
        save_path = output_dir / "04_run_length_distribution.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 5. Diagnostic score distribution.
        score_column = "diagnostic_ranking_score"
        if score_column not in score_table.columns:
            raise KeyError(f"{score_path} missing {score_column}")
        scores = pd.to_numeric(
            score_table[score_column], errors="coerce"
        ).to_numpy(float)
        scores = scores[np.isfinite(scores) & (scores > 0)]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.hist(scores, bins=80)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Diagnostic ranking score")
        ax.set_ylabel("Candidates")
        ax.set_title(
            "Candidate ranking-score distribution "
            "(diagnostic, not classification)"
        )
        ax.grid(True, which="both", alpha=0.25)
        save_path = output_dir / "05_score_distribution.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 6. Reliable switches versus occupancy balance.
        visit_array = np.asarray(visits)
        total_visits = visit_array.sum(axis=2)
        max_fraction = np.divide(
            visit_array.max(axis=2),
            total_visits,
            out=np.full(expected2, np.nan, dtype=float),
            where=total_visits > 0,
        )
        flat_switch = reliable_values
        flat_balance = (1.0 - max_fraction).ravel()
        mask = (
            np.isfinite(flat_balance)
            & (flat_switch > 0)
        )
        indices = np.flatnonzero(mask)
        if indices.size > args.max_scatter_points:
            indices = rng.choice(
                indices,
                size=args.max_scatter_points,
                replace=False,
            )
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(
            flat_switch[indices],
            flat_balance[indices],
            s=5,
            alpha=0.25,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Reliable switch count")
        ax.set_ylabel("Occupancy balance = 1 − maximum state fraction")
        ax.set_title("Switch activity versus occupancy balance")
        ax.grid(True, which="both", alpha=0.25)
        save_path = output_dir / "06_switch_vs_occupancy.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 7. Temperature dependence.
        candidate_any = []
        candidate_bidir = []
        median_switch = []
        for b in range(n_bins):
            candidate_any.append(int(np.count_nonzero(switch[b] > 0)))
            candidate_bidir.append(
                int(np.count_nonzero(bidirectional[b] > 0))
            )
            positive = np.asarray(reliable_switch[b])
            positive = positive[positive > 0]
            median_switch.append(
                float(np.median(positive)) if positive.size else np.nan
            )
        x = np.arange(n_bins)
        fig, ax = plt.subplots(figsize=(10, 5.8))
        ax.plot(x, candidate_any, marker="o", label="Any switch")
        ax.plot(
            x, candidate_bidir, marker="s",
            label="Bidirectional pair"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel("Temperature bin")
        ax.set_ylabel("Candidate count")
        ax.set_title("Temperature-bin dependence of switching candidates")
        ax.legend()
        ax.grid(True, alpha=0.25)
        save_path = output_dir / "07_temperature_dependence.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 8. Observed state-count composition by bin.
        observed_array = np.asarray(observed)
        composition = np.array([
            [
                np.count_nonzero(observed_array[b] == state_count)
                for state_count in (1, 2, 3)
            ]
            for b in range(n_bins)
        ])
        fig, ax = plt.subplots(figsize=(10, 5.8))
        bottom = np.zeros(n_bins)
        for state_index, state_count in enumerate((1, 2, 3)):
            ax.bar(
                x,
                composition[:, state_index],
                bottom=bottom,
                label=f"{state_count} observed state(s)",
            )
            bottom += composition[:, state_index]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel("Temperature bin")
        ax.set_ylabel("Candidates")
        ax.set_title("Observed state-count composition")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
        save_path = output_dir / "08_state_count_composition.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # Candidate gallery.
    ranking = top_table.copy()
    if "rank" in ranking.columns:
        ranking = ranking.sort_values("rank")
    ranking = ranking.head(args.top_n).reset_index(drop=True)

    required_top = {
        "candidate_index", "temperature_bin_index",
        "switch_count", "reliable_switch_count",
        "bidirectional_pair_count",
    }
    missing = required_top - set(ranking.columns)
    if missing:
        raise KeyError(
            f"top100_switching_candidates.csv missing {sorted(missing)}"
        )

    gallery_index_rows = []
    gallery_pdf_path = output_dir / "top_candidate_gallery.pdf"
    per_page = args.gallery_per_page
    page_shape = {
        1: (1, 1),
        2: (2, 1),
        4: (2, 2),
    }[per_page]

    with PdfPages(gallery_pdf_path) as pdf:
        for page_start in range(0, len(ranking), per_page):
            subset = ranking.iloc[page_start:page_start + per_page]
            fig, axes = plt.subplots(
                page_shape[0], page_shape[1],
                figsize=(11.69, 8.27),
                squeeze=False,
            )
            axes_flat = axes.ravel()

            for panel_index, (_, rank_row) in enumerate(subset.iterrows()):
                ax = axes_flat[panel_index]
                candidate = int(rank_row["candidate_index"])
                bin_number = int(rank_row["temperature_bin_index"])
                frames = bin_frames[bin_number]

                centered = (
                    np.asarray(centered_raw[frames, candidate], dtype=float)
                    / centered_scale
                )
                states = np.asarray(
                    frame_state[frames, candidate], dtype=np.int16
                )
                qualities = np.asarray(
                    quality[frames, candidate], dtype=np.int16
                )
                centers = np.asarray(
                    state_center[bin_number, candidate], dtype=float
                )

                local_x = np.arange(len(frames))
                good = qualities == Q_EXCELLENT
                normal = qualities == Q_NORMAL
                valley = qualities == Q_NEAR_VALLEY

                ax.plot(
                    local_x, centered,
                    linewidth=0.7, alpha=0.65,
                    label="Centered ADU",
                )
                ax.scatter(
                    local_x[good], centered[good],
                    s=8, marker="o", label="Excellent",
                )
                ax.scatter(
                    local_x[normal], centered[normal],
                    s=8, marker=".", label="Normal",
                )
                if np.any(valley):
                    ax.scatter(
                        local_x[valley], centered[valley],
                        s=14, marker="x", label="Near valley",
                    )

                state_count = int(np.max(states))
                for state_number in range(1, state_count + 1):
                    center = centers[state_number - 1]
                    if np.isfinite(center):
                        ax.axhline(
                            center,
                            linestyle="--",
                            linewidth=0.9,
                            alpha=0.8,
                            label=f"State {state_number} center",
                        )

                candidate_label = f"candidate {candidate}"
                coordinate_parts = []
                if candidate < len(catalog):
                    catalog_row = catalog.iloc[candidate]
                    for x_col, y_col in (
                        ("x", "y"),
                        ("pixel_x", "pixel_y"),
                        ("sensor_x", "sensor_y"),
                    ):
                        if x_col in catalog.columns and y_col in catalog.columns:
                            coordinate_parts = [
                                f"x={catalog_row[x_col]}",
                                f"y={catalog_row[y_col]}",
                            ]
                            break
                title = (
                    f"Rank {int(rank_row.get('rank', page_start + panel_index + 1))}: "
                    f"{candidate_label}"
                )
                if coordinate_parts:
                    title += " (" + ", ".join(coordinate_parts) + ")"
                ax.set_title(title, fontsize=10)
                ax.set_xlabel(
                    f"Frame position in {labels[bin_number]}"
                )
                ax.set_ylabel("Centered signal [ADU]")
                ax.grid(True, alpha=0.2)

                text = (
                    f"switch={int(rank_row['switch_count'])}, "
                    f"reliable={int(rank_row['reliable_switch_count'])}, "
                    f"bidir={int(rank_row['bidirectional_pair_count'])}, "
                    f"states={int(rank_row.get('observed_state_count', state_count))}"
                )
                ax.text(
                    0.01, 0.98, text,
                    transform=ax.transAxes,
                    va="top", ha="left", fontsize=8,
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        alpha=0.75,
                    ),
                )

                gallery_index_rows.append({
                    "rank": int(
                        rank_row.get(
                            "rank", page_start + panel_index + 1
                        )
                    ),
                    "page": page_start // per_page + 1,
                    "panel": panel_index + 1,
                    "candidate_index": candidate,
                    "temperature_bin_index": bin_number,
                    "temperature_bin_label": labels[bin_number],
                    "switch_count": int(rank_row["switch_count"]),
                    "reliable_switch_count": int(
                        rank_row["reliable_switch_count"]
                    ),
                    "bidirectional_pair_count": int(
                        rank_row["bidirectional_pair_count"]
                    ),
                    "diagnostic_ranking_score": float(
                        rank_row.get(
                            "diagnostic_ranking_score", np.nan
                        )
                    ),
                })

            for unused in range(len(subset), len(axes_flat)):
                axes_flat[unused].axis("off")

            handles, legend_labels = axes_flat[0].get_legend_handles_labels()
            if handles:
                unique = {}
                for handle, label in zip(handles, legend_labels):
                    if label not in unique:
                        unique[label] = handle
                fig.legend(
                    unique.values(), unique.keys(),
                    loc="lower center",
                    ncol=min(5, len(unique)),
                    fontsize=8,
                )
            fig.suptitle(
                "Top ranked candidate / temperature-bin time series",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0.05, 1, 0.95))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    pd.DataFrame(gallery_index_rows).to_csv(
        output_dir / "top_candidate_gallery_index.csv",
        index=False,
    )

    elapsed = time.perf_counter() - started
    report_summary = {
        "step": "08_5_visual_judgment_report",
        "script_version": SCRIPT_VERSION,
        "responsibility": (
            "visual diagnosis before Step09; no RTS classification "
            "and no dictionary generation"
        ),
        "validation_passed": True,
        "state_dir": str(state_dir),
        "transition_dir": str(transition_dir),
        "centered_dir": str(centered_dir),
        "frame_count": n_frames,
        "candidate_count": n_candidates,
        "temperature_bin_count": n_bins,
        "top_n_requested": args.top_n,
        "top_n_generated": len(ranking),
        "centered_residual_input": str(centered_path),
        "centered_residual_scale_divisor": centered_scale,
        "outputs": {
            "overview_report": str(overview_pdf_path),
            "candidate_gallery": str(gallery_pdf_path),
            "candidate_gallery_index": str(
                output_dir / "top_candidate_gallery_index.csv"
            ),
            "plots": [
                str(output_dir / name)
                for name in (
                    "01_switch_count_histogram.png",
                    "02_reliable_switch_count_histogram.png",
                    "03_bidirectional_pair_histogram.png",
                    "04_run_length_distribution.png",
                    "05_score_distribution.png",
                    "06_switch_vs_occupancy.png",
                    "07_temperature_dependence.png",
                    "08_state_count_composition.png",
                )
            ],
        },
        "elapsed_seconds": elapsed,
    }
    (output_dir / "report_summary.json").write_text(
        json.dumps(report_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    readme = f"""Step08.5 visual judgment report
================================

This report is diagnostic only. It does not classify RTS pixels.

Recommended review order
------------------------
1. overview_report.pdf
2. 05_score_distribution.png
3. 06_switch_vs_occupancy.png
4. 07_temperature_dependence.png
5. top_candidate_gallery.pdf

Questions to answer before Step09
---------------------------------
- Is there a natural break in reliable switch count or ranking score?
- Do high-ranked pixels show distinct plateaus rather than broad noise?
- Are both transition directions represented?
- Are minority states occupied long enough to be credible?
- Does the behavior persist across more than one temperature bin?
- Are direct 1<->3 transitions physically plausible for 3-state candidates?
- Are near-valley assignments concentrated around apparent transitions?

Centered residual input:
{centered_path}

Scale divisor applied:
{centered_scale}
"""
    (output_dir / "README_report.txt").write_text(
        readme, encoding="utf-8"
    )

    print(f"PASS: {output_dir}")
    print(f"Overview: {overview_pdf_path}")
    print(f"Gallery:  {gallery_pdf_path}")
    print(f"Elapsed:  {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
