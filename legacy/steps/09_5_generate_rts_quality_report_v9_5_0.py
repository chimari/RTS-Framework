#!/usr/bin/env python3
"""
Step09.5 v9.5.0
Quality-assurance visualization report for Step09 RTS decisions.

Responsibilities
----------------
- Compare reject / review / accept populations.
- Inspect accepted and review candidates with time-series galleries.
- Visualize plateau, separation, occupancy, switch fraction, and temperature dependence.
- Summarize rejection reasons.
- Produce no new RTS classification and no image correction.

Inputs
------
- Step05 output directory
- Step07 output directory
- Step09 output directory

Main outputs
------------
overview_report.pdf
accept_gallery.pdf
review_gallery.pdf
reject_boundary_gallery.pdf
decision_comparison.csv
gallery_index_*.csv
report_summary.json
PNG figures
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


SCRIPT_VERSION = "9.5.0"
MAX_STATES = 3

DECISION_REJECT = 0
DECISION_REVIEW = 1
DECISION_ACCEPT = 2

Q_INVALID = 0
Q_EXCELLENT = 1
Q_NORMAL = 2
Q_NEAR_VALLEY = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a QA report for Step09 RTS decisions without "
            "changing the classification."
        )
    )
    p.add_argument("--state-dir", type=Path, required=True,
                   help="Step07 output directory.")
    p.add_argument("--classification-dir", type=Path, required=True,
                   help="Step09 output directory.")
    p.add_argument("--centered-dir", type=Path, required=True,
                   help="Step05 output directory.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("09_5_rts_quality_report"))
    p.add_argument("--accept-top-n", type=int, default=100)
    p.add_argument("--review-top-n", type=int, default=50)
    p.add_argument("--reject-boundary-n", type=int, default=50)
    p.add_argument("--gallery-per-page", type=int, default=4,
                   choices=(1, 2, 4))
    p.add_argument("--max-scatter-points", type=int, default=100000)
    p.add_argument("--random-seed", type=int, default=12345)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


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
            p = Path(str(value))
            if p.is_file():
                return p
            q = directory / p.name
            if q.is_file():
                return q
    for name in fallbacks:
        p = directory / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Could not resolve {keys} / {fallbacks} in {directory}"
    )


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
    bins = bin_table.sort_values("temperature_bin_index").reset_index(drop=True)
    nums = pd.to_numeric(
        bins["temperature_bin_index"], errors="raise"
    ).to_numpy(np.int64)
    if not np.array_equal(nums, np.arange(len(bins))):
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
    result = []
    for row in bins.itertuples(index=False):
        pieces = []
        for idx in parse_indices(row.dataset_indices):
            if idx not in datasets.index:
                raise KeyError(f"Unknown dataset_index {idx}")
            d = datasets.loc[idx]
            start = int(d.frame_start)
            stop = int(d.frame_stop_exclusive)
            if start < 0 or stop > frame_count or stop <= start:
                raise ValueError(f"Invalid interval [{start}, {stop})")
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
        raise ValueError("Some frames are not assigned to any temperature bin")
    return result


def bin_label(row: pd.Series, b: int) -> str:
    for col in (
        "temperature_bin_label", "bin_label",
        "label", "temperature_label"
    ):
        if col in row and pd.notna(row[col]):
            return str(row[col])
    for col in (
        "temperature_center_C", "temperature_median_C",
        "temperature_mean_C", "temperature_C"
    ):
        if col in row and pd.notna(row[col]):
            return f"{float(row[col]):+.2f} °C"
    return f"Bin {b}"


def positive_log_bins(values: np.ndarray, max_bins: int = 80) -> np.ndarray:
    x = np.asarray(values, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.array([0.5, 1.5])
    vmax = float(x.max())
    if vmax <= 1:
        return np.array([0.5, 1.5])
    n = min(max_bins, max(12, int(math.ceil(math.log2(vmax))) * 5))
    edges = np.unique(np.rint(np.geomspace(1, vmax + 1, n)).astype(int))
    return np.unique(np.r_[0.5, edges + 0.5])


def decision_name(code: int) -> str:
    return ("reject", "review", "accept")[int(code)]


def get_coordinate_text(catalog: pd.DataFrame, candidate: int) -> str:
    for xcol, ycol in (
        ("x", "y"), ("pixel_x", "pixel_y"), ("sensor_x", "sensor_y")
    ):
        if xcol in catalog.columns and ycol in catalog.columns:
            row = catalog.iloc[candidate]
            return f"x={row[xcol]}, y={row[ycol]}"
    return ""


def add_overview_page(
    pdf: PdfPages,
    summary09: dict,
    candidate_table: pd.DataFrame,
    bin_table: pd.DataFrame,
) -> None:
    counts = summary09.get("candidate_counts", {})
    bin_counts = summary09.get("candidate_bin_counts", {})
    thresholds = summary09.get("thresholds", {})

    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("Step09.5 RTS classification QA report", fontsize=18, y=0.96)

    text = [
        f"Step09 version: {summary09.get('script_version', 'unknown')}",
        f"Candidates: {len(candidate_table):,}",
        f"Temperature bins: {len(bin_table)}",
        "",
        "Candidate decisions",
        f"  reject: {counts.get('reject', 0):,}",
        f"  review: {counts.get('review', 0):,}",
        f"  accept: {counts.get('accept', 0):,}",
        "",
        "Candidate/bin decisions",
        f"  reject: {bin_counts.get('reject', 0):,}",
        f"  review: {bin_counts.get('review', 0):,}",
        f"  accept: {bin_counts.get('accept', 0):,}",
        "",
        "Selected thresholds",
        f"  max switch fraction: {thresholds.get('max_switch_fraction', 'n/a')}",
        f"  min maximum plateau: {thresholds.get('min_max_plateau_frames', 'n/a')} frames",
        f"  min median plateau: {thresholds.get('min_median_plateau_frames', 'n/a')} frames",
        f"  min separation: {thresholds.get('min_separation_sigma', 'n/a')} sigma",
        "",
        "This report does not alter Step09 decisions.",
    ]
    fig.text(
        0.07, 0.86, "\n".join(text),
        va="top", ha="left", fontsize=12.5, linespacing=1.45
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_by_decision(
    table: pd.DataFrame,
    column: str,
    xlabel: str,
    title: str,
    output_path: Path,
    pdf: PdfPages,
    log_x: bool = False,
    log_y: bool = True,
    bins: int | np.ndarray = 60,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for code in (DECISION_REJECT, DECISION_REVIEW, DECISION_ACCEPT):
        values = pd.to_numeric(
            table.loc[table["decision_code"] == code, column],
            errors="coerce",
        ).to_numpy(float)
        values = values[np.isfinite(values)]
        if log_x:
            values = values[values > 0]
        if values.size == 0:
            continue
        use_bins = positive_log_bins(values) if log_x else bins
        ax.hist(
            values, bins=use_bins, histtype="step",
            linewidth=1.5, label=decision_name(code)
        )
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Candidate / temperature-bin combinations")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_gallery(
    *,
    rows: pd.DataFrame,
    pdf_path: Path,
    index_path: Path,
    title: str,
    frame_state: np.ndarray,
    quality: np.ndarray,
    centered_raw: np.ndarray,
    centered_scale: float,
    state_center: np.ndarray,
    bin_frames: list[np.ndarray],
    labels: list[str],
    catalog: pd.DataFrame,
    per_page: int,
) -> None:
    page_shape = {1: (1, 1), 2: (2, 1), 4: (2, 2)}[per_page]
    index_rows = []

    with PdfPages(pdf_path) as pdf:
        if rows.empty:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.text(
                0.5, 0.5, f"No rows available for {title}",
                ha="center", va="center", fontsize=16
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        else:
            for start in range(0, len(rows), per_page):
                subset = rows.iloc[start:start + per_page]
                fig, axes = plt.subplots(
                    page_shape[0], page_shape[1],
                    figsize=(11.69, 8.27),
                    squeeze=False,
                )
                axes_flat = axes.ravel()

                for panel, (_, row) in enumerate(subset.iterrows()):
                    ax = axes_flat[panel]
                    c = int(row["candidate_index"])
                    b = int(row["temperature_bin_index"])
                    frames = bin_frames[b]

                    values = (
                        np.asarray(centered_raw[frames, c], dtype=float)
                        / centered_scale
                    )
                    states = np.asarray(frame_state[frames, c], np.int16)
                    quals = np.asarray(quality[frames, c], np.int16)
                    centers = np.asarray(state_center[b, c], float)
                    x = np.arange(len(frames))

                    good = quals == Q_EXCELLENT
                    normal = quals == Q_NORMAL
                    valley = quals == Q_NEAR_VALLEY

                    ax.plot(x, values, linewidth=0.65, alpha=0.55,
                            label="Centered ADU")
                    ax.scatter(x[good], values[good], s=8, marker="o",
                               label="Excellent")
                    ax.scatter(x[normal], values[normal], s=7, marker=".",
                               label="Normal")
                    if np.any(valley):
                        ax.scatter(x[valley], values[valley], s=14,
                                   marker="x", label="Near valley")

                    n_state = int(row.get("observed_state_count", 0))
                    for s in range(min(n_state, MAX_STATES)):
                        if np.isfinite(centers[s]):
                            ax.axhline(
                                centers[s], linestyle="--",
                                linewidth=0.9, alpha=0.8,
                                label=f"State {s+1} center"
                            )

                    coord = get_coordinate_text(catalog, c)
                    rank_text = ""
                    if "qa_rank" in row and pd.notna(row["qa_rank"]):
                        rank_text = f"Rank {int(row['qa_rank'])}: "
                    subtitle = f"{rank_text}candidate {c}"
                    if coord:
                        subtitle += f" ({coord})"
                    ax.set_title(subtitle, fontsize=10)
                    ax.set_xlabel(f"Frame position in {labels[b]}")
                    ax.set_ylabel("Centered signal [ADU]")
                    ax.grid(True, alpha=0.2)

                    metrics = [
                        f"decision={row.get('decision', '')}",
                        f"switch={int(row.get('reliable_switch_count', 0))}",
                        f"plateau max={int(row.get('maximum_plateau_frames', 0))}",
                        f"plateau med={float(row.get('median_plateau_frames', np.nan)):.1f}",
                        f"sep={float(row.get('minimum_state_separation_sigma', np.nan)):.2f}σ",
                        f"switch frac={float(row.get('switch_fraction', np.nan)):.3f}",
                    ]
                    if str(row.get("reason_text", "")).strip():
                        metrics.append(f"reason={row.get('reason_text')}")
                    ax.text(
                        0.01, 0.99, "\n".join(metrics),
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=7.5,
                        bbox=dict(boxstyle="round", facecolor="white",
                                  alpha=0.78)
                    )

                    index_rows.append({
                        "page": start // per_page + 1,
                        "panel": panel + 1,
                        "candidate_index": c,
                        "temperature_bin_index": b,
                        "temperature_bin_label": labels[b],
                        "decision": row.get("decision", ""),
                        "reliable_switch_count": row.get(
                            "reliable_switch_count", np.nan
                        ),
                        "maximum_plateau_frames": row.get(
                            "maximum_plateau_frames", np.nan
                        ),
                        "median_plateau_frames": row.get(
                            "median_plateau_frames", np.nan
                        ),
                        "minimum_state_separation_sigma": row.get(
                            "minimum_state_separation_sigma", np.nan
                        ),
                        "switch_fraction": row.get(
                            "switch_fraction", np.nan
                        ),
                        "reason_text": row.get("reason_text", ""),
                    })

                for unused in range(len(subset), len(axes_flat)):
                    axes_flat[unused].axis("off")

                handles, legend_labels = axes_flat[0].get_legend_handles_labels()
                unique = {}
                for h, label in zip(handles, legend_labels):
                    if label not in unique:
                        unique[label] = h
                if unique:
                    fig.legend(
                        unique.values(), unique.keys(),
                        loc="lower center", ncol=min(5, len(unique)),
                        fontsize=8
                    )
                fig.suptitle(title, fontsize=14)
                fig.tight_layout(rect=(0, 0.05, 1, 0.95))
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    pd.DataFrame(index_rows).to_csv(index_path, index=False)


def normalize_for_rank(values: pd.Series, higher_is_better: bool) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=float)
    if not np.any(finite):
        return out
    lo, hi = np.nanpercentile(x[finite], [5, 95])
    if hi <= lo:
        out[finite] = 0.5
    else:
        out[finite] = np.clip((x[finite] - lo) / (hi - lo), 0, 1)
    if not higher_is_better:
        out[finite] = 1.0 - out[finite]
    return out


def main() -> int:
    args = parse_args()
    for name in (
        "accept_top_n", "review_top_n",
        "reject_boundary_n", "max_scatter_points"
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    started = time.perf_counter()
    state_dir = args.state_dir.expanduser().resolve()
    class_dir = args.classification_dir.expanduser().resolve()
    centered_dir = args.centered_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output(output_dir, args.overwrite)

    s7 = read_json(state_dir / "summary.json")
    s9 = read_json(class_dir / "summary.json")
    s5 = read_json(centered_dir / "summary.json")
    if not s7.get("validation_passed", False):
        raise ValueError("Step07 validation did not pass")
    if not s9.get("validation_passed", False):
        raise ValueError("Step09 validation did not pass")

    state_path = resolve_file(
        state_dir, s7,
        ("state", "frame_state_output"),
        ("frame_state_uint8.npy",)
    )
    quality_path = resolve_file(
        state_dir, s7,
        ("quality", "assignment_quality_output"),
        ("assign_quality_uint8.npy",)
    )
    center_path = resolve_file(
        state_dir, s7,
        ("center", "state_center"),
        ("state_center_refined_ADU_float32.npy",)
    )
    centered_path = resolve_file(
        centered_dir, s5,
        ("centered_residual", "centered_residual_output"),
        ("centered_residual_x4_int16.npy", "centered_residual_float32.npy")
    )

    candidate_table_path = class_dir / "candidate_rts_decision.csv"
    diagnostic_table_path = class_dir / "candidate_bin_diagnostics.csv"
    bin_decision_path = class_dir / "rts_bin_decision_uint8.npy"
    reason_mask_path = class_dir / "rts_bin_reason_mask_uint16.npy"

    for p in (
        candidate_table_path, diagnostic_table_path,
        bin_decision_path, reason_mask_path,
        state_dir / "candidate_catalog.csv",
        state_dir / "temperature_bin_index.csv",
        centered_dir / "dataset_index.csv",
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    frame_state = np.load(state_path, mmap_mode="r")
    quality = np.load(quality_path, mmap_mode="r")
    state_center = np.load(center_path, mmap_mode="r")
    centered_raw = np.load(centered_path, mmap_mode="r")
    bin_decision = np.load(bin_decision_path, mmap_mode="r")
    reason_mask = np.load(reason_mask_path, mmap_mode="r")

    if frame_state.ndim != 2:
        raise ValueError("frame_state must be 2D")
    n_frames, n_candidates = map(int, frame_state.shape)

    candidate_table = pd.read_csv(candidate_table_path)
    diagnostics = pd.read_csv(diagnostic_table_path)
    catalog = pd.read_csv(state_dir / "candidate_catalog.csv")
    bin_table = pd.read_csv(
        state_dir / "temperature_bin_index.csv"
    ).sort_values("temperature_bin_index").reset_index(drop=True)
    dataset_table = pd.read_csv(centered_dir / "dataset_index.csv")
    bin_frames = build_bin_frames(bin_table, dataset_table, n_frames)
    n_bins = len(bin_frames)
    labels = [bin_label(bin_table.iloc[b], b) for b in range(n_bins)]

    if len(candidate_table) != n_candidates:
        raise ValueError("candidate_rts_decision.csv row count mismatch")
    if len(catalog) != n_candidates:
        raise ValueError("candidate_catalog.csv row count mismatch")
    if bin_decision.shape != (n_bins, n_candidates):
        raise ValueError("rts_bin_decision shape mismatch")
    if state_center.shape != (n_bins, n_candidates, MAX_STATES):
        raise ValueError("state_center shape mismatch")
    if centered_raw.shape != frame_state.shape:
        raise ValueError("centered residual shape mismatch")

    centered_scale = 4.0 if "_x4_" in centered_path.name else 1.0
    rng = np.random.default_rng(args.random_seed)

    required_columns = {
        "candidate_index", "temperature_bin_index", "decision_code",
        "decision", "reason_text", "observed_state_count",
        "reliable_switch_count", "switch_fraction",
        "minimum_active_state_occupancy",
        "singleton_run_fraction",
        "median_plateau_frames", "maximum_plateau_frames",
        "minimum_state_separation_sigma",
    }
    missing = required_columns - set(diagnostics.columns)
    if missing:
        raise KeyError(
            f"candidate_bin_diagnostics.csv missing {sorted(missing)}"
        )

    # QA ranking:
    # accept: strong separation + long plateau + balanced occupancy,
    #         while avoiding extreme switch fraction
    qa = diagnostics.copy()
    qa["qa_score"] = (
        0.30 * normalize_for_rank(
            qa["minimum_state_separation_sigma"], True
        )
        + 0.25 * normalize_for_rank(
            qa["maximum_plateau_frames"], True
        )
        + 0.15 * normalize_for_rank(
            qa["median_plateau_frames"], True
        )
        + 0.15 * normalize_for_rank(
            qa["minimum_active_state_occupancy"], True
        )
        + 0.10 * normalize_for_rank(
            qa["singleton_run_fraction"], False
        )
        + 0.05 * normalize_for_rank(
            qa["switch_fraction"], False
        )
    )

    accept_rows = qa[qa["decision_code"] == DECISION_ACCEPT].copy()
    accept_rows = accept_rows.sort_values(
        ["qa_score", "minimum_state_separation_sigma",
         "maximum_plateau_frames"],
        ascending=[False, False, False]
    ).head(args.accept_top_n).reset_index(drop=True)
    accept_rows["qa_rank"] = np.arange(1, len(accept_rows) + 1)

    review_rows = qa[qa["decision_code"] == DECISION_REVIEW].copy()
    review_rows = review_rows.sort_values(
        ["qa_score", "minimum_state_separation_sigma",
         "maximum_plateau_frames"],
        ascending=[False, False, False]
    ).head(args.review_top_n).reset_index(drop=True)
    review_rows["qa_rank"] = np.arange(1, len(review_rows) + 1)

    # Boundary rejects: those closest to acceptance by count of failed criteria
    reject_rows = qa[qa["decision_code"] == DECISION_REJECT].copy()
    if not reject_rows.empty:
        reject_rows["failed_criterion_count"] = reject_rows[
            "reason_text"
        ].fillna("").map(
            lambda x: 0 if not str(x).strip() else len(str(x).split(";"))
        )
        reject_rows = reject_rows.sort_values(
            [
                "failed_criterion_count", "qa_score",
                "minimum_state_separation_sigma",
                "maximum_plateau_frames",
            ],
            ascending=[True, False, False, False],
        ).head(args.reject_boundary_n).reset_index(drop=True)
        reject_rows["qa_rank"] = np.arange(1, len(reject_rows) + 1)

    overview_pdf = output_dir / "overview_report.pdf"
    with PdfPages(overview_pdf) as pdf:
        add_overview_page(pdf, s9, candidate_table, bin_table)

        plot_distribution_by_decision(
            diagnostics, "maximum_plateau_frames",
            "Maximum plateau length [frames]",
            "Maximum plateau length by Step09 decision",
            output_dir / "01_maximum_plateau_by_decision.png",
            pdf, log_x=True
        )
        plot_distribution_by_decision(
            diagnostics, "median_plateau_frames",
            "Median plateau length [frames]",
            "Median plateau length by Step09 decision",
            output_dir / "02_median_plateau_by_decision.png",
            pdf, log_x=True
        )
        plot_distribution_by_decision(
            diagnostics, "minimum_state_separation_sigma",
            "Minimum adjacent-state separation [sigma]",
            "State separation significance by Step09 decision",
            output_dir / "03_separation_sigma_by_decision.png",
            pdf, log_x=True
        )
        plot_distribution_by_decision(
            diagnostics, "switch_fraction",
            "Reliable switch fraction",
            "Switch fraction by Step09 decision",
            output_dir / "04_switch_fraction_by_decision.png",
            pdf, log_x=False, bins=np.linspace(0, 1, 81)
        )
        plot_distribution_by_decision(
            diagnostics, "minimum_active_state_occupancy",
            "Minimum active-state occupancy",
            "Minimum state occupancy by Step09 decision",
            output_dir / "05_minimum_occupancy_by_decision.png",
            pdf, log_x=False, bins=np.linspace(0, 0.5, 81)
        )
        plot_distribution_by_decision(
            diagnostics, "singleton_run_fraction",
            "Singleton-run fraction",
            "Singleton-run fraction by Step09 decision",
            output_dir / "06_singleton_fraction_by_decision.png",
            pdf, log_x=False, bins=np.linspace(0, 1, 81)
        )

        # 2D comparison: plateau versus separation
        fig, ax = plt.subplots(figsize=(9, 6))
        for code in (DECISION_REJECT, DECISION_REVIEW, DECISION_ACCEPT):
            sub = diagnostics[diagnostics["decision_code"] == code]
            x = pd.to_numeric(
                sub["maximum_plateau_frames"], errors="coerce"
            ).to_numpy(float)
            y = pd.to_numeric(
                sub["minimum_state_separation_sigma"], errors="coerce"
            ).to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            idx = np.flatnonzero(mask)
            if idx.size > args.max_scatter_points // 3:
                idx = rng.choice(
                    idx, args.max_scatter_points // 3, replace=False
                )
            ax.scatter(
                x[idx], y[idx], s=7, alpha=0.25,
                label=decision_name(code)
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Maximum plateau [frames]")
        ax.set_ylabel("Minimum separation [sigma]")
        ax.set_title("Plateau length versus state separation")
        ax.legend()
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output_dir / "07_plateau_vs_separation.png",
            dpi=180, bbox_inches="tight"
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Temperature dependence of decisions
        matrix = np.zeros((n_bins, 3), dtype=int)
        for b in range(n_bins):
            for code in (0, 1, 2):
                matrix[b, code] = int(
                    np.count_nonzero(bin_decision[b] == code)
                )
        x = np.arange(n_bins)
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for code in (DECISION_REJECT, DECISION_REVIEW, DECISION_ACCEPT):
            ax.plot(
                x, matrix[:, code], marker="o",
                label=decision_name(code)
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel("Temperature bin")
        ax.set_ylabel("Candidate/bin count")
        ax.set_title("Step09 decision count versus temperature")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output_dir / "08_temperature_decision_counts.png",
            dpi=180, bbox_inches="tight"
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Accepted fraction among multistate candidates
        multi_counts = np.zeros(n_bins, dtype=int)
        accept_counts = np.zeros(n_bins, dtype=int)
        review_counts = np.zeros(n_bins, dtype=int)
        for b in range(n_bins):
            multi_counts[b] = int(
                np.count_nonzero(bin_decision[b] != DECISION_REJECT)
                + np.count_nonzero(
                    (bin_decision[b] == DECISION_REJECT)
                    & (reason_mask[b] != 1)
                )
            )
            accept_counts[b] = int(
                np.count_nonzero(bin_decision[b] == DECISION_ACCEPT)
            )
            review_counts[b] = int(
                np.count_nonzero(bin_decision[b] == DECISION_REVIEW)
            )
        denom = np.maximum(multi_counts, 1)
        fig, ax = plt.subplots(figsize=(10, 5.8))
        ax.plot(
            x, accept_counts / denom,
            marker="o", label="accept / multistate"
        )
        ax.plot(
            x, review_counts / denom,
            marker="s", label="review / multistate"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Temperature bin")
        ax.set_ylabel("Fraction")
        ax.set_title("Decision fractions among multistate bins")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output_dir / "09_temperature_decision_fractions.png",
            dpi=180, bbox_inches="tight"
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Rejection reason counts from summary
        reason_counts = s9.get(
            "failure_reason_counts_among_multistate_bins", {}
        )
        if reason_counts:
            reason_df = pd.DataFrame({
                "reason": list(reason_counts.keys()),
                "count": list(reason_counts.values()),
            }).sort_values("count", ascending=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(reason_df["reason"], reason_df["count"])
            ax.set_xscale("log")
            ax.set_xlabel("Candidate/bin combinations")
            ax.set_title("Step09 failure-reason counts")
            ax.grid(True, axis="x", which="both", alpha=0.25)
            fig.tight_layout()
            fig.savefig(
                output_dir / "10_failure_reason_counts.png",
                dpi=180, bbox_inches="tight"
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    make_gallery(
        rows=accept_rows,
        pdf_path=output_dir / "accept_gallery.pdf",
        index_path=output_dir / "accept_gallery_index.csv",
        title="Accepted RTS candidates — QA ranking",
        frame_state=frame_state,
        quality=quality,
        centered_raw=centered_raw,
        centered_scale=centered_scale,
        state_center=state_center,
        bin_frames=bin_frames,
        labels=labels,
        catalog=catalog,
        per_page=args.gallery_per_page,
    )
    make_gallery(
        rows=review_rows,
        pdf_path=output_dir / "review_gallery.pdf",
        index_path=output_dir / "review_gallery_index.csv",
        title="Review candidates — strongest near-accept examples",
        frame_state=frame_state,
        quality=quality,
        centered_raw=centered_raw,
        centered_scale=centered_scale,
        state_center=state_center,
        bin_frames=bin_frames,
        labels=labels,
        catalog=catalog,
        per_page=args.gallery_per_page,
    )
    make_gallery(
        rows=reject_rows,
        pdf_path=output_dir / "reject_boundary_gallery.pdf",
        index_path=output_dir / "reject_boundary_gallery_index.csv",
        title="Rejected candidates closest to the acceptance boundary",
        frame_state=frame_state,
        quality=quality,
        centered_raw=centered_raw,
        centered_scale=centered_scale,
        state_center=state_center,
        bin_frames=bin_frames,
        labels=labels,
        catalog=catalog,
        per_page=args.gallery_per_page,
    )

    # Summary tables
    comparison_rows = []
    for code in (DECISION_REJECT, DECISION_REVIEW, DECISION_ACCEPT):
        sub = diagnostics[diagnostics["decision_code"] == code]
        row = {
            "decision_code": code,
            "decision": decision_name(code),
            "count": len(sub),
        }
        for col in (
            "reliable_switch_count",
            "switch_fraction",
            "minimum_active_state_occupancy",
            "singleton_run_fraction",
            "median_plateau_frames",
            "maximum_plateau_frames",
            "minimum_state_separation_sigma",
        ):
            values = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            row[f"{col}_median"] = (
                float(np.median(values)) if values.size else np.nan
            )
            row[f"{col}_p16"] = (
                float(np.percentile(values, 16)) if values.size else np.nan
            )
            row[f"{col}_p84"] = (
                float(np.percentile(values, 84)) if values.size else np.nan
            )
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "decision_comparison.csv", index=False
    )

    accept_rows.to_csv(
        output_dir / "accept_qa_ranking.csv", index=False
    )
    review_rows.to_csv(
        output_dir / "review_qa_ranking.csv", index=False
    )
    reject_rows.to_csv(
        output_dir / "reject_boundary_ranking.csv", index=False
    )

    validation = {
        "step09_validation_passed": bool(
            s9.get("validation_passed", False)
        ),
        "candidate_row_count_matches": bool(
            len(candidate_table) == n_candidates
        ),
        "bin_decision_shape_matches": bool(
            bin_decision.shape == (n_bins, n_candidates)
        ),
        "accept_gallery_rows_are_accept": bool(
            accept_rows.empty
            or np.all(
                accept_rows["decision_code"].to_numpy()
                == DECISION_ACCEPT
            )
        ),
        "review_gallery_rows_are_review": bool(
            review_rows.empty
            or np.all(
                review_rows["decision_code"].to_numpy()
                == DECISION_REVIEW
            )
        ),
        "reject_gallery_rows_are_reject": bool(
            reject_rows.empty
            or np.all(
                reject_rows["decision_code"].to_numpy()
                == DECISION_REJECT
            )
        ),
    }
    validation_passed = all(validation.values())

    elapsed = time.perf_counter() - started
    summary = {
        "step": "09_5_rts_classification_quality_report",
        "script_version": SCRIPT_VERSION,
        "validation_passed": validation_passed,
        "responsibility": (
            "quality assurance and visualization of Step09 decisions; "
            "no reclassification and no image correction"
        ),
        "frame_count": n_frames,
        "candidate_count": n_candidates,
        "temperature_bin_count": n_bins,
        "accept_gallery_count": len(accept_rows),
        "review_gallery_count": len(review_rows),
        "reject_boundary_gallery_count": len(reject_rows),
        "centered_residual_input": str(centered_path),
        "centered_residual_scale_divisor": centered_scale,
        "validation": validation,
        "outputs": {
            "overview_report": str(overview_pdf),
            "accept_gallery": str(output_dir / "accept_gallery.pdf"),
            "review_gallery": str(output_dir / "review_gallery.pdf"),
            "reject_boundary_gallery": str(
                output_dir / "reject_boundary_gallery.pdf"
            ),
            "decision_comparison": str(
                output_dir / "decision_comparison.csv"
            ),
            "accept_qa_ranking": str(
                output_dir / "accept_qa_ranking.csv"
            ),
            "review_qa_ranking": str(
                output_dir / "review_qa_ranking.csv"
            ),
            "reject_boundary_ranking": str(
                output_dir / "reject_boundary_ranking.csv"
            ),
        },
        "elapsed_seconds": elapsed,
    }
    (output_dir / "report_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    readme = f"""Step09.5 RTS classification QA report
=====================================

This report does not change Step09 decisions.

Recommended review order
------------------------
1. overview_report.pdf
2. accept_gallery.pdf
3. review_gallery.pdf
4. reject_boundary_gallery.pdf
5. decision_comparison.csv

Questions to answer before Step10
---------------------------------
- Do accepted candidates show discrete plateaus?
- Are accepted states sufficiently separated relative to within-state scatter?
- Are accepted candidates dominated by one-frame alternation?
- Does the accepted fraction vary smoothly with temperature?
- Are review candidates physically plausible enough to justify threshold changes?
- Are rejected boundary cases being rejected for sensible reasons?

QA ranking
----------
The gallery ranking is not a new classifier. It favors:
- larger state-separation significance
- longer maximum and median plateaus
- larger minimum state occupancy
- smaller singleton-run fraction
- smaller switch fraction

Centered residual input:
{centered_path}

Scale divisor:
{centered_scale}
"""
    (output_dir / "README_report.txt").write_text(
        readme, encoding="utf-8"
    )

    print(f"PASS: {output_dir}" if validation_passed
          else f"FAIL: {output_dir}")
    print(f"Overview: {overview_pdf}")
    print(f"Accept gallery: {output_dir / 'accept_gallery.pdf'}")
    print(f"Review gallery: {output_dir / 'review_gallery.pdf'}")
    print(
        "Reject boundary gallery: "
        f"{output_dir / 'reject_boundary_gallery.pdf'}"
    )
    print(f"Elapsed: {elapsed:.1f} s")
    return 0 if validation_passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
