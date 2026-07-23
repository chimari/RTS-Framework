#!/usr/bin/env python3
"""
Step09 QA Report v1.0.0
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


SCRIPT_VERSION = "1.0.0"
MAX_STATES = 3

DECISION_REJECT = 0
DECISION_REVIEW = 1
DECISION_ACCEPT = 2
DECISION_STRONG_ACCEPT = 3
DECISION_CODES = (0, 1, 2, 3)
DECISION_NAMES = ("reject", "review", "accept", "strong_accept")

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
    p.add_argument("--strong-top-n", type=int, default=100)
    p.add_argument("--accept-top-n", type=int, default=100)
    p.add_argument("--review-top-n", type=int, default=50)
    p.add_argument("--reject-top-n", type=int, default=100)
    p.add_argument("--reject-boundary-n", type=int, default=50)
    p.add_argument("--reject-per-reason-n", type=int, default=20)
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
    return DECISION_NAMES[int(code)]


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
    fig.suptitle("RTS Step09 QA Report", fontsize=18, y=0.96)

    text = [
        f"Step09 version: {summary09.get('script_version', 'unknown')}",
        f"Candidates: {len(candidate_table):,}",
        f"Temperature bins: {len(bin_table)}",
        "",
        "Candidate decisions",
        f"  reject: {counts.get('reject', 0):,}",
        f"  review: {counts.get('review', 0):,}",
        f"  accept: {counts.get('accept', 0):,}",
        f"  strong accept: {counts.get('strong_accept', 0):,}",
        "",
        "Candidate/bin decisions",
        f"  reject: {bin_counts.get('reject', 0):,}",
        f"  review: {bin_counts.get('review', 0):,}",
        f"  accept: {bin_counts.get('accept', 0):,}",
        f"  strong accept: {bin_counts.get('strong_accept', 0):,}",
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
    for code in DECISION_CODES:
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



def add_threshold_page(pdf: PdfPages, summary09: dict) -> None:
    thresholds = summary09.get("thresholds", {})
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("Step09 threshold summary", fontsize=17, y=0.96)
    if not thresholds:
        text = "No threshold metadata found in Step09 summary.json"
    else:
        lines = [f"{key}: {thresholds[key]}" for key in sorted(thresholds)]
        midpoint = (len(lines) + 1) // 2
        fig.text(0.06, 0.88, "\n".join(lines[:midpoint]), va="top", fontsize=9.5)
        fig.text(0.53, 0.88, "\n".join(lines[midpoint:]), va="top", fontsize=9.5)
        text = ""
    if text:
        fig.text(0.08, 0.85, text, va="top", fontsize=12)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_state_count_page(pdf: PdfPages, diagnostics: pd.DataFrame,
                         output_path: Path) -> pd.DataFrame:
    groups = [("all", diagnostics)] + [
        (decision_name(code), diagnostics[diagnostics["decision_code"] == code])
        for code in DECISION_CODES
    ]
    rows = []
    for name, sub in groups:
        counts = sub["observed_state_count"].value_counts()
        total = max(len(sub), 1)
        for nstate in (1, 2, 3):
            count = int(counts.get(nstate, 0))
            rows.append({"group": name, "state_count": nstate,
                         "count": count, "fraction": count / total})
    out = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [g[0] for g in groups]
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for nstate in (1, 2, 3):
        vals = np.array([
            out[(out.group == name) & (out.state_count == nstate)]["fraction"].iloc[0]
            for name in names
        ])
        ax.bar(x, vals, bottom=bottom, label=f"{nstate}-state")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction")
    ax.set_title("Observed state-count composition")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return out


def add_assignment_quality_page(pdf: PdfPages, diagnostics: pd.DataFrame,
                                frame_state: np.ndarray, quality: np.ndarray,
                                bin_frames: list[np.ndarray], output_path: Path,
                                max_candidates: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for code in DECISION_CODES:
        sub = diagnostics[diagnostics["decision_code"] == code]
        pairs = sub[["temperature_bin_index", "candidate_index"]].drop_duplicates().to_numpy(int)
        if len(pairs) > max_candidates:
            pairs = pairs[rng.choice(len(pairs), max_candidates, replace=False)]
        counts = np.zeros((MAX_STATES, 4), dtype=np.int64)
        for b, c in pairs:
            frames = bin_frames[int(b)]
            st = np.asarray(frame_state[frames, int(c)], np.int16)
            qu = np.asarray(quality[frames, int(c)], np.int16)
            for state_index in range(MAX_STATES):
                mask = st == state_index
                if np.any(mask):
                    counts[state_index] += np.bincount(qu[mask], minlength=4)[:4]
        for state_index, state_name in enumerate(("Lower", "Middle", "Upper")):
            denom = counts[state_index, 1:4].sum()
            for qcode, qname in ((1, "Excellent"), (2, "Normal"), (3, "Near valley")):
                rows.append({
                    "decision": decision_name(code), "state": state_name,
                    "quality": qname, "count": int(counts[state_index, qcode]),
                    "fraction": float(counts[state_index, qcode] / denom) if denom else np.nan,
                })
    out = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    for ax, state_name in zip(axes, ("Lower", "Middle", "Upper")):
        names = [decision_name(c) for c in DECISION_CODES]
        x = np.arange(4)
        bottom = np.zeros(4)
        for qname in ("Excellent", "Normal", "Near valley"):
            vals = np.array([
                out[(out.decision == name) & (out.state == state_name) &
                    (out.quality == qname)]["fraction"].iloc[0]
                for name in names
            ])
            vals = np.nan_to_num(vals)
            ax.bar(x, vals, bottom=bottom, label=qname)
            bottom += vals
        ax.set_title(state_name)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.2)
    axes[0].set_ylabel("Assignment-quality fraction")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("State assignment quality by decision and state")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return out


def reason_counts_table(diagnostics: pd.DataFrame, code: int) -> pd.DataFrame:
    counts: dict[str, int] = {}
    sub = diagnostics[diagnostics["decision_code"] == code]
    for text in sub["reason_text"].fillna(""):
        for reason in [x.strip() for x in str(text).split(";") if x.strip()]:
            counts[reason] = counts.get(reason, 0) + 1
    return pd.DataFrame([{"reason": k, "count": v} for k, v in counts.items()]).sort_values(
        "count", ascending=False
    ) if counts else pd.DataFrame(columns=["reason", "count"])


def add_reason_page(pdf: PdfPages, diagnostics: pd.DataFrame,
                    output_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    reject = reason_counts_table(diagnostics, DECISION_REJECT)
    review = reason_counts_table(diagnostics, DECISION_REVIEW)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, table, title in zip(axes, (reject, review), ("Reject", "Review")):
        if table.empty:
            ax.text(0.5, 0.5, "No reasons", ha="center")
            ax.axis("off")
            continue
        shown = table.head(15).sort_values("count")
        ax.barh(shown["reason"], shown["count"])
        ax.set_xscale("log")
        ax.set_title(f"{title} reason counts")
        ax.grid(True, axis="x", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return reject, review


def add_candidate_acceptance_page(pdf: PdfPages, candidate_table: pd.DataFrame,
                                  output_path: Path) -> None:
    cols = [c for c in ("accepted_bin_count", "strong_accepted_bin_count",
                         "acceptance_fraction_of_multistate_bins") if c in candidate_table]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4.8), squeeze=False)
    for ax, col in zip(axes.ravel(), cols):
        vals = pd.to_numeric(candidate_table[col], errors="coerce").dropna().to_numpy(float)
        if "count" in col:
            bins = np.arange(-0.5, (vals.max() if vals.size else 0) + 1.5, 1)
        else:
            bins = np.linspace(0, 1, 41)
        ax.hist(vals, bins=bins)
        ax.set_yscale("log")
        ax.set_title(col.replace("_", " "))
        ax.set_ylabel("Candidates")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def auto_qa_text(diagnostics: pd.DataFrame, state_counts: pd.DataFrame,
                 assignment: pd.DataFrame, reject_reasons: pd.DataFrame,
                 review_reasons: pd.DataFrame) -> str:
    lines = ["Automatic QA observations"]
    total = len(diagnostics)
    for code in DECISION_CODES:
        n = int(np.count_nonzero(diagnostics["decision_code"] == code))
        lines.append(f"- {decision_name(code)}: {n:,} candidate/bin rows ({n/max(total,1):.2%})")
    for name in ("accept", "strong_accept"):
        row2 = state_counts[(state_counts.group == name) & (state_counts.state_count == 2)]
        row3 = state_counts[(state_counts.group == name) & (state_counts.state_count == 3)]
        if not row2.empty and not row3.empty:
            lines.append(f"- {name}: 2-state {row2.iloc[0].fraction:.2%}, 3-state {row3.iloc[0].fraction:.2%}")
    upper = assignment[(assignment.state == "Upper") & (assignment.quality == "Excellent")]
    if not upper.empty:
        vals = ", ".join(f"{r.decision}={r.fraction:.2%}" for r in upper.itertuples())
        lines.append(f"- Upper-state Excellent fractions: {vals}")
    if not reject_reasons.empty:
        r = reject_reasons.iloc[0]
        lines.append(f"- Most frequent Reject reason: {r.reason} ({int(r['count']):,})")
    if not review_reasons.empty:
        r = review_reasons.iloc[0]
        lines.append(f"- Most frequent Review reason: {r.reason} ({int(r['count']):,})")
    lines.append("- These statements describe the current classification output; they do not by themselves prove a physical detector property.")
    return "\n".join(lines)


def add_text_page(pdf: PdfPages, title: str, text: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(title, fontsize=17, y=0.96)
    fig.text(0.06, 0.88, text, va="top", ha="left", fontsize=10.5, linespacing=1.5)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    for name in (
        "strong_top_n", "accept_top_n", "review_top_n",
        "reject_top_n", "reject_boundary_n", "reject_per_reason_n",
        "max_scatter_points"
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

    strong_rows = qa[qa["decision_code"] == DECISION_STRONG_ACCEPT].copy()
    strong_rows = strong_rows.sort_values(
        ["qa_score", "minimum_state_separation_sigma", "maximum_plateau_frames"],
        ascending=[False, False, False]
    ).head(args.strong_top_n).reset_index(drop=True)
    strong_rows["qa_rank"] = np.arange(1, len(strong_rows) + 1)

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

    # Reject galleries: representative high-score rejects and boundary rejects.
    all_reject_rows = qa[qa["decision_code"] == DECISION_REJECT].copy()
    reject_gallery_rows = all_reject_rows.sort_values(
        ["qa_score", "minimum_state_separation_sigma", "maximum_plateau_frames"],
        ascending=[False, False, False]
    ).head(args.reject_top_n).reset_index(drop=True)
    reject_gallery_rows["qa_rank"] = np.arange(1, len(reject_gallery_rows) + 1)

    # Boundary rejects: closest to acceptance by failed-criterion count.
    reject_rows = all_reject_rows.copy()
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
        add_threshold_page(pdf, s9)
        state_count_table = add_state_count_page(
            pdf, diagnostics, output_dir / "01_state_count_composition.png"
        )
        assignment_quality_table = add_assignment_quality_page(
            pdf, diagnostics, frame_state, quality, bin_frames,
            output_dir / "02_state_assignment_quality.png",
            max_candidates=min(args.max_scatter_points, 20000), rng=rng,
        )
        add_candidate_acceptance_page(
            pdf, candidate_table, output_dir / "03_candidate_acceptance.png"
        )
        reject_reason_table, review_reason_table = add_reason_page(
            pdf, diagnostics, output_dir / "04_reason_counts.png"
        )

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

        optional_metrics = [
            ("reliable_pair_fraction", "Reliable pair fraction", np.linspace(0, 1, 81)),
            ("near_valley_fraction", "Near-valley frame fraction", np.linspace(0, 1, 81)),
            ("direct_1_3_fraction", "Direct 1↔3 switch fraction", np.linspace(0, 1, 81)),
        ]
        for idx_opt, (column, label_opt, bins_opt) in enumerate(optional_metrics, start=11):
            if column in diagnostics.columns:
                plot_distribution_by_decision(
                    diagnostics, column, label_opt,
                    f"{label_opt} by Step09 decision",
                    output_dir / f"{idx_opt:02d}_{column}_by_decision.png",
                    pdf, log_x=False, bins=bins_opt,
                )

        # 2D comparison: plateau versus separation
        fig, ax = plt.subplots(figsize=(9, 6))
        for code in DECISION_CODES:
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
        matrix = np.zeros((n_bins, 4), dtype=int)
        for b in range(n_bins):
            for code in DECISION_CODES:
                matrix[b, code] = int(
                    np.count_nonzero(bin_decision[b] == code)
                )
        x = np.arange(n_bins)
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for code in DECISION_CODES:
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
                np.count_nonzero((bin_decision[b] == DECISION_ACCEPT) |
                                 (bin_decision[b] == DECISION_STRONG_ACCEPT))
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

        qa_text = auto_qa_text(
            diagnostics, state_count_table, assignment_quality_table,
            reject_reason_table, review_reason_table,
        )
        add_text_page(pdf, "Automatic QA observations", qa_text)

    state_count_table.to_csv(output_dir / "state_count_composition.csv", index=False)
    assignment_quality_table.to_csv(output_dir / "state_assignment_quality.csv", index=False)
    reject_reason_table.to_csv(output_dir / "reject_reason_counts.csv", index=False)
    review_reason_table.to_csv(output_dir / "review_reason_counts.csv", index=False)
    (output_dir / "automatic_qa.txt").write_text(qa_text + "\n", encoding="utf-8")

    make_gallery(
        rows=strong_rows,
        pdf_path=output_dir / "strong_accept_gallery.pdf",
        index_path=output_dir / "strong_accept_gallery_index.csv",
        title="Strong-accept RTS candidates — QA ranking",
        frame_state=frame_state, quality=quality,
        centered_raw=centered_raw, centered_scale=centered_scale,
        state_center=state_center, bin_frames=bin_frames,
        labels=labels, catalog=catalog, per_page=args.gallery_per_page,
    )
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
        rows=reject_gallery_rows,
        pdf_path=output_dir / "reject_gallery.pdf",
        index_path=output_dir / "reject_gallery_index.csv",
        title="Rejected candidates — representative high-score examples",
        frame_state=frame_state, quality=quality,
        centered_raw=centered_raw, centered_scale=centered_scale,
        state_center=state_center, bin_frames=bin_frames,
        labels=labels, catalog=catalog, per_page=args.gallery_per_page,
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

    reason_gallery_dir = output_dir / "reject_by_reason"
    reason_gallery_dir.mkdir(exist_ok=True)
    for reason in reject_reason_table["reason"].head(12).tolist():
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(reason))
        rows_reason = all_reject_rows[
            all_reject_rows["reason_text"].fillna("").map(
                lambda value: reason in [x.strip() for x in str(value).split(";")]
            )
        ].sort_values("qa_score", ascending=False).head(args.reject_per_reason_n).copy()
        rows_reason["qa_rank"] = np.arange(1, len(rows_reason) + 1)
        make_gallery(
            rows=rows_reason,
            pdf_path=reason_gallery_dir / f"reject_{safe}.pdf",
            index_path=reason_gallery_dir / f"reject_{safe}_index.csv",
            title=f"Rejected candidates — reason: {reason}",
            frame_state=frame_state, quality=quality,
            centered_raw=centered_raw, centered_scale=centered_scale,
            state_center=state_center, bin_frames=bin_frames,
            labels=labels, catalog=catalog, per_page=args.gallery_per_page,
        )

    # Boundary gallery: lowest strong/accept and highest review/reject.
    boundary_parts = []
    for code, ascending, label in (
        (DECISION_STRONG_ACCEPT, True, "strong_low"),
        (DECISION_ACCEPT, True, "accept_low"),
        (DECISION_REVIEW, False, "review_high"),
        (DECISION_REJECT, False, "reject_high"),
    ):
        part = qa[qa["decision_code"] == code].sort_values(
            "qa_score", ascending=ascending
        ).head(args.reject_boundary_n).copy()
        part["boundary_group"] = label
        boundary_parts.append(part)
    boundary_rows = pd.concat(boundary_parts, ignore_index=True) if boundary_parts else pd.DataFrame()
    boundary_rows["qa_rank"] = np.arange(1, len(boundary_rows) + 1)
    make_gallery(
        rows=boundary_rows,
        pdf_path=output_dir / "decision_boundary_gallery.pdf",
        index_path=output_dir / "decision_boundary_gallery_index.csv",
        title="Decision-boundary examples",
        frame_state=frame_state, quality=quality,
        centered_raw=centered_raw, centered_scale=centered_scale,
        state_center=state_center, bin_frames=bin_frames,
        labels=labels, catalog=catalog, per_page=args.gallery_per_page,
    )

    # Summary tables
    comparison_rows = []
    for code in DECISION_CODES:
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

    readme = f"""RTS Step09 QA Report
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
