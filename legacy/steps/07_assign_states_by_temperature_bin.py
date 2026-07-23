#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step07: assign Step06 histogram states frame by frame.

This step performs only:
- valley-boundary state assignment
- signed distance from the assigned Step06 peak
- assignment quality flags
- refined centers, sigmas, and occupancies

It does not calculate transitions, dwell times, switching rates, RTS flags,
or dictionary membership.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy.ndimage import gaussian_filter1d

from common.cli import add_common_arguments, validate_common_arguments
from common.io import prepare_output_dir, sha256_file, write_json
from common.version import PIPELINE_VERSION

SCRIPT_VERSION = "7.1.0"
MAX_STATES = 3

Q_INVALID = 0
Q_EXCELLENT = 1
Q_NORMAL = 2
Q_NEAR_VALLEY = 3
Q_OUTSIDE_RANGE = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assign Step06 states using reconstructed histogram valleys."
    )
    p.add_argument("--centered-dir", type=Path,
                   default=Path("05_dataset_centered_timeseries"))
    p.add_argument("--state-dir", type=Path,
                   default=Path("06_histogram_state_detection"))
    p.add_argument("--candidate-block", type=int, default=512)
    p.add_argument("--excellent-distance-ADU", type=float, default=0.50)
    p.add_argument("--near-valley-distance-ADU", type=float, default=0.25)
    p.add_argument("--outside-range-margin-ADU", type=float, default=1.00)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--hash-inputs", action="store_true")
    p.add_argument("--flush-every-blocks", type=int, default=5)
    add_common_arguments(p, output_default="07_frame_state_assignment")
    return p.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def find_centered_residual(directory: Path, summary: dict) -> Path:
    for key in ("centered_residual", "centered_residual_output"):
        value = summary.get(key)
        if value:
            p = Path(str(value))
            if p.is_file():
                return p
            p = directory / p.name
            if p.is_file():
                return p
    found = sorted(directory.glob("centered_residual_x4_*.npy"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected one centered_residual_x4_*.npy in {directory}; "
            f"found {len(found)}"
        )
    return found[0]


def resolve_file(directory: Path, summary: dict,
                 key: str, fallback: str) -> Path:
    value = summary.get(key)
    if value:
        p = Path(str(value))
        if p.is_file():
            return p
        p = directory / p.name
        if p.is_file():
            return p
    p = directory / fallback
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def parse_indices(value: object) -> list[int]:
    text = str(value).strip()
    return [int(v) for v in text.split(";") if v.strip()] if text else []


def build_bin_frames(bin_df: pd.DataFrame, dataset_df: pd.DataFrame,
                     n_frames: int) -> list[np.ndarray]:
    need_b = {"temperature_bin_index", "dataset_indices"}
    need_d = {"dataset_index", "frame_start", "frame_stop_exclusive"}
    if need_b - set(bin_df.columns):
        raise KeyError(f"temperature_bin_index.csv lacks {need_b-set(bin_df.columns)}")
    if need_d - set(dataset_df.columns):
        raise KeyError(f"dataset_index.csv lacks {need_d-set(dataset_df.columns)}")

    bins = bin_df.sort_values("temperature_bin_index").reset_index(drop=True)
    numbers = pd.to_numeric(
        bins["temperature_bin_index"], errors="raise"
    ).to_numpy(np.int64)
    if not np.array_equal(numbers, np.arange(len(bins))):
        raise ValueError("temperature_bin_index must be 0..n_bins-1")

    ds = dataset_df.copy()
    ds["dataset_index"] = pd.to_numeric(
        ds["dataset_index"], errors="raise"
    ).astype(np.int64)
    ds = ds.set_index("dataset_index")
    owner = np.full(n_frames, -1, dtype=np.int32)
    result = []

    for row in bins.itertuples(index=False):
        b = int(row.temperature_bin_index)
        pieces = []
        for d in parse_indices(row.dataset_indices):
            if d not in ds.index:
                raise KeyError(f"Unknown dataset_index {d}")
            r = ds.loc[d]
            start, stop = int(r.frame_start), int(r.frame_stop_exclusive)
            if start < 0 or stop > n_frames or stop <= start:
                raise ValueError(f"Bad frame range [{start},{stop}) for dataset {d}")
            pieces.append(np.arange(start, stop, dtype=np.int64))
        if not pieces:
            raise ValueError(f"Temperature bin {b} has no frames")
        frames = np.concatenate(pieces)
        if np.any(owner[frames] >= 0):
            raise ValueError("A frame belongs to more than one temperature bin")
        owner[frames] = b
        result.append(frames)

    if np.any(owner < 0):
        raise ValueError(
            f"{np.count_nonzero(owner < 0)} frames belong to no temperature bin"
        )
    return result


def validate_catalogs(a: pd.DataFrame, b: pd.DataFrame, n: int) -> None:
    if len(a) != n or len(b) != n:
        raise ValueError("Candidate catalog length mismatch")
    for names in (("timeseries_column",), ("extraction_index",),
                  ("full_x", "full_y"), ("x", "y")):
        if all(x in a.columns and x in b.columns for x in names):
            for name in names:
                av = pd.to_numeric(a[name], errors="raise").to_numpy()
                bv = pd.to_numeric(b[name], errors="raise").to_numpy()
                if not np.array_equal(av, bv):
                    raise ValueError(f"Candidate catalogs disagree in {name}")
            return
    print("WARNING: catalog alignment checked by row count only", file=sys.stderr)


def valleys_x4(values: np.ndarray, centers: np.ndarray,
               hist_min: int, hist_max: int,
               sigma_adu: float) -> np.ndarray:
    """Find first smoothed-histogram minimum between adjacent Step06 peaks."""
    if len(centers) <= 1:
        return np.empty(0, dtype=np.int32)
    idx = values.astype(np.int64, copy=False) - hist_min
    n_hist = hist_max - hist_min + 1
    if np.any(idx < 0) or np.any(idx >= n_hist):
        raise ValueError("Residual value lies outside Step06 histogram range")
    counts = np.bincount(idx, minlength=n_hist).astype(np.float64)
    smooth = gaussian_filter1d(
        counts, sigma_adu * 4.0, mode="nearest"
    ) if sigma_adu > 0 else counts

    peaks = np.rint(centers * 4.0 - hist_min).astype(np.int64)
    peaks = np.clip(peaks, 0, n_hist - 1)
    if not np.all(np.diff(peaks) > 0):
        raise ValueError("Peak centers are not strictly increasing")

    out = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        out.append(hist_min + left + int(np.argmin(smooth[left:right+1])))
    out = np.asarray(out, dtype=np.int32)
    if len(out) > 1 and not np.all(np.diff(out) > 0):
        raise ValueError("Reconstructed valleys are not strictly increasing")
    return out


def assign(values_x4: np.ndarray, centers: np.ndarray,
           cuts_x4: np.ndarray, excellent: float,
           near_valley: float, outside_margin: float):
    if len(centers) == 1:
        labels = np.ones(len(values_x4), dtype=np.uint8)
    else:
        labels = (
            np.searchsorted(cuts_x4, values_x4, side="left") + 1
        ).astype(np.uint8)

    values_adu = values_x4.astype(np.float32) / 4.0
    assigned_centers = centers[labels.astype(np.int64) - 1].astype(np.float32)
    distance = (values_adu - assigned_centers).astype(np.float32)

    quality = np.full(len(values_x4), Q_NORMAL, dtype=np.uint8)
    quality[np.abs(distance) <= excellent] = Q_EXCELLENT
    outside = (
        (values_adu < centers[0] - outside_margin)
        | (values_adu > centers[-1] + outside_margin)
    )
    quality[outside] = Q_OUTSIDE_RANGE
    if len(cuts_x4):
        cuts_adu = cuts_x4.astype(np.float32) / 4.0
        dvalley = np.min(
            np.abs(values_adu[:, None] - cuts_adu[None, :]), axis=1
        )
        quality[(dvalley <= near_valley) & ~outside] = Q_NEAR_VALLEY
    return labels, distance, quality


def checkpoint(shape, n_bins, b, c):
    return {
        "step": "07_assign_states_by_temperature_bin",
        "script_version": SCRIPT_VERSION,
        "shape": list(shape),
        "temperature_bin_count": n_bins,
        "completed_temperature_bin_index": b,
        "completed_candidate_stop": c,
    }


def main() -> int:
    args = parse_args()
    validate_common_arguments(args)
    if args.candidate_block <= 0 or args.flush_every_blocks <= 0:
        raise ValueError("block sizes must be positive")
    for name in ("excellent_distance_ADU", "near_valley_distance_ADU",
                 "outside_range_margin_ADU"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")

    t0 = time.perf_counter()
    cdir = args.centered_dir.expanduser().resolve()
    sdir = args.state_dir.expanduser().resolve()
    outdir = args.output_dir.expanduser().resolve()
    s5_path, s6_path = cdir/"summary.json", sdir/"summary.json"
    s5, s6 = load_json(s5_path), load_json(s6_path)
    if not s5.get("validation_passed", False):
        raise ValueError("Step05 validation_passed is not true")
    if not s6.get("validation_passed", False):
        raise ValueError("Step06 validation_passed is not true")

    step06_max = int(s6.get("max_states", MAX_STATES))
    if not 1 <= step06_max <= MAX_STATES:
        raise ValueError(f"Step06 max_states={step06_max}; supported 1..3")

    residual_path = find_centered_residual(cdir, s5)
    state_count_path = resolve_file(
        sdir, s6, "state_count_output", "state_count_uint8.npy"
    )
    peak_path = resolve_file(
        sdir, s6, "peak_center_output", "peak_center_ADU_float32.npy"
    )
    dataset_path = cdir/"dataset_index.csv"
    metadata_path = cdir/"frame_metadata.csv"
    cat5_path = cdir/"candidate_catalog.csv"
    cat6_path = sdir/"candidate_catalog.csv"
    bins_path = sdir/"temperature_bin_index.csv"
    for p in (dataset_path, metadata_path, cat5_path, cat6_path, bins_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    residual = np.load(residual_path, mmap_mode="r")
    state_count = np.load(state_count_path, mmap_mode="r")
    peaks = np.load(peak_path, mmap_mode="r")
    if residual.ndim != 2 or residual.dtype.kind != "i":
        raise ValueError("Centered residual must be a signed 2-D array")
    n_frames, n_candidates = map(int, residual.shape)
    if state_count.shape[1] != n_candidates:
        raise ValueError("State-count candidate axis mismatch")
    n_bins = int(state_count.shape[0])
    if peaks.shape != (n_bins, n_candidates, step06_max):
        raise ValueError(f"Unexpected peak array shape {peaks.shape}")
    actual_max = int(np.max(state_count))
    if actual_max > MAX_STATES or np.any(state_count < 1):
        raise ValueError(f"Actual state-count range is invalid: 1..{actual_max}")

    cat5, cat6 = pd.read_csv(cat5_path), pd.read_csv(cat6_path)
    validate_catalogs(cat5, cat6, n_candidates)
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != n_frames:
        raise ValueError("frame_metadata row mismatch")
    dataset_df, bins_df = pd.read_csv(dataset_path), pd.read_csv(bins_path)
    if len(bins_df) != n_bins:
        raise ValueError("temperature-bin count mismatch")
    bin_frames = build_bin_frames(bins_df, dataset_df, n_frames)

    hist_min = int(s6["histogram_x4_min"])
    hist_max = int(s6["histogram_x4_max"])
    sigma_adu = float(s6["smooth_sigma_ADU"])

    names = {
        "state": "frame_state_uint8.npy",
        "distance": "frame_distance_ADU_float32.npy",
        "quality": "assign_quality_uint8.npy",
        "valley": "valley_boundary_ADU_float32.npy",
        "center": "state_center_refined_ADU_float32.npy",
        "occupancy": "state_occupancy_refined_float32.npy",
        "sigma": "state_sigma_refined_ADU_float32.npy",
        "count": "assigned_frame_count_uint16.npy",
    }
    frame_shape = (n_frames, n_candidates)
    state_shape = (n_bins, n_candidates, MAX_STATES)
    valley_shape = (n_bins, n_candidates, MAX_STATES-1)
    bc_shape = (n_bins, n_candidates)

    if args.resume:
        cp = load_json(outdir/"checkpoint.json")
        if cp.get("shape") != list(frame_shape) or \
           int(cp.get("temperature_bin_count", -1)) != n_bins:
            raise ValueError("Checkpoint mismatch")
        resume_bin = int(cp.get("completed_temperature_bin_index", 0))
        resume_col = int(cp.get("completed_candidate_stop", 0))
        mode = "r+"
    else:
        prepare_output_dir(outdir, overwrite=args.overwrite)
        resume_bin = resume_col = 0
        mode = "w+"

    fs = open_memmap(outdir/names["state"], mode=mode,
                     dtype=np.uint8, shape=frame_shape)
    fd = open_memmap(outdir/names["distance"], mode=mode,
                     dtype=np.float32, shape=frame_shape)
    fq = open_memmap(outdir/names["quality"], mode=mode,
                     dtype=np.uint8, shape=frame_shape)
    vb = open_memmap(outdir/names["valley"], mode=mode,
                     dtype=np.float32, shape=valley_shape)
    rc = open_memmap(outdir/names["center"], mode=mode,
                     dtype=np.float32, shape=state_shape)
    ro = open_memmap(outdir/names["occupancy"], mode=mode,
                     dtype=np.float32, shape=state_shape)
    rs = open_memmap(outdir/names["sigma"], mode=mode,
                     dtype=np.float32, shape=state_shape)
    ac = open_memmap(outdir/names["count"], mode=mode,
                     dtype=np.uint16, shape=bc_shape)
    arrays = [fs, fd, fq, vb, rc, ro, rs, ac]

    if not args.resume:
        fs[:] = 0
        fd[:] = np.nan
        fq[:] = 0
        vb[:] = np.nan
        rc[:] = ro[:] = rs[:] = np.nan
        ac[:] = 0
        shutil.copy2(cat6_path, outdir/"candidate_catalog.csv")
        shutil.copy2(metadata_path, outdir/"frame_metadata.csv")
        shutil.copy2(bins_path, outdir/"temperature_bin_index.csv")

    blocks_per_bin = math.ceil(n_candidates / args.candidate_block)
    total_blocks = n_bins * blocks_per_bin
    done = resume_bin * blocks_per_bin + resume_col // args.candidate_block
    since_flush = 0

    print(f"Step07: {n_frames:,} frames x {n_candidates:,} candidates; "
          f"{n_bins} temperature bins")
    for b, rows in enumerate(bin_frames):
        if b < resume_bin:
            continue
        start = resume_col if b == resume_bin else 0
        label = bins_df.iloc[b].get("temperature_bin_label", f"bin_{b}")
        print(f"Temperature bin {b+1}/{n_bins}: {label}; {len(rows)} frames")

        for c0 in range(start, n_candidates, args.candidate_block):
            c1 = min(c0 + args.candidate_block, n_candidates)
            cols = np.arange(c0, c1, dtype=np.int64)
            block = np.asarray(residual[np.ix_(rows, cols)])

            for j in range(block.shape[1]):
                c = c0 + j
                k = int(state_count[b, c])
                centers = np.asarray(peaks[b, c, :k], dtype=np.float64)
                if not np.all(np.isfinite(centers)) or \
                   (k > 1 and not np.all(np.diff(centers) > 0)):
                    raise ValueError(f"Bad centers at bin={b}, candidate={c}")

                values = block[:, j]
                cuts = valleys_x4(
                    values, centers, hist_min, hist_max, sigma_adu
                )
                labels, distance, quality = assign(
                    values, centers, cuts,
                    args.excellent_distance_ADU,
                    args.near_valley_distance_ADU,
                    args.outside_range_margin_ADU,
                )
                fs[rows, c] = labels
                fd[rows, c] = distance
                fq[rows, c] = quality
                ac[b, c] = min(len(rows), np.iinfo(np.uint16).max)
                if len(cuts):
                    vb[b, c, :len(cuts)] = cuts.astype(np.float32)/4.0

                values_adu = values.astype(np.float64)/4.0
                for s in range(k):
                    selected = labels == s+1
                    count = int(np.count_nonzero(selected))
                    if count:
                        vv = values_adu[selected]
                        rc[b, c, s] = np.mean(vv)
                        ro[b, c, s] = count/len(labels)
                        rs[b, c, s] = np.std(vv, ddof=0)

            done += 1
            since_flush += 1
            if since_flush >= args.flush_every_blocks or c1 == n_candidates:
                for a in arrays:
                    a.flush()
                nb, nc = (b+1, 0) if c1 == n_candidates else (b, c1)
                write_json(outdir/"checkpoint.json",
                           checkpoint(frame_shape, n_bins, nb, nc))
                since_flush = 0

            if args.progress_every and \
               (done % args.progress_every == 0 or c1 == n_candidates):
                elapsed = time.perf_counter()-t0
                frac = done/total_blocks
                eta = elapsed*(1-frac)/frac if frac else math.nan
                print(f"  candidates {c1:,}/{n_candidates:,}; "
                      f"blocks {done}/{total_blocks}; "
                      f"elapsed {elapsed:.1f}s; ETA {eta:.1f}s")
        resume_col = 0

    for a in arrays:
        a.flush()

    state_arr = np.asarray(fs)
    quality_arr = np.asarray(fq)
    occupancy_arr = np.asarray(ro)
    expected = n_frames*n_candidates
    assigned = int(np.count_nonzero(state_arr))
    occupancy_sum = np.nansum(occupancy_arr, axis=2)
    occupancy_ok = np.allclose(occupancy_sum, 1.0, atol=1e-6, rtol=0)

    bin_rows = []
    for b, rows in enumerate(bin_frames):
        st, qu = np.asarray(fs[rows]), np.asarray(fq[rows])
        row = bins_df.iloc[b].to_dict()
        row.update({
            "assigned_samples": int(st.size),
            "state_1_samples": int(np.count_nonzero(st == 1)),
            "state_2_samples": int(np.count_nonzero(st == 2)),
            "state_3_samples": int(np.count_nonzero(st == 3)),
            "excellent_samples": int(np.count_nonzero(qu == Q_EXCELLENT)),
            "normal_samples": int(np.count_nonzero(qu == Q_NORMAL)),
            "near_valley_samples": int(np.count_nonzero(qu == Q_NEAR_VALLEY)),
            "outside_range_samples": int(np.count_nonzero(qu == Q_OUTSIDE_RANGE)),
            "near_valley_fraction": float(np.mean(qu == Q_NEAR_VALLEY)),
            "outside_range_fraction": float(np.mean(qu == Q_OUTSIDE_RANGE)),
        })
        bin_rows.append(row)
    pd.DataFrame(bin_rows).to_csv(
        outdir/"temperature_bin_assignment_summary.csv", index=False
    )

    cand = cat6.copy()
    sc = np.asarray(state_count)
    cand["max_step06_state_count"] = sc.max(axis=0)
    cand["multistate_temperature_bin_count"] = np.count_nonzero(sc >= 2, axis=0)
    cand["three_state_temperature_bin_count"] = np.count_nonzero(sc == 3, axis=0)
    cand["near_valley_fraction_all_frames"] = np.mean(
        quality_arr == Q_NEAR_VALLEY, axis=0
    )
    cand["outside_range_fraction_all_frames"] = np.mean(
        quality_arr == Q_OUTSIDE_RANGE, axis=0
    )
    cand["mean_absolute_peak_distance_ADU"] = np.nanmean(
        np.abs(np.asarray(fd)), axis=0
    )
    cand.to_csv(outdir/"candidate_assignment_summary.csv", index=False)

    elapsed = time.perf_counter()-t0
    quality_counts = {
        "invalid": int(np.count_nonzero(quality_arr == Q_INVALID)),
        "excellent": int(np.count_nonzero(quality_arr == Q_EXCELLENT)),
        "normal": int(np.count_nonzero(quality_arr == Q_NORMAL)),
        "near_valley": int(np.count_nonzero(quality_arr == Q_NEAR_VALLEY)),
        "outside_range": int(np.count_nonzero(quality_arr == Q_OUTSIDE_RANGE)),
    }
    valid = bool(
        assigned == expected
        and occupancy_ok
        and int(np.max(state_arr)) <= actual_max
    )
    summary = {
        "step": "07_assign_states_by_temperature_bin",
        "script_version": SCRIPT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "validation_passed": valid,
        "responsibility": "state assignment only; no transitions or RTS decision",
        "array_shape": [n_frames, n_candidates],
        "frame_count": n_frames,
        "candidate_count": n_candidates,
        "temperature_bin_count": n_bins,
        "step06_script_version": s6.get("script_version"),
        "step06_max_states": step06_max,
        "actual_max_state_count": actual_max,
        "assignment_algorithm":
            "exact x4 histogram -> Step06 Gaussian smoothing -> "
            "minimum between adjacent peaks -> valley-boundary labels",
        "quality_codes": {
            "0": "invalid",
            "1": "excellent",
            "2": "normal",
            "3": "near_valley",
            "4": "outside_peak_range",
        },
        "excellent_distance_ADU": args.excellent_distance_ADU,
        "near_valley_distance_ADU": args.near_valley_distance_ADU,
        "outside_range_margin_ADU": args.outside_range_margin_ADU,
        "assigned_sample_count": assigned,
        "expected_sample_count": expected,
        "assigned_fraction": assigned/expected,
        "quality_counts": quality_counts,
        "occupancy_sum_validation_passed": bool(occupancy_ok),
        "outputs": {k: str(outdir/v) for k, v in names.items()},
        "elapsed_seconds": elapsed,
    }
    write_json(outdir/"summary.json", summary)

    manifest = {
        "inputs": {
            "centered_residual": str(residual_path),
            "step05_summary": str(s5_path),
            "step06_summary": str(s6_path),
            "dataset_index": str(dataset_path),
            "frame_metadata": str(metadata_path),
            "step05_candidate_catalog": str(cat5_path),
            "step06_candidate_catalog": str(cat6_path),
            "temperature_bin_index": str(bins_path),
            "state_count": str(state_count_path),
            "peak_center": str(peak_path),
        },
        "outputs": [
            *names.values(),
            "candidate_catalog.csv",
            "frame_metadata.csv",
            "temperature_bin_index.csv",
            "temperature_bin_assignment_summary.csv",
            "candidate_assignment_summary.csv",
            "checkpoint.json",
            "summary.json",
            "manifest.json",
        ],
    }
    if args.hash_inputs:
        manifest["input_sha256"] = {
            k: sha256_file(Path(v)) for k, v in manifest["inputs"].items()
        }
    write_json(outdir/"manifest.json", manifest)

    print(f"PASS: Step07 completed in {elapsed:.1f} s")
    print(f"  assigned samples : {assigned:,}")
    print(f"  near valley      : {quality_counts['near_valley']:,} "
          f"({quality_counts['near_valley']/expected:.4%})")
    print(f"  outside range    : {quality_counts['outside_range']:,} "
          f"({quality_counts['outside_range']/expected:.4%})")
    return 0 if valid else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
