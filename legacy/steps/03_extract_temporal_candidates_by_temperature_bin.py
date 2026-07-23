#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step03: temporal-variability candidates by temperature bin, with common ROI support."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from common.cli import add_common_arguments, validate_common_arguments
from common.io import prepare_output_dir, safe_name, sha256_file, write_json
from common.manifest import build_datasets, load_frame_manifest
from common.roi import ROI
from common.statistics import deterministic_sample_indices, robust_std, sigma_clip_1d
from common.temperature import make_temperature_bins
from common.version import PIPELINE_VERSION

SCRIPT_VERSION = "3.1.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select unstable pixels independently in temperature bins, then OR the masks.")
    p.add_argument("--frame-manifest", type=Path, default=Path("01_frame_index_output_19200x12800/frame_manifest.csv"))
    add_common_arguments(p, output_default="03_temporal_candidates_by_temperature_bin")
    p.add_argument("--shape", nargs=2, type=int, default=None, metavar=("HEIGHT", "WIDTH"))
    p.add_argument("--dtype", default=None)
    p.add_argument("--temperature-bin-tolerance", type=float, default=1.5)
    p.add_argument("--unstable-sigma", type=float, default=8.0)
    p.add_argument("--row-tile", type=int, default=128)
    p.add_argument("--sample-pixels", type=int, default=2_000_000)
    p.add_argument("--clip-sigma", type=float, default=5.0)
    p.add_argument("--clip-iterations", type=int, default=5)
    p.add_argument("--edge-margin", type=int, default=4)
    p.add_argument("--density-block", type=int, default=128)
    p.add_argument("--max-csv-candidates", type=int, default=0)
    p.add_argument("--keep-temperature-bin-std", action="store_true")
    p.add_argument("--hash-inputs", action="store_true")
    return p.parse_args()


def within_set_pooled_std_tile(bin_info: dict, full_shape: tuple[int, int], dtype: np.dtype,
                               roi: ROI, local_y0: int, local_y1: int,
                               progress_every: int, tile_index: int, total_tiles: int) -> np.ndarray:
    """Pool within-set M2 values. The returned tile is ROI-width only."""
    full_h, full_w = full_shape
    global_y0, global_y1 = roi.y0 + local_y0, roi.y0 + local_y1
    pooled_m2 = np.zeros((local_y1 - local_y0, roi.width), dtype=np.float64)
    total_df = 0
    started = time.time()
    for set_pos, d in enumerate(bin_info["datasets"], start=1):
        mean = np.zeros_like(pooled_m2)
        m2 = np.zeros_like(pooled_m2)
        count = 0
        for frame_pos, raw_path in enumerate(d["paths"], start=1):
            frame = np.memmap(raw_path, dtype=dtype, mode="r", shape=(full_h, full_w), order="C")
            values = np.asarray(frame[global_y0:global_y1, roi.x0:roi.x1], dtype=np.float64)
            count += 1
            delta = values - mean
            mean += delta / count
            m2 += delta * (values - mean)
            del values, frame
            if frame_pos == 1 or frame_pos % progress_every == 0 or frame_pos == len(d["paths"]):
                print(f"bin={bin_info['temperature_bin']} set={set_pos}/{len(bin_info['datasets'])}:{d['dataset']} "
                      f"tile={tile_index}/{total_tiles} global_rows={global_y0}:{global_y1} "
                      f"frame={frame_pos}/{len(d['paths'])} elapsed={(time.time()-started)/60:.1f} min")
        pooled_m2 += m2
        total_df += count - 1
    if total_df <= 0:
        return np.full((local_y1-local_y0, roi.width), np.nan, dtype=np.float32)
    return np.sqrt(pooled_m2 / total_df).astype(np.float32)


def block_mean(arr: np.ndarray, block: int) -> np.ndarray:
    h, w = arr.shape
    if block <= 0: raise ValueError("density block must be positive")
    hh, ww = (h // block) * block, (w // block) * block
    if hh == 0 or ww == 0:
        return np.asarray(arr, dtype=np.float32)
    a = np.asarray(arr[:hh, :ww], dtype=np.float32)
    return a.reshape(hh//block, block, ww//block, block).mean(axis=(1, 3))


def make_previews(mask_path: Path, global_y: np.ndarray, global_x: np.ndarray, roi: ROI,
                  preview_dir: Path, density_block: int, bin_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(global_x, global_y, s=0.2, marker=".", linewidths=0, rasterized=True)
    ax.set_xlim(roi.x0, roi.x1); ax.set_ylim(roi.y1, roi.y0); ax.set_aspect("equal")
    ax.set_title(f"Temperature-bin candidates: {len(global_y):,}")
    ax.set_xlabel("global x [pixel]"); ax.set_ylabel("global y [pixel]")
    fig.tight_layout(); fig.savefig(preview_dir / "candidate_map.png", dpi=180); plt.close(fig)

    mask = np.load(mask_path, mmap_mode="r")
    density = block_mean(mask, density_block)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(density, origin="upper", aspect="equal", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Candidate fraction")
    ax.set_title(f"ROI candidate density ({density_block} x {density_block} blocks)")
    ax.set_xlabel("ROI x block"); ax.set_ylabel("ROI y block")
    fig.tight_layout(); fig.savefig(preview_dir / "candidate_density.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(bin_summary["temperature_bin"], bin_summary["candidate_count"])
    ax.set_ylabel("Candidate pixels in ROI"); ax.set_xlabel("Temperature bin")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(preview_dir / "temperature_bin_candidate_counts.png", dpi=160); plt.close(fig)


def main() -> int:
    args = parse_args(); validate_common_arguments(args)
    if args.temperature_bin_tolerance < 0: raise ValueError("--temperature-bin-tolerance must be non-negative")
    if args.unstable_sigma <= 0 or args.row_tile <= 0 or args.sample_pixels <= 0: raise ValueError("Invalid positive-valued option")

    manifest_path = args.frame_manifest.expanduser().resolve()
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    preview_dir = output_dir / "previews"; preview_dir.mkdir()
    temp_dir = output_dir / "temporary_std"; temp_dir.mkdir()

    df, full_shape, dtype = load_frame_manifest(manifest_path, shape_arg=tuple(args.shape) if args.shape else None, dtype_arg=args.dtype)
    full_h, full_w = full_shape
    roi = ROI.from_args(args, full_width=full_w, full_height=full_h)
    datasets = build_datasets(df)
    bins = make_temperature_bins(datasets, args.temperature_bin_tolerance)
    if len(bins) > 32: raise ValueError("At most 32 temperature bins are supported")

    rh, rw = roi.shape; roi_pixels = rh * rw
    candidate_mask_path = output_dir / "candidate_mask.npy"
    hit_bits_path = output_dir / "temperature_bin_hit_bitmask.npy"
    best_excess_path = output_dir / "best_excess_sigma.npy"
    best_bin_path = output_dir / "best_temperature_bin_index.npy"
    candidate_mask = open_memmap(candidate_mask_path, mode="w+", dtype=np.uint8, shape=roi.shape)
    hit_bits = open_memmap(hit_bits_path, mode="w+", dtype=np.uint32, shape=roi.shape)
    best_excess = open_memmap(best_excess_path, mode="w+", dtype=np.float32, shape=roi.shape)
    best_bin = open_memmap(best_bin_path, mode="w+", dtype=np.int16, shape=roi.shape)
    candidate_mask[:] = 0; hit_bits[:] = 0; best_excess[:] = -np.inf; best_bin[:] = -1

    sample_idx = deterministic_sample_indices(roi_pixels, args.sample_pixels)
    sample_y, sample_x = sample_idx // rw, sample_idx % rw
    total_tiles = math.ceil(rh / args.row_tile)
    summary_rows = []; started = time.time()

    print(f"Manifest: {manifest_path}\nFull geometry: {full_h} x {full_w}\nROI: x={roi.x0}:{roi.x1}, y={roi.y0}:{roi.y1} ({rh} x {rw})")
    print(f"Temperature bins: {len(bins)}; rule median(std)+{args.unstable_sigma:g}*robust_std(std)")
    for b in bins: print(f"  {b['temperature_bin']}: " + ", ".join(d["dataset"] for d in b["datasets"]))

    for seq, b in enumerate(bins, start=1):
        idx = b["temperature_bin_index"]
        std_path = temp_dir / f"temporal_std__{idx:02d}__{safe_name(b['temperature_bin'])}.npy"
        std_map = open_memmap(std_path, mode="w+", dtype=np.float32, shape=roi.shape)
        print(f"=== Bin {seq}/{len(bins)}: {b['temperature_bin']} ===")
        for tile_i, ly0 in enumerate(range(0, rh, args.row_tile), start=1):
            ly1 = min(rh, ly0 + args.row_tile)
            std_map[ly0:ly1, :] = within_set_pooled_std_tile(
                b, full_shape, dtype, roi, ly0, ly1, args.progress_every, tile_i, total_tiles)
            std_map.flush()

        sampled = np.asarray(std_map[sample_y, sample_x], dtype=np.float64)
        clipped = sigma_clip_1d(sampled[np.isfinite(sampled)], args.clip_sigma, args.clip_iterations)
        if clipped.size == 0: raise RuntimeError(f"No valid sampled std for {b['temperature_bin']}")
        med = float(np.median(clipped)); rs = robust_std(clipped)
        if not np.isfinite(rs) or rs <= 0: raise RuntimeError(f"Invalid robust std for {b['temperature_bin']}: {rs}")
        threshold = med + args.unstable_sigma * rs; bin_count = 0

        for ly0 in range(0, rh, args.row_tile):
            ly1 = min(rh, ly0 + args.row_tile)
            s = np.asarray(std_map[ly0:ly1, :])
            selected = np.isfinite(s) & (s > threshold)
            if args.edge_margin:
                global_ys = roi.y0 + np.arange(ly0, ly1)
                global_xs = roi.x0 + np.arange(rw)
                selected[(global_ys < args.edge_margin) | (global_ys >= full_h-args.edge_margin), :] = False
                selected[:, (global_xs < args.edge_margin) | (global_xs >= full_w-args.edge_margin)] = False
            excess = (s - med) / rs
            old = np.asarray(best_excess[ly0:ly1, :]); better = excess > old
            best_excess[ly0:ly1, :][better] = excess[better]
            best_bin[ly0:ly1, :][better] = idx
            candidate_mask[ly0:ly1, :] |= selected.astype(np.uint8)
            hit_bits[ly0:ly1, :][selected] |= np.uint32(1 << idx)
            bin_count += int(selected.sum())
        for a in (candidate_mask, hit_bits, best_excess, best_bin): a.flush()
        summary_rows.append({"temperature_bin_index": idx, "temperature_bin": b["temperature_bin"],
            "temperature_mean_C": b["temperature_mean_C"], "temperature_min_C": b["temperature_min_C"],
            "temperature_max_C": b["temperature_max_C"], "dataset_count": b["dataset_count"],
            "datasets": ";".join(d["dataset"] for d in b["datasets"]), "frame_count": b["frame_count"],
            "effective_degrees_of_freedom": sum(d["frame_count"]-1 for d in b["datasets"]),
            "temporal_std_median_adu": med, "temporal_std_robust_sigma_adu": rs,
            "threshold_adu": threshold, "candidate_count": bin_count, "candidate_fraction": bin_count/roi_pixels})
        if not args.keep_temperature_bin_std:
            del std_map; std_path.unlink(missing_ok=True)

    bin_summary = pd.DataFrame(summary_rows); bin_summary.to_csv(output_dir / "temperature_bin_summary.csv", index=False)
    mapping_rows = []
    for b in bins:
        for d in b["datasets"]:
            mapping_rows.append({"dataset_index": d["dataset_index"], "dataset": d["dataset"],
                "dataset_temperature_mean_C": d["temperature_mean_C"], "dataset_frame_count": d["frame_count"],
                "temperature_bin_index": b["temperature_bin_index"], "temperature_bin": b["temperature_bin"],
                "temperature_bin_mean_C": b["temperature_mean_C"]})
    pd.DataFrame(mapping_rows).to_csv(output_dir / "dataset_to_temperature_bin.csv", index=False)
    bin_summary[["temperature_bin_index", "temperature_bin", "temperature_mean_C", "dataset_count", "datasets", "frame_count"]].to_csv(output_dir / "temperature_bin_index.csv", index=False)

    local_y, local_x = np.nonzero(candidate_mask)
    global_y, global_x = roi.local_to_global(local_y, local_x)
    n_candidates = len(local_y)
    if args.max_csv_candidates and n_candidates > args.max_csv_candidates: raise RuntimeError(f"Candidates {n_candidates:,} exceed CSV limit")
    bits = np.asarray(hit_bits[local_y, local_x], dtype=np.uint32)
    hit_count = np.fromiter((int(v).bit_count() for v in bits), dtype=np.uint8, count=len(bits))
    catalog = pd.DataFrame({"y": global_y.astype(np.int32), "x": global_x.astype(np.int32),
        "roi_y": local_y.astype(np.int32), "roi_x": local_x.astype(np.int32),
        "linear_index": global_y.astype(np.int64)*full_w + global_x.astype(np.int64),
        "temperature_bin_hit_count": hit_count, "temperature_bin_hit_bitmask": bits,
        "best_excess_sigma": np.asarray(best_excess[local_y, local_x]),
        "best_temperature_bin_index": np.asarray(best_bin[local_y, local_x])})
    for b in bins:
        catalog[f"candidate__{safe_name(b['temperature_bin'])}"] = (bits & (1 << b["temperature_bin_index"])) != 0
    catalog = catalog.sort_values(["best_excess_sigma", "temperature_bin_hit_count", "y", "x"], ascending=[False, False, True, True], kind="stable").reset_index(drop=True)
    catalog.insert(0, "candidate_rank", np.arange(1, len(catalog)+1, dtype=np.int64))
    catalog.to_csv(output_dir / "candidate_catalog.csv", index=False)

    make_previews(candidate_mask_path, global_y, global_x, roi, preview_dir, args.density_block, bin_summary)
    elapsed = time.time() - started
    summary = {"step": "03_extract_temporal_candidates_by_temperature_bin", "script_version": SCRIPT_VERSION,
        "pipeline_version": PIPELINE_VERSION, "validation_passed": True, "frame_manifest": str(manifest_path),
        "full_shape": list(full_shape), "roi": roi.to_dict(), "array_shape": list(roi.shape), "raw_dtype": str(dtype),
        "dataset_count": len(datasets), "temperature_bin_count": len(bins),
        "temperature_bin_tolerance_C": args.temperature_bin_tolerance,
        "set_offset_handling": "pooled within-set variance; between-set offsets excluded",
        "threshold_rule": f"median(std)+{args.unstable_sigma:g}*robust_std(std)",
        "candidate_rule": "independent temperature-bin masks followed by OR union",
        "candidate_count": n_candidates, "candidate_fraction": n_candidates/roi_pixels,
        "roi_pixels": roi_pixels, "full_detector_pixels": full_h*full_w, "elapsed_seconds": elapsed}
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", {"step": summary["step"], "script_version": SCRIPT_VERSION,
        "pipeline_version": PIPELINE_VERSION, "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path) if args.hash_inputs else None,
        "coordinate_convention": {"catalog_x_y": "global detector coordinates, 0-based", "roi_x_y": "ROI-local coordinates, 0-based", "interval": "half-open"},
        "roi": roi.to_dict(), "outputs": ["candidate_catalog.csv", "candidate_mask.npy", "temperature_bin_hit_bitmask.npy",
        "best_excess_sigma.npy", "best_temperature_bin_index.npy", "temperature_bin_index.csv", "temperature_bin_summary.csv",
        "dataset_to_temperature_bin.csv", "summary.json", "manifest.json", "previews/"]})
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
