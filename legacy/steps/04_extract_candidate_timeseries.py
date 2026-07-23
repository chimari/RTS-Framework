#!/usr/bin/env python3
"""
04_extract_candidate_timeseries.py

Step 04 of the IMX811 RTS pipeline.

Extract, for every Step03 candidate and every frame:

  1. center_uint16
       raw value of the candidate pixel

  2. local_residual_x2_int16
       2 * (center - median of the eight immediate neighbours)

The median of eight integer neighbours can be a half integer.  Storing twice
its residual preserves the value exactly without floating-point storage:

    local_residual_x2 = 2*center - (4th + 5th sorted neighbour)

Array layout is (n_frames, n_candidates).  Candidates are sorted by global
linear pixel index for efficient raw-file access.  Frame metadata preserves
manifest order and dataset membership for Step05 dataset-wise centering.

The Step03 ROI is inherited automatically from candidate-dir/summary.json.
Explicit --x0/--x1/--y0/--y1 may be used to further restrict candidates for
debugging.  --full means no additional restriction; it does not add pixels
that are absent from the Step03 catalog.
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

from common.cli import add_common_arguments, validate_common_arguments
from common.io import prepare_output_dir, sha256_file, write_json
from common.manifest import load_frame_manifest
from common.roi import ROI
from common.version import PIPELINE_VERSION

SCRIPT_VERSION = "4.0.0"

NEIGHBOUR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract center and local-neighbour-residual time series for all Step03 candidates."
    )
    p.add_argument(
        "--frame-manifest", type=Path,
        default=Path("01_frame_index_output_19200x12800/frame_manifest.csv"),
    )
    p.add_argument(
        "--candidate-dir", type=Path,
        default=Path("03_temporal_candidates_by_temperature_bin"),
        help="Step03 output directory containing candidate_catalog.csv and summary.json.",
    )
    p.add_argument(
        "--candidate-catalog", type=Path, default=None,
        help="Override candidate-dir/candidate_catalog.csv.",
    )
    p.add_argument(
        "--shape", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=None,
        help="Raw shape only when absent from the frame manifest.",
    )
    p.add_argument(
        "--dtype", default=None,
        help="Raw dtype only when absent from the frame manifest.",
    )
    p.add_argument(
        "--candidate-chunk", type=int, default=250_000,
        help="Candidates processed per indexing chunk. Default: 250000.",
    )
    p.add_argument(
        "--flush-every", type=int, default=10,
        help="Flush output arrays and checkpoint every N frames. Default: 10.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted run using checkpoint.json.",
    )
    p.add_argument(
        "--hash-inputs", action="store_true",
        help="Record SHA-256 hashes of manifest and candidate catalog.",
    )
    p.add_argument(
        "--residual-dtype", choices=("int16", "int32"), default="int16",
        help="Storage dtype for local_residual_x2. int16 is safe for Mono12 data and is range-checked.",
    )
    add_common_arguments(p, output_default="04_candidate_timeseries")
    return p.parse_args()


def read_step03_summary(candidate_dir: Path) -> dict | None:
    path = candidate_dir / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def roi_from_dict(d: dict, full_width: int, full_height: int) -> ROI:
    roi = ROI(
        int(d["x0"]), int(d["x1"]), int(d["y0"]), int(d["y1"]),
        full_width, full_height,
    )
    roi.validate()
    return roi


def resolve_catalog_filter_roi(
    args: argparse.Namespace,
    step03_summary: dict | None,
    full_width: int,
    full_height: int,
) -> tuple[ROI, ROI]:
    """Return (step03_roi, effective_catalog_filter_roi)."""
    if step03_summary and isinstance(step03_summary.get("roi"), dict):
        step03_roi = roi_from_dict(step03_summary["roi"], full_width, full_height)
    else:
        step03_roi = ROI(0, full_width, 0, full_height, full_width, full_height)

    explicit = all(v is not None for v in (args.x0, args.x1, args.y0, args.y1))
    if explicit:
        requested = ROI.from_args(args, full_width=full_width, full_height=full_height)
        x0 = max(step03_roi.x0, requested.x0)
        x1 = min(step03_roi.x1, requested.x1)
        y0 = max(step03_roi.y0, requested.y0)
        y1 = min(step03_roi.y1, requested.y1)
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Requested ROI does not overlap the Step03 ROI")
        effective = ROI(x0, x1, y0, y1, full_width, full_height)
    else:
        effective = step03_roi
    return step03_roi, effective


def load_candidates(
    catalog_path: Path,
    width: int,
    height: int,
    effective_roi: ROI,
) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(catalog_path)
    required = {"x", "y"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Candidate catalog missing columns: {missing}")

    if "candidate_rank" not in df.columns:
        df.insert(0, "candidate_rank", np.arange(1, len(df) + 1, dtype=np.int64))

    x = pd.to_numeric(df["x"], errors="raise").to_numpy(np.int64)
    y = pd.to_numeric(df["y"], errors="raise").to_numpy(np.int64)
    original_count = len(df)

    keep = (
        (x >= effective_roi.x0) & (x < effective_roi.x1) &
        (y >= effective_roi.y0) & (y < effective_roi.y1)
    )
    df = df.loc[keep].copy()
    if df.empty:
        raise ValueError("No candidates remain after ROI filtering")

    x = pd.to_numeric(df["x"], errors="raise").to_numpy(np.int64)
    y = pd.to_numeric(df["y"], errors="raise").to_numpy(np.int64)
    if np.any(x < 1) or np.any(x >= width - 1) or np.any(y < 1) or np.any(y >= height - 1):
        raise ValueError("Candidate catalog contains pixels without complete 3x3 neighbour support")

    expected_linear = y * width + x
    if "linear_index" in df.columns:
        linear = pd.to_numeric(df["linear_index"], errors="raise").to_numpy(np.int64)
        if not np.array_equal(linear, expected_linear):
            raise ValueError("Candidate linear_index does not match y*width+x")
    else:
        df["linear_index"] = expected_linear

    if len(np.unique(expected_linear)) != len(expected_linear):
        raise ValueError("Candidate catalog contains duplicate pixel coordinates")

    if "timeseries_column" in df.columns:
        df = df.rename(columns={"timeseries_column": "previous_timeseries_column"})

    df = df.sort_values(["linear_index", "candidate_rank"], kind="stable").reset_index(drop=True)
    df.insert(0, "timeseries_column", np.arange(len(df), dtype=np.int64))
    return df, original_count


def build_frame_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "manifest_row" not in out.columns:
        out.insert(0, "manifest_row", np.arange(len(out), dtype=np.int64))
    keep_first = [
        "manifest_row", "dataset", "frame_index", "temperature_C",
        "exposure_s", "resolved_path",
    ]
    cols = [c for c in keep_first if c in out.columns]
    extras = [c for c in out.columns if c not in cols]
    out = out[cols + extras].copy()
    out.insert(0, "timeseries_row", np.arange(len(out), dtype=np.int64))

    dataset_names = list(dict.fromkeys(out["dataset"].astype(str)))
    dataset_map = {name: i for i, name in enumerate(dataset_names)}
    out.insert(2, "dataset_index", out["dataset"].astype(str).map(dataset_map).astype(np.int32))
    return out


def validate_resume(
    output_dir: Path,
    n_frames: int,
    n_candidates: int,
    residual_dtype: np.dtype,
) -> int:
    checkpoint_path = output_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError("--resume requested but checkpoint.json is absent")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("shape") != [n_frames, n_candidates]:
        raise ValueError("Checkpoint shape does not match current inputs")
    if checkpoint.get("residual_dtype") != str(residual_dtype):
        raise ValueError("Checkpoint residual dtype does not match current option")
    completed = int(checkpoint.get("completed_frames", 0))
    if not (0 <= completed <= n_frames):
        raise ValueError("Invalid completed_frames in checkpoint")
    for name in ("center_uint16.npy", f"local_residual_x2_{residual_dtype}.npy"):
        if not (output_dir / name).is_file():
            raise FileNotFoundError(output_dir / name)
    return completed


def open_outputs(
    output_dir: Path,
    n_frames: int,
    n_candidates: int,
    residual_dtype: np.dtype,
    resume: bool,
):
    center_path = output_dir / "center_uint16.npy"
    residual_path = output_dir / f"local_residual_x2_{residual_dtype}.npy"
    mode = "r+" if resume else "w+"
    center = open_memmap(center_path, mode=mode, dtype=np.uint16, shape=(n_frames, n_candidates))
    residual = open_memmap(residual_path, mode=mode, dtype=residual_dtype, shape=(n_frames, n_candidates))
    return center_path, residual_path, center, residual


def extract_one_frame(
    raw_path: Path,
    shape: tuple[int, int],
    raw_dtype: np.dtype,
    ys: np.ndarray,
    xs: np.ndarray,
    candidate_chunk: int,
    residual_dtype: np.dtype,
    center_out: np.ndarray,
    residual_out: np.ndarray,
) -> tuple[int, int, float, float]:
    raw = np.memmap(raw_path, mode="r", dtype=raw_dtype, shape=shape)
    n = len(xs)
    min_res = math.inf
    max_res = -math.inf
    center_min = math.inf
    center_max = -math.inf

    for c0 in range(0, n, candidate_chunk):
        c1 = min(n, c0 + candidate_chunk)
        cy = ys[c0:c1]
        cx = xs[c0:c1]

        center = np.asarray(raw[cy, cx], dtype=np.uint16)
        neighbours = np.empty((8, c1 - c0), dtype=np.uint16)
        for k, (dy, dx) in enumerate(NEIGHBOUR_OFFSETS):
            neighbours[k] = raw[cy + dy, cx + dx]

        middle = np.partition(neighbours, kth=(3, 4), axis=0)
        residual_x2 = (
            center.astype(np.int32) * 2
            - middle[3].astype(np.int32)
            - middle[4].astype(np.int32)
        )

        if residual_dtype == np.dtype("int16"):
            rmin = int(residual_x2.min(initial=0))
            rmax = int(residual_x2.max(initial=0))
            if rmin < np.iinfo(np.int16).min or rmax > np.iinfo(np.int16).max:
                raise OverflowError(
                    f"local_residual_x2 outside int16 range in {raw_path}: [{rmin}, {rmax}]. "
                    "Rerun with --residual-dtype int32."
                )

        center_out[c0:c1] = center
        residual_out[c0:c1] = residual_x2.astype(residual_dtype, copy=False)
        center_min = min(center_min, int(center.min(initial=0)))
        center_max = max(center_max, int(center.max(initial=0)))
        min_res = min(min_res, int(residual_x2.min(initial=0)))
        max_res = max(max_res, int(residual_x2.max(initial=0)))

    del raw
    return int(center_min), int(center_max), float(min_res), float(max_res)


def main() -> int:
    args = parse_args()
    validate_common_arguments(args)
    if args.candidate_chunk <= 0 or args.flush_every <= 0:
        raise ValueError("--candidate-chunk and --flush-every must be positive")

    manifest_path = args.frame_manifest.expanduser().resolve()
    candidate_dir = args.candidate_dir.expanduser().resolve()
    catalog_path = (
        args.candidate_catalog.expanduser().resolve()
        if args.candidate_catalog is not None
        else candidate_dir / "candidate_catalog.csv"
    )
    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)

    manifest, full_shape, raw_dtype = load_frame_manifest(
        manifest_path,
        shape_arg=tuple(args.shape) if args.shape else None,
        dtype_arg=args.dtype,
        require_temperature=True,
    )
    full_h, full_w = full_shape
    step03_summary = read_step03_summary(candidate_dir)
    step03_roi, effective_roi = resolve_catalog_filter_roi(
        args, step03_summary, full_w, full_h
    )
    candidates, original_candidate_count = load_candidates(
        catalog_path, full_w, full_h, effective_roi
    )
    frame_metadata = build_frame_metadata(manifest)

    n_frames = len(frame_metadata)
    n_candidates = len(candidates)
    residual_dtype = np.dtype(args.residual_dtype)
    expected_center_bytes = n_frames * n_candidates * np.dtype(np.uint16).itemsize
    expected_residual_bytes = n_frames * n_candidates * residual_dtype.itemsize

    output_dir = args.output_dir.expanduser().resolve()
    if args.resume:
        if args.overwrite:
            raise ValueError("--resume and --overwrite are mutually exclusive")
        if not output_dir.is_dir():
            raise FileNotFoundError(output_dir)
        start_frame = validate_resume(output_dir, n_frames, n_candidates, residual_dtype)
    else:
        output_dir = prepare_output_dir(output_dir, args.overwrite)
        start_frame = 0
        candidates.to_csv(output_dir / "candidate_catalog_extraction_order.csv", index=False)
        frame_metadata.to_csv(output_dir / "frame_metadata.csv", index=False)

    center_path, residual_path, center_mm, residual_mm = open_outputs(
        output_dir, n_frames, n_candidates, residual_dtype, args.resume
    )

    ys = candidates["y"].to_numpy(np.int64)
    xs = candidates["x"].to_numpy(np.int64)

    print(f"Manifest: {manifest_path}")
    print(f"Candidate catalog: {catalog_path}")
    print(f"Detector: {full_h} x {full_w}, dtype={raw_dtype}")
    print(f"Step03 ROI: x={step03_roi.x0}:{step03_roi.x1}, y={step03_roi.y0}:{step03_roi.y1}")
    print(f"Effective candidate ROI: x={effective_roi.x0}:{effective_roi.x1}, y={effective_roi.y0}:{effective_roi.y1}")
    print(f"Frames: {n_frames:,}; candidates: {n_candidates:,} (catalog total {original_candidate_count:,})")
    print(f"Output shape: ({n_frames:,}, {n_candidates:,})")
    print(f"Expected storage: center={expected_center_bytes/2**30:.2f} GiB, residual={expected_residual_bytes/2**30:.2f} GiB")
    if start_frame:
        print(f"Resuming at frame {start_frame}/{n_frames}")

    started = time.time()
    observed_center_min = math.inf
    observed_center_max = -math.inf
    observed_residual_min = math.inf
    observed_residual_max = -math.inf

    for i in range(start_frame, n_frames):
        row = frame_metadata.iloc[i]
        raw_path = Path(str(row["resolved_path"]))
        cmin, cmax, rmin, rmax = extract_one_frame(
            raw_path, full_shape, raw_dtype, ys, xs,
            args.candidate_chunk, residual_dtype,
            center_mm[i], residual_mm[i],
        )
        observed_center_min = min(observed_center_min, cmin)
        observed_center_max = max(observed_center_max, cmax)
        observed_residual_min = min(observed_residual_min, rmin)
        observed_residual_max = max(observed_residual_max, rmax)

        completed = i + 1
        need_flush = completed % args.flush_every == 0 or completed == n_frames
        if need_flush:
            center_mm.flush()
            residual_mm.flush()
            write_json(output_dir / "checkpoint.json", {
                "step": "04_extract_candidate_timeseries",
                "script_version": SCRIPT_VERSION,
                "shape": [n_frames, n_candidates],
                "completed_frames": completed,
                "residual_dtype": str(residual_dtype),
                "updated_unix_time": time.time(),
            })

        if completed % args.progress_every == 0 or completed == n_frames or i == start_frame:
            elapsed = time.time() - started
            done_this_run = completed - start_frame
            rate = done_this_run / elapsed if elapsed > 0 else math.nan
            remaining = n_frames - completed
            eta = remaining / rate if rate > 0 else math.nan
            print(
                f"[{completed:4d}/{n_frames}] {row['dataset']} frame={row['frame_index']} "
                f"elapsed={elapsed/60:.1f} min ETA={eta/60:.1f} min"
            )

    center_mm.flush()
    residual_mm.flush()
    elapsed = time.time() - started

    summary = {
        "step": "04_extract_candidate_timeseries",
        "script_version": SCRIPT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "validation_passed": True,
        "frame_manifest": str(manifest_path),
        "candidate_catalog": str(catalog_path),
        "full_shape": list(full_shape),
        "raw_dtype": str(raw_dtype),
        "step03_roi": step03_roi.to_dict(),
        "effective_candidate_roi": effective_roi.to_dict(),
        "catalog_candidate_count": original_candidate_count,
        "extracted_candidate_count": n_candidates,
        "frame_count": n_frames,
        "dataset_count": int(frame_metadata["dataset"].nunique()),
        "array_shape": [n_frames, n_candidates],
        "center_dtype": "uint16",
        "residual_definition": "2*center-(4th+5th sorted values of 8 immediate neighbours)",
        "residual_dtype": str(residual_dtype),
        "candidate_order": "ascending global linear_index",
        "center_output": str(center_path),
        "residual_output": str(residual_path),
        "expected_center_bytes": expected_center_bytes,
        "expected_residual_bytes": expected_residual_bytes,
        "observed_center_min": None if not np.isfinite(observed_center_min) else int(observed_center_min),
        "observed_center_max": None if not np.isfinite(observed_center_max) else int(observed_center_max),
        "observed_residual_x2_min": None if not np.isfinite(observed_residual_min) else int(observed_residual_min),
        "observed_residual_x2_max": None if not np.isfinite(observed_residual_max) else int(observed_residual_max),
        "elapsed_seconds_this_run": elapsed,
        "resumed_from_frame": start_frame,
    }
    write_json(output_dir / "summary.json", summary)

    manifest_record = {
        "step": summary["step"],
        "script_version": SCRIPT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "inputs": {
            "frame_manifest": str(manifest_path),
            "candidate_catalog": str(catalog_path),
        },
        "outputs": {
            "candidate_catalog_extraction_order": str(output_dir / "candidate_catalog_extraction_order.csv"),
            "frame_metadata": str(output_dir / "frame_metadata.csv"),
            "center": str(center_path),
            "local_residual_x2": str(residual_path),
            "summary": str(output_dir / "summary.json"),
        },
    }
    if args.hash_inputs:
        manifest_record["input_sha256"] = {
            "frame_manifest": sha256_file(manifest_path),
            "candidate_catalog": sha256_file(catalog_path),
        }
    write_json(output_dir / "manifest.json", manifest_record)

    print(f"Done in {elapsed/60:.1f} min")
    print(f"  {center_path}")
    print(f"  {residual_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Use --resume to continue after the next saved checkpoint.", file=sys.stderr)
        raise SystemExit(130)
