#!/usr/bin/env python3
"""
05_center_timeseries_by_dataset.py

Step 05 of the IMX811 RTS pipeline.

For every candidate pixel, subtract an independent temporal median in each
acquisition dataset.  Step 05 performs no RTS/state classification.

Input from Step 04
------------------
  center_uint16.npy
  local_residual_x2_int16.npy (or int32)
  frame_metadata.csv
  candidate_catalog_extraction_order.csv
  summary.json

Exact integer representation
----------------------------
The center series is stored in ADU integers.  A dataset median can be a half
integer, so the centered center series is stored as twice the centered value:

  centered_center_x2 = 2*center_ADU - 2*median(center_ADU)

The Step04 local residual is already stored as residual_x2 = 2*residual_ADU.
Its dataset median can likewise be a half integer in residual_x2 units, so the
centered residual is stored as four times the physical residual:

  centered_residual_x4 = 2*residual_x2 - 2*median(residual_x2)

Thus later steps recover physical ADU values exactly with:

  centered_center_ADU   = centered_center_x2 / 2
  centered_residual_ADU = centered_residual_x4 / 4

Outputs retain the Step04 array shape (n_frames, n_candidates).  Dataset
baseline arrays have shape (n_datasets, n_candidates).
"""

from __future__ import annotations

import argparse
import json
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

SCRIPT_VERSION = "5.0.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Center all Step04 candidate time series by an independent median in each dataset."
    )
    p.add_argument(
        "--timeseries-dir", type=Path, default=Path("04_candidate_timeseries"),
        help="Step04 output directory.",
    )
    p.add_argument(
        "--candidate-block", type=int, default=4096,
        help="Candidates processed per block. Default: 4096.",
    )
    p.add_argument(
        "--centered-dtype", choices=("int16", "int32"), default="int16",
        help="Storage dtype of centered arrays. Values are range-checked. Default: int16.",
    )
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint.json.")
    p.add_argument("--hash-inputs", action="store_true")
    p.add_argument(
        "--flush-every-blocks", type=int, default=10,
        help="Flush arrays and checkpoint every N candidate blocks. Default: 10.",
    )
    add_common_arguments(p, output_default="05_dataset_centered_timeseries")
    return p.parse_args()


def find_residual_file(timeseries_dir: Path, step04_summary: dict) -> Path:
    recorded = step04_summary.get("residual_output")
    if recorded:
        p = Path(recorded)
        if p.is_file():
            return p
        p2 = timeseries_dir / p.name
        if p2.is_file():
            return p2
    matches = sorted(timeseries_dir.glob("local_residual_x2_*.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one local_residual_x2_*.npy in {timeseries_dir}; found {len(matches)}"
        )
    return matches[0]


def build_dataset_index(frame_metadata: pd.DataFrame) -> tuple[pd.DataFrame, list[np.ndarray]]:
    if "dataset" not in frame_metadata.columns:
        raise KeyError("frame_metadata.csv lacks dataset column")

    names = frame_metadata["dataset"].astype(str).drop_duplicates().tolist()
    all_names = frame_metadata["dataset"].astype(str).to_numpy()
    rows: list[dict] = []
    positions: list[np.ndarray] = []

    for dataset_index, name in enumerate(names):
        idx = np.flatnonzero(all_names == name).astype(np.int64)
        if idx.size == 0:
            raise RuntimeError(f"No frames for dataset {name!r}")
        if idx.size > 1 and not np.all(np.diff(idx) == 1):
            raise ValueError(f"Dataset {name!r} is not contiguous in frame_metadata.csv")
        positions.append(idx)
        group = frame_metadata.iloc[idx]
        temp = pd.to_numeric(group.get("temperature_C"), errors="coerce")
        rows.append({
            "dataset_index": dataset_index,
            "dataset": name,
            "frame_start": int(idx[0]),
            "frame_stop_exclusive": int(idx[-1] + 1),
            "frame_count": int(idx.size),
            "temperature_mean_C": float(temp.mean()) if temp.notna().any() else np.nan,
            "temperature_min_C": float(temp.min()) if temp.notna().any() else np.nan,
            "temperature_max_C": float(temp.max()) if temp.notna().any() else np.nan,
        })
    return pd.DataFrame(rows), positions


def twice_median_int(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return exactly 2*median for an integer array, as int64."""
    n = values.shape[axis]
    if n <= 0:
        raise ValueError("Cannot calculate a median of an empty axis")
    # np.partition avoids a complete sort. Work on a copy because partition mutates.
    work = np.asarray(values).copy()
    if n % 2:
        k = n // 2
        part = np.partition(work, k, axis=axis)
        med = np.take(part, k, axis=axis).astype(np.int64)
        return 2 * med
    k0, k1 = n // 2 - 1, n // 2
    part = np.partition(work, (k0, k1), axis=axis)
    lo = np.take(part, k0, axis=axis).astype(np.int64)
    hi = np.take(part, k1, axis=axis).astype(np.int64)
    return lo + hi


def checkpoint_payload(
    n_frames: int,
    n_candidates: int,
    n_datasets: int,
    dtype: np.dtype,
    completed_dataset: int,
    completed_candidate_stop: int,
) -> dict:
    return {
        "shape": [n_frames, n_candidates],
        "dataset_count": n_datasets,
        "centered_dtype": str(dtype),
        "completed_dataset_index": completed_dataset,
        "completed_candidate_stop": completed_candidate_stop,
    }


def main() -> int:
    args = parse_args()
    validate_common_arguments(args)
    if args.candidate_block <= 0 or args.flush_every_blocks <= 0:
        raise ValueError("candidate-block and flush-every-blocks must be positive")

    t0 = time.perf_counter()
    tsdir = args.timeseries_dir.resolve()
    outdir = args.output_dir.resolve()
    step04_summary_path = tsdir / "summary.json"
    if not step04_summary_path.is_file():
        raise FileNotFoundError(step04_summary_path)
    step04_summary = json.loads(step04_summary_path.read_text(encoding="utf-8"))
    if not step04_summary.get("validation_passed", False):
        raise ValueError("Step04 summary does not report validation_passed=true")

    center_path = tsdir / "center_uint16.npy"
    residual_path = find_residual_file(tsdir, step04_summary)
    frame_metadata_path = tsdir / "frame_metadata.csv"
    catalog_path = tsdir / "candidate_catalog_extraction_order.csv"
    for p in (center_path, residual_path, frame_metadata_path, catalog_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    center = np.load(center_path, mmap_mode="r")
    residual = np.load(residual_path, mmap_mode="r")
    if center.ndim != 2 or residual.shape != center.shape:
        raise ValueError("Step04 center and residual arrays must be matching 2-D arrays")
    if center.dtype != np.uint16:
        raise ValueError(f"Expected uint16 center array, got {center.dtype}")
    if residual.dtype.kind != "i":
        raise ValueError(f"Expected signed integer residual array, got {residual.dtype}")
    n_frames, n_candidates = map(int, center.shape)

    frame_metadata = pd.read_csv(frame_metadata_path)
    catalog = pd.read_csv(catalog_path)
    if len(frame_metadata) != n_frames:
        raise ValueError("frame_metadata row count does not match time-series rows")
    if len(catalog) != n_candidates:
        raise ValueError("candidate catalog row count does not match time-series columns")
    if "timeseries_column" in catalog.columns:
        expected = np.arange(n_candidates, dtype=np.int64)
        got = pd.to_numeric(catalog["timeseries_column"], errors="raise").to_numpy(np.int64)
        if not np.array_equal(got, expected):
            raise ValueError("candidate catalog timeseries_column is not 0..N-1")

    dataset_index, frame_positions = build_dataset_index(frame_metadata)
    n_datasets = len(dataset_index)

    dtype = np.dtype(args.centered_dtype)
    centered_center_name = f"centered_center_x2_{dtype}.npy"
    centered_residual_name = f"centered_residual_x4_{dtype}.npy"
    baseline_center_name = "dataset_center_baseline_x2_int32.npy"
    baseline_residual_name = "dataset_residual_baseline_x4_int32.npy"

    if args.resume:
        if not outdir.is_dir():
            raise FileNotFoundError(outdir)
        cp_path = outdir / "checkpoint.json"
        if not cp_path.is_file():
            raise FileNotFoundError(cp_path)
        cp = json.loads(cp_path.read_text(encoding="utf-8"))
        if cp.get("shape") != [n_frames, n_candidates] or cp.get("dataset_count") != n_datasets:
            raise ValueError("Checkpoint does not match current inputs")
        if cp.get("centered_dtype") != str(dtype):
            raise ValueError("Checkpoint dtype does not match")
        resume_dataset = int(cp.get("completed_dataset_index", 0))
        resume_stop = int(cp.get("completed_candidate_stop", 0))
        mode = "r+"
    else:
        prepare_output_dir(outdir, overwrite=args.overwrite)
        resume_dataset, resume_stop = 0, 0
        mode = "w+"

    centered_center = open_memmap(
        outdir / centered_center_name, mode=mode, dtype=dtype, shape=(n_frames, n_candidates)
    )
    centered_residual = open_memmap(
        outdir / centered_residual_name, mode=mode, dtype=dtype, shape=(n_frames, n_candidates)
    )
    baseline_center = open_memmap(
        outdir / baseline_center_name, mode=mode, dtype=np.int32, shape=(n_datasets, n_candidates)
    )
    baseline_residual = open_memmap(
        outdir / baseline_residual_name, mode=mode, dtype=np.int32, shape=(n_datasets, n_candidates)
    )

    if not args.resume:
        shutil.copy2(frame_metadata_path, outdir / "frame_metadata.csv")
        shutil.copy2(catalog_path, outdir / "candidate_catalog.csv")
        dataset_index.to_csv(outdir / "dataset_index.csv", index=False)

    info = np.iinfo(dtype)
    observed_cc_min, observed_cc_max = np.iinfo(np.int64).max, np.iinfo(np.int64).min
    observed_cr_min, observed_cr_max = np.iinfo(np.int64).max, np.iinfo(np.int64).min
    total_blocks = n_datasets * ((n_candidates + args.candidate_block - 1) // args.candidate_block)
    done_blocks = 0

    print(f"Step05: {n_frames:,} frames x {n_candidates:,} candidates, {n_datasets} datasets")
    print(f"Centered storage dtype: {dtype}")

    for d, rows in enumerate(frame_positions):
        if d < resume_dataset:
            continue
        start_candidate = resume_stop if d == resume_dataset else 0
        dname = str(dataset_index.iloc[d]["dataset"])
        print(f"Dataset {d+1}/{n_datasets}: {dname} ({len(rows)} frames)")

        for c0 in range(start_candidate, n_candidates, args.candidate_block):
            c1 = min(c0 + args.candidate_block, n_candidates)
            center_block = np.asarray(center[rows[0]:rows[-1] + 1, c0:c1])
            residual_block = np.asarray(residual[rows[0]:rows[-1] + 1, c0:c1])

            base_c_x2 = twice_median_int(center_block, axis=0)
            base_r_x4 = twice_median_int(residual_block, axis=0)
            cc = 2 * center_block.astype(np.int64) - base_c_x2[None, :]
            cr = 2 * residual_block.astype(np.int64) - base_r_x4[None, :]

            cc_min, cc_max = int(cc.min()), int(cc.max())
            cr_min, cr_max = int(cr.min()), int(cr.max())
            if cc_min < info.min or cc_max > info.max:
                raise OverflowError(
                    f"centered_center_x2 range [{cc_min},{cc_max}] exceeds {dtype}; rerun with --centered-dtype int32"
                )
            if cr_min < info.min or cr_max > info.max:
                raise OverflowError(
                    f"centered_residual_x4 range [{cr_min},{cr_max}] exceeds {dtype}; rerun with --centered-dtype int32"
                )

            centered_center[rows[0]:rows[-1] + 1, c0:c1] = cc.astype(dtype)
            centered_residual[rows[0]:rows[-1] + 1, c0:c1] = cr.astype(dtype)
            baseline_center[d, c0:c1] = base_c_x2.astype(np.int32)
            baseline_residual[d, c0:c1] = base_r_x4.astype(np.int32)

            observed_cc_min = min(observed_cc_min, cc_min)
            observed_cc_max = max(observed_cc_max, cc_max)
            observed_cr_min = min(observed_cr_min, cr_min)
            observed_cr_max = max(observed_cr_max, cr_max)
            done_blocks += 1

            if done_blocks % args.flush_every_blocks == 0 or c1 == n_candidates:
                for arr in (centered_center, centered_residual, baseline_center, baseline_residual):
                    arr.flush()
                next_dataset, next_stop = d, c1
                if c1 == n_candidates:
                    next_dataset, next_stop = d + 1, 0
                write_json(
                    outdir / "checkpoint.json",
                    checkpoint_payload(n_frames, n_candidates, n_datasets, dtype, next_dataset, next_stop),
                )

            if args.progress_every and (done_blocks % args.progress_every == 0 or c1 == n_candidates):
                elapsed = time.perf_counter() - t0
                frac = done_blocks / total_blocks
                eta = elapsed * (1 - frac) / frac if frac > 0 else float("nan")
                print(
                    f"  candidates {c1:,}/{n_candidates:,}; blocks {done_blocks}/{total_blocks}; "
                    f"elapsed {elapsed:.1f}s; ETA {eta:.1f}s"
                )

        resume_stop = 0

    for arr in (centered_center, centered_residual, baseline_center, baseline_residual):
        arr.flush()

    # Validation: each dataset/candidate median must be exactly zero in the scaled representation.
    # Full validation is cheap enough for ROI and reads in candidate blocks.
    validation_passed = True
    validation_nonzero_center = 0
    validation_nonzero_residual = 0
    for d, rows in enumerate(frame_positions):
        for c0 in range(0, n_candidates, args.candidate_block):
            c1 = min(c0 + args.candidate_block, n_candidates)
            # For odd frame counts the median must be zero. For even counts, the sum of
            # the two middle centered values must be zero. twice_median_int captures both.
            mc = twice_median_int(np.asarray(centered_center[rows[0]:rows[-1] + 1, c0:c1]), axis=0)
            mr = twice_median_int(np.asarray(centered_residual[rows[0]:rows[-1] + 1, c0:c1]), axis=0)
            validation_nonzero_center += int(np.count_nonzero(mc))
            validation_nonzero_residual += int(np.count_nonzero(mr))
    if validation_nonzero_center or validation_nonzero_residual:
        validation_passed = False

    elapsed = time.perf_counter() - t0
    summary = {
        "step": "05_center_timeseries_by_dataset",
        "script_version": SCRIPT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "validation_passed": validation_passed,
        "timeseries_dir": str(tsdir),
        "step04_summary": str(step04_summary_path),
        "array_shape": [n_frames, n_candidates],
        "frame_count": n_frames,
        "candidate_count": n_candidates,
        "dataset_count": n_datasets,
        "centering_unit": "independent temporal median for each candidate within each acquisition dataset",
        "centered_center_definition": "2*center_ADU - 2*median_dataset(center_ADU)",
        "centered_residual_definition": "2*local_residual_x2 - 2*median_dataset(local_residual_x2)",
        "centered_center_physical_scale_ADU": 0.5,
        "centered_residual_physical_scale_ADU": 0.25,
        "centered_dtype": str(dtype),
        "centered_center_output": str(outdir / centered_center_name),
        "centered_residual_output": str(outdir / centered_residual_name),
        "center_baseline_output": str(outdir / baseline_center_name),
        "residual_baseline_output": str(outdir / baseline_residual_name),
        "observed_centered_center_x2_min": int(observed_cc_min),
        "observed_centered_center_x2_max": int(observed_cc_max),
        "observed_centered_residual_x4_min": int(observed_cr_min),
        "observed_centered_residual_x4_max": int(observed_cr_max),
        "validation_nonzero_center_medians": validation_nonzero_center,
        "validation_nonzero_residual_medians": validation_nonzero_residual,
        "elapsed_seconds": elapsed,
    }
    write_json(outdir / "summary.json", summary)

    manifest = {
        "inputs": {
            "center": str(center_path),
            "residual": str(residual_path),
            "frame_metadata": str(frame_metadata_path),
            "candidate_catalog": str(catalog_path),
        },
        "outputs": [
            centered_center_name,
            centered_residual_name,
            baseline_center_name,
            baseline_residual_name,
            "frame_metadata.csv",
            "candidate_catalog.csv",
            "dataset_index.csv",
            "summary.json",
        ],
    }
    if args.hash_inputs:
        manifest["input_sha256"] = {
            "center": sha256_file(center_path),
            "residual": sha256_file(residual_path),
            "frame_metadata": sha256_file(frame_metadata_path),
            "candidate_catalog": sha256_file(catalog_path),
        }
    write_json(outdir / "manifest.json", manifest)

    print(f"PASS: Step05 completed in {elapsed:.1f} s")
    print(f"  centered center:   {outdir / centered_center_name}")
    print(f"  centered residual: {outdir / centered_residual_name}")
    return 0 if validation_passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
