#!/usr/bin/env python3
"""
02_build_raw_pixel_statistics_v1_1.py

Step 02 of the rebuilt IMX811 RTS pipeline.

Purpose
-------
Read every validated raw frame listed in the Step 01 frame manifest and build
full-frame, per-pixel RAW statistics. This step performs no RTS selection,
temperature correction, state classification, or dictionary construction.

The 1.2 TB image cube is NOT saved. Images are processed in row tiles.

Statistics
----------
Across all frames:
  mean_ADU.npy
  std_ADU.npy
  min_ADU.npy
  max_ADU.npy
  range_ADU.npy

Within each dataset only (differences are never taken across dataset borders):
  adjacent_diff_rms_ADU.npy
  max_abs_adjacent_diff_ADU.npy

The adjacent-difference products are useful later because they are less
sensitive than the raw range to slow temperature/session drifts.

Input
-----
01_frame_index_output_19200x12800/frame_manifest.csv

Output
------
02_raw_statistics_output/
  mean_ADU.npy
  std_ADU.npy
  min_ADU.npy
  max_ADU.npy
  range_ADU.npy
  adjacent_diff_rms_ADU.npy
  max_abs_adjacent_diff_ADU.npy
  frame_metadata.csv
  summary.json
  manifest.json
  checkpoint.json
  previews/
      *.png
      sampled_histograms.png

Notes
-----
- Image width, height, and dtype are read from the Step 01 frame manifest.
- For iRayple IMX811, the manifest should contain width=19200 and height=12800.
- NumPy arrays use shape=(height, width)=(12800, 19200).
- The script can resume after interruption with --resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap


OUTPUT_SPECS = {
    "mean_ADU.npy": np.float32,
    "std_ADU.npy": np.float32,
    "min_ADU.npy": np.uint16,
    "max_ADU.npy": np.uint16,
    "range_ADU.npy": np.uint16,
    "adjacent_diff_rms_ADU.npy": np.float32,
    "max_abs_adjacent_diff_ADU.npy": np.uint16,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build full-frame raw pixel statistics from the validated manifest."
    )
    p.add_argument(
        "--frame-manifest",
        type=Path,
        default=Path("01_frame_index_output_19200x12800/frame_manifest.csv"),
    )
    p.add_argument(
        "--tile-rows",
        type=int,
        default=256,
        help="Rows processed at once. Reduce if RAM is limited.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("02_raw_statistics_output"),
    )
    p.add_argument(
        "--preview-step",
        type=int,
        default=16,
        help="Preview block size and histogram sampling stride. Default: 16.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print frame progress every N frames within each tile.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint.json in an existing output directory.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory. Incompatible with --resume.",
    )
    p.add_argument("--hash-input", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_output_arrays(
    output_dir: Path,
    shape: tuple[int, int],
    resume: bool,
) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    for filename, dtype in OUTPUT_SPECS.items():
        path = output_dir / filename
        if resume:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume: missing output array {path}"
                )
            arr = np.load(path, mmap_mode="r+")
            if arr.shape != shape or arr.dtype != np.dtype(dtype):
                raise ValueError(
                    f"Resume array mismatch for {path}: "
                    f"shape={arr.shape}, dtype={arr.dtype}"
                )
        else:
            arr = open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        arrays[filename] = arr
    return arrays


def validate_manifest(
    manifest_path: Path,
) -> tuple[pd.DataFrame, int, int, np.dtype]:
    """
    Validate Step 01 output and obtain the authoritative image geometry.

    Step 01 stores:
      image_width  = number of columns
      image_height = number of rows
      pixel_dtype  = raw pixel dtype

    NumPy shape is therefore always:
      (image_height, image_width)
    """
    df = pd.read_csv(manifest_path)

    required = [
        "manifest_row",
        "dataset",
        "frame_index",
        "temperature_C",
        "exposure_s",
        "resolved_path",
        "file_exists",
        "file_size_ok",
        "image_width",
        "image_height",
        "pixel_dtype",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "Frame manifest missing columns: "
            f"{missing}. Re-run Step 01 with v1.1 or later."
        )

    df = df.sort_values("manifest_row", kind="stable").reset_index(drop=True)

    if not df["file_exists"].astype(bool).all():
        raise ValueError("Frame manifest contains missing files.")
    if not df["file_size_ok"].astype(bool).all():
        raise ValueError("Frame manifest contains files with invalid sizes.")
    if df["resolved_path"].duplicated().any():
        raise ValueError("Frame manifest contains duplicate resolved paths.")

    width_values = pd.to_numeric(
        df["image_width"], errors="raise"
    ).astype(np.int64).unique()
    height_values = pd.to_numeric(
        df["image_height"], errors="raise"
    ).astype(np.int64).unique()
    dtype_values = df["pixel_dtype"].astype(str).unique()

    if len(width_values) != 1:
        raise ValueError(
            f"Inconsistent image_width values in manifest: {width_values.tolist()}"
        )
    if len(height_values) != 1:
        raise ValueError(
            f"Inconsistent image_height values in manifest: {height_values.tolist()}"
        )
    if len(dtype_values) != 1:
        raise ValueError(
            f"Inconsistent pixel_dtype values in manifest: {dtype_values.tolist()}"
        )

    width = int(width_values[0])
    height = int(height_values[0])
    raw_dtype = np.dtype(dtype_values[0])

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image geometry in manifest: width={width}, height={height}"
        )

    shape = (height, width)
    expected_bytes = width * height * raw_dtype.itemsize

    # Cross-check redundant Step 01 geometry columns when available.
    if "numpy_rows" in df.columns:
        rows = pd.to_numeric(df["numpy_rows"], errors="raise").astype(np.int64)
        if not (rows == height).all():
            raise ValueError("numpy_rows does not match image_height.")
    if "numpy_columns" in df.columns:
        cols = pd.to_numeric(
            df["numpy_columns"], errors="raise"
        ).astype(np.int64)
        if not (cols == width).all():
            raise ValueError("numpy_columns does not match image_width.")
    if "expected_file_size_bytes" in df.columns:
        expected_from_manifest = pd.to_numeric(
            df["expected_file_size_bytes"], errors="raise"
        ).astype(np.int64)
        if not (expected_from_manifest == expected_bytes).all():
            raise ValueError(
                "expected_file_size_bytes in the manifest does not match "
                "image_width × image_height × dtype."
            )

    for path_str in df["resolved_path"]:
        path = Path(str(path_str))
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Unexpected size for {path}: "
                f"{actual_bytes} != {expected_bytes}"
            )

    return df, width, height, raw_dtype


def create_preview(
    array_path: Path,
    output_path: Path,
    step: int,
    title: str,
    label: str,
    percentile_limits: tuple[float, float] = (1.0, 99.0),
) -> None:
    """
    Create a full-frame diagnostic preview using step x step block means.

    Sparse sampling such as arr[::step, ::step] is intentionally avoided
    because it can create severe aliasing/moiré patterns.
    """
    arr = np.load(array_path, mmap_mode="r")
    nrows, ncols = arr.shape

    block = max(1, int(step))
    used_rows = (nrows // block) * block
    used_cols = (ncols // block) * block

    if block == 1:
        preview = np.asarray(arr, dtype=np.float32)
    else:
        source = np.asarray(
            arr[:used_rows, :used_cols],
            dtype=np.float32,
        )
        preview = source.reshape(
            used_rows // block,
            block,
            used_cols // block,
            block,
        ).mean(axis=(1, 3))

    finite = preview[np.isfinite(preview)]
    if finite.size == 0:
        return

    vmin, vmax = np.percentile(finite, percentile_limits)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0

    plt.figure(figsize=(14, 9))
    image = plt.imshow(
        preview,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(image, label=label)
    plt.title(
        f"{title}\n"
        f"{block} x {block} block mean; display limits "
        f"{percentile_limits[0]:g}-{percentile_limits[1]:g} percentile"
    )
    plt.xlabel(f"x block index (1 block = {block} pixels)")
    plt.ylabel(f"y block index (1 block = {block} pixels)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def create_histograms(output_dir: Path, preview_dir: Path, step: int) -> None:
    items = [
        ("mean_ADU.npy", "Mean [ADU]"),
        ("std_ADU.npy", "Standard deviation [ADU]"),
        ("range_ADU.npy", "Range [ADU]"),
        ("adjacent_diff_rms_ADU.npy", "Adjacent-difference RMS [ADU]"),
        ("max_abs_adjacent_diff_ADU.npy", "Maximum adjacent difference [ADU]"),
    ]

    fig, axes = plt.subplots(len(items), 1, figsize=(10, 18))
    for ax, (filename, label) in zip(axes, items):
        arr = np.load(output_dir / filename, mmap_mode="r")
        sampled = np.asarray(arr[::step, ::step], dtype=np.float64).ravel()
        sampled = sampled[np.isfinite(sampled)]

        if sampled.size:
            upper = np.percentile(sampled, 99.9)
            visible = sampled[sampled <= upper]
            ax.hist(visible, bins=150)
            ax.axvline(np.median(sampled), linewidth=1)
            ax.set_title(
                f"{label}; sampled pixels={sampled.size:,}; "
                f"median={np.median(sampled):.4g}; "
                f"99.9%={upper:.4g}"
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(preview_dir / "sampled_histograms.png", dpi=160)
    plt.close(fig)


def sampled_array_summary(path: Path, step: int) -> dict:
    arr = np.load(path, mmap_mode="r")
    values = np.asarray(arr[::step, ::step], dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"sample_count": 0}

    return {
        "sample_count": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p99": float(np.percentile(values, 99)),
        "p999": float(np.percentile(values, 99.9)),
        "max": float(np.max(values)),
    }


def main() -> int:
    args = parse_args()

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")

    manifest_path = args.frame_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_path = output_dir / "checkpoint.json"

    if args.tile_rows <= 0:
        raise ValueError("--tile-rows must be positive.")
    if args.preview_step <= 0:
        raise ValueError("--preview-step must be positive.")

    frame_df, width, height, raw_dtype = validate_manifest(manifest_path)
    nrows = height
    ncols = width
    shape = (nrows, ncols)
    n_frames = len(frame_df)

    if n_frames < 2:
        raise ValueError("At least two frames are required.")

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            import shutil
            shutil.rmtree(output_dir)
        elif not args.resume:
            raise FileExistsError(
                f"{output_dir} is not empty. Use --overwrite or --resume."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)

    # Freeze the exact frame metadata used by Step 02.
    metadata_columns = [
        c for c in [
            "manifest_row",
            "dataset",
            "directory",
            "environment",
            "frame_index",
            "temperature_C",
            "temperature_start_C",
            "temperature_end_C",
            "temperature_fraction",
            "exposure_s",
            "filename",
            "filepath",
            "resolved_path",
            "image_width",
            "image_height",
            "numpy_rows",
            "numpy_columns",
            "pixel_dtype",
            "byte_order",
        ]
        if c in frame_df.columns
    ]
    frame_df[metadata_columns].to_csv(
        output_dir / "frame_metadata.csv", index=False
    )

    start_row = 0
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint not found: {checkpoint_path}"
            )
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        start_row = int(checkpoint["completed_through_row"])
        if checkpoint.get("shape") != [nrows, ncols]:
            raise ValueError(
                "Checkpoint shape does not match the Step 01 manifest."
            )
        if checkpoint.get("dtype") != str(raw_dtype):
            raise ValueError("Checkpoint dtype does not match the Step 01 manifest.")
        if checkpoint.get("frame_count") != n_frames:
            raise ValueError("Checkpoint frame count does not match manifest.")

    arrays = prepare_output_arrays(output_dir, shape, args.resume)

    dataset_values = frame_df["dataset"].astype(str).to_numpy()
    paths = [Path(str(p)) for p in frame_df["resolved_path"]]

    total_start = time.time()
    total_tiles = math.ceil(nrows / args.tile_rows)
    first_tile_index = start_row // args.tile_rows

    print(f"Frames          : {n_frames}")
    print(f"Image geometry  : width={width}, height={height}")
    print(f"NumPy shape     : {shape}  # (height, width)")
    print(f"Raw dtype       : {raw_dtype}")
    print(f"Expected bytes  : {width * height * raw_dtype.itemsize:,}")
    print(f"Tile rows       : {args.tile_rows}")
    print(f"Total tiles     : {total_tiles}")
    print(f"Starting row    : {start_row}")
    print(f"Output          : {output_dir}")
    print()

    for tile_index, y0 in enumerate(
        range(start_row, nrows, args.tile_rows),
        start=first_tile_index,
    ):
        y1 = min(y0 + args.tile_rows, nrows)
        tile_shape = (y1 - y0, ncols)

        sum_values = np.zeros(tile_shape, dtype=np.float64)
        sumsq_values = np.zeros(tile_shape, dtype=np.float64)
        min_values = np.full(tile_shape, np.iinfo(raw_dtype).max, dtype=raw_dtype)
        max_values = np.zeros(tile_shape, dtype=raw_dtype)

        diff_sumsq = np.zeros(tile_shape, dtype=np.float64)
        max_abs_diff = np.zeros(tile_shape, dtype=np.uint16)
        diff_count = 0
        previous = None
        previous_dataset = None

        tile_start = time.time()

        for frame_pos, (path, dataset) in enumerate(
            zip(paths, dataset_values)
        ):
            frame = np.memmap(
                path,
                dtype=raw_dtype,
                mode="r",
                shape=shape,
                order="C",
            )
            raw_tile = np.asarray(frame[y0:y1, :])
            float_tile = raw_tile.astype(np.float64)

            sum_values += float_tile
            sumsq_values += float_tile * float_tile
            np.minimum(min_values, raw_tile, out=min_values)
            np.maximum(max_values, raw_tile, out=max_values)

            if previous is not None and dataset == previous_dataset:
                diff = raw_tile.astype(np.int32) - previous.astype(np.int32)
                abs_diff = np.abs(diff).astype(np.uint16)
                diff_sumsq += diff.astype(np.float64) ** 2
                np.maximum(max_abs_diff, abs_diff, out=max_abs_diff)
                diff_count += 1

            # A copy is necessary because raw_tile points into the current memmap.
            previous = raw_tile.copy()
            previous_dataset = dataset
            del raw_tile, float_tile, frame

            if (
                frame_pos == 0
                or (frame_pos + 1) % max(1, args.progress_every) == 0
                or frame_pos + 1 == n_frames
            ):
                elapsed = time.time() - tile_start
                rate = (frame_pos + 1) / elapsed if elapsed > 0 else math.nan
                remaining = (
                    (n_frames - frame_pos - 1) / rate
                    if rate > 0 else math.nan
                )
                print(
                    f"tile {tile_index+1:3d}/{total_tiles} "
                    f"rows {y0:5d}:{y1:5d} | "
                    f"frame {frame_pos+1:4d}/{n_frames} | "
                    f"elapsed {elapsed/60:6.1f} min | "
                    f"remaining {remaining/60:6.1f} min"
                )

        mean_values = sum_values / n_frames
        variance = sumsq_values / n_frames - mean_values * mean_values
        np.maximum(variance, 0.0, out=variance)
        std_values = np.sqrt(variance)

        range_values = (
            max_values.astype(np.uint32) - min_values.astype(np.uint32)
        ).astype(np.uint16)

        if diff_count > 0:
            diff_rms = np.sqrt(diff_sumsq / diff_count)
        else:
            diff_rms = np.full(tile_shape, np.nan, dtype=np.float64)

        arrays["mean_ADU.npy"][y0:y1, :] = mean_values.astype(np.float32)
        arrays["std_ADU.npy"][y0:y1, :] = std_values.astype(np.float32)
        arrays["min_ADU.npy"][y0:y1, :] = min_values
        arrays["max_ADU.npy"][y0:y1, :] = max_values
        arrays["range_ADU.npy"][y0:y1, :] = range_values
        arrays["adjacent_diff_rms_ADU.npy"][y0:y1, :] = diff_rms.astype(
            np.float32
        )
        arrays["max_abs_adjacent_diff_ADU.npy"][y0:y1, :] = max_abs_diff

        for arr in arrays.values():
            arr.flush()

        checkpoint = {
            "step": "02_build_raw_pixel_statistics",
            "script_version": "1.1.0",
            "completed_through_row": y1,
            "shape": [nrows, ncols],
            "image_width": width,
            "image_height": height,
            "shape_order": "HEIGHT WIDTH",
            "dtype": str(raw_dtype),
            "frame_count": n_frames,
            "tile_rows": args.tile_rows,
            "last_completed_tile_index": tile_index,
            "updated_unix_time": time.time(),
        }
        write_json(checkpoint_path, checkpoint)

        tile_elapsed = time.time() - tile_start
        total_elapsed = time.time() - total_start
        completed_rows = y1 - start_row
        rows_remaining = nrows - y1
        row_rate = completed_rows / total_elapsed if total_elapsed > 0 else math.nan
        estimated_remaining = (
            rows_remaining / row_rate if row_rate > 0 else math.nan
        )
        print(
            f"COMPLETED rows {y0}:{y1} | "
            f"tile={tile_elapsed/60:.1f} min | "
            f"total={total_elapsed/60:.1f} min | "
            f"estimated remaining={estimated_remaining/60:.1f} min"
        )
        print()

    # Generate diagnostics only after every row is complete.
    preview_specs = [
        ("mean_ADU.npy", "mean_ADU.png", "Raw mean", "ADU", (1.0, 99.0)),
        ("std_ADU.npy", "std_ADU.png", "Raw standard deviation", "ADU", (1.0, 99.5)),
        ("range_ADU.npy", "range_ADU.png", "Raw range", "ADU", (1.0, 99.5)),
        (
            "adjacent_diff_rms_ADU.npy",
            "adjacent_diff_rms_ADU.png",
            "Within-dataset adjacent-difference RMS",
            "ADU",
            (1.0, 99.5),
        ),
        (
            "max_abs_adjacent_diff_ADU.npy",
            "max_abs_adjacent_diff_ADU.png",
            "Within-dataset maximum absolute adjacent difference",
            "ADU",
            (1.0, 99.5),
        ),
    ]
    for source, destination, title, label, limits in preview_specs:
        create_preview(
            output_dir / source,
            preview_dir / destination,
            args.preview_step,
            title,
            label,
            limits,
        )

    create_histograms(output_dir, preview_dir, args.preview_step)

    sampled_summaries = {
        filename: sampled_array_summary(
            output_dir / filename, args.preview_step
        )
        for filename in OUTPUT_SPECS
    }

    dataset_counts = (
        frame_df.groupby("dataset", sort=False)
        .size()
        .astype(int)
        .to_dict()
    )
    expected_diff_count = int(
        sum(max(0, count - 1) for count in dataset_counts.values())
    )

    summary = {
        "step": "02_build_raw_pixel_statistics",
        "script_version": "1.1.0",
        "validation_passed": True,
        "frame_manifest": str(manifest_path),
        "frames": int(n_frames),
        "datasets": int(len(dataset_counts)),
        "frames_per_dataset": dataset_counts,
        "within_dataset_adjacent_pairs": expected_diff_count,
        "shape": [nrows, ncols],
        "image_width": width,
        "image_height": height,
        "shape_order": "HEIGHT WIDTH",
        "dtype": str(raw_dtype),
        "tile_rows": int(args.tile_rows),
        "preview_step": int(args.preview_step),
        "temperature_min_C": float(frame_df["temperature_C"].min()),
        "temperature_max_C": float(frame_df["temperature_C"].max()),
        "elapsed_seconds_this_run": float(time.time() - total_start),
        "sampled_statistics": sampled_summaries,
    }
    write_json(output_dir / "summary.json", summary)

    manifest = {
        "step": "02_build_raw_pixel_statistics",
        "script_version": "1.1.0",
        "input_frame_manifest": str(manifest_path),
        "input_frame_manifest_sha256": (
            sha256_file(manifest_path) if args.hash_input else None
        ),
        "shape": [nrows, ncols],
        "image_width": width,
        "image_height": height,
        "shape_order": "HEIGHT WIDTH",
        "dtype": str(raw_dtype),
        "statistics_definition": {
            "std_ADU": "population standard deviation over all 2518 frames",
            "range_ADU": "max_ADU - min_ADU over all frames",
            "adjacent_diff_rms_ADU": (
                "RMS of frame-to-frame differences; dataset boundaries excluded"
            ),
            "max_abs_adjacent_diff_ADU": (
                "maximum absolute frame-to-frame difference; "
                "dataset boundaries excluded"
            ),
        },
        "outputs": [
            *OUTPUT_SPECS.keys(),
            "frame_metadata.csv",
            "summary.json",
            "manifest.json",
            "checkpoint.json",
            "previews/",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("PASS")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
