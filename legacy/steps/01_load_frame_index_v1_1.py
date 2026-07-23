#!/usr/bin/env python3
"""
01_load_frame_index.py

Step 01 of the rebuilt IMX811 RTS pipeline.

Purpose
-------
Read temperature_index_100us.csv as the authoritative frame list, resolve every
frame path, validate ordering and metadata, validate binary file sizes, and
write a clean frame manifest for all later steps.

The command-line shape convention is explicitly:

    --shape WIDTH HEIGHT

For the iRayple IMX811 output:

    --shape 19200 12800

Internally NumPy uses:

    (HEIGHT, WIDTH) = (12800, 19200)

To prevent a wrong-but-size-compatible reshape from passing unnoticed, this
step also reads the first valid BIN frame and writes:
  - preview_first_frame_full.png
  - preview_first_frame_center2000.png

Outputs
-------
01_frame_index_output/
  frame_manifest.csv
  dataset_summary.csv
  validation_issues.csv
  summary.json
  manifest.json
  temperature_vs_frame.png
  frames_per_dataset.png
  preview_first_frame_full.png
  preview_first_frame_center2000.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "dataset",
    "directory",
    "environment",
    "frame_index",
    "n_frames",
    "temperature_C",
    "temperature_start_C",
    "temperature_end_C",
    "temperature_fraction",
    "exposure_s",
    "filename",
    "filepath",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate and freeze temperature_index_100us.csv."
    )
    p.add_argument(
        "--frame-index",
        type=Path,
        default=Path("temperature_index_100us.csv"),
    )
    p.add_argument(
        "--frame-root",
        type=Path,
        default=Path("."),
        help="Base directory for relative filepath entries.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("01_frame_index_output"),
    )
    p.add_argument(
        "--shape",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(19200, 12800),
        help=(
            "Image size in WIDTH HEIGHT order. "
            "For iRayple IMX811 use: --shape 19200 12800"
        ),
    )
    p.add_argument("--dtype", default="uint16")
    p.add_argument(
        "--byte-order",
        choices=["little", "big", "native"],
        default="little",
        help="Binary byte order. Default: little.",
    )
    p.add_argument(
        "--skip-size-check",
        action="store_true",
        help="Check existence only, not expected binary byte size.",
    )
    p.add_argument(
        "--skip-preview",
        action="store_true",
        help="Do not read the first valid frame or create preview PNGs.",
    )
    p.add_argument(
        "--preview-percentiles",
        nargs=2,
        type=float,
        default=(0.5, 99.5),
        metavar=("LOW", "HIGH"),
    )
    p.add_argument(
        "--preview-max-width",
        type=int,
        default=3000,
    )
    p.add_argument(
        "--preview-max-height",
        type=int,
        default=2000,
    )
    p.add_argument(
        "--preview-crop-size",
        type=int,
        default=2000,
    )
    p.add_argument(
        "--hash-index",
        action="store_true",
        help="Store SHA256 of the input CSV in the manifest.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dtype_with_order(name: str, byte_order: str) -> np.dtype:
    dt = np.dtype(name)
    if dt.itemsize == 1 or byte_order == "native":
        return dt
    return dt.newbyteorder("<" if byte_order == "little" else ">")


def add_issue(
    issues: list[dict],
    severity: str,
    issue_type: str,
    row_number: int | None,
    dataset: str | None,
    detail: str,
) -> None:
    issues.append({
        "severity": severity,
        "issue_type": issue_type,
        "row_number": row_number,
        "dataset": dataset,
        "detail": detail,
    })


def block_mean_preview(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> tuple[np.ndarray, int]:
    """Reduce by integer block mean; do not use sparse pixel decimation."""
    height, width = image.shape
    factor = max(
        1,
        math.ceil(width / max_width),
        math.ceil(height / max_height),
    )

    if factor == 1:
        return np.asarray(image, dtype=np.float32), factor

    trimmed_h = (height // factor) * factor
    trimmed_w = (width // factor) * factor
    trimmed = np.asarray(
        image[:trimmed_h, :trimmed_w],
        dtype=np.float32,
    )
    preview = trimmed.reshape(
        trimmed_h // factor,
        factor,
        trimmed_w // factor,
        factor,
    ).mean(axis=(1, 3))
    return preview, factor


def robust_limits(
    arr: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> tuple[float, float]:
    finite = np.asarray(arr)[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("No finite pixels available for preview.")
    vmin, vmax = np.percentile(
        finite,
        [low_percentile, high_percentile],
    )
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def save_preview(
    image: np.ndarray,
    output_path: Path,
    title: str,
    vmin: float,
    vmax: float,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(
        image,
        cmap="gray",
        origin="upper",
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label="ADU")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_first_frame_previews(
    frame_path: Path,
    output_dir: Path,
    width: int,
    height: int,
    dtype: np.dtype,
    low_percentile: float,
    high_percentile: float,
    max_preview_width: int,
    max_preview_height: int,
    crop_size_requested: int,
) -> dict:
    """
    Read one frame using NumPy shape=(HEIGHT, WIDTH) and create previews.
    """
    frame = np.memmap(
        frame_path,
        dtype=dtype,
        mode="r",
        shape=(height, width),
    )

    preview, factor = block_mean_preview(
        frame,
        max_preview_width,
        max_preview_height,
    )
    full_vmin, full_vmax = robust_limits(
        preview,
        low_percentile,
        high_percentile,
    )

    full_path = output_dir / "preview_first_frame_full.png"
    save_preview(
        preview,
        full_path,
        (
            f"{frame_path.name}\n"
            f"full frame {width} x {height}; "
            f"{factor} x {factor} block mean; "
            f"display={low_percentile:g}-{high_percentile:g} percentile"
        ),
        full_vmin,
        full_vmax,
        xlabel=f"x block index (1 block = {factor} pixels)",
        ylabel=f"y block index (1 block = {factor} pixels)",
    )

    crop_size = min(crop_size_requested, width, height)
    x0 = (width - crop_size) // 2
    y0 = (height - crop_size) // 2
    crop = np.asarray(
        frame[y0:y0 + crop_size, x0:x0 + crop_size],
        dtype=np.float32,
    )
    crop_vmin, crop_vmax = robust_limits(
        crop,
        low_percentile,
        high_percentile,
    )

    crop_path = output_dir / "preview_first_frame_center2000.png"
    save_preview(
        crop,
        crop_path,
        (
            f"{frame_path.name}\n"
            f"center crop x={x0}:{x0 + crop_size}, "
            f"y={y0}:{y0 + crop_size}; no decimation"
        ),
        crop_vmin,
        crop_vmax,
        xlabel="x [pixel within crop]",
        ylabel="y [pixel within crop]",
    )

    return {
        "source_file": str(frame_path),
        "full_preview": full_path.name,
        "center_preview": crop_path.name,
        "block_mean_factor": factor,
        "center_crop": {
            "x_start": x0,
            "x_end": x0 + crop_size,
            "y_start": y0,
            "y_end": y0 + crop_size,
        },
        "display_percentiles": [
            low_percentile,
            high_percentile,
        ],
    }


def main() -> int:
    args = parse_args()

    frame_index_path = args.frame_index.expanduser().resolve()
    frame_root = args.frame_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not frame_index_path.is_file():
        raise FileNotFoundError(frame_index_path)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty. "
            "Use --overwrite or another output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(frame_index_path, encoding="utf-8-sig")
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    df = df.copy()
    df.insert(0, "manifest_row", np.arange(len(df), dtype=np.int64))

    issues: list[dict] = []

    numeric_columns = [
        "frame_index",
        "n_frames",
        "temperature_C",
        "temperature_start_C",
        "temperature_end_C",
        "temperature_fraction",
        "exposure_s",
    ]
    for col in numeric_columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid = converted.isna() & df[col].notna()
        for idx in df.index[invalid]:
            add_issue(
                issues,
                "ERROR",
                "invalid_numeric_value",
                int(idx),
                str(df.at[idx, "dataset"]),
                f"{col}={df.at[idx, col]!r}",
            )
        df[col] = converted

    width, height = (int(v) for v in args.shape)
    if width <= 0 or height <= 0:
        raise ValueError("--shape WIDTH HEIGHT must contain positive integers.")

    low_p, high_p = (float(v) for v in args.preview_percentiles)
    if not (0.0 <= low_p < high_p <= 100.0):
        raise ValueError(
            "--preview-percentiles requires 0 <= LOW < HIGH <= 100."
        )

    dtype = dtype_with_order(args.dtype, args.byte_order)
    numpy_shape = (height, width)
    expected_pixels = width * height
    expected_bytes = expected_pixels * dtype.itemsize

    resolved_paths: list[str] = []
    file_exists: list[bool] = []
    file_size_bytes: list[float] = []
    size_ok: list[bool] = []

    for idx, row in df.iterrows():
        raw = str(row["filepath"]).strip()
        dataset = str(row["dataset"])

        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = frame_root / p
        p = p.resolve()

        exists = p.is_file()
        actual_size = p.stat().st_size if exists else np.nan
        valid_size = bool(
            exists and (
                args.skip_size_check or actual_size == expected_bytes
            )
        )

        resolved_paths.append(str(p))
        file_exists.append(exists)
        file_size_bytes.append(actual_size)
        size_ok.append(valid_size)

        if not exists:
            add_issue(
                issues,
                "ERROR",
                "missing_file",
                int(idx),
                dataset,
                str(p),
            )
        elif not args.skip_size_check and actual_size != expected_bytes:
            add_issue(
                issues,
                "ERROR",
                "unexpected_file_size",
                int(idx),
                dataset,
                (
                    f"{p}: actual={actual_size}, "
                    f"expected={expected_bytes}"
                ),
            )

        if Path(raw).name != str(row["filename"]):
            add_issue(
                issues,
                "ERROR",
                "filename_filepath_mismatch",
                int(idx),
                dataset,
                (
                    f"filename={row['filename']!r}, "
                    f"filepath={raw!r}"
                ),
            )

        parent_name = Path(raw).parent.name
        if parent_name != str(row["directory"]):
            add_issue(
                issues,
                "WARNING",
                "directory_filepath_mismatch",
                int(idx),
                dataset,
                (
                    f"directory={row['directory']!r}, "
                    f"filepath parent={parent_name!r}"
                ),
            )

        if str(row["dataset"]) != str(row["directory"]):
            add_issue(
                issues,
                "WARNING",
                "dataset_directory_mismatch",
                int(idx),
                dataset,
                (
                    f"dataset={row['dataset']!r}, "
                    f"directory={row['directory']!r}"
                ),
            )

    df["resolved_path"] = resolved_paths
    df["file_exists"] = file_exists
    df["file_size_bytes"] = file_size_bytes
    df["expected_file_size_bytes"] = expected_bytes
    df["file_size_ok"] = size_ok

    # Freeze image geometry explicitly in the manifest.
    df["image_width"] = width
    df["image_height"] = height
    df["numpy_rows"] = height
    df["numpy_columns"] = width
    df["pixel_dtype"] = str(dtype)
    df["byte_order"] = args.byte_order

    duplicated_paths = df["resolved_path"].duplicated(keep=False)
    for idx in df.index[duplicated_paths]:
        add_issue(
            issues,
            "ERROR",
            "duplicate_resolved_path",
            int(idx),
            str(df.at[idx, "dataset"]),
            str(df.at[idx, "resolved_path"]),
        )

    duplicated_dataset_frame = df.duplicated(
        subset=["dataset", "frame_index"],
        keep=False,
    )
    for idx in df.index[duplicated_dataset_frame]:
        add_issue(
            issues,
            "ERROR",
            "duplicate_dataset_frame_index",
            int(idx),
            str(df.at[idx, "dataset"]),
            f"frame_index={df.at[idx, 'frame_index']}",
        )

    dataset_rows = []
    for dataset, group in df.groupby("dataset", sort=False):
        group = group.sort_values("manifest_row", kind="stable")
        count = len(group)

        n_frames_values = group["n_frames"].dropna().unique()
        declared_n_frames = (
            int(n_frames_values[0])
            if len(n_frames_values) == 1
            else np.nan
        )

        if len(n_frames_values) != 1:
            add_issue(
                issues,
                "ERROR",
                "inconsistent_n_frames_within_dataset",
                None,
                str(dataset),
                f"values={n_frames_values.tolist()}",
            )
        elif declared_n_frames != count:
            add_issue(
                issues,
                "ERROR",
                "declared_n_frames_mismatch",
                None,
                str(dataset),
                f"declared={declared_n_frames}, actual_rows={count}",
            )

        fi = group["frame_index"].to_numpy(float)
        finite_fi = fi[np.isfinite(fi)]
        monotonic = bool(
            len(finite_fi) <= 1 or np.all(np.diff(finite_fi) > 0)
        )
        if not monotonic:
            add_issue(
                issues,
                "ERROR",
                "non_monotonic_frame_index",
                None,
                str(dataset),
                "frame_index is not strictly increasing in manifest order",
            )

        if len(finite_fi):
            expected_sequence_0 = np.arange(len(finite_fi))
            expected_sequence_1 = np.arange(1, len(finite_fi) + 1)
            contiguous = bool(
                np.array_equal(finite_fi, expected_sequence_0)
                or np.array_equal(finite_fi, expected_sequence_1)
            )
        else:
            contiguous = False

        if not contiguous:
            add_issue(
                issues,
                "WARNING",
                "non_contiguous_frame_index",
                None,
                str(dataset),
                "frame_index is not exactly 0..N-1 or 1..N",
            )

        temp = group["temperature_C"].to_numpy(float)
        frac = group["temperature_fraction"].to_numpy(float)

        frac_in_range = np.all(
            ~np.isfinite(frac) | ((frac >= 0.0) & (frac <= 1.0))
        )
        if not frac_in_range:
            add_issue(
                issues,
                "ERROR",
                "temperature_fraction_out_of_range",
                None,
                str(dataset),
                "temperature_fraction contains values outside [0, 1]",
            )

        start_values = group["temperature_start_C"].dropna().unique()
        end_values = group["temperature_end_C"].dropna().unique()
        exposure_values = group["exposure_s"].dropna().unique()
        environment_values = group["environment"].astype(str).unique()

        dataset_rows.append({
            "dataset": dataset,
            "environment": (
                environment_values[0]
                if len(environment_values) == 1
                else "|".join(environment_values)
            ),
            "rows": count,
            "declared_n_frames": declared_n_frames,
            "frame_index_min": (
                np.nanmin(fi)
                if np.any(np.isfinite(fi))
                else np.nan
            ),
            "frame_index_max": (
                np.nanmax(fi)
                if np.any(np.isfinite(fi))
                else np.nan
            ),
            "frame_index_strictly_increasing": monotonic,
            "frame_index_contiguous": contiguous,
            "temperature_min_C": (
                np.nanmin(temp)
                if np.any(np.isfinite(temp))
                else np.nan
            ),
            "temperature_max_C": (
                np.nanmax(temp)
                if np.any(np.isfinite(temp))
                else np.nan
            ),
            "temperature_median_C": (
                np.nanmedian(temp)
                if np.any(np.isfinite(temp))
                else np.nan
            ),
            "temperature_start_C_values": "|".join(
                map(str, start_values)
            ),
            "temperature_end_C_values": "|".join(
                map(str, end_values)
            ),
            "exposure_s_values": "|".join(
                map(str, exposure_values)
            ),
            "existing_files": int(group["file_exists"].sum()),
            "size_valid_files": int(group["file_size_ok"].sum()),
            "image_width": width,
            "image_height": height,
            "expected_file_size_bytes": expected_bytes,
        })

    dataset_summary = pd.DataFrame(dataset_rows)
    issues_df = pd.DataFrame(
        issues,
        columns=[
            "severity",
            "issue_type",
            "row_number",
            "dataset",
            "detail",
        ],
    )

    df.to_csv(output_dir / "frame_manifest.csv", index=False)
    dataset_summary.to_csv(
        output_dir / "dataset_summary.csv",
        index=False,
    )
    issues_df.to_csv(
        output_dir / "validation_issues.csv",
        index=False,
    )

    plt.figure(figsize=(11, 6))
    for dataset, group in df.groupby("dataset", sort=False):
        plt.plot(
            group["manifest_row"],
            group["temperature_C"],
            marker=".",
            markersize=2,
            linewidth=0.8,
            label=str(dataset),
        )
    plt.xlabel("Manifest row")
    plt.ylabel("Temperature [deg C]")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_vs_frame.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(
        dataset_summary["dataset"].astype(str),
        dataset_summary["rows"],
    )
    plt.xlabel("Dataset")
    plt.ylabel("Number of frames")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "frames_per_dataset.png", dpi=180)
    plt.close()

    error_count = (
        int((issues_df["severity"] == "ERROR").sum())
        if len(issues_df)
        else 0
    )
    warning_count = (
        int((issues_df["severity"] == "WARNING").sum())
        if len(issues_df)
        else 0
    )

    preview_info = None
    if not args.skip_preview:
        preview_candidates = df.loc[
            df["file_exists"] & df["file_size_ok"],
            "resolved_path",
        ]
        if len(preview_candidates) == 0:
            add_issue(
                issues,
                "ERROR",
                "no_valid_frame_for_preview",
                None,
                None,
                "No existing size-valid frame was available.",
            )
            error_count += 1
        else:
            preview_path = Path(preview_candidates.iloc[0])
            preview_info = make_first_frame_previews(
                frame_path=preview_path,
                output_dir=output_dir,
                width=width,
                height=height,
                dtype=dtype,
                low_percentile=low_p,
                high_percentile=high_p,
                max_preview_width=args.preview_max_width,
                max_preview_height=args.preview_max_height,
                crop_size_requested=args.preview_crop_size,
            )

    # Rewrite issues after possible preview issue.
    issues_df = pd.DataFrame(
        issues,
        columns=[
            "severity",
            "issue_type",
            "row_number",
            "dataset",
            "detail",
        ],
    )
    issues_df.to_csv(
        output_dir / "validation_issues.csv",
        index=False,
    )
    error_count = (
        int((issues_df["severity"] == "ERROR").sum())
        if len(issues_df)
        else 0
    )
    warning_count = (
        int((issues_df["severity"] == "WARNING").sum())
        if len(issues_df)
        else 0
    )

    summary = {
        "input_csv": str(frame_index_path),
        "frame_root": str(frame_root),
        "rows": int(len(df)),
        "datasets": int(df["dataset"].nunique()),
        "dataset_names": (
            df["dataset"].astype(str).drop_duplicates().tolist()
        ),
        "existing_files": int(df["file_exists"].sum()),
        "missing_files": int((~df["file_exists"]).sum()),
        "size_valid_files": int(df["file_size_ok"].sum()),
        "image_width": width,
        "image_height": height,
        "shape_argument_order": "WIDTH HEIGHT",
        "numpy_shape_order": "HEIGHT WIDTH",
        "numpy_shape": [height, width],
        "pixels_per_frame": expected_pixels,
        "expected_file_size_bytes": expected_bytes,
        "dtype": str(dtype),
        "byte_order": args.byte_order,
        "temperature_min_C": float(df["temperature_C"].min()),
        "temperature_max_C": float(df["temperature_C"].max()),
        "exposure_s_values": sorted(
            float(v)
            for v in df["exposure_s"].dropna().unique()
        ),
        "preview": preview_info,
        "errors": error_count,
        "warnings": warning_count,
        "validation_passed": error_count == 0,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    outputs = [
        "frame_manifest.csv",
        "dataset_summary.csv",
        "validation_issues.csv",
        "summary.json",
        "temperature_vs_frame.png",
        "frames_per_dataset.png",
    ]
    if preview_info is not None:
        outputs.extend([
            preview_info["full_preview"],
            preview_info["center_preview"],
        ])

    manifest = {
        "step": "01_load_frame_index",
        "script_version": "1.1.0",
        "input_csv": str(frame_index_path),
        "input_csv_sha256": (
            sha256_file(frame_index_path)
            if args.hash_index
            else None
        ),
        "frame_root": str(frame_root),
        "output_dir": str(output_dir),
        "image_geometry": {
            "width": width,
            "height": height,
            "command_line_shape": [width, height],
            "command_line_order": "WIDTH HEIGHT",
            "numpy_shape": [height, width],
            "numpy_order": "HEIGHT WIDTH",
        },
        "dtype": str(dtype),
        "byte_order": args.byte_order,
        "size_check_enabled": not args.skip_size_check,
        "preview_enabled": not args.skip_preview,
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("=== Image geometry ===")
    print(f"command line     : --shape {width} {height}")
    print(f"meaning          : WIDTH={width}, HEIGHT={height}")
    print(f"NumPy shape      : ({height}, {width})")
    print(f"pixels/frame     : {expected_pixels:,}")
    print(f"expected bytes   : {expected_bytes:,}")
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(f"Output directory: {output_dir}")
    if preview_info is not None:
        print("Please visually confirm:")
        print(f"  {output_dir / preview_info['full_preview']}")
        print(f"  {output_dir / preview_info['center_preview']}")
    print("PASS" if error_count == 0 else "FAIL")

    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
