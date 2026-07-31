"""Shared helpers for detector-characterization scripts."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np

MAD_TO_SIGMA = 1.482602218505602

@dataclass(frozen=True, slots=True)
class RobustClipResult:
    sigma: float
    median: float
    kept_count: int
    rejected_count: int
    rejected_fraction: float
    lower_limit: float
    upper_limit: float

def robust_std(values: np.ndarray, *, axis=None):
    array = np.asarray(values)
    median = np.nanmedian(array, axis=axis, keepdims=True)
    sigma = MAD_TO_SIGMA * np.nanmedian(np.abs(array - median), axis=axis)
    return float(sigma) if np.ndim(sigma) == 0 else sigma

def robust_sigma_clip(values: np.ndarray, *, clip_sigma: float = 5.0) -> RobustClipResult:
    if clip_sigma <= 0:
        raise ValueError("clip_sigma must be positive")
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite values are available for clipping")
    center = float(np.median(finite))
    initial_sigma = float(robust_std(finite))
    if initial_sigma == 0.0:
        lower = upper = center
        keep = finite == center
    else:
        lower = center - clip_sigma * initial_sigma
        upper = center + clip_sigma * initial_sigma
        keep = (finite >= lower) & (finite <= upper)
    kept = finite[keep]
    if kept.size == 0:
        raise ValueError("Sigma clipping rejected all values")
    rejected = int(finite.size - kept.size)
    return RobustClipResult(
        sigma=float(robust_std(kept)), median=float(np.median(kept)),
        kept_count=int(kept.size), rejected_count=rejected,
        rejected_fraction=rejected / float(finite.size),
        lower_limit=float(lower), upper_limit=float(upper),
    )

def center_roi(image_shape: tuple[int, int], *, size: int):
    h, w = image_shape
    if size <= 0 or size > min(h, w):
        raise ValueError(f"Invalid ROI size {size} for image shape {image_shape}")
    return (w - size) // 2, (h - size) // 2, size, size

def validate_roi(image_shape, *, x, y, width, height):
    h, w = image_shape
    if min(x, y) < 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid ROI coordinates or size")
    if x + width > w or y + height > h:
        raise ValueError(f"ROI extends outside image bounds: roi={(x,y,width,height)}, image={image_shape}")
    return x, y, width, height

def crop_roi(image: np.ndarray, roi):
    x, y, width, height = roi
    return image[y:y+height, x:x+width]

def correct_frame_level(image: np.ndarray, method: str):
    values = np.asarray(image, dtype=np.float32)
    if method == "median":
        level = float(np.nanmedian(values))
    elif method == "mean":
        level = float(np.nanmean(values, dtype=np.float64))
    elif method == "none":
        level = 0.0
    else:
        raise ValueError(f"Unknown frame-level correction: {method}")
    return values - level, level

def finite_percentiles(values: np.ndarray, percentiles: Iterable[float]):
    finite = np.asarray(values)
    finite = finite[np.isfinite(finite)]
    requested = tuple(float(v) for v in percentiles)
    computed = np.percentile(finite, requested)
    return dict(zip(requested, map(float, computed)))

def save_histogram(
    values,
    output_path,
    *,
    xlabel,
    title,
    bins=300,
    log_y=False,
    percentile_range=(0.1, 99.9),
):
    """Save a histogram and explicitly record any display-range truncation.

    percentile_range=None displays the complete finite data range.  A tuple such
    as (0.1, 99.9) displays the central 99.8 percent while leaving the analysis
    values themselves unchanged.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    finite = np.asarray(values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite values are available for histogram output")

    hist_range = None
    omitted_fraction = 0.0
    range_note = "full finite range"
    if percentile_range is not None:
        q_low, q_high = map(float, percentile_range)
        low, high = np.percentile(finite, [q_low, q_high])
        if low < high:
            hist_range = (float(low), float(high))
            inside = (finite >= low) & (finite <= high)
            omitted_fraction = 1.0 - float(np.count_nonzero(inside)) / float(finite.size)
            range_note = (
                f"display: P{q_low:g}..P{q_high:g}; "
                f"outside display={100.0 * omitted_fraction:.4g}%"
            )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(finite, bins=bins, range=hist_range)
    ax.set(xlabel=xlabel, ylabel="Pixel count", title=title)
    if log_y:
        ax.set_yscale("log")
    ax.text(
        0.99,
        0.98,
        range_note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize="small",
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def save_noise_map(values, output_path, *, title, colorbar_label):
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    finite = np.asarray(values); finite = finite[np.isfinite(finite)]
    vmin, vmax = np.percentile(finite, [1, 99])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(values, origin="lower", vmin=vmin if vmin < vmax else None, vmax=vmax if vmin < vmax else None)
    ax.set(xlabel="ROI x pixel", ylabel="ROI y pixel", title=title)
    fig.colorbar(im, ax=ax).set_label(colorbar_label)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
