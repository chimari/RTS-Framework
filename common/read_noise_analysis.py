"""Reusable robust read-noise characterization for one Step 02 dataset."""

from __future__ import annotations

__version__ = "1.5.0-dev"

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from common.image_io import read_image
from common.dataset_characterization import (
    DatasetCharacterization,
    write_dataset_characterization,
)


MAD_TO_SIGMA = 1.482602218505602


class DatasetGroupLike(Protocol):
    name: str
    frames: tuple
    environment: str
    image_shape: tuple[int, int]
    pixel_dtype: str
    exposure_s: float
    temperature_min_C: float
    temperature_max_C: float

    @property
    def n_frames(self) -> int: ...


ProgressCallback = Callable[[int, int, object], None]


@dataclass(frozen=True, slots=True)
class ReadNoiseConfig:
    output_dir: Path
    frame_level_correction: str = "median"
    clip_sigma: float = 5.0
    hist_bins: int = 300
    roi: tuple[int, int, int, int] | None = None
    temporal_chunk_rows: int = 64


@dataclass(frozen=True, slots=True)
class PairNoiseRecord:
    pair_index: int
    frame_index_a: int
    frame_index_b: int
    filename_a: str
    filename_b: str
    pair_offset_median_adu: float
    initial_sigma_adu_rms: float
    noise_adu_rms: float
    clip_lower_adu: float
    clip_upper_adu: float
    total_pixels: int
    kept_pixels: int
    rejected_pixels: int
    rejected_fraction: float


@dataclass(frozen=True, slots=True)
class FrameLevelRecord:
    frame_index: int
    filename: str
    temperature_C: float
    level_adu: float


from typing import Any

@dataclass(slots=True)
class _AnalysisContext:
    """Intermediate results produced by dataset analysis."""

    frame_levels: list[FrameLevelRecord]
    pair_records: list[PairNoiseRecord]
    pair_values: np.ndarray
    first_pair_difference: np.ndarray | None
    temporal_noise: np.ndarray
    temporal_finite: np.ndarray
    quantization: dict[str, Any]
    quantization_level_rows: list[dict[str, Any]]
    histogram_samples: list[np.ndarray]
    temporal_path: Path
    
    
@dataclass(frozen=True, slots=True)
class ReadNoiseResult:
    dataset: str
    status: str
    frames: int
    pairs: int
    pair_noise_median_adu_rms: float
    clipping_rejected_fraction: float
    temporal_noise_median_adu_rms: float
    output_directory: str
    error: str = ""


class ReadNoiseAnalysisError(Exception):
    """Raised when robust read-noise characterization cannot be completed."""


def safe_dataset_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    return stem or "dataset"


def robust_std(values: np.ndarray, *, axis=None):
    array = np.asarray(values)
    median = np.nanmedian(array, axis=axis, keepdims=True)
    sigma = MAD_TO_SIGMA * np.nanmedian(np.abs(array - median), axis=axis)
    return float(sigma) if np.ndim(sigma) == 0 else sigma


def _resolve_roi(
    image_shape: tuple[int, int],
    roi: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    height, width = image_shape
    if roi is None:
        return (0, 0, width, height)
    x, y, roi_width, roi_height = roi
    if min(x, y) < 0 or roi_width <= 0 or roi_height <= 0:
        raise ReadNoiseAnalysisError(f"Invalid ROI: {roi}")
    if x + roi_width > width or y + roi_height > height:
        raise ReadNoiseAnalysisError(
            f"ROI {roi} exceeds image shape {image_shape}."
        )
    return roi


def _correct_level(image: np.ndarray, method: str) -> tuple[np.ndarray, float]:
    values = np.asarray(image, dtype=np.float32)
    if method == "median":
        level = float(np.nanmedian(values))
    elif method == "mean":
        level = float(np.nanmean(values, dtype=np.float64))
    elif method == "none":
        level = 0.0
    else:
        raise ReadNoiseAnalysisError(
            f"Unknown frame-level correction: {method!r}"
        )
    return values - level, level


def _clip_pair(values: np.ndarray, clip_sigma: float):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ReadNoiseAnalysisError("Pair difference contains no finite pixels.")
    center = float(np.median(finite))
    initial_sigma = float(robust_std(finite))
    if initial_sigma == 0.0:
        keep = finite == center
        lower = upper = center
    else:
        lower = center - clip_sigma * initial_sigma
        upper = center + clip_sigma * initial_sigma
        keep = (finite >= lower) & (finite <= upper)
    kept = finite[keep]
    if kept.size == 0:
        raise ReadNoiseAnalysisError("Pair clipping rejected every pixel.")
    return {
        "center": center,
        "initial_sigma": initial_sigma,
        "sigma": float(robust_std(kept)),
        "lower": float(lower),
        "upper": float(upper),
        "total": int(finite.size),
        "kept": int(kept.size),
        "rejected": int(finite.size - kept.size),
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ReadNoiseAnalysisError(f"No rows available for {path.name}.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_histogram(
    values: np.ndarray,
    path: Path,
    *,
    xlabel: str,
    title: str,
    bins: int,
    log_y: bool,
    percentile_range: tuple[float, float] | None,
) -> None:
    finite = np.asarray(values)
    finite = finite[np.isfinite(finite)]
    histogram_range = None
    note = "display: full finite sampled range"
    if percentile_range is not None:
        q0, q1 = percentile_range
        low, high = np.percentile(finite, [q0, q1])
        if low < high:
            histogram_range = (float(low), float(high))
            inside = (finite >= low) & (finite <= high)
            omitted = 1.0 - np.count_nonzero(inside) / float(finite.size)
            note = (
                f"display: P{q0:g}..P{q1:g}; "
                f"outside display={100.0 * omitted:.4g}%"
            )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(finite, bins=bins, range=histogram_range)
    ax.set(xlabel=xlabel, ylabel="Pixel count", title=title)
    if log_y:
        ax.set_yscale("log")
    ax.text(
        0.99, 0.98, note, transform=ax.transAxes,
        ha="right", va="top", fontsize="small"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_line_plot(
    x: Iterable[float],
    y: Iterable[float],
    path: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(x), list(y), marker="o")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _quantization_summary(values: np.ndarray):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    low, high = np.percentile(finite, [0.1, 99.9])
    central = np.round(finite[(finite >= low) & (finite <= high)], 10)
    levels, counts = np.unique(central, return_counts=True)
    positive_spacings = np.diff(levels)
    positive_spacings = positive_spacings[positive_spacings > 1e-10]
    spacing = (
        float(np.min(positive_spacings))
        if positive_spacings.size else float("nan")
    )
    total = int(np.sum(counts))
    rows = [
        {
            "level_index": index,
            "pair_difference_level_adu": float(level),
            "count": int(count),
            "fraction": int(count) / total,
        }
        for index, (level, count) in enumerate(zip(levels, counts))
    ]
    return {
        "pair_quantization_level_count": int(levels.size),
        "pair_quantization_min_level_spacing_adu": spacing,
        "estimated_original_difference_step_adu": (
            spacing * math.sqrt(2.0) if np.isfinite(spacing) else float("nan")
        ),
    }, rows


def _analyze_dataset(
    group: DatasetGroupLike,
    config: ReadNoiseConfig,
    *,
    progress: ProgressCallback | None = None,
) -> _AnalysisContext:
    if group.n_frames < 2:
        raise ReadNoiseAnalysisError("At least two frames are required.")
    if config.clip_sigma <= 0:
        raise ReadNoiseAnalysisError("clip_sigma must be positive.")
    if config.temporal_chunk_rows <= 0:
        raise ReadNoiseAnalysisError("temporal_chunk_rows must be positive.")

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    x, y, width, height = _resolve_roi(group.image_shape, config.roi)

    stack_path = output / ".read_noise_stack.npy"
    temporal_path = output / ".temporal_noise.npy"

    frames = np.lib.format.open_memmap(
        stack_path,
        mode="w+",
        dtype=np.float32,
        shape=(group.n_frames, height, width),
    )

    frame_levels: list[FrameLevelRecord] = []
    try:
        for current, frame in enumerate(group.frames, start=1):
            if progress is not None:
                progress(current, group.n_frames, frame)

            image = read_image(frame)
            if tuple(image.shape) != tuple(group.image_shape):
                raise ReadNoiseAnalysisError(
                    f"Image shape changed: {frame.filepath}: "
                    f"expected={group.image_shape}, actual={image.shape}"
                )

            cropped = image[y:y + height, x:x + width]
            corrected, level = _correct_level(
                cropped, config.frame_level_correction
            )
            frames[current - 1] = corrected
            frame_levels.append(
                FrameLevelRecord(
                    frame_index=frame.frame_index,
                    filename=frame.filepath.name,
                    temperature_C=frame.temperature_C,
                    level_adu=level,
                )
            )
        frames.flush()

        pair_records: list[PairNoiseRecord] = []
        histogram_samples: list[np.ndarray] = []
        first_pair_difference: np.ndarray | None = None

        for pair_index in range(group.n_frames // 2):
            a = pair_index * 2
            b = a + 1
            difference = (
                frames[a].astype(np.float64)
                - frames[b].astype(np.float64)
            ) / math.sqrt(2.0)

            if first_pair_difference is None:
                first_pair_difference = difference.astype(np.float32)

            clipped = _clip_pair(difference.ravel(), config.clip_sigma)
            frame_a = group.frames[a]
            frame_b = group.frames[b]
            pair_records.append(
                PairNoiseRecord(
                    pair_index=pair_index,
                    frame_index_a=frame_a.frame_index,
                    frame_index_b=frame_b.frame_index,
                    filename_a=frame_a.filepath.name,
                    filename_b=frame_b.filepath.name,
                    pair_offset_median_adu=clipped["center"],
                    initial_sigma_adu_rms=clipped["initial_sigma"],
                    noise_adu_rms=clipped["sigma"],
                    clip_lower_adu=clipped["lower"],
                    clip_upper_adu=clipped["upper"],
                    total_pixels=clipped["total"],
                    kept_pixels=clipped["kept"],
                    rejected_pixels=clipped["rejected"],
                    rejected_fraction=(
                        clipped["rejected"] / clipped["total"]
                    ),
                )
            )

            flat = difference.ravel()
            stride = max(1, flat.size // 250_000)
            histogram_samples.append(flat[::stride])

        temporal_noise = np.lib.format.open_memmap(
            temporal_path,
            mode="w+",
            dtype=np.float32,
            shape=(height, width),
        )
        for row0 in range(0, height, config.temporal_chunk_rows):
            row1 = min(height, row0 + config.temporal_chunk_rows)
            block = np.asarray(
                frames[:, row0:row1, :],
                dtype=np.float32,
            )
            temporal_noise[row0:row1] = np.asarray(
                robust_std(block, axis=0),
                dtype=np.float32,
            )
        temporal_noise.flush()

        pair_values = np.asarray(
            [record.noise_adu_rms for record in pair_records],
            dtype=np.float64,
        )
        temporal_finite = np.asarray(
            temporal_noise[np.isfinite(temporal_noise)],
            dtype=np.float32,
        )
        if temporal_finite.size == 0:
            raise ReadNoiseAnalysisError(
                "Temporal-noise map contains no finite pixels."
            )

        pair_sample = np.concatenate(histogram_samples)
        quantization, level_rows = _quantization_summary(pair_sample)

        return _AnalysisContext(
            frame_levels=frame_levels,
            pair_records=pair_records,
            pair_values=pair_values,
            first_pair_difference=first_pair_difference,
            temporal_noise=temporal_noise,
            temporal_finite=temporal_finite,
            quantization=quantization,
            quantization_level_rows=level_rows,
            histogram_samples=histogram_samples,
            temporal_path=temporal_path,
        )
    finally:
        del frames
        stack_path.unlink(missing_ok=True)


def _build_dataset_characterization(
    *,
    group: DatasetGroupLike,
    frame_levels: list[FrameLevelRecord],
    pair_values: np.ndarray,
    temporal_finite: np.ndarray,
    temporal_noise: np.ndarray,
    quantization: dict,
) -> DatasetCharacterization:
    """Build an immutable dataset characterization."""

    step = quantization["estimated_original_difference_step_adu"]
    if not np.isfinite(step):
        step = None

    return DatasetCharacterization(
        dataset=group.name,
        n_frames=group.n_frames,

        pair_noise_median_adu_rms=float(
            np.median(pair_values)
        ),

        temporal_noise_median_adu_rms=float(
            np.median(temporal_finite)
        ),

        frame_offset_sigma_adu=float(
            robust_std(
                np.asarray(
                    [r.level_adu for r in frame_levels],
                    dtype=np.float64,
                )
            )
        ),

        quantization_step_adu=step,

        # Step02ではまだ計算していない
        saturated_pixel_fraction=0.0,

        finite_pixel_fraction=float(
            np.count_nonzero(np.isfinite(temporal_noise))
            / temporal_noise.size
        ),
    )


def characterize_dataset(
    group: DatasetGroupLike,
    config: ReadNoiseConfig,
    *,
    progress: ProgressCallback | None = None,
) -> DatasetCharacterization:
    """Characterize one dataset without producing report artifacts."""

    context = _analyze_dataset(
        group,
        config,
        progress=progress,
    )

    try:
        return _build_dataset_characterization(
            group=group,
            frame_levels=context.frame_levels,
            pair_values=context.pair_values,
            temporal_finite=context.temporal_finite,
            temporal_noise=context.temporal_noise,
            quantization=context.quantization,
        )
    finally:
        temporal_path = context.temporal_path
        del context
        temporal_path.unlink(missing_ok=True)


def _build_read_noise_summary(
    *,
    group: DatasetGroupLike,
    config: ReadNoiseConfig,
    roi: tuple[int, int, int, int],
    context: _AnalysisContext,
) -> dict[str, Any]:
    """Build the detailed Step 02 read-noise summary."""
    x, y, width, height = roi
    pair_records = context.pair_records
    pair_values = context.pair_values
    temporal_finite = context.temporal_finite

    rejected = sum(record.rejected_pixels for record in pair_records)
    total = sum(record.total_pixels for record in pair_records)
    temporal_p = np.percentile(
        temporal_finite,
        [1, 5, 16, 84, 95, 99],
    )

    return {
        "schema": "rts-framework.read-noise-summary",
        "schema_version": 1,
        "analysis_version": __version__,
        "dataset": group.name,
        "environment": group.environment,
        "frames": group.n_frames,
        "pairs": len(pair_records),
        "unpaired_frames": group.n_frames % 2,
        "exposure_s": group.exposure_s,
        "temperature_min_C": group.temperature_min_C,
        "temperature_max_C": group.temperature_max_C,
        "image_height": group.image_shape[0],
        "image_width": group.image_shape[1],
        "roi_x": x,
        "roi_y": y,
        "roi_width": width,
        "roi_height": height,
        "frame_level_correction": config.frame_level_correction,
        "clip_sigma": config.clip_sigma,
        "pair_noise_median_adu_rms": float(
            np.median(pair_values)
        ),
        "pair_noise_mean_adu_rms": float(np.mean(pair_values)),
        "pair_noise_p16_adu_rms": float(
            np.percentile(pair_values, 16)
        ),
        "pair_noise_p84_adu_rms": float(
            np.percentile(pair_values, 84)
        ),
        "clipping_rejected_pixels": rejected,
        "clipping_total_pixels": total,
        "clipping_rejected_fraction": rejected / total,
        "pair_median_rejected_fraction": float(
            np.median(
                [
                    record.rejected_fraction
                    for record in pair_records
                ]
            )
        ),
        "pair_max_rejected_fraction": float(
            np.max(
                [
                    record.rejected_fraction
                    for record in pair_records
                ]
            )
        ),
        "temporal_noise_median_adu_rms": float(
            np.median(temporal_finite)
        ),
        "temporal_noise_mean_adu_rms": float(
            np.mean(temporal_finite)
        ),
        "temporal_noise_p01_adu_rms": float(temporal_p[0]),
        "temporal_noise_p05_adu_rms": float(temporal_p[1]),
        "temporal_noise_p16_adu_rms": float(temporal_p[2]),
        "temporal_noise_p84_adu_rms": float(temporal_p[3]),
        "temporal_noise_p95_adu_rms": float(temporal_p[4]),
        "temporal_noise_p99_adu_rms": float(temporal_p[5]),
        "temporal_noise_max_adu_rms": float(
            np.max(temporal_finite)
        ),
        **context.quantization,
    }


def _write_read_noise_tables(
    *,
    output: Path,
    context: _AnalysisContext,
    summary: dict[str, Any],
) -> None:
    """Write CSV and JSON table products."""
    _write_rows(
        output / "frame_level.csv",
        [asdict(record) for record in context.frame_levels],
    )
    _write_rows(
        output / "pair_noise_values.csv",
        [asdict(record) for record in context.pair_records],
    )
    _write_rows(
        output / "pair_value_levels.csv",
        context.quantization_level_rows,
    )
    _write_rows(output / "read_noise_summary.csv", [summary])
    (output / "read_noise_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_read_noise_report(
    *,
    output: Path,
    group: DatasetGroupLike,
    config: ReadNoiseConfig,
    roi: tuple[int, int, int, int],
    pair_count: int,
    summary: dict[str, Any],
) -> None:
    """Write the human-readable Step 02 report."""
    report = [
        "RTS Framework robust read-noise characterization",
        "================================================",
        f"Dataset                         : {group.name}",
        f"Frames / pairs                  : "
        f"{group.n_frames} / {pair_count}",
        f"ROI (x,y,w,h)                   : {roi}",
        f"Frame-level correction          : "
        f"{config.frame_level_correction}",
        f"Pair noise median [ADU rms]     : "
        f"{summary['pair_noise_median_adu_rms']:.9g}",
        f"Overall rejected fraction       : "
        f"{100 * summary['clipping_rejected_fraction']:.9g} %",
        f"Temporal noise median [ADU rms] : "
        f"{summary['temporal_noise_median_adu_rms']:.9g}",
        "",
        "The rejected fraction is an outlier-quality indicator, "
        "not an RTS rate.",
        "No spatial median filter, RTS classification, or RTS "
        "mask is applied.",
        "Standard histograms display P0.1..P99.9; full-range "
        "output is separate.",
    ]
    (output / "read_noise_report.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


def _write_read_noise_plots(
    *,
    output: Path,
    config: ReadNoiseConfig,
    context: _AnalysisContext,
) -> None:
    """Write all Step 02 diagnostic plots."""
    _save_line_plot(
        [record.frame_index for record in context.frame_levels],
        [record.level_adu for record in context.frame_levels],
        output / "frame_level_drift.png",
        xlabel="Frame index",
        ylabel="Removed frame level [ADU]",
        title="Frame-level drift",
    )
    _save_line_plot(
        [record.pair_index for record in context.pair_records],
        [record.noise_adu_rms for record in context.pair_records],
        output / "pair_noise_by_pair.png",
        xlabel="Pair index",
        ylabel="Robust pair noise [ADU rms]",
        title="Pair-difference noise",
    )
    _save_line_plot(
        [record.pair_index for record in context.pair_records],
        [
            100 * record.rejected_fraction
            for record in context.pair_records
        ],
        output / "pair_rejected_fraction_by_pair.png",
        xlabel="Pair index",
        ylabel="Rejected pixels [%]",
        title="MAD clipping rejection fraction",
    )

    pair_sample = np.concatenate(context.histogram_samples)
    for filename, log_y, percentile_range in (
        ("pair_difference_histogram.png", False, (0.1, 99.9)),
        ("pair_difference_histogram_log.png", True, (0.1, 99.9)),
        (
            "pair_difference_histogram_full_range_log.png",
            True,
            None,
        ),
    ):
        _save_histogram(
            pair_sample,
            output / filename,
            xlabel="(Frame A - Frame B) / sqrt(2) [ADU]",
            title="Pair-difference distribution",
            bins=config.hist_bins,
            log_y=log_y,
            percentile_range=percentile_range,
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    vmin, vmax = np.percentile(
        context.temporal_finite,
        [1, 99],
    )
    shown = ax.imshow(
        context.temporal_noise,
        origin="lower",
        vmin=float(vmin),
        vmax=float(vmax),
    )
    ax.set(
        xlabel="ROI x pixel",
        ylabel="ROI y pixel",
        title="Per-pixel robust temporal noise",
    )
    fig.colorbar(
        shown,
        ax=ax,
        label="Temporal noise [ADU rms]",
    )
    fig.tight_layout()
    fig.savefig(output / "temporal_noise_map.png", dpi=160)
    plt.close(fig)

    for filename, log_y in (
        ("temporal_noise_histogram.png", False),
        ("temporal_noise_histogram_log.png", True),
    ):
        _save_histogram(
            context.temporal_noise,
            output / filename,
            xlabel="Per-pixel temporal noise [ADU rms]",
            title="Temporal-noise distribution",
            bins=config.hist_bins,
            log_y=log_y,
            percentile_range=(0.1, 99.9),
        )


def _write_read_noise_fits(
    *,
    output: Path,
    context: _AnalysisContext,
) -> None:
    """Write Step 02 FITS data products."""
    fits.PrimaryHDU(context.temporal_noise).writeto(
        output / "temporal_noise_map.fits",
        overwrite=True,
    )
    if context.first_pair_difference is not None:
        fits.PrimaryHDU(
            context.first_pair_difference
        ).writeto(
            output / "pair_difference_0000.fits",
            overwrite=True,
        )


def _write_read_noise_outputs(
    *,
    output: Path,
    group: DatasetGroupLike,
    config: ReadNoiseConfig,
    roi: tuple[int, int, int, int],
    context: _AnalysisContext,
    characterization: DatasetCharacterization,
    summary: dict[str, Any],
) -> None:
    """Write the complete deterministic Step 02 output set."""
    _write_read_noise_tables(
        output=output,
        context=context,
        summary=summary,
    )
    write_dataset_characterization(
        characterization,
        output / "dataset_characterization.json",
    )
    _write_read_noise_report(
        output=output,
        group=group,
        config=config,
        roi=roi,
        pair_count=len(context.pair_records),
        summary=summary,
    )
    _write_read_noise_plots(
        output=output,
        config=config,
        context=context,
    )
    _write_read_noise_fits(
        output=output,
        context=context,
    )


def analyze_read_noise_dataset(
    group: DatasetGroupLike,
    config: ReadNoiseConfig,
    *,
    progress: ProgressCallback | None = None,
) -> ReadNoiseResult:
    """Analyze one immutable Step 02 dataset and write science outputs."""
    if config.hist_bins <= 0:
        raise ReadNoiseAnalysisError("hist_bins must be positive.")

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    roi = _resolve_roi(group.image_shape, config.roi)

    context = _analyze_dataset(
        group,
        config,
        progress=progress,
    )
    temporal_path = context.temporal_path

    try:
        characterization = _build_dataset_characterization(
            group=group,
            frame_levels=context.frame_levels,
            pair_values=context.pair_values,
            temporal_finite=context.temporal_finite,
            temporal_noise=context.temporal_noise,
            quantization=context.quantization,
        )
        summary = _build_read_noise_summary(
            group=group,
            config=config,
            roi=roi,
            context=context,
        )

        _write_read_noise_outputs(
            output=output,
            group=group,
            config=config,
            roi=roi,
            context=context,
            characterization=characterization,
            summary=summary,
        )

        return ReadNoiseResult(
            dataset=group.name,
            status="PASSED",
            frames=group.n_frames,
            pairs=len(context.pair_records),
            pair_noise_median_adu_rms=summary[
                "pair_noise_median_adu_rms"
            ],
            clipping_rejected_fraction=summary[
                "clipping_rejected_fraction"
            ],
            temporal_noise_median_adu_rms=summary[
                "temporal_noise_median_adu_rms"
            ],
            output_directory=str(output),
        )
    finally:
        del context
        temporal_path.unlink(missing_ok=True)

