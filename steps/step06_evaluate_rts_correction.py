"""Step 06: deterministic numerical quality assessment of RTS correction.

Version 6.4.0 adds optional mask-based astronomical CMOS science metrics to the deterministic scientific assessment and recommendations to the multi-page PDF report to the PNG visualization to the numerical quality assessment introduced in v6.0.0. It provides a focused comparison between an original FITS image and
its corrected counterpart.  It performs validation and reports reproducible
whole-frame difference and noise statistics, plus five reproducible diagnostic PNG files and a reproducible PDF report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import sys
import tempfile
from typing import Sequence

import numpy as np

# Support both package import and direct execution.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from common.image_io import ImageIOError, read_image

__version__ = "6.4.0"

__all__ = [
    "RTSCorrectionEvaluation",
    "RTSCorrectionPlotOutputs",
    "RTSCorrectionAssessment",
    "RTSScienceMetrics",
    "Step06Error",
    "evaluate_rts_correction",
    "assess_rts_correction",
    "calculate_rts_science_metrics",
    "write_rts_science_metrics_json",
    "generate_rts_evaluation_plots",
    "generate_rts_evaluation_pdf",
    "run_rts_evaluation_cli",
    "write_rts_evaluation_json",
]


class Step06Error(Exception):
    """Raised when Step06 cannot evaluate an RTS-correction result."""


@dataclass(slots=True, frozen=True)
class RTSCorrectionEvaluation:
    """Immutable numerical comparison of original and corrected images.

    ``difference`` is defined as ``corrected - original``.  Signed correction
    statistics therefore retain the direction of the change, while absolute
    statistics describe its magnitude.
    """

    original_path: Path
    corrected_path: Path
    image_shape: tuple[int, int]
    original_dtype: str
    corrected_dtype: str
    pixel_count: int
    finite_pixel_count: int
    changed_pixel_count: int
    changed_pixel_fraction: float
    change_tolerance: float
    difference_sum: float
    difference_mean: float
    difference_median: float
    difference_minimum: float
    difference_maximum: float
    absolute_difference_sum: float
    absolute_difference_mean: float
    absolute_difference_median: float
    absolute_difference_maximum: float
    difference_rms: float
    original_mean: float
    corrected_mean: float
    original_std: float
    corrected_std: float
    noise_std_change: float
    noise_std_ratio: float | None

    @property
    def unchanged_pixel_count(self) -> int:
        """Return the number of finite pixels not changed above tolerance."""
        return self.finite_pixel_count - self.changed_pixel_count

    def summary(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable summary."""
        return {
            "step06_version": __version__,
            "original_path": str(self.original_path),
            "corrected_path": str(self.corrected_path),
            "image_shape": list(self.image_shape),
            "original_dtype": self.original_dtype,
            "corrected_dtype": self.corrected_dtype,
            "pixel_count": self.pixel_count,
            "finite_pixel_count": self.finite_pixel_count,
            "changed_pixel_count": self.changed_pixel_count,
            "unchanged_pixel_count": self.unchanged_pixel_count,
            "changed_pixel_fraction": self.changed_pixel_fraction,
            "change_tolerance": self.change_tolerance,
            "difference_definition": "corrected - original",
            "difference": {
                "sum": self.difference_sum,
                "mean": self.difference_mean,
                "median": self.difference_median,
                "minimum": self.difference_minimum,
                "maximum": self.difference_maximum,
                "rms": self.difference_rms,
            },
            "absolute_difference": {
                "sum": self.absolute_difference_sum,
                "mean": self.absolute_difference_mean,
                "median": self.absolute_difference_median,
                "maximum": self.absolute_difference_maximum,
            },
            "image_statistics": {
                "original_mean": self.original_mean,
                "corrected_mean": self.corrected_mean,
                "original_std": self.original_std,
                "corrected_std": self.corrected_std,
                "noise_std_change": self.noise_std_change,
                "noise_std_ratio": self.noise_std_ratio,
            },
        }


@dataclass(slots=True, frozen=True)
class RTSCorrectionAssessment:
    """Immutable interpretation of one numerical RTS-correction evaluation.

    The assessment is intentionally conservative and transparent.  It uses
    whole-frame standard-deviation ratio and mean-shift size; it does not claim
    that a correction is scientifically valid without domain-specific review.
    """

    grade: str
    recommendation: str
    noise_reduction_fraction: float | None
    mean_shift_sigma: float | None
    concerns: tuple[str, ...]
    criteria_version: str = "1.0"

    def summary(self) -> dict[str, object]:
        return {
            "criteria_version": self.criteria_version,
            "grade": self.grade,
            "recommendation": self.recommendation,
            "noise_reduction_fraction": self.noise_reduction_fraction,
            "mean_shift_sigma": self.mean_shift_sigma,
            "concerns": list(self.concerns),
        }


def assess_rts_correction(
    evaluation: RTSCorrectionEvaluation,
) -> RTSCorrectionAssessment:
    """Classify correction quality using fixed, documented thresholds.

    Grades are based on whole-frame noise ratio and absolute mean shift in
    units of the original standard deviation:

    * Excellent: ratio <= 0.90 and mean shift <= 0.05 sigma
    * Good: ratio <= 1.00 and mean shift <= 0.10 sigma
    * Acceptable: ratio <= 1.05 and mean shift <= 0.25 sigma
    * Warning: otherwise, or when the ratio is undefined
    """
    ratio = evaluation.noise_std_ratio
    if evaluation.original_std == 0.0:
        mean_shift_sigma = (
            0.0 if evaluation.difference_mean == 0.0 else None
        )
    else:
        mean_shift_sigma = abs(evaluation.difference_mean) / evaluation.original_std

    if ratio is None or mean_shift_sigma is None:
        grade = "Warning"
    elif ratio <= 0.90 and mean_shift_sigma <= 0.05:
        grade = "Excellent"
    elif ratio <= 1.00 and mean_shift_sigma <= 0.10:
        grade = "Good"
    elif ratio <= 1.05 and mean_shift_sigma <= 0.25:
        grade = "Acceptable"
    else:
        grade = "Warning"

    recommendation_by_grade = {
        "Excellent": "Correction substantially reduces whole-frame scatter without a material mean shift.",
        "Good": "Correction improves or preserves whole-frame scatter with a small mean shift.",
        "Acceptable": "Correction is numerically plausible, but diagnostic plots should be reviewed before scientific use.",
        "Warning": "Correction requires review before scientific use; one or more whole-frame checks exceeded the fixed criteria.",
    }

    concerns: list[str] = []
    if evaluation.changed_pixel_count == 0:
        concerns.append("No pixels exceeded the configured change tolerance.")
    if ratio is None:
        concerns.append("Noise ratio is undefined because the original standard deviation is zero.")
    elif ratio > 1.05:
        concerns.append("Corrected whole-frame standard deviation increased by more than 5%.")
    elif ratio > 1.00:
        concerns.append("Corrected whole-frame standard deviation increased slightly.")
    if mean_shift_sigma is None:
        concerns.append("Mean-shift significance is undefined because the original standard deviation is zero.")
    elif mean_shift_sigma > 0.25:
        concerns.append("Absolute mean shift exceeds 0.25 times the original standard deviation.")
    if evaluation.changed_pixel_fraction > 0.10:
        concerns.append("More than 10% of finite pixels were changed; verify dictionary selectivity and tolerance.")
    if not concerns:
        concerns.append("No automatic whole-frame concerns were detected.")

    reduction = None if ratio is None else 1.0 - ratio
    return RTSCorrectionAssessment(
        grade=grade,
        recommendation=recommendation_by_grade[grade],
        noise_reduction_fraction=reduction,
        mean_shift_sigma=mean_shift_sigma,
        concerns=tuple(concerns),
    )


@dataclass(slots=True, frozen=True)
class RTSScienceMetrics:
    """Immutable mask-based metrics for astronomical CMOS RTS correction.

    The reference mask identifies pixels known or suspected to exhibit RTS.
    A reference pixel is considered covered when its absolute correction is
    greater than ``change_tolerance``.  Unchanged reference pixels are reported
    as residual *candidates*, not proven residual RTS events.
    """

    mask_path: Path
    image_shape: tuple[int, int]
    finite_pixel_count: int
    reference_pixel_count: int
    rts_pixel_fraction: float
    changed_pixel_count: int
    changed_reference_pixel_count: int
    correction_coverage_fraction: float | None
    residual_candidate_count: int
    residual_candidate_fraction: float | None
    off_mask_changed_pixel_count: int
    off_mask_changed_fraction: float
    correction_selectivity_fraction: float | None
    reference_cluster_count: int
    largest_reference_cluster_size: int
    changed_reference_cluster_count: int
    largest_changed_reference_cluster_size: int
    connectivity: int = 8
    definition_version: str = "1.0"

    def summary(self) -> dict[str, object]:
        return {
            "definition_version": self.definition_version,
            "mask_path": str(self.mask_path),
            "image_shape": list(self.image_shape),
            "finite_pixel_count": self.finite_pixel_count,
            "reference_pixel_count": self.reference_pixel_count,
            "rts_pixel_fraction": self.rts_pixel_fraction,
            "changed_pixel_count": self.changed_pixel_count,
            "changed_reference_pixel_count": self.changed_reference_pixel_count,
            "correction_coverage_fraction": self.correction_coverage_fraction,
            "residual_candidate_count": self.residual_candidate_count,
            "residual_candidate_fraction": self.residual_candidate_fraction,
            "off_mask_changed_pixel_count": self.off_mask_changed_pixel_count,
            "off_mask_changed_fraction": self.off_mask_changed_fraction,
            "correction_selectivity_fraction": self.correction_selectivity_fraction,
            "reference_cluster_count": self.reference_cluster_count,
            "largest_reference_cluster_size": self.largest_reference_cluster_size,
            "changed_reference_cluster_count": self.changed_reference_cluster_count,
            "largest_changed_reference_cluster_size": self.largest_changed_reference_cluster_size,
            "connectivity": self.connectivity,
            "residual_definition": "reference-mask pixels not changed above tolerance; candidates only",
        }


def _cluster_statistics(mask: np.ndarray) -> tuple[int, int]:
    """Return deterministic 8-connected component count and largest size."""
    work = np.asarray(mask, dtype=bool)
    if work.ndim != 2:
        raise Step06Error(f"Cluster mask must be two-dimensional: shape={work.shape}")
    visited = np.zeros(work.shape, dtype=bool)
    components = 0
    largest = 0
    height, width = work.shape
    for y in range(height):
        for x in range(width):
            if not work[y, x] or visited[y, x]:
                continue
            components += 1
            size = 0
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                size += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < height and 0 <= nx < width and
                                work[ny, nx] and not visited[ny, nx]):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            largest = max(largest, size)
    return components, largest


def calculate_rts_science_metrics(
    original_path: str | Path,
    corrected_path: str | Path,
    mask_path: str | Path,
    *,
    change_tolerance: float = 0.0,
) -> RTSScienceMetrics:
    """Calculate deterministic mask-based RTS science metrics.

    Nonzero finite mask pixels are treated as reference RTS pixels.  The mask
    must be a numeric two-dimensional image with the same shape as the input
    pair.  At least one reference pixel must be present.
    """
    evaluation = evaluate_rts_correction(
        original_path, corrected_path, change_tolerance=change_tolerance
    )
    mask = _normalize_path(mask_path, label="RTS mask")
    try:
        original_image = np.asarray(read_image(evaluation.original_path), dtype=np.float64)
        corrected_image = np.asarray(read_image(evaluation.corrected_path), dtype=np.float64)
        mask_image_raw = read_image(mask)
    except (ImageIOError, NotImplementedError, TypeError, ValueError) as exc:
        raise Step06Error(str(exc)) from exc
    if mask_image_raw.shape != original_image.shape:
        raise Step06Error(
            "RTS mask shape differs from image shape: "
            f"mask={mask_image_raw.shape}, image={original_image.shape}"
        )
    if not np.issubdtype(mask_image_raw.dtype, np.number) and mask_image_raw.dtype != np.bool_:
        raise Step06Error(f"RTS mask dtype must be numeric or boolean: dtype={mask_image_raw.dtype}")
    mask_float = np.asarray(mask_image_raw, dtype=np.float64)
    if np.any(np.isinf(mask_float)):
        raise Step06Error("RTS mask must not contain infinite values")

    finite = np.isfinite(original_image) & np.isfinite(corrected_image)
    reference = finite & np.isfinite(mask_float) & (mask_float != 0.0)
    reference_count = int(np.count_nonzero(reference))
    if reference_count == 0:
        raise Step06Error("RTS mask contains no nonzero reference pixels on finite image pixels")

    difference = corrected_image - original_image
    changed = finite & (np.abs(difference) > evaluation.change_tolerance)
    changed_reference = changed & reference
    residual_candidates = reference & ~changed
    off_mask_changed = changed & ~reference

    changed_count = int(np.count_nonzero(changed))
    changed_reference_count = int(np.count_nonzero(changed_reference))
    residual_count = int(np.count_nonzero(residual_candidates))
    off_mask_count = int(np.count_nonzero(off_mask_changed))
    reference_clusters, largest_reference = _cluster_statistics(reference)
    changed_clusters, largest_changed = _cluster_statistics(changed_reference)

    return RTSScienceMetrics(
        mask_path=mask,
        image_shape=evaluation.image_shape,
        finite_pixel_count=evaluation.finite_pixel_count,
        reference_pixel_count=reference_count,
        rts_pixel_fraction=reference_count / evaluation.finite_pixel_count,
        changed_pixel_count=changed_count,
        changed_reference_pixel_count=changed_reference_count,
        correction_coverage_fraction=changed_reference_count / reference_count,
        residual_candidate_count=residual_count,
        residual_candidate_fraction=residual_count / reference_count,
        off_mask_changed_pixel_count=off_mask_count,
        off_mask_changed_fraction=off_mask_count / evaluation.finite_pixel_count,
        correction_selectivity_fraction=(
            None if changed_count == 0 else changed_reference_count / changed_count
        ),
        reference_cluster_count=reference_clusters,
        largest_reference_cluster_size=largest_reference,
        changed_reference_cluster_count=changed_clusters,
        largest_changed_reference_cluster_size=largest_changed,
    )


@dataclass(slots=True, frozen=True)
class RTSCorrectionPlotOutputs:
    """Immutable paths for the five Step06 diagnostic PNG files."""

    difference_image: Path
    correction_map: Path
    original_histogram: Path
    corrected_histogram: Path
    difference_histogram: Path

    def as_dict(self) -> dict[str, str]:
        """Return deterministic string paths keyed by plot type."""
        return {
            "difference_image": str(self.difference_image),
            "correction_map": str(self.correction_map),
            "original_histogram": str(self.original_histogram),
            "corrected_histogram": str(self.corrected_histogram),
            "difference_histogram": str(self.difference_histogram),
        }


def _normalize_path(path: str | Path, *, label: str) -> Path:
    normalized = Path(path).expanduser()
    if not normalized.is_file():
        raise Step06Error(f"{label} FITS file does not exist: {normalized}")
    return normalized.resolve()


def _finite_float64(image: np.ndarray, *, label: str) -> np.ndarray:
    if not np.issubdtype(image.dtype, np.number):
        raise Step06Error(
            f"{label} image dtype must be numeric: dtype={image.dtype}"
        )
    try:
        return np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Step06Error(
            f"Unable to convert {label} image to float64: {exc}"
        ) from exc


def evaluate_rts_correction(
    original_path: str | Path,
    corrected_path: str | Path,
    *,
    change_tolerance: float = 0.0,
) -> RTSCorrectionEvaluation:
    """Evaluate one original/corrected FITS pair numerically.

    Parameters
    ----------
    original_path, corrected_path
        Two-dimensional FITS images to compare.
    change_tolerance
        A non-negative absolute threshold.  A finite pixel is counted as
        changed only when ``abs(corrected - original) > change_tolerance``.

    Raises
    ------
    Step06Error
        If files are missing or unreadable, shapes differ, tolerance is
        invalid, non-finite masks differ, or no finite pixels remain.
    """
    if isinstance(change_tolerance, bool):
        raise Step06Error("change_tolerance must be a finite non-negative number")
    try:
        tolerance = float(change_tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Step06Error(
            "change_tolerance must be a finite non-negative number"
        ) from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise Step06Error("change_tolerance must be a finite non-negative number")

    original = _normalize_path(original_path, label="Original")
    corrected = _normalize_path(corrected_path, label="Corrected")
    if original == corrected:
        raise Step06Error("Original and corrected paths must be different")

    try:
        original_image = read_image(original)
        corrected_image = read_image(corrected)
    except (ImageIOError, NotImplementedError) as exc:
        raise Step06Error(str(exc)) from exc

    if original_image.shape != corrected_image.shape:
        raise Step06Error(
            "Original and corrected image shapes differ: "
            f"original={original_image.shape}, corrected={corrected_image.shape}"
        )

    original_float = _finite_float64(original_image, label="Original")
    corrected_float = _finite_float64(corrected_image, label="Corrected")

    original_finite = np.isfinite(original_float)
    corrected_finite = np.isfinite(corrected_float)
    if not np.array_equal(original_finite, corrected_finite):
        mismatch_count = int(np.count_nonzero(original_finite != corrected_finite))
        raise Step06Error(
            "Original and corrected images have different finite-pixel masks: "
            f"mismatched_pixels={mismatch_count}"
        )

    finite_mask = original_finite
    finite_count = int(np.count_nonzero(finite_mask))
    if finite_count == 0:
        raise Step06Error("Images contain no mutually finite pixels")

    original_values = original_float[finite_mask]
    corrected_values = corrected_float[finite_mask]
    difference = corrected_values - original_values
    absolute_difference = np.abs(difference)
    changed_count = int(np.count_nonzero(absolute_difference > tolerance))

    original_std = float(np.std(original_values, ddof=0))
    corrected_std = float(np.std(corrected_values, ddof=0))
    if original_std == 0.0:
        noise_std_ratio: float | None = (
            1.0 if corrected_std == 0.0 else None
        )
    else:
        noise_std_ratio = corrected_std / original_std

    return RTSCorrectionEvaluation(
        original_path=original,
        corrected_path=corrected,
        image_shape=(int(original_image.shape[0]), int(original_image.shape[1])),
        original_dtype=str(original_image.dtype),
        corrected_dtype=str(corrected_image.dtype),
        pixel_count=int(original_image.size),
        finite_pixel_count=finite_count,
        changed_pixel_count=changed_count,
        changed_pixel_fraction=changed_count / finite_count,
        change_tolerance=tolerance,
        difference_sum=float(np.sum(difference, dtype=np.float64)),
        difference_mean=float(np.mean(difference)),
        difference_median=float(np.median(difference)),
        difference_minimum=float(np.min(difference)),
        difference_maximum=float(np.max(difference)),
        absolute_difference_sum=float(
            np.sum(absolute_difference, dtype=np.float64)
        ),
        absolute_difference_mean=float(np.mean(absolute_difference)),
        absolute_difference_median=float(np.median(absolute_difference)),
        absolute_difference_maximum=float(np.max(absolute_difference)),
        difference_rms=float(np.sqrt(np.mean(np.square(difference)))),
        original_mean=float(np.mean(original_values)),
        corrected_mean=float(np.mean(corrected_values)),
        original_std=original_std,
        corrected_std=corrected_std,
        noise_std_change=corrected_std - original_std,
        noise_std_ratio=noise_std_ratio,
    )


def _validate_plot_options(
    output_directory: str | Path,
    *,
    bins: int,
    dpi: int,
) -> tuple[Path, int, int]:
    directory = Path(output_directory).expanduser()
    if not directory.is_dir():
        raise Step06Error(f"Plot output directory does not exist: {directory}")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise Step06Error("bins must be an integer greater than or equal to 2")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 50:
        raise Step06Error("dpi must be an integer greater than or equal to 50")
    return directory.resolve(), bins, dpi


def _plot_output_paths(directory: Path) -> RTSCorrectionPlotOutputs:
    return RTSCorrectionPlotOutputs(
        difference_image=directory / "difference_image.png",
        correction_map=directory / "correction_map.png",
        original_histogram=directory / "original_histogram.png",
        corrected_histogram=directory / "corrected_histogram.png",
        difference_histogram=directory / "difference_histogram.png",
    )


def _save_figure_atomic(figure, path: Path, *, dpi: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise Step06Error(f"Plot output already exists: {path}")
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        figure.savefig(
            temporary,
            dpi=dpi,
            bbox_inches="tight",
            metadata={"Software": f"RTS-Framework Step06 {__version__}"},
        )
        temporary.replace(path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise Step06Error(f"Unable to write plot {path}: {exc}") from exc


def generate_rts_evaluation_plots(
    original_path: str | Path,
    corrected_path: str | Path,
    output_directory: str | Path,
    *,
    change_tolerance: float = 0.0,
    bins: int = 128,
    dpi: int = 120,
    overwrite: bool = False,
) -> RTSCorrectionPlotOutputs:
    """Generate five deterministic diagnostic PNG files for one FITS pair.

    The correction map marks pixels for which
    ``abs(corrected - original) > change_tolerance``.  The difference image is
    displayed with symmetric robust limits derived from the 99.5th percentile
    of finite absolute differences.  Histogram bin edges are calculated once
    per relevant dataset and are therefore reproducible for identical inputs.
    """
    directory, bins, dpi = _validate_plot_options(
        output_directory, bins=bins, dpi=dpi
    )
    evaluation = evaluate_rts_correction(
        original_path, corrected_path, change_tolerance=change_tolerance
    )
    outputs = _plot_output_paths(directory)
    for path in outputs.__dict__.values() if hasattr(outputs, "__dict__") else (
        outputs.difference_image, outputs.correction_map,
        outputs.original_histogram, outputs.corrected_histogram,
        outputs.difference_histogram,
    ):
        if Path(path).exists() and not overwrite:
            raise Step06Error(f"Plot output already exists: {path}")

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise Step06Error("matplotlib is required for Step06 plot generation") from exc

    try:
        original_image = np.asarray(read_image(evaluation.original_path), dtype=np.float64)
        corrected_image = np.asarray(read_image(evaluation.corrected_path), dtype=np.float64)
    except (ImageIOError, NotImplementedError) as exc:
        raise Step06Error(str(exc)) from exc

    finite = np.isfinite(original_image) & np.isfinite(corrected_image)
    difference = corrected_image - original_image
    changed = finite & (np.abs(difference) > evaluation.change_tolerance)
    original_values = original_image[finite]
    corrected_values = corrected_image[finite]
    difference_values = difference[finite]

    figures = []
    try:
        robust_limit = float(np.percentile(np.abs(difference_values), 99.5))
        if robust_limit == 0.0:
            robust_limit = 1.0

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        image = ax.imshow(
            difference, origin="lower", interpolation="nearest",
            vmin=-robust_limit, vmax=robust_limit, cmap="coolwarm"
        )
        ax.set_title("RTS correction difference (corrected - original)")
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")
        fig.colorbar(image, ax=ax, label="Difference [ADU]")
        figures.append((fig, outputs.difference_image))

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        image = ax.imshow(
            changed.astype(np.uint8), origin="lower", interpolation="nearest",
            vmin=0, vmax=1, cmap="binary"
        )
        ax.set_title(
            f"Changed-pixel map (|difference| > {evaluation.change_tolerance:.6g})"
        )
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")
        fig.colorbar(image, ax=ax, ticks=[0, 1], label="Changed")
        figures.append((fig, outputs.correction_map))

        combined_min = float(min(np.min(original_values), np.min(corrected_values)))
        combined_max = float(max(np.max(original_values), np.max(corrected_values)))
        if combined_min == combined_max:
            combined_min -= 0.5
            combined_max += 0.5
        image_edges = np.linspace(combined_min, combined_max, bins + 1)

        for values, title, path in (
            (original_values, "Original image histogram", outputs.original_histogram),
            (corrected_values, "Corrected image histogram", outputs.corrected_histogram),
        ):
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            ax.hist(values, bins=image_edges)
            ax.set_title(title)
            ax.set_xlabel("Pixel value [ADU]")
            ax.set_ylabel("Count")
            figures.append((fig, path))

        diff_min = float(np.min(difference_values))
        diff_max = float(np.max(difference_values))
        if diff_min == diff_max:
            diff_min -= 0.5
            diff_max += 0.5
        diff_edges = np.linspace(diff_min, diff_max, bins + 1)
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.hist(difference_values, bins=diff_edges)
        ax.set_title("Correction difference histogram")
        ax.set_xlabel("Corrected - original [ADU]")
        ax.set_ylabel("Count")
        figures.append((fig, outputs.difference_histogram))

        written: list[Path] = []
        try:
            for figure, path in figures:
                _save_figure_atomic(figure, path, dpi=dpi, overwrite=overwrite)
                written.append(path)
        except Step06Error:
            for path in written:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
    finally:
        for figure, _ in figures:
            plt.close(figure)

    return RTSCorrectionPlotOutputs(
        difference_image=outputs.difference_image.resolve(),
        correction_map=outputs.correction_map.resolve(),
        original_histogram=outputs.original_histogram.resolve(),
        corrected_histogram=outputs.corrected_histogram.resolve(),
        difference_histogram=outputs.difference_histogram.resolve(),
    )



def generate_rts_evaluation_pdf(
    original_path: str | Path,
    corrected_path: str | Path,
    output_path: str | Path,
    *,
    change_tolerance: float = 0.0,
    bins: int = 128,
    plot_dpi: int = 120,
    overwrite: bool = False,
    rts_mask_path: str | Path | None = None,
) -> Path:
    """Generate a deterministic multi-page PDF evaluation report.

    The report contains a numerical summary followed by all five Step06
    diagnostic plots. Plot files are produced in a temporary directory and
    removed after the PDF has been written.
    """
    path = Path(output_path).expanduser()
    if path.suffix.lower() != ".pdf":
        raise Step06Error(f"PDF output path must end with .pdf: {path}")
    if path.exists() and not overwrite:
        raise Step06Error(f"Output PDF already exists: {path}")
    if not path.parent.is_dir():
        raise Step06Error(f"Output directory does not exist: {path.parent}")

    # Reuse the validation and numerical definitions used by PNG generation.
    _, bins, plot_dpi = _validate_plot_options(
        path.parent, bins=bins, dpi=plot_dpi
    )
    evaluation = evaluate_rts_correction(
        original_path, corrected_path, change_tolerance=change_tolerance
    )
    assessment = assess_rts_correction(evaluation)
    science_metrics = (
        None if rts_mask_path is None else calculate_rts_science_metrics(
            evaluation.original_path, evaluation.corrected_path, rts_mask_path,
            change_tolerance=evaluation.change_tolerance,
        )
    )

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            Image as RLImage, PageBreak, KeepTogether,
        )
    except ImportError as exc:
        raise Step06Error(
            "reportlab is required for Step06 PDF report generation"
        ) from exc

    temporary_pdf = path.with_name(f".{path.name}.tmp")
    try:
        with tempfile.TemporaryDirectory(prefix="rts_step06_pdf_") as temp_name:
            plot_directory = Path(temp_name)
            plots = generate_rts_evaluation_plots(
                evaluation.original_path,
                evaluation.corrected_path,
                plot_directory,
                change_tolerance=evaluation.change_tolerance,
                bins=bins,
                dpi=plot_dpi,
                overwrite=False,
            )

            document = SimpleDocTemplate(
                str(temporary_pdf),
                pagesize=A4,
                rightMargin=16 * mm,
                leftMargin=16 * mm,
                topMargin=15 * mm,
                bottomMargin=15 * mm,
                title="RTS correction evaluation report",
                author="RTS-Framework",
                subject=f"Step06 {__version__}",
                creator=f"RTS-Framework Step06 {__version__}",
                invariant=1,
                pageCompression=1,
            )
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name="RTSCentered",
                parent=styles["Heading1"],
                alignment=TA_CENTER,
                spaceAfter=8 * mm,
            ))
            styles.add(ParagraphStyle(
                name="RTSPath",
                parent=styles["BodyText"],
                fontSize=7.5,
                leading=9,
            ))

            story = [
                Paragraph("RTS Correction Evaluation", styles["RTSCentered"]),
                Paragraph(
                    f"RTS-Framework Step06 version {__version__}",
                    styles["BodyText"],
                ),
                Spacer(1, 4 * mm),
            ]
            source_rows = [
                ["Original FITS", Paragraph(str(evaluation.original_path), styles["RTSPath"])],
                ["Corrected FITS", Paragraph(str(evaluation.corrected_path), styles["RTSPath"])],
                ["Image shape", f"{evaluation.image_shape[0]} x {evaluation.image_shape[1]}"],
                ["Difference", "corrected - original"],
                ["Change tolerance", f"{evaluation.change_tolerance:.12g} ADU"],
            ]
            source_table = Table(source_rows, colWidths=[42 * mm, 132 * mm])
            source_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8E8E8")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#808080")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([source_table, Spacer(1, 5 * mm)])

            story.append(Paragraph("Automatic scientific assessment", styles["Heading2"]))
            reduction_text = (
                "undefined" if assessment.noise_reduction_fraction is None
                else f"{100.0 * assessment.noise_reduction_fraction:.3f}%"
            )
            shift_text = (
                "undefined" if assessment.mean_shift_sigma is None
                else f"{assessment.mean_shift_sigma:.6g} sigma"
            )
            assessment_rows = [
                ["Grade", assessment.grade],
                ["Noise reduction", reduction_text],
                ["Absolute mean shift", shift_text],
                ["Recommendation", Paragraph(assessment.recommendation, styles["BodyText"])],
            ]
            assessment_table = Table(assessment_rows, colWidths=[42 * mm, 132 * mm])
            assessment_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8E8E8")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#808080")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([assessment_table, Spacer(1, 3 * mm)])
            story.append(Paragraph("Automatic concerns", styles["Heading3"]))
            for concern in assessment.concerns:
                story.append(Paragraph(f"- {concern}", styles["BodyText"]))
            story.extend([
                Paragraph(
                    "This automated assessment uses fixed whole-frame criteria and does not replace inspection of the diagnostic plots or science-specific validation.",
                    styles["RTSPath"],
                ),
                Spacer(1, 4 * mm),
            ])

            ratio = (
                "undefined" if evaluation.noise_std_ratio is None
                else f"{evaluation.noise_std_ratio:.12g}"
            )
            metrics = [
                ["Metric", "Value"],
                ["Finite pixels", f"{evaluation.finite_pixel_count}"],
                ["Changed pixels", f"{evaluation.changed_pixel_count}"],
                ["Changed fraction", f"{evaluation.changed_pixel_fraction:.12g}"],
                ["Difference mean [ADU]", f"{evaluation.difference_mean:.12g}"],
                ["Difference median [ADU]", f"{evaluation.difference_median:.12g}"],
                ["Absolute difference mean [ADU]", f"{evaluation.absolute_difference_mean:.12g}"],
                ["Absolute difference maximum [ADU]", f"{evaluation.absolute_difference_maximum:.12g}"],
                ["Difference RMS [ADU]", f"{evaluation.difference_rms:.12g}"],
                ["Original mean [ADU]", f"{evaluation.original_mean:.12g}"],
                ["Corrected mean [ADU]", f"{evaluation.corrected_mean:.12g}"],
                ["Original standard deviation [ADU]", f"{evaluation.original_std:.12g}"],
                ["Corrected standard deviation [ADU]", f"{evaluation.corrected_std:.12g}"],
                ["Noise standard-deviation change [ADU]", f"{evaluation.noise_std_change:.12g}"],
                ["Noise standard-deviation ratio", ratio],
            ]
            metric_table = Table(metrics, colWidths=[105 * mm, 69 * mm], repeatRows=1)
            metric_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#808080")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]))
            story.append(metric_table)
            if science_metrics is not None:
                story.extend([Spacer(1, 5 * mm), Paragraph("Mask-based science metrics", styles["Heading2"])])
                coverage = science_metrics.correction_coverage_fraction
                residual = science_metrics.residual_candidate_fraction
                selectivity = science_metrics.correction_selectivity_fraction
                science_rows = [
                    ["Metric", "Value"],
                    ["Reference RTS pixels", f"{science_metrics.reference_pixel_count}"],
                    ["RTS pixel fraction", f"{science_metrics.rts_pixel_fraction:.12g}"],
                    ["Changed reference pixels", f"{science_metrics.changed_reference_pixel_count}"],
                    ["Correction coverage", "undefined" if coverage is None else f"{coverage:.12g}"],
                    ["Residual candidates", f"{science_metrics.residual_candidate_count}"],
                    ["Residual candidate fraction", "undefined" if residual is None else f"{residual:.12g}"],
                    ["Off-mask changed pixels", f"{science_metrics.off_mask_changed_pixel_count}"],
                    ["Off-mask changed fraction", f"{science_metrics.off_mask_changed_fraction:.12g}"],
                    ["Correction selectivity", "undefined" if selectivity is None else f"{selectivity:.12g}"],
                    ["Reference clusters (8-connected)", f"{science_metrics.reference_cluster_count}"],
                    ["Largest reference cluster", f"{science_metrics.largest_reference_cluster_size}"],
                ]
                science_table = Table(science_rows, colWidths=[105 * mm, 69 * mm], repeatRows=1)
                science_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#808080")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ]))
                story.extend([science_table, Paragraph(
                    "Residual candidates are reference-mask pixels not changed above tolerance; this is not proof of residual RTS behavior.",
                    styles["RTSPath"],
                )])
            story.append(PageBreak())

            def add_plot(title: str, image_path: Path, caption: str) -> None:
                image = RLImage(str(image_path))
                image._restrictSize(176 * mm, 102 * mm)
                story.append(KeepTogether([
                    Paragraph(title, styles["Heading2"]),
                    image,
                    Spacer(1, 2 * mm),
                    Paragraph(caption, styles["BodyText"]),
                    Spacer(1, 5 * mm),
                ]))

            add_plot(
                "Difference image", plots.difference_image,
                "Pixel-by-pixel correction, defined as corrected minus original. "
                "Display limits use the robust 99.5th percentile of absolute differences.",
            )
            add_plot(
                "Changed-pixel map", plots.correction_map,
                "Pixels are marked when the absolute correction exceeds the configured change tolerance.",
            )
            story.append(PageBreak())
            add_plot(
                "Original image histogram", plots.original_histogram,
                "Distribution of mutually finite pixel values before RTS correction.",
            )
            add_plot(
                "Corrected image histogram", plots.corrected_histogram,
                "Distribution of mutually finite pixel values after RTS correction, using the same range and bin edges.",
            )
            story.append(PageBreak())
            add_plot(
                "Difference histogram", plots.difference_histogram,
                "Distribution of corrected-minus-original pixel differences.",
            )

            document.build(story)
        temporary_pdf.replace(path)
    except Step06Error:
        temporary_pdf.unlink(missing_ok=True)
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        temporary_pdf.unlink(missing_ok=True)
        raise Step06Error(f"Unable to write output PDF {path}: {exc}") from exc

    return path.resolve()


def write_rts_science_metrics_json(
    metrics: RTSScienceMetrics,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write mask-based science metrics atomically as deterministic JSON."""
    path = Path(output_path).expanduser()
    if path.exists() and not overwrite:
        raise Step06Error(f"Science metrics JSON already exists: {path}")
    if not path.parent.is_dir():
        raise Step06Error(f"Output directory does not exist: {path.parent}")
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(metrics.summary(), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary_path.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise Step06Error(f"Unable to write science metrics JSON {path}: {exc}") from exc
    return path.resolve()

def write_rts_evaluation_json(
    evaluation: RTSCorrectionEvaluation,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one evaluation summary atomically as deterministic JSON."""
    path = Path(output_path).expanduser()
    if path.exists() and not overwrite:
        raise Step06Error(f"Output JSON already exists: {path}")
    if not path.parent.is_dir():
        raise Step06Error(f"Output directory does not exist: {path.parent}")

    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                evaluation.summary(),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
        temporary_path.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise Step06Error(f"Unable to write output JSON {path}: {exc}") from exc
    return path.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one original/corrected RTS FITS pair."
    )
    parser.add_argument("--original", required=True, help="Original FITS image")
    parser.add_argument("--corrected", required=True, help="Corrected FITS image")
    parser.add_argument(
        "--change-tolerance",
        type=float,
        default=0.0,
        help="Absolute difference threshold for counting changed pixels",
    )
    parser.add_argument("--output-json", help="Write deterministic JSON report")
    parser.add_argument("--rts-mask", help="Optional FITS mask of reference RTS pixels")
    parser.add_argument("--science-json", help="Write mask-based science metrics JSON (requires --rts-mask)")
    parser.add_argument("--output-pdf", help="Write deterministic multi-page PDF report")
    parser.add_argument(
        "--plot-directory",
        help="Write five diagnostic PNG files to an existing directory",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=128,
        help="Number of bins used for histogram PNG files (default: 128)",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=120,
        help="PNG resolution in dots per inch (default: 120)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing JSON, PNG, or PDF outputs to be replaced",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete evaluation as JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout; errors are still written to stderr",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _format_text(
    evaluation: RTSCorrectionEvaluation,
    science_metrics: RTSScienceMetrics | None = None,
) -> str:
    assessment = assess_rts_correction(evaluation)
    ratio = (
        "undefined"
        if evaluation.noise_std_ratio is None
        else f"{evaluation.noise_std_ratio:.12g}"
    )
    lines = [
            f"Step06 version          : {__version__}",
            f"Assessment grade        : {assessment.grade}",
            f"Recommendation          : {assessment.recommendation}",
            f"Original                : {evaluation.original_path}",
            f"Corrected               : {evaluation.corrected_path}",
            f"Image shape             : {evaluation.image_shape}",
            f"Finite pixels           : {evaluation.finite_pixel_count}",
            f"Changed pixels          : {evaluation.changed_pixel_count}",
            f"Changed fraction        : {evaluation.changed_pixel_fraction:.12g}",
            f"Difference mean         : {evaluation.difference_mean:.12g}",
            f"Absolute difference mean: {evaluation.absolute_difference_mean:.12g}",
            f"Difference RMS          : {evaluation.difference_rms:.12g}",
            f"Original std            : {evaluation.original_std:.12g}",
            f"Corrected std           : {evaluation.corrected_std:.12g}",
            f"Noise std ratio         : {ratio}",
        ]
    if science_metrics is not None:
        coverage = science_metrics.correction_coverage_fraction
        residual = science_metrics.residual_candidate_fraction
        selectivity = science_metrics.correction_selectivity_fraction
        lines.extend([
            f"RTS mask                : {science_metrics.mask_path}",
            f"RTS pixel fraction      : {science_metrics.rts_pixel_fraction:.12g}",
            f"Correction coverage     : {'undefined' if coverage is None else f'{coverage:.12g}'}",
            f"Residual candidate frac : {'undefined' if residual is None else f'{residual:.12g}'}",
            f"Off-mask changed frac   : {science_metrics.off_mask_changed_fraction:.12g}",
            f"Correction selectivity  : {'undefined' if selectivity is None else f'{selectivity:.12g}'}",
            f"Reference clusters      : {science_metrics.reference_cluster_count}",
            f"Largest RTS cluster     : {science_metrics.largest_reference_cluster_size}",
        ])
    return "\n".join(lines)


def run_rts_evaluation_cli(argv: Sequence[str] | None = None) -> int:
    """Run the Step06 command-line interface and return its exit status."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        evaluation = evaluate_rts_correction(
            args.original,
            args.corrected,
            change_tolerance=args.change_tolerance,
        )
        if args.science_json and not args.rts_mask:
            raise Step06Error("--science-json requires --rts-mask")
        science_metrics = (
            None if not args.rts_mask else calculate_rts_science_metrics(
                args.original, args.corrected, args.rts_mask,
                change_tolerance=args.change_tolerance,
            )
        )
        if args.output_json:
            write_rts_evaluation_json(
                evaluation,
                args.output_json,
                overwrite=args.overwrite,
            )
        if args.science_json and science_metrics is not None:
            write_rts_science_metrics_json(
                science_metrics, args.science_json, overwrite=args.overwrite
            )
        if args.output_pdf:
            generate_rts_evaluation_pdf(
                args.original,
                args.corrected,
                args.output_pdf,
                change_tolerance=args.change_tolerance,
                bins=args.histogram_bins,
                plot_dpi=args.plot_dpi,
                overwrite=args.overwrite,
                rts_mask_path=args.rts_mask,
            )
        if args.plot_directory:
            generate_rts_evaluation_plots(
                args.original,
                args.corrected,
                args.plot_directory,
                change_tolerance=args.change_tolerance,
                bins=args.histogram_bins,
                dpi=args.plot_dpi,
                overwrite=args.overwrite,
            )
        if not args.quiet:
            if args.json:
                print(
                    json.dumps(
                        evaluation.summary(),
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
            else:
                print(_format_text(evaluation, science_metrics))
        return 0
    except Step06Error as exc:
        print(f"Step06 error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_rts_evaluation_cli())
