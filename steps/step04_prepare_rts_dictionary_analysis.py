"""Step 04: prepare an RTS dictionary analysis plan.

Version 4.3.0 adds a deterministic two-state score for one pixel time series.
It exactly searches every non-empty split of the sorted values and compares
the best two-center residual with the single-center residual.
"""

from __future__ import annotations

__version__ = "4.3.0"

from dataclasses import dataclass

import numpy as np

from steps.step03_prepare_bias_analysis import (
    BiasAnalysisPlan,
    Step03Error,
    iter_bias_frames,
)


__all__ = [
    "PixelTimeSeries",
    "PixelTimeSeriesStatistics",
    "RTSDictionaryPlan",
    "Step04Error",
    "TwoStateScoreResult",
    "compute_pixel_timeseries_statistics",
    "compute_two_state_score",
    "load_pixel_timeseries",
    "prepare_rts_dictionary_analysis",
]


class Step04Error(Exception):
    """Raised when Step 04 cannot prepare an RTS dictionary analysis."""





@dataclass(slots=True, frozen=True)
class TwoStateScoreResult:
    """Immutable result of deterministic one- versus two-state fitting."""

    series: "PixelTimeSeries"
    n_frames: int
    lower_state_count: int
    upper_state_count: int
    lower_state_center: float
    upper_state_center: float
    state_separation: float
    single_state_residual: float
    two_state_residual: float
    score: float

    def summary(self) -> dict[str, object]:
        """Return a canonical JSON-serializable score summary."""
        return {
            "dataset": self.series.dataset,
            "row": self.series.row,
            "column": self.series.column,
            "n_frames": self.n_frames,
            "lower_state_count": self.lower_state_count,
            "upper_state_count": self.upper_state_count,
            "lower_state_center": self.lower_state_center,
            "upper_state_center": self.upper_state_center,
            "state_separation": self.state_separation,
            "single_state_residual": self.single_state_residual,
            "two_state_residual": self.two_state_residual,
            "score": self.score,
        }


@dataclass(slots=True, frozen=True)
class PixelTimeSeriesStatistics:
    """Immutable algorithm-neutral statistics for one pixel time series."""

    series: "PixelTimeSeries"
    n_frames: int
    minimum: float
    maximum: float
    mean: float
    median: float
    standard_deviation: float
    median_absolute_deviation: float
    peak_to_peak: float

    def summary(self) -> dict[str, object]:
        """Return a canonical JSON-serializable statistics summary."""
        return {
            "dataset": self.series.dataset,
            "row": self.series.row,
            "column": self.series.column,
            "n_frames": self.n_frames,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "standard_deviation": self.standard_deviation,
            "median_absolute_deviation": self.median_absolute_deviation,
            "peak_to_peak": self.peak_to_peak,
        }


@dataclass(slots=True, frozen=True)
class PixelTimeSeries:
    """Immutable time series for one image coordinate."""

    plan: "RTSDictionaryPlan"
    dataset: str
    row: int
    column: int
    n_frames: int
    values: np.ndarray

    def summary(self) -> dict[str, object]:
        """Return a canonical JSON-serializable metadata summary."""
        return {
            "dataset": self.dataset,
            "row": self.row,
            "column": self.column,
            "n_frames": self.n_frames,
            "dtype": self.values.dtype.name,
        }


@dataclass(slots=True, frozen=True)
class RTSDictionaryPlan:
    """Immutable metadata plan for later RTS dictionary generation."""

    bias_plan: BiasAnalysisPlan
    dataset: str
    n_frames: int
    image_shape: tuple[int, int]
    minimum_frames: int

    def summary(self) -> str:
        """Return a deterministic human-readable plan summary."""
        height, width = self.image_shape
        return "\n".join(
            [
                "RTS Framework Step 04",
                "=====================",
                "Status         : READY",
                f"Dataset        : {self.dataset}",
                f"Frames         : {self.n_frames}",
                f"Minimum frames : {self.minimum_frames}",
                f"Shape          : {height}x{width}",
            ]
        )





def compute_two_state_score(
    series: PixelTimeSeries,
) -> TwoStateScoreResult:
    """Compare exact single-center and two-center fits to one time series.

    The values are sorted, then every split position from 1 through ``n-1`` is
    evaluated. Each side is represented by its arithmetic mean. The selected
    split is the one with the smallest total squared residual. If multiple
    splits have exactly equal residuals, the smallest split index is retained.

    Definitions
    -----------
    ``single_state_residual``
        Sum of squared deviations from the global arithmetic mean.
    ``two_state_residual``
        Minimum sum of squared deviations from two arithmetic means over every
        non-empty split of the sorted values.
    ``score``
        Fractional residual improvement:

        ``(single_state_residual - two_state_residual) /
        single_state_residual``.

        A constant series has zero single-state residual and receives score
        ``0.0``.

    Notes
    -----
    This score measures how strongly a two-center representation improves the
    fit. It is not, by itself, an RTS classification. It does not account for
    temporal switching order, minimum state occupancy, state dwell time, or
    read-noise significance.

    Parameters
    ----------
    series
        Pixel time series returned by :func:`load_pixel_timeseries`.

    Returns
    -------
    TwoStateScoreResult
        Immutable exact-fit result retaining the source time series.

    Raises
    ------
    Step04Error
        If the series is invalid, contains fewer than three values, is
        inconsistent with its metadata, or contains non-finite values.
    """
    if not isinstance(series, PixelTimeSeries):
        raise Step04Error(
            "series must be a PixelTimeSeries returned by "
            "load_pixel_timeseries()."
        )

    values = series.values
    if values.ndim != 1:
        raise Step04Error("Pixel time-series values must be one-dimensional.")
    if values.size < 3:
        raise Step04Error(
            "Two-state scoring requires at least 3 pixel values."
        )
    if values.size != series.n_frames:
        raise Step04Error(
            f"Pixel time series contains {values.size} value(s); "
            f"metadata requires {series.n_frames}."
        )
    if not np.all(np.isfinite(values)):
        raise Step04Error("Pixel time-series values must all be finite.")

    sorted_values = np.sort(values, kind="stable")
    n_values = sorted_values.size

    cumulative_sum = np.cumsum(sorted_values, dtype=np.float64)
    cumulative_square_sum = np.cumsum(
        sorted_values * sorted_values,
        dtype=np.float64,
    )

    total_sum = float(cumulative_sum[-1])
    total_square_sum = float(cumulative_square_sum[-1])
    global_mean = total_sum / n_values
    single_state_residual = max(
        0.0,
        total_square_sum - total_sum * global_mean,
    )

    best_split = 1
    best_residual = np.inf
    best_lower_center = float(sorted_values[0])
    best_upper_center = float(np.mean(sorted_values[1:], dtype=np.float64))

    for split in range(1, n_values):
        lower_count = split
        upper_count = n_values - split

        lower_sum = float(cumulative_sum[split - 1])
        lower_square_sum = float(cumulative_square_sum[split - 1])
        upper_sum = total_sum - lower_sum
        upper_square_sum = total_square_sum - lower_square_sum

        lower_center = lower_sum / lower_count
        upper_center = upper_sum / upper_count

        lower_residual = max(
            0.0,
            lower_square_sum - lower_sum * lower_center,
        )
        upper_residual = max(
            0.0,
            upper_square_sum - upper_sum * upper_center,
        )
        residual = lower_residual + upper_residual

        if residual < best_residual:
            best_split = split
            best_residual = residual
            best_lower_center = lower_center
            best_upper_center = upper_center

    best_residual = float(max(0.0, best_residual))
    if single_state_residual == 0.0:
        score = 0.0
    else:
        score = (
            single_state_residual - best_residual
        ) / single_state_residual
        score = float(min(1.0, max(0.0, score)))

    return TwoStateScoreResult(
        series=series,
        n_frames=series.n_frames,
        lower_state_count=best_split,
        upper_state_count=n_values - best_split,
        lower_state_center=float(best_lower_center),
        upper_state_center=float(best_upper_center),
        state_separation=float(best_upper_center - best_lower_center),
        single_state_residual=float(single_state_residual),
        two_state_residual=best_residual,
        score=score,
    )

def compute_pixel_timeseries_statistics(
    series: PixelTimeSeries,
) -> PixelTimeSeriesStatistics:
    """Compute deterministic basic statistics for one pixel time series.

    Definitions
    -----------
    ``standard_deviation``
        Population standard deviation, equivalent to ``numpy.std(ddof=0)``.
    ``median_absolute_deviation``
        Raw, unscaled median of ``abs(values - median(values))``.
    ``peak_to_peak``
        ``maximum - minimum``.

    This function does not estimate RTS states, select thresholds, remove
    outliers, or apply Gaussian-equivalent scaling to the MAD.

    Parameters
    ----------
    series
        Pixel time series returned by :func:`load_pixel_timeseries`.

    Returns
    -------
    PixelTimeSeriesStatistics
        Immutable scalar statistics retaining the source series.

    Raises
    ------
    Step04Error
        If ``series`` is invalid or its values are empty or non-finite.
    """
    if not isinstance(series, PixelTimeSeries):
        raise Step04Error(
            "series must be a PixelTimeSeries returned by "
            "load_pixel_timeseries()."
        )

    values = series.values
    if values.ndim != 1:
        raise Step04Error("Pixel time-series values must be one-dimensional.")
    if values.size == 0:
        raise Step04Error("Pixel time-series values must not be empty.")
    if values.size != series.n_frames:
        raise Step04Error(
            f"Pixel time series contains {values.size} value(s); "
            f"metadata requires {series.n_frames}."
        )
    if not np.all(np.isfinite(values)):
        raise Step04Error("Pixel time-series values must all be finite.")

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    mean = float(np.mean(values, dtype=np.float64))
    median = float(np.median(values))
    standard_deviation = float(np.std(values, ddof=0, dtype=np.float64))
    median_absolute_deviation = float(
        np.median(np.abs(values - median))
    )

    return PixelTimeSeriesStatistics(
        series=series,
        n_frames=series.n_frames,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        median=median,
        standard_deviation=standard_deviation,
        median_absolute_deviation=median_absolute_deviation,
        peak_to_peak=maximum - minimum,
    )

def load_pixel_timeseries(
    plan: RTSDictionaryPlan,
    row: int,
    column: int,
) -> PixelTimeSeries:
    """Load one pixel value from every bias frame in canonical order.

    The function reads frames lazily through :func:`iter_bias_frames`, retains
    only a one-dimensional float64 result array, and does not construct a full
    ``(frames, height, width)`` image cube.

    Parameters
    ----------
    plan
        RTS dictionary plan returned by
        :func:`prepare_rts_dictionary_analysis`.
    row, column
        Zero-based image coordinates.

    Returns
    -------
    PixelTimeSeries
        Immutable metadata plus a read-only, C-contiguous float64 vector.

    Raises
    ------
    Step04Error
        If the plan or coordinates are invalid, Step 03 cannot read a frame,
        or the iterator yields an unexpected frame count.
    """
    if not isinstance(plan, RTSDictionaryPlan):
        raise Step04Error(
            "plan must be an RTSDictionaryPlan returned by "
            "prepare_rts_dictionary_analysis()."
        )

    validated_row = _validate_coordinate("row", row, plan.image_shape[0])
    validated_column = _validate_coordinate(
        "column",
        column,
        plan.image_shape[1],
    )

    values = np.empty(plan.n_frames, dtype=np.float64)
    count = 0

    try:
        for image in iter_bias_frames(plan.bias_plan):
            if count >= plan.n_frames:
                raise Step04Error(
                    f"Dataset {plan.dataset!r} yielded more than "
                    f"{plan.n_frames} frame(s)."
                )
            if image.shape != plan.image_shape:
                raise Step04Error(
                    f"Bias frame {count} has shape {image.shape!r}; "
                    f"expected {plan.image_shape!r}."
                )
            values[count] = image[validated_row, validated_column]
            count += 1
    except Step03Error as exc:
        raise Step04Error(
            f"Could not load pixel time series for dataset "
            f"{plan.dataset!r}: {exc}"
        ) from exc

    if count == 0:
        raise Step04Error(
            f"Dataset {plan.dataset!r} yielded no frames."
        )
    if count != plan.n_frames:
        raise Step04Error(
            f"Dataset {plan.dataset!r} yielded {count} frame(s); "
            f"the RTS dictionary plan requires {plan.n_frames}."
        )

    values.setflags(write=False)

    return PixelTimeSeries(
        plan=plan,
        dataset=plan.dataset,
        row=validated_row,
        column=validated_column,
        n_frames=count,
        values=values,
    )


def _validate_coordinate(name: str, value: int, limit: int) -> int:
    """Validate one zero-based image coordinate."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise Step04Error(f"{name} must be an integer.")
    if value < 0 or value >= limit:
        raise Step04Error(
            f"{name}={value} is outside the valid range 0..{limit - 1}."
        )
    return value

def prepare_rts_dictionary_analysis(
    bias_plan: BiasAnalysisPlan,
    *,
    min_frames: int = 3,
) -> RTSDictionaryPlan:
    """Validate Step 03 metadata for later RTS dictionary generation.

    This function deliberately performs no FITS reads, image allocation,
    pixel-statistics calculation, threshold selection, or RTS classification.

    Parameters
    ----------
    bias_plan
        Bias-analysis plan returned by
        :func:`steps.step03_prepare_bias_analysis.prepare_bias_analysis`.
    min_frames
        Minimum frame count required at this planning stage. It must be an
        integer of at least three. More demanding algorithms may impose a
        larger requirement in later Step 04 milestones.

    Returns
    -------
    RTSDictionaryPlan
        Immutable metadata-only RTS dictionary analysis plan.

    Raises
    ------
    Step04Error
        If ``bias_plan`` is invalid, ``min_frames`` is invalid, or the selected
        dataset contains too few frames.
    """
    if not isinstance(bias_plan, BiasAnalysisPlan):
        raise Step04Error(
            "bias_plan must be a BiasAnalysisPlan returned by "
            "prepare_bias_analysis()."
        )

    minimum_frames = _validate_min_frames(min_frames)

    if bias_plan.n_frames < minimum_frames:
        raise Step04Error(
            f"Dataset {bias_plan.dataset!r} contains {bias_plan.n_frames} "
            f"frame(s); RTS dictionary analysis requires at least "
            f"{minimum_frames}."
        )

    return RTSDictionaryPlan(
        bias_plan=bias_plan,
        dataset=bias_plan.dataset,
        n_frames=bias_plan.n_frames,
        image_shape=bias_plan.image_shape,
        minimum_frames=minimum_frames,
    )


def _validate_min_frames(value: int) -> int:
    """Return a validated Step 04 minimum frame count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise Step04Error("min_frames must be an integer of at least 3.")
    if value < 3:
        raise Step04Error("min_frames must be an integer of at least 3.")
    return value
