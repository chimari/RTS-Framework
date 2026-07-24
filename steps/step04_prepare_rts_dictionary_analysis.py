"""Step 04: prepare an RTS dictionary analysis plan.

Version 4.2.0 adds deterministic basic statistics for one pixel time series.
The statistics are algorithm-neutral and do not perform RTS classification.
"""

from __future__ import annotations

__version__ = "4.2.0"

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
    "compute_pixel_timeseries_statistics",
    "load_pixel_timeseries",
    "prepare_rts_dictionary_analysis",
]


class Step04Error(Exception):
    """Raised when Step 04 cannot prepare an RTS dictionary analysis."""




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
