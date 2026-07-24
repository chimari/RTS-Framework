"""Step 04: prepare an RTS dictionary analysis plan.

Version 4.12.0 adds deterministic flat-row serialization for one final
RTS candidate without performing file I/O.
"""

from __future__ import annotations

__version__ = "4.12.0"

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
    "RTSCandidateResult",
    "RTSDictionaryPlan",
    "RTSPixelAnalysisResult",
    "Step04Error",
    "TemporalRTSCandidateResult",
    "TwoStateScoreResult",
    "TwoStateTransitionResult",
    "analyze_rts_pixel",
    "analyze_two_state_transitions",
    "classify_rts_candidate",
    "classify_temporal_rts_candidate",
    "compute_pixel_timeseries_statistics",
    "compute_two_state_score",
    "iter_image_coordinates",
    "iter_image_rts_analyses",
    "iter_rts_candidates",
    "iter_rts_pixel_analyses",
    "load_pixel_timeseries",
    "prepare_rts_dictionary_analysis",
    "rts_candidate_to_row",
]


class Step04Error(Exception):
    """Raised when Step 04 cannot prepare an RTS dictionary analysis."""









@dataclass(slots=True, frozen=True)
class RTSPixelAnalysisResult:
    """Immutable aggregate of every Step 04 result for one pixel."""

    series: "PixelTimeSeries"
    statistics: "PixelTimeSeriesStatistics"
    score: "TwoStateScoreResult"
    candidate: "RTSCandidateResult"
    transitions: "TwoStateTransitionResult"
    temporal_candidate: "TemporalRTSCandidateResult"

    @property
    def is_candidate(self) -> bool:
        """Return the final temporal RTS-candidate decision."""
        return self.temporal_candidate.is_candidate

    def summary(self) -> dict[str, object]:
        """Return a canonical JSON-serializable one-pixel analysis summary."""
        return {
            "dataset": self.series.dataset,
            "row": self.series.row,
            "column": self.series.column,
            "n_frames": self.series.n_frames,
            "statistics": self.statistics.summary(),
            "score": self.score.summary(),
            "candidate": self.candidate.summary(),
            "transitions": self.transitions.summary(),
            "temporal_candidate": self.temporal_candidate.summary(),
            "is_candidate": self.is_candidate,
        }


@dataclass(slots=True, frozen=True)
class TemporalRTSCandidateResult:
    """Immutable temporal extension of one RTS candidate classification."""

    candidate_result: "RTSCandidateResult"
    transition_result: "TwoStateTransitionResult"
    minimum_transition_count: int
    minimum_lower_run: int
    minimum_upper_run: int
    passes_base_candidate: bool
    passes_transition_count: bool
    passes_lower_run: bool
    passes_upper_run: bool
    is_candidate: bool

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        """Return failed condition names in canonical order."""
        failed: list[str] = []
        if not self.passes_base_candidate:
            failed.append("base_candidate")
        if not self.passes_transition_count:
            failed.append("transition_count")
        if not self.passes_lower_run:
            failed.append("lower_run")
        if not self.passes_upper_run:
            failed.append("upper_run")
        return tuple(failed)

    def summary(self) -> dict[str, object]:
        """Return a canonical JSON-serializable temporal classification summary."""
        transition = self.transition_result
        return {
            "dataset": transition.series.dataset,
            "row": transition.series.row,
            "column": transition.series.column,
            "n_frames": transition.series.n_frames,
            "base_candidate": self.candidate_result.is_candidate,
            "transition_count": transition.transition_count,
            "longest_lower_run": transition.longest_lower_run,
            "longest_upper_run": transition.longest_upper_run,
            "minimum_transition_count": self.minimum_transition_count,
            "minimum_lower_run": self.minimum_lower_run,
            "minimum_upper_run": self.minimum_upper_run,
            "passes_base_candidate": self.passes_base_candidate,
            "passes_transition_count": self.passes_transition_count,
            "passes_lower_run": self.passes_lower_run,
            "passes_upper_run": self.passes_upper_run,
            "is_candidate": self.is_candidate,
            "failed_conditions": list(self.failed_conditions),
        }


@dataclass(slots=True, frozen=True)
class TwoStateTransitionResult:
    """Immutable temporal transition analysis for one fitted two-state series."""

    series: "PixelTimeSeries"
    score_result: "TwoStateScoreResult"
    state_sequence: tuple[str, ...]
    lower_state_count: int
    upper_state_count: int
    transition_count: int
    lower_to_upper_count: int
    upper_to_lower_count: int
    longest_lower_run: int
    longest_upper_run: int

    def summary(self) -> dict[str, object]:
        """Return a canonical JSON-serializable transition summary."""
        return {
            "dataset": self.series.dataset,
            "row": self.series.row,
            "column": self.series.column,
            "n_frames": self.series.n_frames,
            "lower_state_count": self.lower_state_count,
            "upper_state_count": self.upper_state_count,
            "transition_count": self.transition_count,
            "lower_to_upper_count": self.lower_to_upper_count,
            "upper_to_lower_count": self.upper_to_lower_count,
            "longest_lower_run": self.longest_lower_run,
            "longest_upper_run": self.longest_upper_run,
            "state_sequence": list(self.state_sequence),
        }


@dataclass(slots=True, frozen=True)
class RTSCandidateResult:
    """Immutable threshold-based classification of one two-state score."""

    score_result: "TwoStateScoreResult"
    minimum_score: float
    minimum_state_count: int
    minimum_separation: float
    passes_score: bool
    passes_state_count: bool
    passes_separation: bool
    is_candidate: bool

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        """Return failed condition names in canonical order."""
        failed: list[str] = []
        if not self.passes_score:
            failed.append("score")
        if not self.passes_state_count:
            failed.append("state_count")
        if not self.passes_separation:
            failed.append("separation")
        return tuple(failed)

    def summary(self) -> dict[str, object]:
        """Return a canonical JSON-serializable classification summary."""
        score = self.score_result
        return {
            "dataset": score.series.dataset,
            "row": score.series.row,
            "column": score.series.column,
            "n_frames": score.n_frames,
            "score": score.score,
            "lower_state_count": score.lower_state_count,
            "upper_state_count": score.upper_state_count,
            "state_separation": score.state_separation,
            "minimum_score": self.minimum_score,
            "minimum_state_count": self.minimum_state_count,
            "minimum_separation": self.minimum_separation,
            "passes_score": self.passes_score,
            "passes_state_count": self.passes_state_count,
            "passes_separation": self.passes_separation,
            "is_candidate": self.is_candidate,
            "failed_conditions": list(self.failed_conditions),
        }


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











def iter_image_coordinates(
    plan: RTSDictionaryPlan,
    *,
    row_start: int = 0,
    row_stop: int | None = None,
    column_start: int = 0,
    column_stop: int | None = None,
):
    """Yield validated image coordinates in deterministic row-major order.

    The stop bounds are exclusive, matching Python ``range`` semantics. ``None``
    means the full image height or width. Empty regions are valid when a start
    equals its corresponding stop.

    Parameters
    ----------
    plan
        Plan returned by :func:`prepare_rts_dictionary_analysis`.
    row_start, row_stop
        Inclusive row start and exclusive row stop.
    column_start, column_stop
        Inclusive column start and exclusive column stop.

    Yields
    ------
    tuple[int, int]
        ``(row, column)`` pairs in row-major order.

    Raises
    ------
    Step04Error
        If the plan is invalid, a bound is not an integer, a bound lies outside
        the image, or a start is greater than its corresponding stop.
    """
    if not isinstance(plan, RTSDictionaryPlan):
        raise Step04Error(
            "plan must be an RTSDictionaryPlan returned by "
            "prepare_rts_dictionary_analysis()."
        )

    height, width = plan.image_shape

    def require_bound(name: str, value: object, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise Step04Error(f"{name} must be an integer.")
        converted = int(value)
        if converted < 0 or converted > maximum:
            raise Step04Error(
                f"{name} must satisfy 0 <= {name} <= {maximum}."
            )
        return converted

    row_start = require_bound("row_start", row_start, height)
    column_start = require_bound("column_start", column_start, width)

    if row_stop is None:
        row_stop = height
    else:
        row_stop = require_bound("row_stop", row_stop, height)

    if column_stop is None:
        column_stop = width
    else:
        column_stop = require_bound("column_stop", column_stop, width)

    if row_start > row_stop:
        raise Step04Error("row_start must be less than or equal to row_stop.")
    if column_start > column_stop:
        raise Step04Error(
            "column_start must be less than or equal to column_stop."
        )

    for row in range(row_start, row_stop):
        for column in range(column_start, column_stop):
            yield (row, column)




def rts_candidate_to_row(result: RTSPixelAnalysisResult) -> dict[str, object]:
    """Return one final RTS candidate as a deterministic flat CSV-ready row.

    The returned dictionary uses insertion order as the canonical column order.
    Values are limited to strings, integers, finite floats, and booleans. No
    file is opened and the source result is not modified.

    Parameters
    ----------
    result
        Final candidate result produced by :func:`analyze_rts_pixel`.

    Returns
    -------
    dict[str, object]
        A new flat dictionary in canonical RTS-dictionary column order.

    Raises
    ------
    Step04Error
        If ``result`` is not an :class:`RTSPixelAnalysisResult` or is not a
        final RTS candidate.
    """
    if not isinstance(result, RTSPixelAnalysisResult):
        raise Step04Error("result must be an RTSPixelAnalysisResult.")
    if not result.is_candidate:
        raise Step04Error("result must be a final RTS candidate.")

    series = result.series
    statistics = result.statistics
    score = result.score
    candidate = result.candidate
    transitions = result.transitions
    temporal = result.temporal_candidate

    return {
        "dataset": series.dataset,
        "row": series.row,
        "column": series.column,
        "n_frames": series.n_frames,
        "minimum": statistics.minimum,
        "maximum": statistics.maximum,
        "mean": statistics.mean,
        "median": statistics.median,
        "standard_deviation": statistics.standard_deviation,
        "median_absolute_deviation": statistics.median_absolute_deviation,
        "peak_to_peak": statistics.peak_to_peak,
        "lower_state_count": score.lower_state_count,
        "upper_state_count": score.upper_state_count,
        "lower_state_center": score.lower_state_center,
        "upper_state_center": score.upper_state_center,
        "state_separation": score.state_separation,
        "single_state_residual": score.single_state_residual,
        "two_state_residual": score.two_state_residual,
        "two_state_score": score.score,
        "minimum_score": candidate.minimum_score,
        "minimum_state_count": candidate.minimum_state_count,
        "minimum_separation": candidate.minimum_separation,
        "transition_count": transitions.transition_count,
        "lower_to_upper_count": transitions.lower_to_upper_count,
        "upper_to_lower_count": transitions.upper_to_lower_count,
        "longest_lower_run": transitions.longest_lower_run,
        "longest_upper_run": transitions.longest_upper_run,
        "minimum_transition_count": temporal.minimum_transition_count,
        "minimum_lower_run": temporal.minimum_lower_run,
        "minimum_upper_run": temporal.minimum_upper_run,
        "is_candidate": result.is_candidate,
    }

def iter_rts_candidates(results):
    """Yield only final RTS candidates from an analysis-result iterable.

    Input order is preserved, duplicate object references are preserved, and
    accepted :class:`RTSPixelAnalysisResult` instances are yielded unchanged.
    The iterable is consumed lazily and may therefore be a one-shot generator.

    Parameters
    ----------
    results
        Iterable of :class:`RTSPixelAnalysisResult` objects.

    Yields
    ------
    RTSPixelAnalysisResult
        Original result objects whose ``is_candidate`` property is true.

    Raises
    ------
    Step04Error
        Lazily, when an input item is not an ``RTSPixelAnalysisResult``.
    """
    try:
        iterator = iter(results)
    except TypeError as exc:
        raise Step04Error(
            "results must be an iterable of RTSPixelAnalysisResult objects."
        ) from exc

    for index, result in enumerate(iterator):
        if not isinstance(result, RTSPixelAnalysisResult):
            raise Step04Error(
                "results item "
                f"{index} must be an RTSPixelAnalysisResult."
            )
        if result.is_candidate:
            yield result

def iter_image_rts_analyses(
    plan: RTSDictionaryPlan,
    *,
    row_start: int = 0,
    row_stop: int | None = None,
    column_start: int = 0,
    column_stop: int | None = None,
    minimum_score: float,
    minimum_state_count: int,
    minimum_separation: float,
    minimum_transition_count: int,
    minimum_lower_run: int,
    minimum_upper_run: int,
):
    """Yield RTS analyses for a full image or rectangular ROI in row-major order.

    This function is a thin lazy composition of
    :func:`iter_image_coordinates` and :func:`iter_rts_pixel_analyses`.
    It adds no new analysis, candidate, transition, caching, parallelization,
    progress-reporting, or serialization behavior.

    Parameters
    ----------
    plan
        Plan returned by :func:`prepare_rts_dictionary_analysis`.
    row_start, row_stop, column_start, column_stop
        ROI bounds forwarded unchanged to :func:`iter_image_coordinates`.
        Stop bounds are exclusive and ``None`` means the image edge.
    minimum_score, minimum_state_count, minimum_separation
        Base-candidate thresholds forwarded unchanged.
    minimum_transition_count, minimum_lower_run, minimum_upper_run
        Temporal-candidate thresholds forwarded unchanged.

    Yields
    ------
    RTSPixelAnalysisResult
        One result per coordinate in deterministic row-major order.

    Raises
    ------
    Step04Error
        Propagated unchanged and lazily from coordinate generation or one-pixel
        analysis.
    """
    coordinates = iter_image_coordinates(
        plan,
        row_start=row_start,
        row_stop=row_stop,
        column_start=column_start,
        column_stop=column_stop,
    )
    yield from iter_rts_pixel_analyses(
        plan,
        coordinates,
        minimum_score=minimum_score,
        minimum_state_count=minimum_state_count,
        minimum_separation=minimum_separation,
        minimum_transition_count=minimum_transition_count,
        minimum_lower_run=minimum_lower_run,
        minimum_upper_run=minimum_upper_run,
    )

def iter_rts_pixel_analyses(
    plan: RTSDictionaryPlan,
    coordinates: object,
    *,
    minimum_score: float,
    minimum_state_count: int,
    minimum_separation: float,
    minimum_transition_count: int,
    minimum_lower_run: int,
    minimum_upper_run: int,
):
    """Yield one-pixel RTS analyses in the exact supplied coordinate order.

    This function is intentionally lazy and delegates every pixel to
    :func:`analyze_rts_pixel`. Duplicate coordinates are preserved and analyzed
    again. No sorting, de-duplication, full-frame expansion, caching,
    multiprocessing, progress reporting, or output serialization is performed.

    Parameters
    ----------
    plan
        Plan returned by :func:`prepare_rts_dictionary_analysis`.
    coordinates
        An iterable of ``(row, column)`` coordinate pairs. Each pair must be a
        two-item tuple or list whose values are integer coordinates. Boolean
        values are rejected even though ``bool`` is a subclass of ``int``.
    minimum_score, minimum_state_count, minimum_separation
        Thresholds forwarded unchanged to :func:`analyze_rts_pixel`.
    minimum_transition_count, minimum_lower_run, minimum_upper_run
        Temporal thresholds forwarded unchanged to :func:`analyze_rts_pixel`.

    Yields
    ------
    RTSPixelAnalysisResult
        Results in exactly the same order as the supplied coordinates.

    Raises
    ------
    Step04Error
        When ``coordinates`` is not iterable, a coordinate item is malformed,
        or an underlying one-pixel analysis fails. Validation occurs lazily as
        each coordinate is requested.
    """
    try:
        iterator = iter(coordinates)
    except TypeError as exc:
        raise Step04Error("coordinates must be an iterable of (row, column) pairs.") from exc

    for coordinate_index, coordinate in enumerate(iterator):
        if not isinstance(coordinate, (tuple, list)) or len(coordinate) != 2:
            raise Step04Error(
                f"coordinates[{coordinate_index}] must be a two-item "
                "(row, column) tuple or list."
            )

        row, column = coordinate
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)):
            raise Step04Error(
                f"coordinates[{coordinate_index}][0] row must be an integer."
            )
        if isinstance(column, bool) or not isinstance(column, (int, np.integer)):
            raise Step04Error(
                f"coordinates[{coordinate_index}][1] column must be an integer."
            )

        yield analyze_rts_pixel(
            plan,
            row=int(row),
            column=int(column),
            minimum_score=minimum_score,
            minimum_state_count=minimum_state_count,
            minimum_separation=minimum_separation,
            minimum_transition_count=minimum_transition_count,
            minimum_lower_run=minimum_lower_run,
            minimum_upper_run=minimum_upper_run,
        )

def analyze_rts_pixel(
    plan: RTSDictionaryPlan,
    *,
    row: int,
    column: int,
    minimum_score: float,
    minimum_state_count: int,
    minimum_separation: float,
    minimum_transition_count: int,
    minimum_lower_run: int,
    minimum_upper_run: int,
) -> RTSPixelAnalysisResult:
    """Run the existing Step 04 analysis pipeline for exactly one pixel.

    This function is a thin orchestration layer. It adds no new scoring or
    classification rules. The following public APIs are called in order:

    1. :func:`load_pixel_timeseries`
    2. :func:`compute_pixel_timeseries_statistics`
    3. :func:`compute_two_state_score`
    4. :func:`classify_rts_candidate`
    5. :func:`analyze_two_state_transitions`
    6. :func:`classify_temporal_rts_candidate`

    Parameters
    ----------
    plan
        Plan returned by :func:`prepare_rts_dictionary_analysis`.
    row, column
        Zero-based pixel coordinates.
    minimum_score, minimum_state_count, minimum_separation
        Thresholds forwarded unchanged to :func:`classify_rts_candidate`.
    minimum_transition_count, minimum_lower_run, minimum_upper_run
        Thresholds forwarded unchanged to
        :func:`classify_temporal_rts_candidate`.

    Returns
    -------
    RTSPixelAnalysisResult
        Immutable aggregate retaining every intermediate result.

    Raises
    ------
    Step04Error
        Propagated unchanged from the underlying public APIs.
    """
    series = load_pixel_timeseries(plan, row=row, column=column)
    statistics = compute_pixel_timeseries_statistics(series)
    score = compute_two_state_score(series)
    candidate = classify_rts_candidate(
        score,
        minimum_score=minimum_score,
        minimum_state_count=minimum_state_count,
        minimum_separation=minimum_separation,
    )
    transitions = analyze_two_state_transitions(series, score)
    temporal_candidate = classify_temporal_rts_candidate(
        candidate,
        transitions,
        minimum_transition_count=minimum_transition_count,
        minimum_lower_run=minimum_lower_run,
        minimum_upper_run=minimum_upper_run,
    )

    return RTSPixelAnalysisResult(
        series=series,
        statistics=statistics,
        score=score,
        candidate=candidate,
        transitions=transitions,
        temporal_candidate=temporal_candidate,
    )

def classify_temporal_rts_candidate(
    candidate_result: RTSCandidateResult,
    transition_result: TwoStateTransitionResult,
    *,
    minimum_transition_count: int,
    minimum_lower_run: int,
    minimum_upper_run: int,
) -> TemporalRTSCandidateResult:
    """Extend one RTS candidate decision with explicit temporal conditions.

    A result is a temporal RTS candidate only when all of the following are true:

    1. ``candidate_result.is_candidate`` is true
    2. ``transition_result.transition_count >= minimum_transition_count``
    3. ``transition_result.longest_lower_run >= minimum_lower_run``
    4. ``transition_result.longest_upper_run >= minimum_upper_run``

    Threshold comparisons are inclusive. This function does not reinterpret the
    two-state fit, alter the base candidate thresholds, or apply elapsed-time,
    cadence, dwell-duration, hysteresis, read-noise, or neighboring-pixel tests.

    Parameters
    ----------
    candidate_result
        Result returned by :func:`classify_rts_candidate`.
    transition_result
        Result returned by :func:`analyze_two_state_transitions` for the same
        exact PixelTimeSeries and TwoStateScoreResult objects.
    minimum_transition_count
        Inclusive minimum number of state changes.
    minimum_lower_run
        Inclusive minimum longest consecutive run in the lower state.
    minimum_upper_run
        Inclusive minimum longest consecutive run in the upper state.

    Returns
    -------
    TemporalRTSCandidateResult
        Immutable combined classification with each temporal condition retained
        separately.

    Raises
    ------
    Step04Error
        If inputs are invalid, inconsistent, or thresholds are not integers
        greater than or equal to zero for transition count and one for runs.
    """
    if not isinstance(candidate_result, RTSCandidateResult):
        raise Step04Error(
            "candidate_result must be an RTSCandidateResult returned by "
            "classify_rts_candidate()."
        )
    if not isinstance(transition_result, TwoStateTransitionResult):
        raise Step04Error(
            "transition_result must be a TwoStateTransitionResult returned by "
            "analyze_two_state_transitions()."
        )

    if candidate_result.score_result is not transition_result.score_result:
        raise Step04Error(
            "candidate_result and transition_result must refer to the same "
            "TwoStateScoreResult object."
        )
    if candidate_result.score_result.series is not transition_result.series:
        raise Step04Error(
            "candidate_result and transition_result must refer to the same "
            "PixelTimeSeries object."
        )

    def require_integer(name: str, value: object, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            comparator = ">= 0" if minimum == 0 else ">= 1"
            raise Step04Error(f"{name} must be an integer {comparator}.")
        converted = int(value)
        if converted < minimum:
            comparator = ">= 0" if minimum == 0 else ">= 1"
            raise Step04Error(f"{name} must be an integer {comparator}.")
        return converted

    minimum_transition_count = require_integer(
        "minimum_transition_count",
        minimum_transition_count,
        0,
    )
    minimum_lower_run = require_integer(
        "minimum_lower_run",
        minimum_lower_run,
        1,
    )
    minimum_upper_run = require_integer(
        "minimum_upper_run",
        minimum_upper_run,
        1,
    )

    passes_base_candidate = candidate_result.is_candidate
    passes_transition_count = (
        transition_result.transition_count >= minimum_transition_count
    )
    passes_lower_run = (
        transition_result.longest_lower_run >= minimum_lower_run
    )
    passes_upper_run = (
        transition_result.longest_upper_run >= minimum_upper_run
    )
    is_candidate = (
        passes_base_candidate
        and passes_transition_count
        and passes_lower_run
        and passes_upper_run
    )

    return TemporalRTSCandidateResult(
        candidate_result=candidate_result,
        transition_result=transition_result,
        minimum_transition_count=minimum_transition_count,
        minimum_lower_run=minimum_lower_run,
        minimum_upper_run=minimum_upper_run,
        passes_base_candidate=passes_base_candidate,
        passes_transition_count=passes_transition_count,
        passes_lower_run=passes_lower_run,
        passes_upper_run=passes_upper_run,
        is_candidate=is_candidate,
    )

def analyze_two_state_transitions(
    series: PixelTimeSeries,
    score_result: TwoStateScoreResult,
) -> TwoStateTransitionResult:
    """Assign each frame to the nearest fitted state center and count transitions.

    Each value is assigned to either the lower or upper fitted state center.
    Exact midpoint ties are assigned to the lower state, providing a fixed and
    deterministic rule.

    The original frame order is preserved. The function reports total
    transitions, directional transitions, state occupancies, and the longest
    consecutive run in each state.

    This function does not apply candidate thresholds, minimum dwell-time
    requirements, transition-rate criteria, or read-noise significance tests.

    Parameters
    ----------
    series
        Pixel time series returned by :func:`load_pixel_timeseries`.
    score_result
        Two-state fit returned by :func:`compute_two_state_score` for the same
        exact PixelTimeSeries object.

    Returns
    -------
    TwoStateTransitionResult
        Immutable temporal state analysis.

    Raises
    ------
    Step04Error
        If either input is invalid, if they do not refer to the same source
        series, or if the series data are inconsistent or non-finite.
    """
    if not isinstance(series, PixelTimeSeries):
        raise Step04Error(
            "series must be a PixelTimeSeries returned by "
            "load_pixel_timeseries()."
        )
    if not isinstance(score_result, TwoStateScoreResult):
        raise Step04Error(
            "score_result must be a TwoStateScoreResult returned by "
            "compute_two_state_score()."
        )
    if score_result.series is not series:
        raise Step04Error(
            "score_result must have been computed from the same "
            "PixelTimeSeries object."
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
    if score_result.n_frames != series.n_frames:
        raise Step04Error(
            "score_result frame count does not match the pixel time series."
        )
    if not np.all(np.isfinite(values)):
        raise Step04Error("Pixel time-series values must all be finite.")

    lower_center = score_result.lower_state_center
    upper_center = score_result.upper_state_center
    if not np.isfinite(lower_center) or not np.isfinite(upper_center):
        raise Step04Error("State centers must be finite.")
    if upper_center < lower_center:
        raise Step04Error(
            "upper_state_center must be greater than or equal to "
            "lower_state_center."
        )

    midpoint = (lower_center + upper_center) / 2.0
    labels = tuple(
        "lower" if float(value) <= midpoint else "upper"
        for value in values
    )

    lower_state_count = labels.count("lower")
    upper_state_count = labels.count("upper")

    transition_count = 0
    lower_to_upper_count = 0
    upper_to_lower_count = 0
    longest_lower_run = 0
    longest_upper_run = 0
    current_label = labels[0]
    current_run = 1

    for previous, current in zip(labels, labels[1:]):
        if current == previous:
            current_run += 1
            continue

        if previous == "lower":
            longest_lower_run = max(longest_lower_run, current_run)
            lower_to_upper_count += 1
        else:
            longest_upper_run = max(longest_upper_run, current_run)
            upper_to_lower_count += 1

        transition_count += 1
        current_label = current
        current_run = 1

    if current_label == "lower":
        longest_lower_run = max(longest_lower_run, current_run)
    else:
        longest_upper_run = max(longest_upper_run, current_run)

    return TwoStateTransitionResult(
        series=series,
        score_result=score_result,
        state_sequence=labels,
        lower_state_count=lower_state_count,
        upper_state_count=upper_state_count,
        transition_count=transition_count,
        lower_to_upper_count=lower_to_upper_count,
        upper_to_lower_count=upper_to_lower_count,
        longest_lower_run=longest_lower_run,
        longest_upper_run=longest_upper_run,
    )

def classify_rts_candidate(
    score_result: TwoStateScoreResult,
    *,
    minimum_score: float,
    minimum_state_count: int,
    minimum_separation: float,
) -> RTSCandidateResult:
    """Classify one two-state fit using three explicit threshold conditions.

    A result is an RTS candidate only when all of the following are true:

    1. ``score_result.score >= minimum_score``
    2. both fitted states contain at least ``minimum_state_count`` samples
    3. ``score_result.state_separation >= minimum_separation``

    Threshold comparisons are inclusive. This function does not inspect the
    original temporal ordering, dwell times, transition counts, read noise, or
    neighboring pixels.

    Parameters
    ----------
    score_result
        Result returned by :func:`compute_two_state_score`.
    minimum_score
        Inclusive score threshold in the closed interval [0, 1].
    minimum_state_count
        Inclusive minimum occupancy required for each fitted state.
    minimum_separation
        Inclusive non-negative state-center separation threshold.

    Returns
    -------
    RTSCandidateResult
        Immutable classification result with each condition recorded
        separately.

    Raises
    ------
    Step04Error
        If the result or any threshold is invalid.
    """
    if not isinstance(score_result, TwoStateScoreResult):
        raise Step04Error(
            "score_result must be a TwoStateScoreResult returned by "
            "compute_two_state_score()."
        )

    if isinstance(minimum_score, bool) or not isinstance(
        minimum_score, (int, float, np.integer, np.floating)
    ):
        raise Step04Error("minimum_score must be a finite number in [0, 1].")
    minimum_score = float(minimum_score)
    if not np.isfinite(minimum_score) or not 0.0 <= minimum_score <= 1.0:
        raise Step04Error("minimum_score must be a finite number in [0, 1].")

    if isinstance(minimum_state_count, bool) or not isinstance(
        minimum_state_count, (int, np.integer)
    ):
        raise Step04Error("minimum_state_count must be an integer >= 1.")
    minimum_state_count = int(minimum_state_count)
    if minimum_state_count < 1:
        raise Step04Error("minimum_state_count must be an integer >= 1.")

    if isinstance(minimum_separation, bool) or not isinstance(
        minimum_separation, (int, float, np.integer, np.floating)
    ):
        raise Step04Error(
            "minimum_separation must be a finite number >= 0."
        )
    minimum_separation = float(minimum_separation)
    if not np.isfinite(minimum_separation) or minimum_separation < 0.0:
        raise Step04Error(
            "minimum_separation must be a finite number >= 0."
        )

    passes_score = score_result.score >= minimum_score
    passes_state_count = (
        score_result.lower_state_count >= minimum_state_count
        and score_result.upper_state_count >= minimum_state_count
    )
    passes_separation = (
        score_result.state_separation >= minimum_separation
    )
    is_candidate = (
        passes_score and passes_state_count and passes_separation
    )

    return RTSCandidateResult(
        score_result=score_result,
        minimum_score=minimum_score,
        minimum_state_count=minimum_state_count,
        minimum_separation=minimum_separation,
        passes_score=passes_score,
        passes_state_count=passes_state_count,
        passes_separation=passes_separation,
        is_candidate=is_candidate,
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
