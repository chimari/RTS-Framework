"""Step 04: prepare an RTS dictionary analysis plan.

Version 4.26.0 adds validated input-file inventory checks for RTS
dictionary metadata while preserving all existing public APIs.
"""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import csv
import json
import os
import tempfile

__version__ = "4.26.0"

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
    "RTSDictionaryBuildResult",
    "RTSDictionaryArtifactResult",
    "RTSDictionaryMetadata",
    "RTSDictionaryRow",
    "RTSDictionaryCSV",
    "RTSDictionaryArtifacts",
    "RTSInputFileValidation",
    "RTSProgressState",
    "RTSCancellationInfo",
    "RTSDictionaryBuildParameters",
    "Step04Cancelled",
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
    "make_interval_progress_callback",
    "make_timed_progress_callback",
    "iter_rts_pixel_analyses",
    "load_pixel_timeseries",
    "prepare_rts_dictionary_analysis",
    "rts_candidate_to_row",
    "write_rts_dictionary_csv",
    "write_rts_dictionary_metadata_json",
    "load_rts_dictionary_metadata_json",
    "load_rts_dictionary_csv",
    "load_rts_dictionary_artifacts",
    "validate_rts_dictionary_input_files",
    "build_rts_dictionary_artifacts",
    "build_rts_dictionary_csv",
    "build_rts_dictionary_csv_result",
]


class Step04Error(Exception):
    """Raised when Step 04 cannot prepare an RTS dictionary analysis."""



class Step04Cancelled(Step04Error):
    """Raised when cooperative Step 04 cancellation is requested."""

    def __init__(self, info: "RTSCancellationInfo") -> None:
        if not isinstance(info, RTSCancellationInfo):
            raise TypeError("info must be an RTSCancellationInfo.")
        self.info = info
        super().__init__(
            "RTS dictionary build cancelled after "
            f"{info.completed_pixel_count} completed pixels."
        )









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
class RTSCancellationInfo:
    """Immutable context describing one cancelled dictionary build."""

    completed_pixel_count: int
    total_pixel_count: int
    output_path: Path

    def __post_init__(self) -> None:
        if isinstance(self.completed_pixel_count, bool) or not isinstance(
            self.completed_pixel_count, int
        ):
            raise Step04Error("completed_pixel_count must be an integer.")
        if isinstance(self.total_pixel_count, bool) or not isinstance(
            self.total_pixel_count, int
        ):
            raise Step04Error("total_pixel_count must be an integer.")
        if self.completed_pixel_count < 0:
            raise Step04Error("completed_pixel_count must be non-negative.")
        if self.total_pixel_count < 0:
            raise Step04Error("total_pixel_count must be non-negative.")
        if self.completed_pixel_count > self.total_pixel_count:
            raise Step04Error(
                "completed_pixel_count must not exceed total_pixel_count."
            )
        if not isinstance(self.output_path, Path):
            raise Step04Error("output_path must be a pathlib.Path.")

    @property
    def remaining_pixel_count(self) -> int:
        return self.total_pixel_count - self.completed_pixel_count

    @property
    def fraction_complete(self) -> float:
        if self.total_pixel_count == 0:
            return 1.0
        return self.completed_pixel_count / self.total_pixel_count

    @property
    def percent_complete(self) -> float:
        return 100.0 * self.fraction_complete

    def summary(self) -> dict[str, object]:
        return {
            "completed_pixel_count": self.completed_pixel_count,
            "total_pixel_count": self.total_pixel_count,
            "remaining_pixel_count": self.remaining_pixel_count,
            "fraction_complete": self.fraction_complete,
            "percent_complete": self.percent_complete,
            "output_path": str(self.output_path),
        }


@dataclass(slots=True, frozen=True)
class RTSProgressState:
    """Immutable timing information for one progress event."""

    completed: int
    total: int
    elapsed_seconds: float
    pixels_per_second: float | None
    remaining_seconds: float | None

    @property
    def fraction_complete(self) -> float:
        """Return completion as a deterministic fraction in ``[0, 1]``."""
        if self.total == 0:
            return 1.0
        return self.completed / self.total

    @property
    def percent_complete(self) -> float:
        """Return completion percentage."""
        return 100.0 * self.fraction_complete

    @property
    def is_complete(self) -> bool:
        """Return whether this event represents completion."""
        return self.completed == self.total

    def summary(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable timing summary."""
        return {
            "completed": self.completed,
            "total": self.total,
            "fraction_complete": self.fraction_complete,
            "percent_complete": self.percent_complete,
            "elapsed_seconds": self.elapsed_seconds,
            "pixels_per_second": self.pixels_per_second,
            "remaining_seconds": self.remaining_seconds,
            "is_complete": self.is_complete,
        }



@dataclass(slots=True, frozen=True)
class RTSDictionaryBuildParameters:
    """Immutable normalized parameters for one dictionary build."""

    minimum_score: float
    minimum_state_count: int
    minimum_separation: float
    minimum_transition_count: int
    minimum_lower_run: int
    minimum_upper_run: int
    row_start: int
    row_stop: int
    column_start: int
    column_stop: int

    @property
    def row_count(self) -> int:
        return self.row_stop - self.row_start

    @property
    def column_count(self) -> int:
        return self.column_stop - self.column_start

    @property
    def pixel_count(self) -> int:
        return self.row_count * self.column_count

    def summary(self) -> dict[str, object]:
        return {
            "minimum_score": float(self.minimum_score),
            "minimum_state_count": self.minimum_state_count,
            "minimum_separation": float(self.minimum_separation),
            "minimum_transition_count": self.minimum_transition_count,
            "minimum_lower_run": self.minimum_lower_run,
            "minimum_upper_run": self.minimum_upper_run,
            "row_start": self.row_start,
            "row_stop": self.row_stop,
            "column_start": self.column_start,
            "column_stop": self.column_stop,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "pixel_count": self.pixel_count,
        }


@dataclass(slots=True, frozen=True)
class RTSDictionaryBuildResult:
    """Immutable summary of one completed RTS dictionary CSV build."""

    output_path: Path
    dataset: str
    row_start: int
    row_stop: int
    column_start: int
    column_stop: int
    analyzed_pixel_count: int
    candidate_count: int
    parameters: RTSDictionaryBuildParameters

    @property
    def region_shape(self) -> tuple[int, int]:
        """Return the analyzed region shape as ``(height, width)``."""
        return (
            self.row_stop - self.row_start,
            self.column_stop - self.column_start,
        )

    def summary(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable build summary."""
        return {
            "output_path": str(self.output_path),
            "dataset": self.dataset,
            "row_start": self.row_start,
            "row_stop": self.row_stop,
            "column_start": self.column_start,
            "column_stop": self.column_stop,
            "region_shape": self.region_shape,
            "analyzed_pixel_count": self.analyzed_pixel_count,
            "candidate_count": self.candidate_count,
            "parameters": self.parameters.summary(),
        }





@dataclass(slots=True, frozen=True)
class RTSDictionaryRow:
    """Immutable validated representation of one dictionary CSV row."""

    dataset: str
    row: int
    column: int
    n_frames: int
    minimum: float
    maximum: float
    mean: float
    median: float
    standard_deviation: float
    median_absolute_deviation: float
    peak_to_peak: float
    lower_state_count: int
    upper_state_count: int
    lower_state_center: float
    upper_state_center: float
    state_separation: float
    single_state_residual: float
    two_state_residual: float
    two_state_score: float
    minimum_score: float
    minimum_state_count: int
    minimum_separation: float
    transition_count: int
    lower_to_upper_count: int
    upper_to_lower_count: int
    longest_lower_run: int
    longest_upper_run: int
    minimum_transition_count: int
    minimum_lower_run: int
    minimum_upper_run: int
    is_candidate: bool

    @property
    def coordinate(self) -> tuple[int, int]:
        """Return ``(row, column)``."""
        return (self.row, self.column)

    def summary(self) -> dict[str, object]:
        """Return values in canonical dictionary-column order."""
        return {
            name: getattr(self, name)
            for name in RTS_DICTIONARY_COLUMNS
        }


@dataclass(slots=True, frozen=True)
class RTSDictionaryCSV:
    """Immutable validated representation of one dictionary CSV file."""

    path: Path
    rows: tuple[RTSDictionaryRow, ...]

    @property
    def candidate_count(self) -> int:
        """Return the number of validated candidate rows."""
        return len(self.rows)

    @property
    def datasets(self) -> tuple[str, ...]:
        """Return unique dataset names in first-occurrence order."""
        return tuple(dict.fromkeys(row.dataset for row in self.rows))

    def summary(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable file summary."""
        return {
            "path": str(self.path),
            "candidate_count": self.candidate_count,
            "datasets": self.datasets,
            "coordinates": tuple(row.coordinate for row in self.rows),
        }



@dataclass(slots=True, frozen=True)
class RTSDictionaryArtifacts:
    """Immutable validated RTS dictionary CSV/metadata artifact pair."""

    dictionary: RTSDictionaryCSV
    metadata: "RTSDictionaryMetadata"

    @property
    def candidate_count(self) -> int:
        """Return the mutually validated candidate count."""
        return self.dictionary.candidate_count

    @property
    def dataset(self) -> str:
        """Return the mutually validated dataset name."""
        return self.metadata.dataset

    def summary(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable artifact summary."""
        return {
            "dataset": self.dataset,
            "candidate_count": self.candidate_count,
            "dictionary": self.dictionary.summary(),
            "metadata": self.metadata.summary(),
        }



@dataclass(slots=True, frozen=True)
class RTSInputFileValidation:
    """Immutable result of validating metadata-recorded input files."""

    metadata_path: Path
    expected_file_count: int
    validated_filepaths: tuple[Path, ...]

    @property
    def validated_file_count(self) -> int:
        """Return the number of successfully validated input files."""
        return len(self.validated_filepaths)

    def summary(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable validation summary."""
        return {
            "metadata_path": str(self.metadata_path),
            "expected_file_count": self.expected_file_count,
            "validated_file_count": self.validated_file_count,
            "validated_filepaths": tuple(
                str(path) for path in self.validated_filepaths
            ),
        }


@dataclass(slots=True, frozen=True)
class RTSDictionaryMetadata:
    """Immutable validated representation of one metadata sidecar."""

    metadata_path: Path
    schema: str
    schema_version: int
    step04_version: str
    csv_path: Path
    dataset: str
    analyzed_pixel_count: int
    candidate_count: int
    n_frames: int
    image_shape: tuple[int, int]
    minimum_frames: int
    pixel_dtype: str
    exposure_s: float
    temperature_min_C: float
    temperature_max_C: float
    filepaths: tuple[Path, ...]
    parameters: RTSDictionaryBuildParameters

    def summary(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable metadata summary."""
        return {
            "metadata_path": str(self.metadata_path),
            "schema": self.schema,
            "schema_version": self.schema_version,
            "step04_version": self.step04_version,
            "csv_path": str(self.csv_path),
            "dataset": self.dataset,
            "analyzed_pixel_count": self.analyzed_pixel_count,
            "candidate_count": self.candidate_count,
            "n_frames": self.n_frames,
            "image_shape": self.image_shape,
            "minimum_frames": self.minimum_frames,
            "pixel_dtype": self.pixel_dtype,
            "exposure_s": self.exposure_s,
            "temperature_min_C": self.temperature_min_C,
            "temperature_max_C": self.temperature_max_C,
            "filepaths": tuple(str(path) for path in self.filepaths),
            "parameters": self.parameters.summary(),
        }


@dataclass(slots=True, frozen=True)
class RTSDictionaryArtifactResult:
    """Immutable paths and build metadata for a CSV/JSON artifact pair."""

    build_result: RTSDictionaryBuildResult
    metadata_path: Path

    @property
    def output_path(self) -> Path:
        """Return the RTS dictionary CSV path."""
        return self.build_result.output_path

    def summary(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable artifact summary."""
        return {
            "output_path": str(self.output_path),
            "metadata_path": str(self.metadata_path),
            "build": self.build_result.summary(),
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





RTS_DICTIONARY_COLUMNS = (
    "dataset",
    "row",
    "column",
    "n_frames",
    "minimum",
    "maximum",
    "mean",
    "median",
    "standard_deviation",
    "median_absolute_deviation",
    "peak_to_peak",
    "lower_state_count",
    "upper_state_count",
    "lower_state_center",
    "upper_state_center",
    "state_separation",
    "single_state_residual",
    "two_state_residual",
    "two_state_score",
    "minimum_score",
    "minimum_state_count",
    "minimum_separation",
    "transition_count",
    "lower_to_upper_count",
    "upper_to_lower_count",
    "longest_lower_run",
    "longest_upper_run",
    "minimum_transition_count",
    "minimum_lower_run",
    "minimum_upper_run",
    "is_candidate",
)





def make_interval_progress_callback(callback, *, every: int):
    """Return a callback that forwards deterministic interval progress events.

    The wrapped callback receives the initial event, every ``every`` completed
    pixels, and the final event. Duplicate events are suppressed. The helper
    does not alter or catch exceptions raised by the wrapped callback.
    """
    if not callable(callback):
        raise Step04Error("callback must be callable.")
    if isinstance(every, bool) or not isinstance(every, int):
        raise Step04Error("every must be an integer.")
    if every <= 0:
        raise Step04Error("every must be greater than zero.")

    last_forwarded: tuple[int, int] | None = None

    def interval_callback(completed: int, total: int) -> None:
        nonlocal last_forwarded

        event = (completed, total)
        should_forward = (
            completed == 0
            or completed == total
            or completed % every == 0
        )
        if should_forward and event != last_forwarded:
            callback(completed, total)
            last_forwarded = event

    return interval_callback




def make_timed_progress_callback(callback, *, clock=None):
    """Adapt integer progress events into immutable :class:`RTSProgressState`.

    ``clock`` must be a zero-argument callable returning monotonic seconds.
    The first received event defines the start time. Speed and remaining time
    are ``None`` until at least one pixel has completed.
    """
    if not callable(callback):
        raise Step04Error("callback must be callable.")
    if clock is None:
        clock = monotonic
    elif not callable(clock):
        raise Step04Error("clock must be callable or None.")

    start_time: float | None = None
    previous_completed: int | None = None
    previous_total: int | None = None

    def timed_callback(completed: int, total: int) -> None:
        nonlocal start_time, previous_completed, previous_total

        if isinstance(completed, bool) or not isinstance(completed, int):
            raise Step04Error("completed must be an integer.")
        if isinstance(total, bool) or not isinstance(total, int):
            raise Step04Error("total must be an integer.")
        if completed < 0:
            raise Step04Error("completed must be non-negative.")
        if total < 0:
            raise Step04Error("total must be non-negative.")
        if completed > total:
            raise Step04Error("completed must not exceed total.")
        if previous_total is not None and total != previous_total:
            raise Step04Error("total must remain unchanged.")
        if previous_completed is not None and completed < previous_completed:
            raise Step04Error("completed must not decrease.")

        now = float(clock())
        if start_time is None:
            start_time = now
        elapsed = max(0.0, now - start_time)

        if completed > 0 and elapsed > 0.0:
            rate = completed / elapsed
            remaining = (total - completed) / rate
        else:
            rate = None
            remaining = None

        state = RTSProgressState(
            completed=completed,
            total=total,
            elapsed_seconds=elapsed,
            pixels_per_second=rate,
            remaining_seconds=remaining,
        )
        callback(state)

        previous_completed = completed
        previous_total = total

    return timed_callback



def build_rts_dictionary_csv(
    plan: RTSDictionaryPlan,
    output_path,
    *,
    row_start: int | None = None,
    row_stop: int | None = None,
    column_start: int | None = None,
    column_stop: int | None = None,
    minimum_score: float = 0.0,
    minimum_state_count: int = 1,
    minimum_separation: float = 0.0,
    minimum_transition_count: int = 0,
    minimum_lower_run: int = 1,
    minimum_upper_run: int = 1,
    progress_callback=None,
    cancel_requested=None,
) -> Path:
    """Analyze an image region and atomically write its final RTS candidates.

    This compatibility API preserves the v4.14.0 return type. Use
    :func:`build_rts_dictionary_csv_result` when build counts and resolved ROI
    metadata are needed.
    """
    result = build_rts_dictionary_csv_result(
        plan,
        output_path,
        row_start=row_start,
        row_stop=row_stop,
        column_start=column_start,
        column_stop=column_stop,
        minimum_score=minimum_score,
        minimum_state_count=minimum_state_count,
        minimum_separation=minimum_separation,
        minimum_transition_count=minimum_transition_count,
        minimum_lower_run=minimum_lower_run,
        minimum_upper_run=minimum_upper_run,
        progress_callback=progress_callback,
        cancel_requested=cancel_requested,
    )
    return result.output_path


def build_rts_dictionary_csv_result(
    plan: RTSDictionaryPlan,
    output_path,
    *,
    row_start: int | None = None,
    row_stop: int | None = None,
    column_start: int | None = None,
    column_stop: int | None = None,
    minimum_score: float = 0.0,
    minimum_state_count: int = 1,
    minimum_separation: float = 0.0,
    minimum_transition_count: int = 0,
    minimum_lower_run: int = 1,
    minimum_upper_run: int = 1,
    progress_callback=None,
    cancel_requested=None,
) -> RTSDictionaryBuildResult:
    """Build an RTS dictionary CSV and return immutable execution metadata.

    No new RTS analysis or candidate logic is introduced. Counts are collected
    while the existing lazy analysis and filtering pipeline is consumed by the
    atomic CSV writer.
    """
    if not isinstance(plan, RTSDictionaryPlan):
        raise Step04Error(
            "plan must be an RTSDictionaryPlan returned by "
            "prepare_rts_dictionary_analysis()."
        )

    height, width = plan.image_shape
    resolved_row_start = 0 if row_start is None else row_start
    resolved_row_stop = height if row_stop is None else row_stop
    resolved_column_start = 0 if column_start is None else column_start
    resolved_column_stop = width if column_stop is None else column_stop

    # Trigger the existing canonical ROI validation before output creation.
    validation_iterator = iter_image_coordinates(
        plan,
        row_start=resolved_row_start,
        row_stop=resolved_row_stop,
        column_start=resolved_column_start,
        column_stop=resolved_column_stop,
    )
    next(validation_iterator, None)

    if progress_callback is not None and not callable(progress_callback):
        raise Step04Error("progress_callback must be callable or None.")
    if cancel_requested is not None and not callable(cancel_requested):
        raise Step04Error("cancel_requested must be callable or None.")

    try:
        resolved_output_path = Path(output_path)
    except TypeError as exc:
        raise Step04Error("output_path must be path-like.") from exc

    parameters = RTSDictionaryBuildParameters(
        minimum_score=float(minimum_score),
        minimum_state_count=int(minimum_state_count),
        minimum_separation=float(minimum_separation),
        minimum_transition_count=int(minimum_transition_count),
        minimum_lower_run=int(minimum_lower_run),
        minimum_upper_run=int(minimum_upper_run),
        row_start=int(resolved_row_start),
        row_stop=int(resolved_row_stop),
        column_start=int(resolved_column_start),
        column_stop=int(resolved_column_stop),
    )
    total_pixel_count = parameters.pixel_count

    def check_cancellation(completed: int) -> None:
        if cancel_requested is None:
            return
        try:
            requested = cancel_requested()
        except Exception as exc:
            raise Step04Error(
                f"cancel_requested failed after {completed} completed pixels: {exc}"
            ) from exc
        if not isinstance(requested, bool):
            raise Step04Error("cancel_requested must return bool.")
        if requested:
            raise Step04Cancelled(
                RTSCancellationInfo(
                    completed_pixel_count=completed,
                    total_pixel_count=total_pixel_count,
                    output_path=resolved_output_path,
                )
            )

    def notify_progress(completed: int) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(completed, total_pixel_count)
        except Exception as exc:
            raise Step04Error(
                f"progress_callback failed at {completed}/"
                f"{total_pixel_count}: {exc}"
            ) from exc

    check_cancellation(0)
    notify_progress(0)

    analyzed_pixel_count = 0
    candidate_count = 0

    analyses = iter_image_rts_analyses(
        plan,
        row_start=resolved_row_start,
        row_stop=resolved_row_stop,
        column_start=resolved_column_start,
        column_stop=resolved_column_stop,
        minimum_score=minimum_score,
        minimum_state_count=minimum_state_count,
        minimum_separation=minimum_separation,
        minimum_transition_count=minimum_transition_count,
        minimum_lower_run=minimum_lower_run,
        minimum_upper_run=minimum_upper_run,
    )

    def counted_analyses():
        nonlocal analyzed_pixel_count
        iterator = iter(analyses)
        while True:
            check_cancellation(analyzed_pixel_count)
            try:
                analysis = next(iterator)
            except StopIteration:
                break
            analyzed_pixel_count += 1
            notify_progress(analyzed_pixel_count)
            yield analysis

    candidates = iter_rts_candidates(counted_analyses())

    def counted_candidates():
        nonlocal candidate_count
        for candidate in candidates:
            candidate_count += 1
            yield candidate

    written_path = write_rts_dictionary_csv(output_path, counted_candidates())

    return RTSDictionaryBuildResult(
        output_path=written_path,
        dataset=plan.dataset,
        row_start=int(resolved_row_start),
        row_stop=int(resolved_row_stop),
        column_start=int(resolved_column_start),
        column_stop=int(resolved_column_stop),
        analyzed_pixel_count=analyzed_pixel_count,
        candidate_count=candidate_count,
        parameters=parameters,
    )


def _default_metadata_path(output_path: Path) -> Path:
    """Return the canonical sidecar metadata path for a CSV destination."""
    return output_path.with_name(f"{output_path.name}.metadata.json")


def _dictionary_metadata_document(
    plan: RTSDictionaryPlan,
    result: RTSDictionaryBuildResult,
) -> dict[str, object]:
    """Return the canonical metadata document for one completed build."""
    bias = plan.bias_plan
    return {
        "schema": "rts-framework.step04.dictionary-metadata",
        "schema_version": 1,
        "step04_version": __version__,
        "dictionary": {
            "csv_path": str(result.output_path),
            "dataset": result.dataset,
            "analyzed_pixel_count": result.analyzed_pixel_count,
            "candidate_count": result.candidate_count,
        },
        "input": {
            "dataset": plan.dataset,
            "n_frames": plan.n_frames,
            "image_shape": list(plan.image_shape),
            "minimum_frames": plan.minimum_frames,
            "pixel_dtype": bias.pixel_dtype,
            "exposure_s": float(bias.exposure_s),
            "temperature_min_C": float(bias.temperature_min_C),
            "temperature_max_C": float(bias.temperature_max_C),
            "filepaths": [str(path) for path in bias.filepaths],
        },
        "parameters": result.parameters.summary(),
    }




def _parse_csv_int(value: str, name: str, row_number: int,
                   *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Step04Error(
            f"CSV row {row_number} field {name} must be an integer."
        ) from exc
    if str(parsed) != value.strip():
        raise Step04Error(
            f"CSV row {row_number} field {name} must use canonical "
            "integer formatting."
        )
    if parsed < minimum:
        raise Step04Error(
            f"CSV row {row_number} field {name} must be at least {minimum}."
        )
    return parsed


def _parse_csv_float(value: str, name: str, row_number: int,
                     *, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise Step04Error(
            f"CSV row {row_number} field {name} must be a real number."
        ) from exc
    if not np.isfinite(parsed):
        raise Step04Error(
            f"CSV row {row_number} field {name} must be finite."
        )
    if minimum is not None and parsed < minimum:
        raise Step04Error(
            f"CSV row {row_number} field {name} must be at least {minimum}."
        )
    return parsed


def _parse_csv_candidate_bool(value: str, row_number: int) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise Step04Error(
        f"CSV row {row_number} field is_candidate must be True or False."
    )


def _dictionary_row_from_csv(raw: dict[str, str],
                             row_number: int) -> RTSDictionaryRow:
    dataset = raw["dataset"]
    if not dataset:
        raise Step04Error(
            f"CSV row {row_number} field dataset must be non-empty."
        )

    row = RTSDictionaryRow(
        dataset=dataset,
        row=_parse_csv_int(raw["row"], "row", row_number),
        column=_parse_csv_int(raw["column"], "column", row_number),
        n_frames=_parse_csv_int(
            raw["n_frames"], "n_frames", row_number, minimum=1
        ),
        minimum=_parse_csv_float(raw["minimum"], "minimum", row_number),
        maximum=_parse_csv_float(raw["maximum"], "maximum", row_number),
        mean=_parse_csv_float(raw["mean"], "mean", row_number),
        median=_parse_csv_float(raw["median"], "median", row_number),
        standard_deviation=_parse_csv_float(
            raw["standard_deviation"], "standard_deviation", row_number,
            minimum=0.0,
        ),
        median_absolute_deviation=_parse_csv_float(
            raw["median_absolute_deviation"],
            "median_absolute_deviation",
            row_number,
            minimum=0.0,
        ),
        peak_to_peak=_parse_csv_float(
            raw["peak_to_peak"], "peak_to_peak", row_number, minimum=0.0
        ),
        lower_state_count=_parse_csv_int(
            raw["lower_state_count"], "lower_state_count", row_number,
            minimum=1,
        ),
        upper_state_count=_parse_csv_int(
            raw["upper_state_count"], "upper_state_count", row_number,
            minimum=1,
        ),
        lower_state_center=_parse_csv_float(
            raw["lower_state_center"], "lower_state_center", row_number
        ),
        upper_state_center=_parse_csv_float(
            raw["upper_state_center"], "upper_state_center", row_number
        ),
        state_separation=_parse_csv_float(
            raw["state_separation"], "state_separation", row_number,
            minimum=0.0,
        ),
        single_state_residual=_parse_csv_float(
            raw["single_state_residual"], "single_state_residual",
            row_number, minimum=0.0,
        ),
        two_state_residual=_parse_csv_float(
            raw["two_state_residual"], "two_state_residual",
            row_number, minimum=0.0,
        ),
        two_state_score=_parse_csv_float(
            raw["two_state_score"], "two_state_score", row_number
        ),
        minimum_score=_parse_csv_float(
            raw["minimum_score"], "minimum_score", row_number
        ),
        minimum_state_count=_parse_csv_int(
            raw["minimum_state_count"], "minimum_state_count",
            row_number, minimum=1,
        ),
        minimum_separation=_parse_csv_float(
            raw["minimum_separation"], "minimum_separation",
            row_number, minimum=0.0,
        ),
        transition_count=_parse_csv_int(
            raw["transition_count"], "transition_count", row_number
        ),
        lower_to_upper_count=_parse_csv_int(
            raw["lower_to_upper_count"], "lower_to_upper_count", row_number
        ),
        upper_to_lower_count=_parse_csv_int(
            raw["upper_to_lower_count"], "upper_to_lower_count", row_number
        ),
        longest_lower_run=_parse_csv_int(
            raw["longest_lower_run"], "longest_lower_run",
            row_number, minimum=1,
        ),
        longest_upper_run=_parse_csv_int(
            raw["longest_upper_run"], "longest_upper_run",
            row_number, minimum=1,
        ),
        minimum_transition_count=_parse_csv_int(
            raw["minimum_transition_count"], "minimum_transition_count",
            row_number,
        ),
        minimum_lower_run=_parse_csv_int(
            raw["minimum_lower_run"], "minimum_lower_run",
            row_number, minimum=1,
        ),
        minimum_upper_run=_parse_csv_int(
            raw["minimum_upper_run"], "minimum_upper_run",
            row_number, minimum=1,
        ),
        is_candidate=_parse_csv_candidate_bool(
            raw["is_candidate"], row_number
        ),
    )

    if not row.is_candidate:
        raise Step04Error(
            f"CSV row {row_number} is not a final RTS candidate."
        )
    if row.maximum < row.minimum:
        raise Step04Error(
            f"CSV row {row_number} maximum must not be smaller than minimum."
        )
    if row.peak_to_peak != row.maximum - row.minimum:
        raise Step04Error(
            f"CSV row {row_number} peak_to_peak is inconsistent."
        )
    if row.lower_state_count + row.upper_state_count != row.n_frames:
        raise Step04Error(
            f"CSV row {row_number} state counts must sum to n_frames."
        )
    if row.upper_state_center < row.lower_state_center:
        raise Step04Error(
            f"CSV row {row_number} upper_state_center must not be smaller "
            "than lower_state_center."
        )
    if row.state_separation != (
        row.upper_state_center - row.lower_state_center
    ):
        raise Step04Error(
            f"CSV row {row_number} state_separation is inconsistent."
        )
    if row.lower_to_upper_count + row.upper_to_lower_count != (
        row.transition_count
    ):
        raise Step04Error(
            f"CSV row {row_number} transition counts are inconsistent."
        )
    if row.two_state_score < row.minimum_score:
        raise Step04Error(
            f"CSV row {row_number} does not satisfy minimum_score."
        )
    if min(row.lower_state_count, row.upper_state_count) < (
        row.minimum_state_count
    ):
        raise Step04Error(
            f"CSV row {row_number} does not satisfy minimum_state_count."
        )
    if row.state_separation < row.minimum_separation:
        raise Step04Error(
            f"CSV row {row_number} does not satisfy minimum_separation."
        )
    if row.transition_count < row.minimum_transition_count:
        raise Step04Error(
            f"CSV row {row_number} does not satisfy "
            "minimum_transition_count."
        )
    if row.longest_lower_run < row.minimum_lower_run:
        raise Step04Error(
            f"CSV row {row_number} does not satisfy minimum_lower_run."
        )
    if row.longest_upper_run < row.minimum_upper_run:
        raise Step04Error(
            f"CSV row {row_number} does not satisfy minimum_upper_run."
        )
    return row


def load_rts_dictionary_csv(path) -> RTSDictionaryCSV:
    """Load and validate one canonical Step 04 RTS dictionary CSV."""
    try:
        source = Path(path)
    except TypeError as exc:
        raise Step04Error("path must be path-like.") from exc
    if not source.is_file():
        raise Step04Error(f"dictionary CSV does not exist: {source}")

    try:
        with source.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise Step04Error("dictionary CSV is missing its header.")
            if tuple(reader.fieldnames) != RTS_DICTIONARY_COLUMNS:
                raise Step04Error(
                    "dictionary CSV header does not match "
                    "RTS_DICTIONARY_COLUMNS."
                )
            rows = tuple(
                _dictionary_row_from_csv(raw, row_number)
                for row_number, raw in enumerate(reader, start=2)
            )
    except Step04Error:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise Step04Error(
            f"Could not read RTS dictionary CSV '{source}': {exc}"
        ) from exc

    coordinates: set[tuple[str, int, int]] = set()
    for row_number, row in enumerate(rows, start=2):
        key = (row.dataset, row.row, row.column)
        if key in coordinates:
            raise Step04Error(
                f"CSV row {row_number} duplicates a dataset coordinate."
            )
        coordinates.add(key)

    return RTSDictionaryCSV(path=source, rows=rows)




def _normalized_artifact_path(path: Path) -> Path:
    """Return a non-strict absolute path for artifact identity checks."""
    return path.expanduser().resolve(strict=False)


def load_rts_dictionary_artifacts(
    csv_path,
    metadata_path=None,
) -> RTSDictionaryArtifacts:
    """Load and cross-validate one RTS dictionary CSV/metadata artifact pair.

    When ``metadata_path`` is omitted, the canonical
    ``<csv name>.metadata.json`` sidecar path is used.
    """
    try:
        resolved_csv_path = Path(csv_path)
    except TypeError as exc:
        raise Step04Error("csv_path must be path-like.") from exc

    if metadata_path is None:
        resolved_metadata_path = _default_metadata_path(resolved_csv_path)
    else:
        try:
            resolved_metadata_path = Path(metadata_path)
        except TypeError as exc:
            raise Step04Error("metadata_path must be path-like or None.") from exc

    dictionary = load_rts_dictionary_csv(resolved_csv_path)
    metadata = load_rts_dictionary_metadata_json(resolved_metadata_path)

    if _normalized_artifact_path(metadata.csv_path) != (
        _normalized_artifact_path(dictionary.path)
    ):
        raise Step04Error(
            "metadata csv_path does not match the loaded dictionary CSV."
        )

    if metadata.candidate_count != dictionary.candidate_count:
        raise Step04Error(
            "metadata candidate_count does not match the dictionary CSV."
        )

    if dictionary.rows:
        if dictionary.datasets != (metadata.dataset,):
            raise Step04Error(
                "dictionary CSV dataset does not match metadata dataset."
            )

    p = metadata.parameters
    for row_number, row in enumerate(dictionary.rows, start=2):
        if row.dataset != metadata.dataset:
            raise Step04Error(
                f"CSV row {row_number} dataset does not match metadata."
            )
        if row.n_frames != metadata.n_frames:
            raise Step04Error(
                f"CSV row {row_number} n_frames does not match metadata."
            )
        if not (p.row_start <= row.row < p.row_stop):
            raise Step04Error(
                f"CSV row {row_number} row lies outside metadata ROI."
            )
        if not (p.column_start <= row.column < p.column_stop):
            raise Step04Error(
                f"CSV row {row_number} column lies outside metadata ROI."
            )
        if row.minimum_score != p.minimum_score:
            raise Step04Error(
                f"CSV row {row_number} minimum_score does not match metadata."
            )
        if row.minimum_state_count != p.minimum_state_count:
            raise Step04Error(
                f"CSV row {row_number} minimum_state_count does not match "
                "metadata."
            )
        if row.minimum_separation != p.minimum_separation:
            raise Step04Error(
                f"CSV row {row_number} minimum_separation does not match "
                "metadata."
            )
        if row.minimum_transition_count != p.minimum_transition_count:
            raise Step04Error(
                f"CSV row {row_number} minimum_transition_count does not "
                "match metadata."
            )
        if row.minimum_lower_run != p.minimum_lower_run:
            raise Step04Error(
                f"CSV row {row_number} minimum_lower_run does not match "
                "metadata."
            )
        if row.minimum_upper_run != p.minimum_upper_run:
            raise Step04Error(
                f"CSV row {row_number} minimum_upper_run does not match "
                "metadata."
            )

    return RTSDictionaryArtifacts(
        dictionary=dictionary,
        metadata=metadata,
    )




def validate_rts_dictionary_input_files(
    metadata,
) -> RTSInputFileValidation:
    """Validate the input-file inventory recorded in dictionary metadata.

    ``metadata`` may be an already loaded :class:`RTSDictionaryMetadata`
    instance or a metadata JSON path. Validation checks that the recorded
    file count matches ``n_frames``, paths are unique after normalization,
    and every path exists as a regular file.
    """
    if isinstance(metadata, RTSDictionaryMetadata):
        loaded = metadata
    else:
        loaded = load_rts_dictionary_metadata_json(metadata)

    if len(loaded.filepaths) != loaded.n_frames:
        raise Step04Error(
            "metadata input file count does not match n_frames."
        )

    normalized_paths: list[Path] = []
    seen: set[Path] = set()
    for file_index, filepath in enumerate(loaded.filepaths):
        normalized = _normalized_artifact_path(filepath)
        if normalized in seen:
            raise Step04Error(
                f"metadata input file {file_index} duplicates another path."
            )
        seen.add(normalized)

        if not normalized.exists():
            raise Step04Error(
                f"metadata input file {file_index} does not exist: "
                f"{normalized}"
            )
        if not normalized.is_file():
            raise Step04Error(
                f"metadata input file {file_index} is not a regular file: "
                f"{normalized}"
            )
        normalized_paths.append(normalized)

    return RTSInputFileValidation(
        metadata_path=loaded.metadata_path,
        expected_file_count=loaded.n_frames,
        validated_filepaths=tuple(normalized_paths),
    )



def _require_metadata_mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise Step04Error(f"{name} must be a JSON object.")
    return value


def _require_metadata_string(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise Step04Error(f"{name} must be a non-empty string.")
    return value


def _require_metadata_int(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Step04Error(f"{name} must be an integer.")
    if value < minimum:
        raise Step04Error(f"{name} must be at least {minimum}.")
    return value


def _require_metadata_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Step04Error(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result):
        raise Step04Error(f"{name} must be finite.")
    return result


def load_rts_dictionary_metadata_json(path) -> RTSDictionaryMetadata:
    """Load and validate one Step 04 dictionary metadata JSON sidecar."""
    try:
        source = Path(path)
    except TypeError as exc:
        raise Step04Error("path must be path-like.") from exc

    if not source.is_file():
        raise Step04Error(f"metadata JSON does not exist: {source}")

    try:
        with source.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Step04Error(
            f"Could not read RTS dictionary metadata JSON '{source}': {exc}"
        ) from exc

    root = _require_metadata_mapping(document, "metadata root")
    schema = _require_metadata_string(root.get("schema"), "schema")
    if schema != "rts-framework.step04.dictionary-metadata":
        raise Step04Error(f"Unsupported metadata schema: {schema}")

    schema_version = _require_metadata_int(
        root.get("schema_version"), "schema_version", minimum=1
    )
    if schema_version != 1:
        raise Step04Error(
            f"Unsupported metadata schema_version: {schema_version}"
        )

    step04_version = _require_metadata_string(
        root.get("step04_version"), "step04_version"
    )

    dictionary = _require_metadata_mapping(
        root.get("dictionary"), "dictionary"
    )
    input_info = _require_metadata_mapping(root.get("input"), "input")
    raw_parameters = _require_metadata_mapping(
        root.get("parameters"), "parameters"
    )

    csv_path = Path(
        _require_metadata_string(dictionary.get("csv_path"),
                                 "dictionary.csv_path")
    )
    dataset = _require_metadata_string(
        dictionary.get("dataset"), "dictionary.dataset"
    )
    analyzed_pixel_count = _require_metadata_int(
        dictionary.get("analyzed_pixel_count"),
        "dictionary.analyzed_pixel_count",
    )
    candidate_count = _require_metadata_int(
        dictionary.get("candidate_count"),
        "dictionary.candidate_count",
    )
    if candidate_count > analyzed_pixel_count:
        raise Step04Error(
            "dictionary.candidate_count must not exceed "
            "dictionary.analyzed_pixel_count."
        )

    input_dataset = _require_metadata_string(
        input_info.get("dataset"), "input.dataset"
    )
    if input_dataset != dataset:
        raise Step04Error(
            "input.dataset must match dictionary.dataset."
        )

    n_frames = _require_metadata_int(
        input_info.get("n_frames"), "input.n_frames", minimum=1
    )
    minimum_frames = _require_metadata_int(
        input_info.get("minimum_frames"),
        "input.minimum_frames",
        minimum=1,
    )
    if n_frames < minimum_frames:
        raise Step04Error(
            "input.n_frames must not be smaller than input.minimum_frames."
        )

    image_shape_raw = input_info.get("image_shape")
    if (
        not isinstance(image_shape_raw, list)
        or len(image_shape_raw) != 2
    ):
        raise Step04Error(
            "input.image_shape must be a two-element JSON array."
        )
    image_shape = (
        _require_metadata_int(
            image_shape_raw[0], "input.image_shape[0]", minimum=1
        ),
        _require_metadata_int(
            image_shape_raw[1], "input.image_shape[1]", minimum=1
        ),
    )

    pixel_dtype = _require_metadata_string(
        input_info.get("pixel_dtype"), "input.pixel_dtype"
    )
    exposure_s = _require_metadata_float(
        input_info.get("exposure_s"), "input.exposure_s"
    )
    temperature_min_C = _require_metadata_float(
        input_info.get("temperature_min_C"), "input.temperature_min_C"
    )
    temperature_max_C = _require_metadata_float(
        input_info.get("temperature_max_C"), "input.temperature_max_C"
    )
    if temperature_max_C < temperature_min_C:
        raise Step04Error(
            "input.temperature_max_C must not be smaller than "
            "input.temperature_min_C."
        )

    filepaths_raw = input_info.get("filepaths")
    if not isinstance(filepaths_raw, list):
        raise Step04Error("input.filepaths must be a JSON array.")
    if len(filepaths_raw) != n_frames:
        raise Step04Error(
            "input.filepaths length must match input.n_frames."
        )
    filepaths = tuple(
        Path(_require_metadata_string(value, f"input.filepaths[{index}]"))
        for index, value in enumerate(filepaths_raw)
    )

    parameter_names = (
        "minimum_score",
        "minimum_state_count",
        "minimum_separation",
        "minimum_transition_count",
        "minimum_lower_run",
        "minimum_upper_run",
        "row_start",
        "row_stop",
        "column_start",
        "column_stop",
    )
    missing = [name for name in parameter_names if name not in raw_parameters]
    if missing:
        raise Step04Error(
            "parameters is missing required fields: " + ", ".join(missing)
        )

    parameters = RTSDictionaryBuildParameters(
        minimum_score=_require_metadata_float(
            raw_parameters["minimum_score"], "parameters.minimum_score"
        ),
        minimum_state_count=_require_metadata_int(
            raw_parameters["minimum_state_count"],
            "parameters.minimum_state_count",
            minimum=1,
        ),
        minimum_separation=_require_metadata_float(
            raw_parameters["minimum_separation"],
            "parameters.minimum_separation",
        ),
        minimum_transition_count=_require_metadata_int(
            raw_parameters["minimum_transition_count"],
            "parameters.minimum_transition_count",
        ),
        minimum_lower_run=_require_metadata_int(
            raw_parameters["minimum_lower_run"],
            "parameters.minimum_lower_run",
            minimum=1,
        ),
        minimum_upper_run=_require_metadata_int(
            raw_parameters["minimum_upper_run"],
            "parameters.minimum_upper_run",
            minimum=1,
        ),
        row_start=_require_metadata_int(
            raw_parameters["row_start"], "parameters.row_start"
        ),
        row_stop=_require_metadata_int(
            raw_parameters["row_stop"], "parameters.row_stop"
        ),
        column_start=_require_metadata_int(
            raw_parameters["column_start"], "parameters.column_start"
        ),
        column_stop=_require_metadata_int(
            raw_parameters["column_stop"], "parameters.column_stop"
        ),
    )

    if parameters.row_stop > image_shape[0]:
        raise Step04Error(
            "parameters.row_stop exceeds input.image_shape."
        )
    if parameters.column_stop > image_shape[1]:
        raise Step04Error(
            "parameters.column_stop exceeds input.image_shape."
        )
    if parameters.pixel_count != analyzed_pixel_count:
        raise Step04Error(
            "parameters.pixel_count must match "
            "dictionary.analyzed_pixel_count."
        )

    return RTSDictionaryMetadata(
        metadata_path=source,
        schema=schema,
        schema_version=schema_version,
        step04_version=step04_version,
        csv_path=csv_path,
        dataset=dataset,
        analyzed_pixel_count=analyzed_pixel_count,
        candidate_count=candidate_count,
        n_frames=n_frames,
        image_shape=image_shape,
        minimum_frames=minimum_frames,
        pixel_dtype=pixel_dtype,
        exposure_s=exposure_s,
        temperature_min_C=temperature_min_C,
        temperature_max_C=temperature_max_C,
        filepaths=filepaths,
        parameters=parameters,
    )



def write_rts_dictionary_metadata_json(
    path,
    plan: RTSDictionaryPlan,
    result: RTSDictionaryBuildResult,
) -> Path:
    """Atomically write deterministic UTF-8 metadata JSON for a CSV build."""
    if not isinstance(plan, RTSDictionaryPlan):
        raise Step04Error("plan must be an RTSDictionaryPlan.")
    if not isinstance(result, RTSDictionaryBuildResult):
        raise Step04Error("result must be an RTSDictionaryBuildResult.")

    try:
        destination = Path(path)
    except TypeError as exc:
        raise Step04Error("path must be path-like.") from exc

    if destination.exists() and destination.is_dir():
        raise Step04Error(f"path must not be a directory: {destination}")

    parent = destination.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise Step04Error(
            f"Could not create output directory '{parent}': {exc}"
        ) from exc

    document = _dictionary_metadata_document(plan, result)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(
                document,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary_path, destination)
        temporary_path = None
        return destination

    except (OSError, TypeError, ValueError) as exc:
        raise Step04Error(
            f"Could not write RTS dictionary metadata JSON "
            f"'{destination}': {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def build_rts_dictionary_artifacts(
    plan: RTSDictionaryPlan,
    output_path,
    *,
    metadata_path=None,
    row_start: int | None = None,
    row_stop: int | None = None,
    column_start: int | None = None,
    column_stop: int | None = None,
    minimum_score: float = 0.0,
    minimum_state_count: int = 1,
    minimum_separation: float = 0.0,
    minimum_transition_count: int = 0,
    minimum_lower_run: int = 1,
    minimum_upper_run: int = 1,
    progress_callback=None,
    cancel_requested=None,
) -> RTSDictionaryArtifactResult:
    """Build an RTS dictionary CSV and its deterministic metadata sidecar."""
    result = build_rts_dictionary_csv_result(
        plan,
        output_path,
        row_start=row_start,
        row_stop=row_stop,
        column_start=column_start,
        column_stop=column_stop,
        minimum_score=minimum_score,
        minimum_state_count=minimum_state_count,
        minimum_separation=minimum_separation,
        minimum_transition_count=minimum_transition_count,
        minimum_lower_run=minimum_lower_run,
        minimum_upper_run=minimum_upper_run,
        progress_callback=progress_callback,
        cancel_requested=cancel_requested,
    )
    resolved_metadata_path = (
        _default_metadata_path(result.output_path)
        if metadata_path is None
        else Path(metadata_path)
    )
    written_metadata_path = write_rts_dictionary_metadata_json(
        resolved_metadata_path,
        plan,
        result,
    )
    return RTSDictionaryArtifactResult(
        build_result=result,
        metadata_path=written_metadata_path,
    )



def write_rts_dictionary_csv(path, candidates) -> Path:
    """Atomically write final RTS candidates to a deterministic UTF-8 CSV.

    The input iterable is consumed once and lazily. Input order and duplicate
    references are preserved. Empty input produces a valid header-only CSV.
    The destination is replaced only after the entire input has been validated
    and serialized successfully.

    Parameters
    ----------
    path
        Destination path accepted by :class:`pathlib.Path`.
    candidates
        Iterable of final :class:`RTSPixelAnalysisResult` objects.

    Returns
    -------
    pathlib.Path
        The normalized destination path.

    Raises
    ------
    Step04Error
        If the path is invalid, the candidate source is not iterable, an item
        is invalid or not a final candidate, or writing/replacing fails.
    """
    try:
        destination = Path(path)
    except TypeError as exc:
        raise Step04Error("path must be path-like.") from exc

    if destination.exists() and destination.is_dir():
        raise Step04Error(f"path must not be a directory: {destination}")

    try:
        iterator = iter(candidates)
    except TypeError as exc:
        raise Step04Error(
            "candidates must be an iterable of final RTS candidates."
        ) from exc

    parent = destination.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise Step04Error(
            f"Could not create output directory '{parent}': {exc}"
        ) from exc

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(
                stream,
                fieldnames=RTS_DICTIONARY_COLUMNS,
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()

            for index, candidate in enumerate(iterator):
                try:
                    row = rts_candidate_to_row(candidate)
                except Step04Error as exc:
                    raise Step04Error(
                        f"Could not serialize candidates item {index}: {exc}"
                    ) from exc
                writer.writerow(row)

            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary_path, destination)
        temporary_path = None
        return destination

    except Step04Error:
        raise
    except (OSError, csv.Error, ValueError) as exc:
        raise Step04Error(
            f"Could not write RTS dictionary CSV '{destination}': {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

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
