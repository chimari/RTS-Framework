"""Step 05: prepare deterministic RTS correction plans.

Version 5.6.0 adds deterministic batch correction for multiple FITS inputs
using one validated RTS dictionary metadata artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
from astropy.io import fits

# Support both package import and direct execution:
#   python -m steps.step05_apply_rts_correction --version
#   python steps/step05_apply_rts_correction.py --version
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from steps.step04_prepare_rts_dictionary_analysis import (
    RTSDictionaryArtifactValidation,
    RTSDictionaryRow,
    Step04Error,
    validate_rts_dictionary_artifacts,
)

__version__ = "5.6.0"

__all__ = [
    "RTSCandidateClassification",
    "RTSCandidateState",
    "RTSCorrectionApplicationResult",
    "RTSCorrectionCandidate",
    "RTSCorrectionClassificationResult",
    "RTSCorrectionDecision",
    "RTSCorrectionDecisionReason",
    "RTSCorrectionDecisionResult",
    "RTSCorrectionBatchItem",
    "RTSCorrectionBatchResult",
    "RTSCorrectionOutput",
    "RTSCorrectionPlan",
    "Step05Error",
    "apply_rts_correction_in_memory",
    "build_rts_correction_decisions",
    "classify_rts_correction_candidates",
    "prepare_rts_correction",
    "run_rts_correction_batch",
    "run_rts_correction_cli",
    "write_rts_corrected_fits",
]


class Step05Error(Exception):
    """Raised when Step 05 cannot prepare an RTS correction plan."""



class RTSCandidateState(str, Enum):
    """Classification of one current RTS candidate-pixel value."""

    LOWER = "LOWER"
    UPPER = "UPPER"
    MIDPOINT = "MIDPOINT"
    OUTSIDE = "OUTSIDE"



class RTSCorrectionDecisionReason(str, Enum):
    """Reason associated with one deterministic correction decision."""

    LOWER_STATE = "LOWER_STATE"
    UPPER_STATE = "UPPER_STATE"
    MIDPOINT_UNCERTAIN = "MIDPOINT_UNCERTAIN"
    OUTSIDE_DICTIONARY = "OUTSIDE_DICTIONARY"


@dataclass(slots=True, frozen=True)
class RTSCorrectionCandidate:
    """Immutable correction parameters for one RTS dictionary coordinate."""

    row: int
    column: int
    lower_state_center: float
    upper_state_center: float
    state_separation: float
    transition_count: int
    two_state_score: float

    @property
    def coordinate(self) -> tuple[int, int]:
        """Return ``(row, column)``."""
        return (self.row, self.column)

    @property
    def midpoint(self) -> float:
        """Return the midpoint between the two dictionary state centers."""
        return (self.lower_state_center + self.upper_state_center) / 2.0

    def summary(self) -> dict[str, object]:
        """Return one deterministic JSON-serializable candidate summary."""
        return {
            "row": self.row,
            "column": self.column,
            "coordinate": self.coordinate,
            "lower_state_center": self.lower_state_center,
            "upper_state_center": self.upper_state_center,
            "state_separation": self.state_separation,
            "midpoint": self.midpoint,
            "transition_count": self.transition_count,
            "two_state_score": self.two_state_score,
        }


@dataclass(slots=True, frozen=True)
class RTSCorrectionPlan:
    """Immutable validated plan for applying one RTS dictionary to one FITS."""

    input_path: Path
    metadata_path: Path
    dictionary_csv_path: Path
    image_shape: tuple[int, int]
    pixel_dtype: str
    dataset: str
    candidates: tuple[RTSCorrectionCandidate, ...]
    artifact_validation: RTSDictionaryArtifactValidation

    @property
    def candidate_count(self) -> int:
        """Return the number of planned candidate coordinates."""
        return len(self.candidates)

    @property
    def coordinates(self) -> tuple[tuple[int, int], ...]:
        """Return candidate coordinates in canonical dictionary order."""
        return tuple(candidate.coordinate for candidate in self.candidates)

    def summary(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable plan summary."""
        return {
            "step05_version": __version__,
            "input_path": str(self.input_path),
            "metadata_path": str(self.metadata_path),
            "dictionary_csv_path": str(self.dictionary_csv_path),
            "image_shape": self.image_shape,
            "pixel_dtype": self.pixel_dtype,
            "dataset": self.dataset,
            "candidate_count": self.candidate_count,
            "coordinates": self.coordinates,
            "candidates": tuple(
                candidate.summary() for candidate in self.candidates
            ),
        }



@dataclass(slots=True, frozen=True)
class RTSCandidateClassification:
    """Immutable state classification for one candidate pixel."""

    candidate: RTSCorrectionCandidate
    pixel_value: float
    state: RTSCandidateState
    tolerance: float
    distance_to_lower: float
    distance_to_upper: float

    @property
    def coordinate(self) -> tuple[int, int]:
        """Return the classified candidate coordinate."""
        return self.candidate.coordinate

    @property
    def nearest_state(self) -> RTSCandidateState:
        """Return LOWER or UPPER according to the nearest state center."""
        if self.distance_to_lower <= self.distance_to_upper:
            return RTSCandidateState.LOWER
        return RTSCandidateState.UPPER

    def summary(self) -> dict[str, object]:
        """Return one deterministic JSON-serializable classification."""
        return {
            "row": self.candidate.row,
            "column": self.candidate.column,
            "coordinate": self.coordinate,
            "pixel_value": self.pixel_value,
            "state": self.state.value,
            "nearest_state": self.nearest_state.value,
            "tolerance": self.tolerance,
            "distance_to_lower": self.distance_to_lower,
            "distance_to_upper": self.distance_to_upper,
            "lower_state_center": self.candidate.lower_state_center,
            "upper_state_center": self.candidate.upper_state_center,
            "midpoint": self.candidate.midpoint,
        }


@dataclass(slots=True, frozen=True)
class RTSCorrectionClassificationResult:
    """Immutable classifications for every candidate in one correction plan."""

    plan: RTSCorrectionPlan
    state_tolerance_fraction: float
    classifications: tuple[RTSCandidateClassification, ...]

    @property
    def candidate_count(self) -> int:
        """Return the number of classified candidates."""
        return len(self.classifications)

    def count(self, state: RTSCandidateState) -> int:
        """Count classifications with one state."""
        if not isinstance(state, RTSCandidateState):
            raise Step05Error("state must be an RTSCandidateState.")
        return sum(item.state is state for item in self.classifications)

    @property
    def lower_count(self) -> int:
        return self.count(RTSCandidateState.LOWER)

    @property
    def upper_count(self) -> int:
        return self.count(RTSCandidateState.UPPER)

    @property
    def midpoint_count(self) -> int:
        return self.count(RTSCandidateState.MIDPOINT)

    @property
    def outside_count(self) -> int:
        return self.count(RTSCandidateState.OUTSIDE)

    def summary(self) -> dict[str, object]:
        """Return one deterministic JSON-serializable result summary."""
        return {
            "step05_version": __version__,
            "input_path": str(self.plan.input_path),
            "state_tolerance_fraction": self.state_tolerance_fraction,
            "candidate_count": self.candidate_count,
            "lower_count": self.lower_count,
            "upper_count": self.upper_count,
            "midpoint_count": self.midpoint_count,
            "outside_count": self.outside_count,
            "classifications": tuple(
                item.summary() for item in self.classifications
            ),
        }



@dataclass(slots=True, frozen=True)
class RTSCorrectionDecision:
    """Immutable correction decision for one classified RTS candidate."""

    classification: RTSCandidateClassification
    target_value: float | None
    correction_value: float | None
    is_correctable: bool
    reason: RTSCorrectionDecisionReason

    @property
    def candidate(self) -> RTSCorrectionCandidate:
        """Return the candidate associated with this decision."""
        return self.classification.candidate

    @property
    def coordinate(self) -> tuple[int, int]:
        """Return the candidate coordinate."""
        return self.classification.coordinate

    @property
    def current_value(self) -> float:
        """Return the currently observed pixel value."""
        return self.classification.pixel_value

    @property
    def current_state(self) -> RTSCandidateState:
        """Return the classified current state."""
        return self.classification.state

    def summary(self) -> dict[str, object]:
        """Return one deterministic JSON-serializable decision summary."""
        return {
            "row": self.candidate.row,
            "column": self.candidate.column,
            "coordinate": self.coordinate,
            "current_value": self.current_value,
            "current_state": self.current_state.value,
            "target_value": self.target_value,
            "correction_value": self.correction_value,
            "is_correctable": self.is_correctable,
            "reason": self.reason.value,
        }


@dataclass(slots=True, frozen=True)
class RTSCorrectionDecisionResult:
    """Immutable correction decisions derived from one classification result."""

    classification_result: RTSCorrectionClassificationResult
    decisions: tuple[RTSCorrectionDecision, ...]

    @property
    def plan(self) -> RTSCorrectionPlan:
        """Return the correction plan associated with these decisions."""
        return self.classification_result.plan

    @property
    def decision_count(self) -> int:
        """Return the number of correction decisions."""
        return len(self.decisions)

    @property
    def correctable_count(self) -> int:
        """Return the number of decisions that may be applied."""
        return sum(decision.is_correctable for decision in self.decisions)

    @property
    def rejected_count(self) -> int:
        """Return the number of decisions that must not be applied."""
        return self.decision_count - self.correctable_count

    def count_reason(self, reason: RTSCorrectionDecisionReason) -> int:
        """Count decisions with one reason."""
        if not isinstance(reason, RTSCorrectionDecisionReason):
            raise Step05Error(
                "reason must be an RTSCorrectionDecisionReason."
            )
        return sum(decision.reason is reason for decision in self.decisions)

    def summary(self) -> dict[str, object]:
        """Return one deterministic JSON-serializable decision summary."""
        return {
            "step05_version": __version__,
            "input_path": str(self.plan.input_path),
            "decision_count": self.decision_count,
            "correctable_count": self.correctable_count,
            "rejected_count": self.rejected_count,
            "lower_state_count": self.count_reason(
                RTSCorrectionDecisionReason.LOWER_STATE
            ),
            "upper_state_count": self.count_reason(
                RTSCorrectionDecisionReason.UPPER_STATE
            ),
            "midpoint_uncertain_count": self.count_reason(
                RTSCorrectionDecisionReason.MIDPOINT_UNCERTAIN
            ),
            "outside_dictionary_count": self.count_reason(
                RTSCorrectionDecisionReason.OUTSIDE_DICTIONARY
            ),
            "decisions": tuple(
                decision.summary() for decision in self.decisions
            ),
        }



@dataclass(slots=True, frozen=True)
class RTSCorrectionApplicationResult:
    """Immutable metadata plus an independent corrected image array."""

    decision_result: RTSCorrectionDecisionResult
    corrected_image: np.ndarray
    applied_count: int
    preserved_count: int
    output_dtype: str

    def __post_init__(self) -> None:
        image = np.asarray(self.corrected_image)
        if image.ndim != 2:
            raise Step05Error("corrected_image must be two-dimensional.")
        if tuple(int(value) for value in image.shape) != self.plan.image_shape:
            raise Step05Error(
                "corrected_image shape does not match the correction plan."
            )
        if self.applied_count < 0 or self.preserved_count < 0:
            raise Step05Error("application counts must be non-negative.")
        if (
            self.applied_count + self.preserved_count
            != self.decision_result.decision_count
        ):
            raise Step05Error(
                "application counts do not match the decision count."
            )

        independent = np.array(image, copy=True)
        independent.setflags(write=False)
        object.__setattr__(self, "corrected_image", independent)

    @property
    def plan(self) -> RTSCorrectionPlan:
        """Return the correction plan associated with the result."""
        return self.decision_result.plan

    def summary(self) -> dict[str, object]:
        """Return deterministic metadata without serializing image pixels."""
        return {
            "step05_version": __version__,
            "input_path": str(self.plan.input_path),
            "image_shape": self.plan.image_shape,
            "output_dtype": self.output_dtype,
            "decision_count": self.decision_result.decision_count,
            "applied_count": self.applied_count,
            "preserved_count": self.preserved_count,
        }




@dataclass(slots=True, frozen=True)
class RTSCorrectionBatchItem:
    """Immutable result for one input in a batch correction run."""

    input_path: Path
    output_path: Path
    succeeded: bool
    output: RTSCorrectionOutput | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        input_path = Path(self.input_path).expanduser().resolve()
        output_path = Path(self.output_path).expanduser().resolve()
        object.__setattr__(self, "input_path", input_path)
        object.__setattr__(self, "output_path", output_path)

        if self.succeeded:
            if self.output is None:
                raise Step05Error(
                    "successful batch items require an output result."
                )
            if self.error is not None:
                raise Step05Error(
                    "successful batch items must not contain an error."
                )
            if self.output.input_path != input_path:
                raise Step05Error(
                    "batch item input_path does not match output result."
                )
            if self.output.output_path != output_path:
                raise Step05Error(
                    "batch item output_path does not match output result."
                )
        else:
            if self.output is not None:
                raise Step05Error(
                    "failed batch items must not contain an output result."
                )
            if not self.error:
                raise Step05Error(
                    "failed batch items require a non-empty error."
                )

    def summary(self) -> dict[str, object]:
        """Return one deterministic JSON-serializable item summary."""
        payload: dict[str, object] = {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "succeeded": self.succeeded,
        }
        if self.output is not None:
            payload["output"] = self.output.summary()
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(slots=True, frozen=True)
class RTSCorrectionBatchResult:
    """Immutable aggregate result for one batch correction run."""

    metadata_path: Path
    output_directory: Path
    items: tuple[RTSCorrectionBatchItem, ...]
    continue_on_error: bool
    overwrite: bool

    def __post_init__(self) -> None:
        metadata_path = Path(self.metadata_path).expanduser().resolve()
        output_directory = Path(self.output_directory).expanduser().resolve()
        object.__setattr__(self, "metadata_path", metadata_path)
        object.__setattr__(self, "output_directory", output_directory)
        if not self.items:
            raise Step05Error("batch result must contain at least one item.")

    @property
    def total_count(self) -> int:
        return len(self.items)

    @property
    def succeeded_count(self) -> int:
        return sum(1 for item in self.items if item.succeeded)

    @property
    def failed_count(self) -> int:
        return self.total_count - self.succeeded_count

    @property
    def all_succeeded(self) -> bool:
        return self.failed_count == 0

    def summary(self) -> dict[str, object]:
        """Return one deterministic JSON-serializable batch summary."""
        return {
            "step05_version": __version__,
            "metadata_path": str(self.metadata_path),
            "output_directory": str(self.output_directory),
            "continue_on_error": self.continue_on_error,
            "overwrite": self.overwrite,
            "total_count": self.total_count,
            "succeeded_count": self.succeeded_count,
            "failed_count": self.failed_count,
            "all_succeeded": self.all_succeeded,
            "items": tuple(item.summary() for item in self.items),
        }


@dataclass(slots=True, frozen=True)
class RTSCorrectionOutput:
    """Immutable description of one verified corrected FITS artifact."""

    application_result: RTSCorrectionApplicationResult
    output_path: Path
    sha256: str
    image_shape: tuple[int, int]
    pixel_dtype: str
    history_entries: tuple[str, ...]
    written: bool = True
    verified: bool = True

    def __post_init__(self) -> None:
        output_path = Path(self.output_path).expanduser().resolve()
        object.__setattr__(self, "output_path", output_path)

        if len(self.sha256) != 64:
            raise Step05Error("sha256 must contain 64 hexadecimal characters.")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise Step05Error("sha256 is not hexadecimal.") from exc

        if self.image_shape != self.application_result.plan.image_shape:
            raise Step05Error(
                "output image_shape does not match the correction plan."
            )
        if self.pixel_dtype != self.application_result.output_dtype:
            raise Step05Error(
                "output pixel_dtype does not match the application result."
            )
        if not self.history_entries:
            raise Step05Error("history_entries must not be empty.")
        if not self.written or not self.verified:
            raise Step05Error(
                "RTSCorrectionOutput requires a written and verified artifact."
            )

    @property
    def input_path(self) -> Path:
        """Return the source FITS path."""
        return self.application_result.plan.input_path

    @property
    def applied_count(self) -> int:
        """Return the number of corrected candidate pixels."""
        return self.application_result.applied_count

    @property
    def preserved_count(self) -> int:
        """Return the number of preserved candidate pixels."""
        return self.application_result.preserved_count

    def summary(self) -> dict[str, object]:
        """Return one deterministic JSON-serializable output summary."""
        return {
            "step05_version": __version__,
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "written": self.written,
            "verified": self.verified,
            "sha256": self.sha256,
            "image_shape": self.image_shape,
            "pixel_dtype": self.pixel_dtype,
            "applied_count": self.applied_count,
            "preserved_count": self.preserved_count,
            "history_entries": self.history_entries,
        }


def _normalized_path(path) -> Path:
    """Return one absolute non-strict path or raise a Step05Error."""
    try:
        return Path(path).expanduser().resolve(strict=False)
    except TypeError as exc:
        raise Step05Error("path must be path-like.") from exc


def _logical_fits_dtype_name(header) -> str:
    """Return the logical NumPy dtype represented by a FITS image header."""
    try:
        bitpix = int(header["BITPIX"])
        bscale = float(header.get("BSCALE", 1.0))
        bzero = float(header.get("BZERO", 0.0))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise Step05Error("input FITS has invalid image scaling metadata.") from exc

    if bscale == 1.0:
        unsigned_offsets = {
            16: 2**15,
            32: 2**31,
            64: 2**63,
        }
        if bitpix in unsigned_offsets and bzero == float(
            unsigned_offsets[bitpix]
        ):
            return f"uint{bitpix}"

    dtype_names = {
        8: "uint8",
        16: "int16",
        32: "int32",
        64: "int64",
        -32: "float32",
        -64: "float64",
    }
    try:
        return dtype_names[bitpix]
    except KeyError as exc:
        raise Step05Error(
            f"input FITS uses unsupported BITPIX value: {bitpix}."
        ) from exc


def _load_primary_image_metadata(path: Path) -> tuple[tuple[int, int], str]:
    """Read primary FITS shape and logical dtype without loading pixel data."""
    if not path.is_file():
        raise Step05Error(f"input FITS does not exist: {path}")

    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=True,
            do_not_scale_image_data=True,
        ) as hdul:
            header = hdul[0].header
            naxis = int(header.get("NAXIS", 0))
            if naxis == 0:
                raise Step05Error("input FITS primary HDU contains no image.")
            if naxis != 2:
                raise Step05Error(
                    "input FITS primary image must be two-dimensional."
                )

            width = int(header["NAXIS1"])
            height = int(header["NAXIS2"])
            if width <= 0 or height <= 0:
                raise Step05Error(
                    "input FITS primary image dimensions must be positive."
                )

            image_shape = (height, width)
            pixel_dtype = _logical_fits_dtype_name(header)
    except Step05Error:
        raise
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise Step05Error(f"Could not read input FITS '{path}': {exc}") from exc

    return image_shape, pixel_dtype


def _candidate_from_dictionary_row(
    row: RTSDictionaryRow,
) -> RTSCorrectionCandidate:
    """Convert one validated Step 04 row into one Step 05 candidate."""
    return RTSCorrectionCandidate(
        row=row.row,
        column=row.column,
        lower_state_center=row.lower_state_center,
        upper_state_center=row.upper_state_center,
        state_separation=row.state_separation,
        transition_count=row.transition_count,
        two_state_score=row.two_state_score,
    )


def prepare_rts_correction(
    metadata_path,
    input_path,
    *,
    fingerprint_json_path=None,
    comparison_json_path=None,
) -> RTSCorrectionPlan:
    """Validate Step 04 artifacts and prepare a no-modification correction plan.

    The complete canonical Step 04 artifact set is validated first.  The
    primary FITS image must then be two-dimensional and have exactly the image
    shape recorded by the dictionary metadata.  No pixel values are changed.
    """
    resolved_metadata_path = _normalized_path(metadata_path)
    resolved_input_path = _normalized_path(input_path)

    try:
        validation = validate_rts_dictionary_artifacts(
            resolved_metadata_path,
            fingerprint_json_path=fingerprint_json_path,
            comparison_json_path=comparison_json_path,
        )
    except Step04Error as exc:
        raise Step05Error(
            f"Could not validate Step 04 dictionary artifacts: {exc}"
        ) from exc

    image_shape, pixel_dtype = _load_primary_image_metadata(
        resolved_input_path
    )
    expected_shape = validation.artifacts.metadata.image_shape
    if image_shape != expected_shape:
        raise Step05Error(
            "input FITS image_shape does not match dictionary metadata: "
            f"expected {expected_shape}, got {image_shape}."
        )

    candidates = tuple(
        _candidate_from_dictionary_row(row)
        for row in validation.artifacts.dictionary.rows
    )

    height, width = image_shape
    for candidate in candidates:
        if not (0 <= candidate.row < height):
            raise Step05Error(
                f"dictionary row coordinate is outside input image: "
                f"{candidate.coordinate}."
            )
        if not (0 <= candidate.column < width):
            raise Step05Error(
                f"dictionary column coordinate is outside input image: "
                f"{candidate.coordinate}."
            )

    return RTSCorrectionPlan(
        input_path=resolved_input_path,
        metadata_path=validation.metadata_path,
        dictionary_csv_path=validation.dictionary_csv_path,
        image_shape=image_shape,
        pixel_dtype=pixel_dtype,
        dataset=validation.artifacts.dataset,
        candidates=candidates,
        artifact_validation=validation,
    )


def _validated_state_tolerance_fraction(value) -> float:
    """Return a finite tolerance fraction in the supported interval."""
    if isinstance(value, bool):
        raise Step05Error(
            "state_tolerance_fraction must be a real number."
        )
    try:
        fraction = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Step05Error(
            "state_tolerance_fraction must be a real number."
        ) from exc

    if not np.isfinite(fraction):
        raise Step05Error(
            "state_tolerance_fraction must be finite."
        )
    if not 0.0 <= fraction < 0.5:
        raise Step05Error(
            "state_tolerance_fraction must satisfy 0 <= value < 0.5."
        )
    return fraction


def _classify_candidate_value(
    candidate: RTSCorrectionCandidate,
    pixel_value: float,
    tolerance_fraction: float,
) -> RTSCandidateClassification:
    """Classify one finite current pixel value deterministically."""
    if not np.isfinite(pixel_value):
        raise Step05Error(
            f"candidate pixel value is not finite at {candidate.coordinate}."
        )

    tolerance = candidate.state_separation * tolerance_fraction
    distance_to_lower = abs(pixel_value - candidate.lower_state_center)
    distance_to_upper = abs(pixel_value - candidate.upper_state_center)

    if distance_to_lower <= tolerance:
        state = RTSCandidateState.LOWER
    elif distance_to_upper <= tolerance:
        state = RTSCandidateState.UPPER
    elif (
        candidate.lower_state_center
        < pixel_value
        < candidate.upper_state_center
    ):
        state = RTSCandidateState.MIDPOINT
    else:
        state = RTSCandidateState.OUTSIDE

    return RTSCandidateClassification(
        candidate=candidate,
        pixel_value=float(pixel_value),
        state=state,
        tolerance=float(tolerance),
        distance_to_lower=float(distance_to_lower),
        distance_to_upper=float(distance_to_upper),
    )


def classify_rts_correction_candidates(
    plan: RTSCorrectionPlan,
    *,
    state_tolerance_fraction=0.25,
) -> RTSCorrectionClassificationResult:
    """Read and classify every planned RTS candidate without modifying FITS.

    A value within ``state_separation * state_tolerance_fraction`` of a state
    center is classified as LOWER or UPPER.  A value strictly between the two
    state centers but outside both tolerance bands is MIDPOINT.  Any other
    finite value is OUTSIDE.
    """
    if not isinstance(plan, RTSCorrectionPlan):
        raise Step05Error("plan must be an RTSCorrectionPlan.")

    fraction = _validated_state_tolerance_fraction(
        state_tolerance_fraction
    )

    try:
        with fits.open(
            plan.input_path,
            mode="readonly",
            memmap=False,
            do_not_scale_image_data=False,
            uint=True,
        ) as hdul:
            data = hdul[0].data
            if data is None or data.ndim != 2:
                raise Step05Error(
                    "input FITS primary image is no longer two-dimensional."
                )
            current_shape = tuple(int(value) for value in data.shape)
            if current_shape != plan.image_shape:
                raise Step05Error(
                    "input FITS image_shape changed after plan preparation: "
                    f"expected {plan.image_shape}, got {current_shape}."
                )

            classifications = tuple(
                _classify_candidate_value(
                    candidate,
                    float(data[candidate.row, candidate.column]),
                    fraction,
                )
                for candidate in plan.candidates
            )
    except Step05Error:
        raise
    except (OSError, ValueError, TypeError, IndexError) as exc:
        raise Step05Error(
            f"Could not classify RTS candidates in '{plan.input_path}': {exc}"
        ) from exc

    return RTSCorrectionClassificationResult(
        plan=plan,
        state_tolerance_fraction=fraction,
        classifications=classifications,
    )


def _build_rts_correction_decision(
    classification: RTSCandidateClassification,
) -> RTSCorrectionDecision:
    """Build one deterministic correction decision."""
    state = classification.state

    if state is RTSCandidateState.LOWER:
        target_value = classification.candidate.lower_state_center
        reason = RTSCorrectionDecisionReason.LOWER_STATE
        is_correctable = True
    elif state is RTSCandidateState.UPPER:
        target_value = classification.candidate.upper_state_center
        reason = RTSCorrectionDecisionReason.UPPER_STATE
        is_correctable = True
    elif state is RTSCandidateState.MIDPOINT:
        target_value = None
        reason = RTSCorrectionDecisionReason.MIDPOINT_UNCERTAIN
        is_correctable = False
    elif state is RTSCandidateState.OUTSIDE:
        target_value = None
        reason = RTSCorrectionDecisionReason.OUTSIDE_DICTIONARY
        is_correctable = False
    else:
        raise Step05Error(
            f"unsupported RTS candidate state: {state!r}."
        )

    correction_value = (
        float(target_value - classification.pixel_value)
        if target_value is not None
        else None
    )

    return RTSCorrectionDecision(
        classification=classification,
        target_value=(
            float(target_value) if target_value is not None else None
        ),
        correction_value=correction_value,
        is_correctable=is_correctable,
        reason=reason,
    )


def build_rts_correction_decisions(
    classification_result: RTSCorrectionClassificationResult,
) -> RTSCorrectionDecisionResult:
    """Build deterministic correction decisions without modifying the image."""
    if not isinstance(
        classification_result,
        RTSCorrectionClassificationResult,
    ):
        raise Step05Error(
            "classification_result must be an "
            "RTSCorrectionClassificationResult."
        )

    decisions = tuple(
        _build_rts_correction_decision(classification)
        for classification in classification_result.classifications
    )

    if len(decisions) != classification_result.candidate_count:
        raise Step05Error(
            "correction decision count does not match candidate count."
        )

    return RTSCorrectionDecisionResult(
        classification_result=classification_result,
        decisions=decisions,
    )


def _coerce_target_value_for_dtype(
    target_value: float,
    dtype: np.dtype,
    coordinate: tuple[int, int],
):
    """Convert one target value safely for the output dtype."""
    if not np.isfinite(target_value):
        raise Step05Error(
            f"non-finite correction target at {coordinate}."
        )

    if np.issubdtype(dtype, np.integer):
        rounded = float(np.rint(target_value))
        info = np.iinfo(dtype)
        if rounded < info.min or rounded > info.max:
            raise Step05Error(
                f"correction target {target_value} at {coordinate} "
                f"is outside dtype range [{info.min}, {info.max}]."
            )
        return dtype.type(rounded)

    if np.issubdtype(dtype, np.floating):
        cast_value = dtype.type(target_value)
        if not np.isfinite(cast_value):
            raise Step05Error(
                f"correction target {target_value} at {coordinate} "
                f"cannot be represented by dtype {dtype.name}."
            )
        return cast_value

    raise Step05Error(
        f"unsupported correction image dtype: {dtype.name}."
    )


def apply_rts_correction_in_memory(
    decision_result: RTSCorrectionDecisionResult,
) -> RTSCorrectionApplicationResult:
    """Apply correctable decisions to an independent in-memory image copy.

    The source FITS file is opened read-only.  LOWER and UPPER decisions are
    applied to a newly allocated array.  Rejected MIDPOINT and OUTSIDE
    decisions preserve their original pixel values.
    """
    if not isinstance(decision_result, RTSCorrectionDecisionResult):
        raise Step05Error(
            "decision_result must be an RTSCorrectionDecisionResult."
        )

    plan = decision_result.plan
    try:
        with fits.open(
            plan.input_path,
            mode="readonly",
            memmap=False,
            do_not_scale_image_data=False,
            uint=True,
        ) as hdul:
            data = hdul[0].data
            if data is None or data.ndim != 2:
                raise Step05Error(
                    "input FITS primary image is no longer two-dimensional."
                )

            current_shape = tuple(int(value) for value in data.shape)
            if current_shape != plan.image_shape:
                raise Step05Error(
                    "input FITS image_shape changed after plan preparation: "
                    f"expected {plan.image_shape}, got {current_shape}."
                )

            corrected = np.array(data, copy=True)
    except Step05Error:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise Step05Error(
            f"Could not read correction input FITS '{plan.input_path}': {exc}"
        ) from exc

    output_dtype = corrected.dtype
    applied_count = 0
    preserved_count = 0

    for decision in decision_result.decisions:
        row, column = decision.coordinate
        current_value = float(corrected[row, column])

        if not np.isclose(
            current_value,
            decision.current_value,
            rtol=0.0,
            atol=0.0,
            equal_nan=False,
        ):
            raise Step05Error(
                "input FITS candidate value changed after classification at "
                f"{decision.coordinate}: expected {decision.current_value}, "
                f"got {current_value}."
            )

        if decision.is_correctable:
            if decision.target_value is None:
                raise Step05Error(
                    f"correctable decision has no target at {decision.coordinate}."
                )
            corrected[row, column] = _coerce_target_value_for_dtype(
                decision.target_value,
                output_dtype,
                decision.coordinate,
            )
            applied_count += 1
        else:
            preserved_count += 1

    return RTSCorrectionApplicationResult(
        decision_result=decision_result,
        corrected_image=corrected,
        applied_count=applied_count,
        preserved_count=preserved_count,
        output_dtype=output_dtype.name,
    )


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Step05Error(
            f"Could not calculate SHA256 for '{path}': {exc}"
        ) from exc
    return digest.hexdigest()


def _read_primary_header(path: Path) -> fits.Header:
    """Read and copy the source primary FITS header."""
    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=False,
            do_not_scale_image_data=False,
            uint=True,
        ) as hdul:
            if hdul[0].data is None or hdul[0].data.ndim != 2:
                raise Step05Error(
                    "input FITS primary image is no longer two-dimensional."
                )
            return hdul[0].header.copy()
    except Step05Error:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise Step05Error(
            f"Could not read input FITS header '{path}': {exc}"
        ) from exc


def _verify_written_rts_fits(
    output_path: Path,
    application_result: RTSCorrectionApplicationResult,
) -> None:
    """Verify shape, dtype, and pixel values of a written FITS artifact."""
    expected = application_result.corrected_image
    try:
        with fits.open(
            output_path,
            mode="readonly",
            memmap=False,
            do_not_scale_image_data=False,
            uint=True,
        ) as hdul:
            data = hdul[0].data
            if data is None or data.ndim != 2:
                raise Step05Error(
                    "written FITS primary image is not two-dimensional."
                )
            actual = np.asarray(data)
    except Step05Error:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise Step05Error(
            f"Could not verify corrected FITS '{output_path}': {exc}"
        ) from exc

    actual_shape = tuple(int(value) for value in actual.shape)
    if actual_shape != application_result.plan.image_shape:
        raise Step05Error(
            "written FITS shape verification failed: "
            f"expected {application_result.plan.image_shape}, "
            f"got {actual_shape}."
        )

    if actual.dtype.name != application_result.output_dtype:
        raise Step05Error(
            "written FITS dtype verification failed: "
            f"expected {application_result.output_dtype}, "
            f"got {actual.dtype.name}."
        )

    if not np.array_equal(actual, expected, equal_nan=True):
        raise Step05Error(
            "written FITS pixel verification failed."
        )


def write_rts_corrected_fits(
    application_result: RTSCorrectionApplicationResult,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> RTSCorrectionOutput:
    """Write and verify a corrected FITS artifact.

    The source FITS file is never modified.  By default, an existing output
    path is rejected.  The source primary header is copied, RTS provenance is
    appended with HISTORY cards, and the written image is re-read for exact
    shape, dtype, and pixel verification.
    """
    if not isinstance(application_result, RTSCorrectionApplicationResult):
        raise Step05Error(
            "application_result must be an "
            "RTSCorrectionApplicationResult."
        )
    if not isinstance(overwrite, bool):
        raise Step05Error("overwrite must be a bool.")

    destination = Path(output_path).expanduser().resolve()
    source = application_result.plan.input_path.expanduser().resolve()

    if destination == source:
        raise Step05Error("output_path must not overwrite the input FITS.")

    if destination.exists() and not overwrite:
        raise Step05Error(
            f"output FITS already exists: '{destination}'."
        )

    if not destination.parent.exists():
        raise Step05Error(
            f"output directory does not exist: '{destination.parent}'."
        )
    if not destination.parent.is_dir():
        raise Step05Error(
            f"output parent is not a directory: '{destination.parent}'."
        )

    header = _read_primary_header(source)
    history_entries = (
        "RTS correction applied",
        f"RTS Framework {__version__}",
        f"RTS corrected pixels = {application_result.applied_count}",
        f"RTS preserved candidates = {application_result.preserved_count}",
    )
    for entry in history_entries:
        header.add_history(entry)

    hdu = fits.PrimaryHDU(
        data=np.array(application_result.corrected_image, copy=True),
        header=header,
    )

    try:
        hdu.writeto(destination, overwrite=overwrite, checksum=True)
    except (OSError, ValueError, TypeError) as exc:
        raise Step05Error(
            f"Could not write corrected FITS '{destination}': {exc}"
        ) from exc

    try:
        _verify_written_rts_fits(destination, application_result)
        digest = _sha256_file(destination)
    except Exception:
        # Avoid leaving an unverified artifact from this call.
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return RTSCorrectionOutput(
        application_result=application_result,
        output_path=destination,
        sha256=digest,
        image_shape=application_result.plan.image_shape,
        pixel_dtype=application_result.output_dtype,
        history_entries=history_entries,
    )




def run_rts_correction_batch(
    metadata_path: str | Path,
    input_paths: list[str | Path] | tuple[str | Path, ...],
    output_directory: str | Path,
    *,
    output_suffix: str = "_rts_corrected",
    state_tolerance_fraction: float = 0.25,
    overwrite: bool = False,
    continue_on_error: bool = False,
) -> RTSCorrectionBatchResult:
    """Correct multiple FITS inputs using one RTS dictionary metadata file.

    Output filenames preserve each input stem and suffix:
        image.fit -> image_rts_corrected.fit

    When continue_on_error is False, the first failure raises Step05Error.
    When True, failures are recorded and subsequent inputs continue.
    """
    metadata = Path(metadata_path).expanduser().resolve()
    destination_dir = Path(output_directory).expanduser().resolve()

    if not isinstance(input_paths, (list, tuple)) or not input_paths:
        raise Step05Error("input_paths must be a non-empty list or tuple.")
    if not isinstance(output_suffix, str) or not output_suffix:
        raise Step05Error("output_suffix must be a non-empty string.")
    if "/" in output_suffix or "\\" in output_suffix:
        raise Step05Error("output_suffix must not contain path separators.")
    if not isinstance(overwrite, bool):
        raise Step05Error("overwrite must be a bool.")
    if not isinstance(continue_on_error, bool):
        raise Step05Error("continue_on_error must be a bool.")
    if not destination_dir.exists():
        raise Step05Error(
            f"output directory does not exist: '{destination_dir}'."
        )
    if not destination_dir.is_dir():
        raise Step05Error(
            f"output path is not a directory: '{destination_dir}'."
        )

    normalized_inputs = tuple(
        Path(path).expanduser().resolve() for path in input_paths
    )
    if len(set(normalized_inputs)) != len(normalized_inputs):
        raise Step05Error("input_paths contains duplicate paths.")

    items: list[RTSCorrectionBatchItem] = []
    used_outputs: set[Path] = set()

    for input_path in normalized_inputs:
        output_name = (
            f"{input_path.stem}{output_suffix}{input_path.suffix}"
        )
        output_path = destination_dir / output_name

        if output_path in used_outputs:
            raise Step05Error(
                f"multiple inputs map to the same output: '{output_path}'."
            )
        used_outputs.add(output_path)

        try:
            plan = prepare_rts_correction(metadata, input_path)
            classification = classify_rts_correction_candidates(
                plan,
                state_tolerance_fraction=state_tolerance_fraction,
            )
            decisions = build_rts_correction_decisions(classification)
            application = apply_rts_correction_in_memory(decisions)
            output = write_rts_corrected_fits(
                application,
                output_path,
                overwrite=overwrite,
            )
            items.append(
                RTSCorrectionBatchItem(
                    input_path=input_path,
                    output_path=output_path,
                    succeeded=True,
                    output=output,
                )
            )
        except Step05Error as exc:
            if not continue_on_error:
                raise
            items.append(
                RTSCorrectionBatchItem(
                    input_path=input_path,
                    output_path=output_path,
                    succeeded=False,
                    error=str(exc),
                )
            )

    return RTSCorrectionBatchResult(
        metadata_path=metadata,
        output_directory=destination_dir,
        items=tuple(items),
        continue_on_error=continue_on_error,
        overwrite=overwrite,
    )


def _build_rts_correction_cli_parser() -> argparse.ArgumentParser:
    """Build the Step 05 command-line parser."""
    parser = argparse.ArgumentParser(
        prog="step05_apply_rts_correction",
        description=(
            "Validate an RTS dictionary, classify candidate pixels, "
            "apply safe corrections in memory, and write a verified FITS."
        ),
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to the Step 04 RTS dictionary metadata JSON.",
    )
    parser.add_argument(
        "--input",
        required=True,
        dest="input_path",
        help="Path to the input FITS image.",
    )
    parser.add_argument(
        "--output",
        required=True,
        dest="output_path",
        help="Path for the corrected FITS image.",
    )
    parser.add_argument(
        "--state-tolerance-fraction",
        type=float,
        default=0.25,
        help=(
            "Fraction of state separation used for LOWER/UPPER "
            "classification tolerance (default: 0.25)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable success and error output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Write one machine-readable JSON object to stdout.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _rts_correction_cli_summary(
    output: RTSCorrectionOutput,
    classification: RTSCorrectionClassificationResult,
) -> dict[str, object]:
    """Return the deterministic machine-readable CLI summary."""
    return {
        **output.summary(),
        "status": "OK",
        "exit_code": 0,
        "state_tolerance_fraction": (
            classification.state_tolerance_fraction
        ),
        "candidate_count": classification.candidate_count,
        "lower_count": classification.lower_count,
        "upper_count": classification.upper_count,
        "midpoint_count": classification.midpoint_count,
        "outside_count": classification.outside_count,
    }


def run_rts_correction_cli(
    argv: list[str] | tuple[str, ...] | None = None,
) -> int:
    """Run the complete Step 05 correction pipeline.

    Exit codes:
        0: success
        1: operational or validation failure
        2: argument parsing failure from argparse
    """
    parser = _build_rts_correction_cli_parser()
    args = parser.parse_args(argv)

    try:
        plan = prepare_rts_correction(args.metadata, args.input_path)
        classification = classify_rts_correction_candidates(
            plan,
            state_tolerance_fraction=args.state_tolerance_fraction,
        )
        decisions = build_rts_correction_decisions(classification)
        application = apply_rts_correction_in_memory(decisions)
        output = write_rts_corrected_fits(
            application,
            args.output_path,
            overwrite=args.overwrite,
        )
    except Step05Error as exc:
        if args.json_output:
            print(
                json.dumps(
                    {
                        "step05_version": __version__,
                        "status": "ERROR",
                        "exit_code": 1,
                        "error": str(exc),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        elif not args.quiet:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = _rts_correction_cli_summary(output, classification)

    if args.json_output:
        print(
            json.dumps(
                summary,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    elif not args.quiet:
        print("RTS correction completed")
        print(f"Input FITS     : {output.input_path}")
        print(f"Output FITS    : {output.output_path}")
        print(f"Candidates     : {classification.candidate_count}")
        print(f"Corrected      : {output.applied_count}")
        print(f"Preserved      : {output.preserved_count}")
        print(f"LOWER          : {classification.lower_count}")
        print(f"UPPER          : {classification.upper_count}")
        print(f"MIDPOINT       : {classification.midpoint_count}")
        print(f"OUTSIDE        : {classification.outside_count}")
        print(f"SHA256         : {output.sha256}")
        print("Verified       : True")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_rts_correction_cli())
