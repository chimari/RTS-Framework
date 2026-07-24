"""Step 05: prepare deterministic RTS correction plans.

Version 5.1.0 adds deterministic classification of current candidate-pixel
values without modifying the input image.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from astropy.io import fits

from steps.step04_prepare_rts_dictionary_analysis import (
    RTSDictionaryArtifactValidation,
    RTSDictionaryRow,
    Step04Error,
    validate_rts_dictionary_artifacts,
)

__version__ = "5.1.0"

__all__ = [
    "RTSCandidateClassification",
    "RTSCandidateState",
    "RTSCorrectionCandidate",
    "RTSCorrectionClassificationResult",
    "RTSCorrectionPlan",
    "Step05Error",
    "classify_rts_correction_candidates",
    "prepare_rts_correction",
]


class Step05Error(Exception):
    """Raised when Step 05 cannot prepare an RTS correction plan."""



class RTSCandidateState(str, Enum):
    """Classification of one current RTS candidate-pixel value."""

    LOWER = "LOWER"
    UPPER = "UPPER"
    MIDPOINT = "MIDPOINT"
    OUTSIDE = "OUTSIDE"


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
