"""Step 05: prepare deterministic RTS correction plans.

Version 5.0.0 introduces the safe boundary between the validated Step 04
dictionary artifacts and later image-correction algorithms.  This release
does not modify pixel values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from steps.step04_prepare_rts_dictionary_analysis import (
    RTSDictionaryArtifactValidation,
    RTSDictionaryRow,
    Step04Error,
    validate_rts_dictionary_artifacts,
)

__version__ = "5.0.0"

__all__ = [
    "RTSCorrectionCandidate",
    "RTSCorrectionPlan",
    "Step05Error",
    "prepare_rts_correction",
]


class Step05Error(Exception):
    """Raised when Step 05 cannot prepare an RTS correction plan."""


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
