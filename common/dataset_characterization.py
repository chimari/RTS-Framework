"""Dataset-level characterization data products.

This module defines immutable data structures and stable JSON I/O used
to communicate deterministic dataset characteristics between pipeline
steps.
"""

from __future__ import annotations

__version__ = "1.3.0-dev"

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_NAME = "rts-framework.dataset-characterization"
SCHEMA_VERSION = 1


class DatasetCharacterizationError(ValueError):
    """Raised when characterization data cannot be serialized or loaded."""


@dataclass(frozen=True, slots=True)
class DatasetCharacterization:
    """
    Immutable dataset-level characterization produced by Step 02.

    This object summarizes one immutable dataset and is intended to be
    consumed by later pipeline steps without recomputing statistics.
    """

    # dataset identity
    dataset: str
    n_frames: int

    # representative statistics
    pair_noise_median_adu_rms: float
    temporal_noise_median_adu_rms: float
    frame_offset_sigma_adu: float

    # detector characteristics
    quantization_step_adu: float | None

    # data quality
    saturated_pixel_fraction: float
    finite_pixel_fraction: float


def _validate_characterization(
    value: DatasetCharacterization,
) -> None:
    if not value.dataset.strip():
        raise DatasetCharacterizationError("dataset must not be empty.")
    if value.n_frames <= 0:
        raise DatasetCharacterizationError("n_frames must be positive.")

    finite_fields = {
        "pair_noise_median_adu_rms": value.pair_noise_median_adu_rms,
        "temporal_noise_median_adu_rms": (
            value.temporal_noise_median_adu_rms
        ),
        "frame_offset_sigma_adu": value.frame_offset_sigma_adu,
        "saturated_pixel_fraction": value.saturated_pixel_fraction,
        "finite_pixel_fraction": value.finite_pixel_fraction,
    }
    for name, number in finite_fields.items():
        if not math.isfinite(number):
            raise DatasetCharacterizationError(
                f"{name} must be finite."
            )

    if (
        value.quantization_step_adu is not None
        and not math.isfinite(value.quantization_step_adu)
    ):
        raise DatasetCharacterizationError(
            "quantization_step_adu must be finite or None."
        )

    for name, fraction in (
        ("saturated_pixel_fraction", value.saturated_pixel_fraction),
        ("finite_pixel_fraction", value.finite_pixel_fraction),
    ):
        if not 0.0 <= fraction <= 1.0:
            raise DatasetCharacterizationError(
                f"{name} must be between 0 and 1."
            )


def characterization_to_dict(
    value: DatasetCharacterization,
) -> dict[str, Any]:
    """Return a stable JSON-compatible representation."""
    _validate_characterization(value)
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        **asdict(value),
    }


def characterization_from_dict(
    payload: Mapping[str, Any],
) -> DatasetCharacterization:
    """Build and validate a characterization from decoded JSON data."""
    if payload.get("schema") != SCHEMA_NAME:
        raise DatasetCharacterizationError(
            f"Unsupported schema: {payload.get('schema')!r}"
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise DatasetCharacterizationError(
            "Unsupported schema_version: "
            f"{payload.get('schema_version')!r}"
        )

    field_names = {
        "dataset",
        "n_frames",
        "pair_noise_median_adu_rms",
        "temporal_noise_median_adu_rms",
        "frame_offset_sigma_adu",
        "quantization_step_adu",
        "saturated_pixel_fraction",
        "finite_pixel_fraction",
    }
    missing = sorted(field_names.difference(payload))
    if missing:
        raise DatasetCharacterizationError(
            f"Missing fields: {', '.join(missing)}"
        )

    try:
        result = DatasetCharacterization(
            dataset=str(payload["dataset"]),
            n_frames=int(payload["n_frames"]),
            pair_noise_median_adu_rms=float(
                payload["pair_noise_median_adu_rms"]
            ),
            temporal_noise_median_adu_rms=float(
                payload["temporal_noise_median_adu_rms"]
            ),
            frame_offset_sigma_adu=float(
                payload["frame_offset_sigma_adu"]
            ),
            quantization_step_adu=(
                None
                if payload["quantization_step_adu"] is None
                else float(payload["quantization_step_adu"])
            ),
            saturated_pixel_fraction=float(
                payload["saturated_pixel_fraction"]
            ),
            finite_pixel_fraction=float(
                payload["finite_pixel_fraction"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise DatasetCharacterizationError(
            f"Invalid characterization value: {exc}"
        ) from exc

    _validate_characterization(result)
    return result


def write_dataset_characterization(
    value: DatasetCharacterization,
    path: str | Path,
) -> Path:
    """Write one characterization as deterministic UTF-8 JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = characterization_to_dict(value)
    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def read_dataset_characterization(
    path: str | Path,
) -> DatasetCharacterization:
    """Read and validate one characterization JSON file."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetCharacterizationError(
            f"Cannot read characterization file: {source}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise DatasetCharacterizationError(
            f"Invalid JSON in characterization file: {source}"
        ) from exc

    if not isinstance(payload, dict):
        raise DatasetCharacterizationError(
            "Characterization JSON root must be an object."
        )
    return characterization_from_dict(payload)
