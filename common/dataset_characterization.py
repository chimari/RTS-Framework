"""Dataset-level characterization data products.

This module defines immutable data structures used to communicate
deterministic dataset characteristics between pipeline steps.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    
