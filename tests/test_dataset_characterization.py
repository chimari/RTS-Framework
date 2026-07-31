"""Tests for DatasetCharacterization."""

from dataclasses import FrozenInstanceError

import pytest

from common.dataset_characterization import DatasetCharacterization


def make_characterization() -> DatasetCharacterization:
    """Return a representative characterization object for testing."""
    return DatasetCharacterization(
        dataset="cold_bias",
        n_frames=100,
        pair_noise_median_adu_rms=2.4,
        temporal_noise_median_adu_rms=2.5,
        frame_offset_sigma_adu=0.3,
        quantization_step_adu=1.0,
        saturated_pixel_fraction=0.001,
        finite_pixel_fraction=1.0,
    )


def test_dataset_characterization_stores_values() -> None:
    result = make_characterization()

    assert result.dataset == "cold_bias"
    assert result.n_frames == 100
    assert result.temporal_noise_median_adu_rms == pytest.approx(2.5)
    assert result.pair_noise_median_adu_rms == pytest.approx(2.4)
    assert result.frame_offset_sigma_adu == pytest.approx(0.3)
    assert result.quantization_step_adu == pytest.approx(1.0)
    assert result.saturated_pixel_fraction == pytest.approx(0.001)
    assert result.finite_pixel_fraction == pytest.approx(1.0)


def test_quantization_step_may_be_none() -> None:

    result = DatasetCharacterization(
        dataset="room_bias",
        n_frames=50,
        temporal_noise_median_adu_rms=3.0,
        pair_noise_median_adu_rms=2.9,
        frame_offset_sigma_adu=0.5,
        quantization_step_adu=None,
        saturated_pixel_fraction=0.0,
        finite_pixel_fraction=0.999,
    )

    assert result.quantization_step_adu is None


def test_dataset_characterization_is_immutable() -> None:
    result = make_characterization()

    with pytest.raises(FrozenInstanceError):
        result.n_frames = 200  # type: ignore[misc]
