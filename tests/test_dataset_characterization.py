"""Tests for DatasetCharacterization and its JSON representation."""

from dataclasses import FrozenInstanceError
import json

import pytest

from common.dataset_characterization import (
    DatasetCharacterization,
    DatasetCharacterizationError,
    characterization_from_dict,
    characterization_to_dict,
    read_dataset_characterization,
    write_dataset_characterization,
)


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


def test_characterization_dict_round_trip() -> None:
    original = make_characterization()

    payload = characterization_to_dict(original)
    restored = characterization_from_dict(payload)

    assert payload["schema"] == (
        "rts-framework.dataset-characterization"
    )
    assert payload["schema_version"] == 1
    assert restored == original


def test_characterization_json_round_trip(tmp_path) -> None:
    original = make_characterization()
    path = tmp_path / "dataset_characterization.json"

    returned_path = write_dataset_characterization(original, path)
    restored = read_dataset_characterization(path)

    assert returned_path == path
    assert restored == original

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "cold_bias"
    assert payload["quantization_step_adu"] == pytest.approx(1.0)


def test_read_rejects_unknown_schema(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "schema": "other.schema",
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetCharacterizationError,
        match="Unsupported schema",
    ):
        read_dataset_characterization(path)


def test_write_rejects_invalid_fraction(tmp_path) -> None:
    invalid = DatasetCharacterization(
        dataset="bias",
        n_frames=10,
        pair_noise_median_adu_rms=2.0,
        temporal_noise_median_adu_rms=2.1,
        frame_offset_sigma_adu=0.1,
        quantization_step_adu=None,
        saturated_pixel_fraction=1.1,
        finite_pixel_fraction=1.0,
    )

    with pytest.raises(
        DatasetCharacterizationError,
        match="saturated_pixel_fraction",
    ):
        write_dataset_characterization(
            invalid,
            tmp_path / "invalid.json",
        )
