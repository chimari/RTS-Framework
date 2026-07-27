from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.manifest import FrameManifest, FrameRecord


def make_frame(
    *,
    manifest_row: int = 2,
    dataset: str = "cold",
    frame_index: int = 0,
    n_frames: int = 1,
    exposure_s: float = 0.0,
    filename: str = "frame0001.fit",
    filepath: str | Path = "frame0001.fit",
) -> FrameRecord:
    """Create a minimal valid frame record for validation tests."""
    return FrameRecord(
        manifest_row=manifest_row,
        dataset=dataset,
        directory="cold",
        environment="cold",
        frame_index=frame_index,
        n_frames=n_frames,
        temperature_C=-12.0,
        temperature_start_C=-12.0,
        temperature_end_C=-12.0,
        temperature_fraction=0.0,
        exposure_s=exposure_s,
        filename=filename,
        filepath=Path(filepath),
    )


def issue_types(manifest: FrameManifest) -> set[str]:
    """Return all validation issue types."""
    validation = manifest.validate_structure()
    return {issue.issue_type for issue in validation.issues}


def test_valid_manifest_has_no_errors() -> None:
    manifest = FrameManifest([make_frame()])

    validation = manifest.validate_structure()

    assert validation.valid
    assert validation.error_count == 0
    assert validation.errors == ()


def test_negative_exposure_is_invalid() -> None:
    manifest = FrameManifest(
        [
            make_frame(exposure_s=-1.0),
        ]
    )

    validation = manifest.validate_structure()

    assert not validation.valid
    assert "invalid_exposure" in {
        issue.issue_type for issue in validation.errors
    }


def test_duplicate_filepath_is_invalid() -> None:
    manifest = FrameManifest(
        [
            make_frame(
                manifest_row=2,
                frame_index=0,
                n_frames=2,
                filename="frame0001.fit",
                filepath="same.fit",
            ),
            make_frame(
                manifest_row=3,
                frame_index=1,
                n_frames=2,
                filename="frame0002.fit",
                filepath="same.fit",
            ),
        ]
    )

    validation = manifest.validate_structure()

    assert not validation.valid
    assert "duplicate_filepath" in {
        issue.issue_type for issue in validation.errors
    }


def test_declared_frame_count_mismatch_is_invalid() -> None:
    manifest = FrameManifest(
        [
            make_frame(
                frame_index=0,
                n_frames=2,
            ),
        ]
    )

    validation = manifest.validate_structure()

    assert not validation.valid
    assert "declared_frame_count_mismatch" in {
        issue.issue_type for issue in validation.errors
    }


def test_warning_only_manifest_is_still_valid() -> None:
    manifest = FrameManifest(
        [
            make_frame(),
        ]
    )

    # warningになる値
    frame = manifest.frames[0]

    manifest = FrameManifest(
        [
            replace(
                frame,
                temperature_fraction=1.5,
            )
        ]
    )

    validation = manifest.validate_structure()

    assert validation.valid
    assert validation.error_count == 0
    assert validation.warning_count == 1
    assert "temperature_fraction_out_of_range" in {
        issue.issue_type
        for issue in validation.warnings
    }


def test_errors_and_warnings_are_separated() -> None:

    frame = replace(
        make_frame(),
        exposure_s=-1.0,
        temperature_fraction=1.5,
    )

    validation = FrameManifest([frame]).validate_structure()

    assert validation.error_count == 1
    assert validation.warning_count == 1

    assert {i.issue_type for i in validation.errors} == {
        "invalid_exposure",
    }

    assert {i.issue_type for i in validation.warnings} == {
        "temperature_fraction_out_of_range",
    }
