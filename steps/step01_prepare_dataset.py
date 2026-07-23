"""Step 01: load and validate an RTS Framework dataset manifest.

This module is the pipeline entry point.  It loads a CSV manifest, validates
its metadata structure, inspects every referenced image, and returns one
immutable result object.  No RTS detection or image statistics are performed.
"""

from __future__ import annotations

__version__ = "1.0.0"

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from common.image_io import ImageIOError, get_image_shape, validate_image
from common.manifest import FrameManifest, FrameRecord, ManifestError, ManifestValidation


ValidationMode = Literal["shape", "full"]
ProgressCallback = Callable[[int, int, FrameRecord], None]


class Step01Error(Exception):
    """Base exception raised when Step 01 cannot prepare a dataset."""


@dataclass(slots=True, frozen=True)
class ImageValidationIssue:
    """One image-file problem discovered during Step 01."""

    manifest_row: int
    dataset: str
    filepath: Path
    detail: str

    def format(self) -> str:
        return (
            f"row={self.manifest_row}, dataset={self.dataset!r}, "
            f"path={self.filepath}: {self.detail}"
        )


@dataclass(slots=True, frozen=True)
class Step01Result:
    """Immutable result returned by :func:`prepare_dataset`."""

    manifest: FrameManifest
    manifest_validation: ManifestValidation
    image_issues: tuple[ImageValidationIssue, ...]
    validation_mode: ValidationMode

    @property
    def valid(self) -> bool:
        """Return ``True`` when metadata and all referenced images are valid."""
        return self.manifest_validation.valid and not self.image_issues

    @property
    def n_checked_images(self) -> int:
        return self.manifest.n_frames

    def summary(self, *, max_issues: int | None = 20) -> str:
        """Return a human-readable Step 01 report."""
        lines = [
            "RTS Framework Step 01",
            "=====================",
            f"Status          : {'PASSED' if self.valid else 'FAILED'}",
            f"Validation mode : {self.validation_mode}",
            f"Frames checked  : {self.n_checked_images}",
            f"Datasets        : {self.manifest.n_datasets}",
            f"Manifest errors : {self.manifest_validation.error_count}",
            f"Manifest warns  : {self.manifest_validation.warning_count}",
            f"Image errors    : {len(self.image_issues)}",
        ]

        all_issue_lines = [
            issue.format() for issue in self.manifest_validation.issues
        ] + [issue.format() for issue in self.image_issues]

        if all_issue_lines:
            shown = all_issue_lines if max_issues is None else all_issue_lines[:max_issues]
            lines.extend(["", "Issues", "------", *shown])
            omitted = len(all_issue_lines) - len(shown)
            if omitted > 0:
                lines.append(f"... {omitted} additional issue(s) omitted")

        return "\n".join(lines)


def prepare_dataset(
    manifest_path: str | Path,
    *,
    frame_root: str | Path | None = None,
    validation_mode: ValidationMode = "shape",
    progress: ProgressCallback | None = None,
) -> Step01Result:
    """Load and validate one dataset manifest.

    Parameters
    ----------
    manifest_path
        CSV manifest to load.
    frame_root
        Optional root used to resolve relative ``filepath`` values.
    validation_mode
        ``"shape"`` inspects FITS dimensions without loading pixel arrays.
        ``"full"`` reads each image and also validates dtype metadata.
    progress
        Optional callback receiving ``(current, total, frame)`` before each
        image is checked.

    Returns
    -------
    Step01Result
        The loaded manifest and all validation findings.

    Raises
    ------
    Step01Error
        If the CSV cannot be parsed or ``validation_mode`` is unsupported.
    """
    if validation_mode not in ("shape", "full"):
        raise Step01Error(
            f"Unsupported validation mode: {validation_mode!r}. "
            "Expected 'shape' or 'full'."
        )

    try:
        manifest = FrameManifest.from_csv(manifest_path, frame_root=frame_root)
    except ManifestError as exc:
        raise Step01Error(f"Unable to load manifest: {exc}") from exc

    manifest_validation = manifest.validate_structure()
    image_issues: list[ImageValidationIssue] = []
    total = manifest.n_frames

    for current, frame in enumerate(manifest.frames, start=1):
        if progress is not None:
            progress(current, total, frame)

        try:
            if validation_mode == "full":
                validate_image(frame)
            else:
                _validate_shape(frame)
        except (ImageIOError, NotImplementedError) as exc:
            image_issues.append(
                ImageValidationIssue(
                    manifest_row=frame.manifest_row,
                    dataset=frame.dataset,
                    filepath=frame.filepath,
                    detail=str(exc),
                )
            )

    return Step01Result(
        manifest=manifest,
        manifest_validation=manifest_validation,
        image_issues=tuple(image_issues),
        validation_mode=validation_mode,
    )


def _validate_shape(frame: FrameRecord) -> None:
    """Validate readability, dimensionality, and optional geometry metadata."""
    actual_shape = get_image_shape(frame)

    if frame.image_width is None and frame.image_height is None:
        return

    if frame.image_width is None or frame.image_height is None:
        raise ImageIOError(
            "FrameRecord image geometry is incomplete: "
            f"path={frame.filepath}, image_width={frame.image_width}, "
            f"image_height={frame.image_height}"
        )

    expected_shape = (frame.image_height, frame.image_width)
    if actual_shape != expected_shape:
        raise ImageIOError(
            "Image shape does not match FrameRecord metadata: "
            f"path={frame.filepath}, expected={expected_shape}, "
            f"actual={actual_shape}"
        )
