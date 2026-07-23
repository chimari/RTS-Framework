"""
Image input/output helpers for the RTS Framework.

This first implementation provides only source normalization and image-format
detection. Image reading, shape inspection, and validation will be added after
this API has been tested with real data.

Public API
----------
- ImageFormat
- ImageSource
- detect_format
"""

from __future__ import annotations

__version__ = "0.2.0-dev"

from enum import Enum
from pathlib import Path
from typing import TypeAlias

from .manifest import FrameRecord


ImageSource: TypeAlias = str | Path | FrameRecord


class ImageIOError(Exception):
    """Base exception for image-I/O related errors."""


class UnsupportedImageSourceError(ImageIOError, TypeError):
    """Raised when an object cannot be converted into an image path."""


class ImageFormat(Enum):
    """Image formats supported by the RTS Framework."""

    FITS = "fits"
    RAW = "raw"


_FITS_SUFFIXES = frozenset({".fit", ".fits", ".fts"})
_RAW_SUFFIXES = frozenset({".bin", ".raw", ".dat"})


def _normalize_source(source: ImageSource) -> Path:
    """
    Convert an image source into a :class:`pathlib.Path`.

    Parameters
    ----------
    source
        A path string, ``Path``, or ``FrameRecord``.

    Returns
    -------
    pathlib.Path
        The path represented by ``source``.

    Raises
    ------
    UnsupportedImageSourceError
        If ``source`` is not a supported object.
    """
    if isinstance(source, FrameRecord):
        return source.filepath

    if isinstance(source, Path):
        return source

    if isinstance(source, str):
        return Path(source)

    raise UnsupportedImageSourceError(
        "Unsupported image source type: "
        f"{type(source).__name__}. Expected str, Path, or FrameRecord."
    )


def detect_format(source: ImageSource) -> ImageFormat:
    """
    Detect whether an image is FITS or headerless RAW.

    Detection order
    ---------------
    1. A recognized filename suffix.
    2. The first FITS header card (``SIMPLE  =``), when the file exists.
    3. Fallback to RAW.

    Notes
    -----
    The fallback to RAW is intentional because headerless detector files often
    use project-specific filename suffixes. Detailed readability and size
    checks belong to ``validate_image()``, which will be implemented later.
    """
    path = _normalize_source(source)
    suffix = path.suffix.lower()

    if suffix in _FITS_SUFFIXES:
        return ImageFormat.FITS

    if suffix in _RAW_SUFFIXES:
        return ImageFormat.RAW

    if path.is_file() and _has_fits_signature(path):
        return ImageFormat.FITS

    return ImageFormat.RAW


def _has_fits_signature(path: Path) -> bool:
    """Return whether the first FITS card begins with ``SIMPLE  =``."""
    try:
        with path.open("rb") as stream:
            first_card = stream.read(80)
    except OSError as exc:
        raise ImageIOError(f"Unable to inspect image file: {path}") from exc

    return first_card.startswith(b"SIMPLE  =")
