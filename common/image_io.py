"""
Image input/output helpers for the RTS Framework.

This implementation provides source normalization, image-format detection,
and two-dimensional FITS image reading. RAW reading, shape inspection, and
validation will be added in later milestones.

Public API
----------
- ImageFormat
- ImageSource
- detect_format
- read_image
"""

from __future__ import annotations

__version__ = "0.3.0"

from enum import Enum
from pathlib import Path
from typing import TypeAlias

import numpy as np
from astropy.io import fits

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



def read_image(source: ImageSource) -> np.ndarray:
    """
    Read an image and return it as a two-dimensional NumPy array.

    Parameters
    ----------
    source
        A path string, ``Path``, or ``FrameRecord``.

    Returns
    -------
    numpy.ndarray
        A two-dimensional image array.

    Raises
    ------
    ImageIOError
        If the file does not exist, cannot be read, contains no image data, or
        cannot be reduced to exactly two dimensions.
    NotImplementedError
        If RAW input is requested. RAW support will be added later.
    """
    path = _normalize_source(source)

    if not path.is_file():
        raise ImageIOError(f"Image file does not exist: {path}")

    image_format = detect_format(path)

    if image_format is ImageFormat.FITS:
        image = _read_fits(path)
    else:
        raise NotImplementedError(
            f"RAW image reading is not implemented yet: {path}"
        )

    image = np.squeeze(np.asarray(image))

    if image.ndim != 2:
        raise ImageIOError(
            "Image must be two-dimensional after removing singleton axes: "
            f"path={path}, shape={image.shape}"
        )

    return image


def _read_fits(path: Path) -> np.ndarray:
    """Read image data from the first FITS HDU containing an array."""
    try:
        with fits.open(path, mode="readonly", memmap=False) as hdul:
            for hdu in hdul:
                if hdu.data is not None:
                    return np.asarray(hdu.data)
    except (OSError, ValueError, TypeError) as exc:
        raise ImageIOError(f"Unable to read FITS image: {path}: {exc}") from exc

    raise ImageIOError(f"FITS file contains no image data: {path}")

def _has_fits_signature(path: Path) -> bool:
    """Return whether the first FITS card begins with ``SIMPLE  =``."""
    try:
        with path.open("rb") as stream:
            first_card = stream.read(80)
    except OSError as exc:
        raise ImageIOError(f"Unable to inspect image file: {path}") from exc

    return first_card.startswith(b"SIMPLE  =")
