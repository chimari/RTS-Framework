"""
Image input/output helpers for the RTS Framework.

This implementation provides source normalization, image-format detection,
two-dimensional FITS and headerless RAW image reading, image-shape inspection,
and image validation.

Headerless RAW input requires image geometry and pixel-layout metadata from a
FrameRecord.

Public API
----------
- ImageFormat
- ImageSource
- detect_format
- read_image
- get_image_shape
- validate_image
"""
from __future__ import annotations

__version__ = "0.6.1"

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

    image_format = detect_format(source)

    if image_format is ImageFormat.FITS:
        image = _read_fits(path)
    else:
        image = _read_raw(source, path)
    
    image = np.squeeze(np.asarray(image))

    if image.ndim != 2:
        raise ImageIOError(
            "Image must be two-dimensional after removing singleton axes: "
            f"path={path}, shape={image.shape}"
        )

    return image




def get_image_shape(source: ImageSource) -> tuple[int, int]:
    """
    Return image shape as ``(height, width)`` without loading FITS pixels.

    Singleton axes are removed before the dimensionality check, matching the
    behavior of :func:`read_image`. For example, ``(1, height, width)`` becomes
    ``(height, width)``.
    """
    path = _normalize_source(source)

    if not path.is_file():
        raise ImageIOError(f"Image file does not exist: {path}")

    image_format = detect_format(source)

    if image_format is ImageFormat.FITS:
        shape = _get_fits_shape(path)
    else:
        shape, _ = _get_raw_layout(source)

    squeezed_shape = tuple(dimension for dimension in shape if dimension != 1)

    if len(squeezed_shape) != 2:
        raise ImageIOError(
            "Image must be two-dimensional after removing singleton axes: "
            f"path={path}, shape={shape}"
        )

    return squeezed_shape

def validate_image(source: ImageSource) -> None:
    """
    Validate that an image can be read as a two-dimensional array.

    When ``source`` is a :class:`FrameRecord`, optional image metadata are also
    checked against the actual image:

    - ``image_width`` and ``image_height``
    - ``pixel_dtype``

    Parameters
    ----------
    source
        A path string, ``Path``, or ``FrameRecord``.

    Raises
    ------
    ImageIOError
        If the file is missing, unreadable, unsupported, not two-dimensional,
        or inconsistent with metadata stored in a ``FrameRecord``.
    """
    image = read_image(source)

    if not isinstance(source, FrameRecord):
        return

    if source.image_width is not None or source.image_height is not None:
        if source.image_width is None or source.image_height is None:
            raise ImageIOError(
                "FrameRecord image geometry is incomplete: "
                f"path={source.filepath}, "
                f"image_width={source.image_width}, "
                f"image_height={source.image_height}"
            )

        expected_shape = (source.image_height, source.image_width)
        if image.shape != expected_shape:
            raise ImageIOError(
                "Image shape does not match FrameRecord metadata: "
                f"path={source.filepath}, "
                f"expected={expected_shape}, actual={image.shape}"
            )

    if source.pixel_dtype is not None:
        try:
            expected_dtype = np.dtype(source.pixel_dtype)
        except TypeError as exc:
            raise ImageIOError(
                "FrameRecord pixel_dtype is not NumPy-compatible: "
                f"path={source.filepath}, pixel_dtype={source.pixel_dtype!r}"
            ) from exc

        if image.dtype != expected_dtype:
            raise ImageIOError(
                "Image dtype does not match FrameRecord metadata: "
                f"path={source.filepath}, "
                f"expected={expected_dtype}, actual={image.dtype}"
            )


def _get_raw_layout(
    source: ImageSource,
) -> tuple[tuple[int, int], np.dtype]:
    """
    Return RAW image shape and NumPy dtype from a FrameRecord.

    Headerless RAW files do not contain geometry or dtype metadata. Therefore,
    RAW input must be supplied as a FrameRecord containing image_width,
    image_height, pixel_dtype, and optionally byte_order.
    """
    if not isinstance(source, FrameRecord):
        path = _normalize_source(source)
        raise ImageIOError(
            "Headerless RAW input requires a FrameRecord containing "
            "image_width, image_height, pixel_dtype, and byte_order: "
            f"path={path}"
        )

    if source.image_width is None or source.image_height is None:
        raise ImageIOError(
            "FrameRecord RAW image geometry is incomplete: "
            f"path={source.filepath}, "
            f"image_width={source.image_width}, "
            f"image_height={source.image_height}"
        )

    if source.image_width <= 0 or source.image_height <= 0:
        raise ImageIOError(
            "FrameRecord RAW image dimensions must be positive: "
            f"path={source.filepath}, "
            f"image_width={source.image_width}, "
            f"image_height={source.image_height}"
        )

    if source.pixel_dtype is None:
        raise ImageIOError(
            "FrameRecord RAW pixel_dtype is missing: "
            f"path={source.filepath}"
        )

    try:
        dtype = np.dtype(source.pixel_dtype)
    except TypeError as exc:
        raise ImageIOError(
            "FrameRecord RAW pixel_dtype is not NumPy-compatible: "
            f"path={source.filepath}, pixel_dtype={source.pixel_dtype!r}"
        ) from exc

    if source.byte_order in (None, "native", "="):
        pass
    elif source.byte_order == "little":
        dtype = dtype.newbyteorder("<")
    elif source.byte_order == "big":
        dtype = dtype.newbyteorder(">")
    else:
        raise ImageIOError(
            "FrameRecord RAW byte_order is unsupported: "
            f"path={source.filepath}, byte_order={source.byte_order!r}"
        )

    shape = (source.image_height, source.image_width)
    return shape, dtype


def _read_raw(source: ImageSource, path: Path) -> np.ndarray:
    """Read a headerless RAW image using metadata stored in a FrameRecord."""
    shape, dtype = _get_raw_layout(source)

    expected_pixels = shape[0] * shape[1]
    expected_bytes = expected_pixels * dtype.itemsize

    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise ImageIOError(
            f"Unable to inspect RAW image file size: {path}: {exc}"
        ) from exc

    if actual_bytes != expected_bytes:
        raise ImageIOError(
            "RAW file size does not match FrameRecord metadata: "
            f"path={path}, expected_bytes={expected_bytes}, "
            f"actual_bytes={actual_bytes}, shape={shape}, dtype={dtype}"
        )

    try:
        image = np.fromfile(path, dtype=dtype, count=expected_pixels)
    except (OSError, ValueError) as exc:
        raise ImageIOError(
            f"Unable to read RAW image: {path}: {exc}"
        ) from exc

    if image.size != expected_pixels:
        raise ImageIOError(
            "RAW image contains an unexpected number of pixels: "
            f"path={path}, expected={expected_pixels}, actual={image.size}"
        )

    return image.reshape(shape)


def _get_fits_shape(path: Path) -> tuple[int, ...]:
    """Return the first non-empty FITS image shape without loading pixel data."""
    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=True,
            do_not_scale_image_data=True,
        ) as hdul:
            for hdu in hdul:
                shape = tuple(getattr(hdu, "shape", ()) or ())
                if shape:
                    return shape
    except (OSError, ValueError, TypeError) as exc:
        raise ImageIOError(
            f"Unable to inspect FITS image shape: {path}: {exc}"
        ) from exc

    raise ImageIOError(f"FITS file contains no image data: {path}")

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
