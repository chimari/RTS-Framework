from __future__ import annotations
import argparse
from pathlib import Path
from .config import DEFAULT_PROGRESS_EVERY


def add_common_arguments(parser: argparse.ArgumentParser, *, output_default: str | None = None) -> None:
    """Add arguments shared by Step03 and later pipeline stages."""
    roi = parser.add_argument_group("ROI / execution mode")
    roi.add_argument("--full", action="store_true", help="Process the full detector. Mutually exclusive with ROI coordinates.")
    roi.add_argument("--x0", type=int, default=None, help="ROI first x, inclusive.")
    roi.add_argument("--x1", type=int, default=None, help="ROI last x, exclusive.")
    roi.add_argument("--y0", type=int, default=None, help="ROI first y, inclusive.")
    roi.add_argument("--y1", type=int, default=None, help="ROI last y, exclusive.")
    roi.add_argument("--dev", action="store_true", help="Development mode; currently records mode only. Give explicit ROI coordinates for reproducibility.")

    io = parser.add_argument_group("common I/O")
    if output_default is not None:
        io.add_argument("--output-dir", type=Path, default=Path(output_default))
    io.add_argument("--overwrite", action="store_true")
    io.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    io.add_argument("--verbose", action="store_true")


def validate_common_arguments(args: argparse.Namespace) -> None:
    coords = [args.x0, args.x1, args.y0, args.y1]
    any_coord = any(v is not None for v in coords)
    all_coord = all(v is not None for v in coords)
    if args.full and any_coord:
        raise ValueError("--full cannot be combined with --x0/--x1/--y0/--y1")
    if any_coord and not all_coord:
        raise ValueError("Specify all four ROI coordinates: --x0 --x1 --y0 --y1")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
