#!/usr/bin/env python3
"""Standalone CLI for robust read-noise characterization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.read_noise_analysis import (
    ReadNoiseAnalysisError,
    ReadNoiseConfig,
    analyze_read_noise_dataset,
)
from steps import step02_prepare_frame_groups as step02


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--frame-level-correction",
        choices=("median", "mean", "none"),
        default="median",
    )
    parser.add_argument("--clip-sigma", type=float, default=5.0)
    parser.add_argument("--hist-bins", type=int, default=300)
    parser.add_argument("--roi-x", type=int)
    parser.add_argument("--roi-y", type=int)
    parser.add_argument("--roi-width", type=int)
    parser.add_argument("--roi-height", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = step02.prepare_frame_groups(
            args.manifest, frame_root=args.frame_root
        )
        if args.dataset is None:
            if result.n_datasets != 1:
                raise ReadNoiseAnalysisError(
                    "--dataset is required for a multi-dataset manifest."
                )
            group = result.groups[0]
        else:
            try:
                group = result.get_group(args.dataset)
            except KeyError as exc:
                raise ReadNoiseAnalysisError(
                    f"Dataset not found: {args.dataset!r}"
                ) from exc

        explicit = (
            args.roi_x, args.roi_y, args.roi_width, args.roi_height
        )
        if all(value is None for value in explicit):
            roi = None
        elif any(value is None for value in explicit):
            raise ReadNoiseAnalysisError(
                "Explicit ROI requires x, y, width, and height."
            )
        else:
            roi = explicit

        analyze_read_noise_dataset(
            group,
            ReadNoiseConfig(
                output_dir=args.output_dir,
                frame_level_correction=args.frame_level_correction,
                clip_sigma=args.clip_sigma,
                hist_bins=args.hist_bins,
                roi=roi,
            ),
            progress=lambda current, total, frame: print(
                f"[{current}/{total}] {frame.filepath.name}", flush=True
            ),
        )
    except (step02.Step02Error, ReadNoiseAnalysisError, OSError) as exc:
        print(f"read-noise analysis error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
