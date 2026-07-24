"""Step 04: prepare an RTS dictionary analysis plan.

Version 4.0.0 introduces the metadata-only planning boundary for RTS dictionary
generation. It validates a completed Step 03 bias-analysis plan and does not
read image pixels or choose an RTS detection algorithm.
"""

from __future__ import annotations

__version__ = "4.0.0"

from dataclasses import dataclass

from steps.step03_prepare_bias_analysis import BiasAnalysisPlan


__all__ = [
    "RTSDictionaryPlan",
    "Step04Error",
    "prepare_rts_dictionary_analysis",
]


class Step04Error(Exception):
    """Raised when Step 04 cannot prepare an RTS dictionary analysis."""


@dataclass(slots=True, frozen=True)
class RTSDictionaryPlan:
    """Immutable metadata plan for later RTS dictionary generation."""

    bias_plan: BiasAnalysisPlan
    dataset: str
    n_frames: int
    image_shape: tuple[int, int]
    minimum_frames: int

    def summary(self) -> str:
        """Return a deterministic human-readable plan summary."""
        height, width = self.image_shape
        return "\n".join(
            [
                "RTS Framework Step 04",
                "=====================",
                "Status         : READY",
                f"Dataset        : {self.dataset}",
                f"Frames         : {self.n_frames}",
                f"Minimum frames : {self.minimum_frames}",
                f"Shape          : {height}x{width}",
            ]
        )


def prepare_rts_dictionary_analysis(
    bias_plan: BiasAnalysisPlan,
    *,
    min_frames: int = 3,
) -> RTSDictionaryPlan:
    """Validate Step 03 metadata for later RTS dictionary generation.

    This function deliberately performs no FITS reads, image allocation,
    pixel-statistics calculation, threshold selection, or RTS classification.

    Parameters
    ----------
    bias_plan
        Bias-analysis plan returned by
        :func:`steps.step03_prepare_bias_analysis.prepare_bias_analysis`.
    min_frames
        Minimum frame count required at this planning stage. It must be an
        integer of at least three. More demanding algorithms may impose a
        larger requirement in later Step 04 milestones.

    Returns
    -------
    RTSDictionaryPlan
        Immutable metadata-only RTS dictionary analysis plan.

    Raises
    ------
    Step04Error
        If ``bias_plan`` is invalid, ``min_frames`` is invalid, or the selected
        dataset contains too few frames.
    """
    if not isinstance(bias_plan, BiasAnalysisPlan):
        raise Step04Error(
            "bias_plan must be a BiasAnalysisPlan returned by "
            "prepare_bias_analysis()."
        )

    minimum_frames = _validate_min_frames(min_frames)

    if bias_plan.n_frames < minimum_frames:
        raise Step04Error(
            f"Dataset {bias_plan.dataset!r} contains {bias_plan.n_frames} "
            f"frame(s); RTS dictionary analysis requires at least "
            f"{minimum_frames}."
        )

    return RTSDictionaryPlan(
        bias_plan=bias_plan,
        dataset=bias_plan.dataset,
        n_frames=bias_plan.n_frames,
        image_shape=bias_plan.image_shape,
        minimum_frames=minimum_frames,
    )


def _validate_min_frames(value: int) -> int:
    """Return a validated Step 04 minimum frame count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise Step04Error("min_frames must be an integer of at least 3.")
    if value < 3:
        raise Step04Error("min_frames must be an integer of at least 3.")
    return value
