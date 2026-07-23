from __future__ import annotations
from dataclasses import dataclass
import argparse

@dataclass(frozen=True)
class ROI:
    x0: int
    x1: int
    y0: int
    y1: int
    full_width: int
    full_height: int

    @classmethod
    def from_args(cls, args: argparse.Namespace, *, full_width: int, full_height: int) -> "ROI":
        if getattr(args, "full", False) or args.x0 is None:
            return cls(0, full_width, 0, full_height, full_width, full_height)
        roi = cls(args.x0, args.x1, args.y0, args.y1, full_width, full_height)
        roi.validate()
        return roi

    def validate(self) -> None:
        if not (0 <= self.x0 < self.x1 <= self.full_width):
            raise ValueError(f"Invalid x ROI [{self.x0}, {self.x1}) for width {self.full_width}")
        if not (0 <= self.y0 < self.y1 <= self.full_height):
            raise ValueError(f"Invalid y ROI [{self.y0}, {self.y1}) for height {self.full_height}")

    @property
    def width(self) -> int: return self.x1 - self.x0
    @property
    def height(self) -> int: return self.y1 - self.y0
    @property
    def shape(self) -> tuple[int, int]: return (self.height, self.width)
    @property
    def slice_x(self) -> slice: return slice(self.x0, self.x1)
    @property
    def slice_y(self) -> slice: return slice(self.y0, self.y1)
    @property
    def is_full(self) -> bool:
        return self.x0 == 0 and self.y0 == 0 and self.x1 == self.full_width and self.y1 == self.full_height

    def local_to_global(self, local_y, local_x): return local_y + self.y0, local_x + self.x0
    def global_to_local(self, global_y, global_x): return global_y - self.y0, global_x - self.x0
    def contains_global(self, y, x): return (self.y0 <= y < self.y1) and (self.x0 <= x < self.x1)
    def to_dict(self) -> dict:
        return {"x0": self.x0, "x1": self.x1, "y0": self.y0, "y1": self.y1,
                "width": self.width, "height": self.height, "is_full": self.is_full}
