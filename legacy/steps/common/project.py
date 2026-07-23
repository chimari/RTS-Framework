from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
@dataclass(frozen=True)
class RTSProject:
    root: Path
    @classmethod
    def from_path(cls, root: str | Path = "."): return cls(Path(root).expanduser().resolve())
    def step_dir(self, step: int, name: str) -> Path: return self.root / f"{step:02d}_{name}"
