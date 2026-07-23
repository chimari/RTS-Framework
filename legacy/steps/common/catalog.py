from __future__ import annotations
from pathlib import Path
import pandas as pd
from .roi import ROI

def load_candidate_catalog(path: Path, roi: ROI | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"x", "y"}.issubset(df.columns): raise KeyError("Candidate catalog requires x and y columns")
    if roi is not None:
        df = df[(df.x >= roi.x0) & (df.x < roi.x1) & (df.y >= roi.y0) & (df.y < roi.y1)].copy()
    return df.reset_index(drop=True)
