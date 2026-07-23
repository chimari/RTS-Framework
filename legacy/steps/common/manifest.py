from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def resolve_geometry(df: pd.DataFrame, shape_arg, dtype_arg):
    if {"image_height", "image_width"}.issubset(df.columns):
        hs = pd.to_numeric(df["image_height"], errors="raise").unique(); ws = pd.to_numeric(df["image_width"], errors="raise").unique()
        if len(hs) != 1 or len(ws) != 1: raise ValueError("Manifest contains inconsistent image geometry")
        shape = (int(hs[0]), int(ws[0]))
    elif shape_arg is not None: shape = tuple(map(int, shape_arg))
    else: raise ValueError("Provide --shape because geometry is absent from manifest")
    if "pixel_dtype" in df.columns:
        ds = df["pixel_dtype"].astype(str).unique()
        if len(ds) != 1: raise ValueError("Manifest contains inconsistent pixel_dtype")
        dtype = np.dtype(ds[0])
    elif dtype_arg is not None: dtype = np.dtype(dtype_arg)
    else: raise ValueError("Provide --dtype because pixel_dtype is absent")
    return shape, dtype

def load_frame_manifest(path: Path, *, shape_arg=None, dtype_arg=None, require_temperature=True):
    path = path.expanduser().resolve(); df = pd.read_csv(path)
    required = ["dataset", "frame_index", "resolved_path", "file_exists", "file_size_ok"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise KeyError(f"Manifest missing columns: {missing}")
    if require_temperature and "temperature_C" not in df.columns: raise KeyError("Manifest needs temperature_C")
    sort_cols = ["manifest_row"] if "manifest_row" in df.columns else ["dataset", "frame_index"]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    if not df["file_exists"].astype(bool).all(): raise ValueError("Manifest contains missing files")
    if not df["file_size_ok"].astype(bool).all(): raise ValueError("Manifest contains invalid-size files")
    if df["resolved_path"].duplicated().any(): raise ValueError("Manifest contains duplicate paths")
    shape, dtype = resolve_geometry(df, shape_arg, dtype_arg)
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    for raw in df["resolved_path"].astype(str):
        p = Path(raw)
        if not p.is_file(): raise FileNotFoundError(p)
        if p.stat().st_size != expected: raise ValueError(f"Unexpected raw size: {p}")
    return df, shape, dtype

def build_datasets(df: pd.DataFrame) -> list[dict]:
    result = []
    for dataset_index, name in enumerate(dict.fromkeys(df["dataset"].astype(str))):
        g = df[df["dataset"].astype(str) == name].sort_values("frame_index", kind="stable")
        paths = [Path(str(x)) for x in g["resolved_path"]]
        if len(paths) < 2: continue
        temps = pd.to_numeric(g["temperature_C"], errors="coerce")
        if not np.isfinite(temps).any(): raise ValueError(f"No valid temperature for dataset {name}")
        result.append({"dataset_index": dataset_index, "dataset": name, "frame_count": len(paths),
                       "temperature_mean_C": float(temps.mean()), "temperature_min_C": float(temps.min()),
                       "temperature_max_C": float(temps.max()), "paths": paths})
    return result
