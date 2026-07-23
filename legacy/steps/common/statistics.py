from __future__ import annotations
import math
import numpy as np

def robust_std(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0: return math.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))

def sigma_clip_1d(values: np.ndarray, sigma: float = 5.0, max_iter: int = 5) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    good = np.isfinite(x)
    for _ in range(max_iter):
        cur = x[good]
        if cur.size == 0: break
        med = float(np.median(cur)); rs = robust_std(cur)
        if not np.isfinite(rs) or rs <= 0: break
        new_good = np.isfinite(x) & (np.abs(x - med) <= sigma * rs)
        if np.array_equal(new_good, good): break
        good = new_good
    return x[good]

def deterministic_sample_indices(total: int, n: int) -> np.ndarray:
    if total <= 0 or n <= 0: raise ValueError("total and n must be positive")
    return np.linspace(0, total - 1, num=min(total, n), dtype=np.int64)
