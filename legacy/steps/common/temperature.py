from __future__ import annotations
import math

def make_temperature_bins(datasets: list[dict], tolerance: float) -> list[dict]:
    if tolerance < 0: raise ValueError("temperature tolerance must be non-negative")
    bins: list[dict] = []
    for d in sorted(datasets, key=lambda x: x["temperature_mean_C"], reverse=True):
        best_i = None; best_distance = math.inf
        for i, b in enumerate(bins):
            distance = abs(d["temperature_mean_C"] - b["temperature_mean_C"])
            if distance <= tolerance and distance < best_distance: best_i, best_distance = i, distance
        if best_i is None: bins.append({"datasets": [d]})
        else: bins[best_i]["datasets"].append(d)
        for b in bins:
            n = sum(x["frame_count"] for x in b["datasets"])
            b["temperature_mean_C"] = sum(x["temperature_mean_C"] * x["frame_count"] for x in b["datasets"]) / n
    bins.sort(key=lambda b: b["temperature_mean_C"], reverse=True)
    for i, b in enumerate(bins):
        b.update(temperature_bin_index=i, temperature_bin=f"Tbin_{b['temperature_mean_C']:+06.2f}C",
                 frame_count=sum(x["frame_count"] for x in b["datasets"]), dataset_count=len(b["datasets"]),
                 temperature_min_C=min(x["temperature_min_C"] for x in b["datasets"]),
                 temperature_max_C=max(x["temperature_max_C"] for x in b["datasets"]))
    return bins
