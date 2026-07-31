Copy the scripts directory contents into the RTS-Framework repository.

Example:
python scripts/analyze_read_noise.py normalized_manifest.csv \
  --dataset bias_-12dec \
  --frame-root /path/to/data \
  --roi-size 2000 \
  --output-dir output/bias_-12dec

The script calls read_image(FrameRecord), preserving RAW geometry/dtype/byte-order metadata.

Version 1.1 additions:
- Explicit histogram display-range annotation.
- Full sampled-range logarithmic pair-difference histogram.
- Discrete pair-value level table and plot.
- Quantization-level spacing descriptors in summary CSV/JSON/report.

The normal pair-difference histogram displays P0.1..P99.9 only. This is a
display choice and does not truncate the values used for noise statistics.
