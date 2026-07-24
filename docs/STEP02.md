# RTS Framework Step 02

Step 02 consumes the normalized manifest produced by Step 01, validates
dataset-level consistency, exposes lazy image iteration, computes deterministic
per-frame statistics, and exports CSV and dataset-summary JSON products.

## Public API

- `prepare_frame_groups()`
- `iter_dataset_images()`
- `compute_dataset_statistics()`
- `write_statistics_csv()`
- `write_all_statistics_csv()`
- `build_statistics_summary()`
- `write_statistics_summary_json()`
- `write_all_statistics_summary_json()`
- `statistics_filename()`
- `statistics_summary_filename()`
- `DatasetGroup`
- `Step02Result`
- `FrameStatistics`
- `DatasetStatistics`
- `Step02Error`

The complete machine-readable public surface is available as
`steps.step02_prepare_frame_groups.__all__`.

## CLI

```bash
python -m steps.step02_prepare_frame_groups MANIFEST
```

CSV and JSON export:

```bash
python -m steps.step02_prepare_frame_groups MANIFEST     --statistics-dir output/statistics     --summary-dir output/summaries
```

Other options:

- `--frame-root DIRECTORY`
- `--quiet`
- `--version`
- `--help`

Exit status is `0` on success and `2` for command-line or Step 02 errors.

## Completion criterion

Step 02 is complete when all milestone integration tests through v2.6.0 and
`tests/test_step02_completion.py` pass.
