# Legacy RTS Pipeline

This directory contains the production version of the IMX811 RTS analysis pipeline
as of July 2026.

The purpose of this directory is to preserve the original working implementation
before refactoring into the RTS Framework.

## Pipeline

01_load_frame_index_v1_1.py

↓

02_build_raw_pixel_statistics_v1_1.py

↓

03_extract_temporal_candidates_by_temperature_bin.py

↓

04_extract_candidate_timeseries.py

↓

05_center_timeseries_by_dataset.py

↓

06_detect_histogram_states_by_temperature_bin_v6_2.py

↓

06b_validate_histograms.py

↓

07_assign_states_by_temperature_bin.py

↓

08_measure_state_transition_statistics_v8_0_2.py

↓

08_5_generate_visual_judgment_report_v8_5_0.py

↓

09_classify_rts_and_generate_dictionary_v9_1_0.py

↓

09_5_generate_rts_quality_report_v9_5_0.py

## Additional tools

- Step09_QA_Report_v1_0_0.py
  - Generates QA reports for manual inspection of RTS classification.

## Notes

- These files are preserved without modification.
- Future development will be carried out under `src/`.
- Common utility modules are located in `steps/common/`.