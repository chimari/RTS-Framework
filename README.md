# RTS Framework

Human-in-the-loop RTS analysis framework for scientific CMOS sensors.

## Migration policy

1. Import the currently validated scripts without changing behavior.
2. Preserve original filenames and versions under `legacy/steps/`.
3. Register every step in `config/pipeline_registry.yaml`.
4. Add smoke tests before refactoring.
5. Refactor into `src/rtsfw/` only after the legacy pipeline is reproducible.

## GitHub bootstrap

```bash
git init
git add .
git commit -m "chore: bootstrap RTS Framework repository"
git branch -M main
git remote add origin <GITHUB_REPOSITORY_URL>
git push -u origin main
```


# Planning Pipeline Framework

Frame Index
    │
    ▼
Raw Statistics
    │
    ▼
Candidate Extraction
    │
    ▼
Time Series
    │
    ▼
Histogram Analysis
    │
    ▼
State Assignment
    │
    ▼
Transition Statistics
    │
    ▼
RTS Classification
    │
    ├── Dictionary
    ├── QA Report
    └── (Future)
         Reviewer
              │
              ▼
        Learning Dataset
              │
              ▼
      Threshold Optimizer

## Step05 CLI example
python step05_apply_rts_correction.py \
    --metadata dictionary.metadata.json \
    --input-dir science \
    --pattern "*.fits" \
    --output-directory corrected \
    --workers 4 \
    --continue-on-error
