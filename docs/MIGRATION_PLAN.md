# Migration plan

## A. Archival import
Copy the latest validated scripts unchanged into `legacy/steps/`.

For each script record:
- original filename and version
- responsibility
- inputs and outputs
- validation command
- last validated dataset
- known limitations

## B. Reproducibility
Run a small ROI through the complete available chain and compare shapes, dtypes,
row counts, validation flags, summary statistics, and hashes.

## C. Package migration
Move reusable code into `src/rtsfw/common/` while retaining compatible entry points.

## D. Reviewer
Add Learning Set and Human Reviewer after Step01-Step09 are registered.
