# Step01 Validation

## Purpose

Step01 prepares the normalized manifest that serves as the input for all
subsequent processing steps.

This validation verifies that identical input manifests always produce
identical normalized manifests.

Its primary objective is to detect unintended changes in manifest
generation before they propagate to later stages of the pipeline.

---

## Scope

This validation covers:

- dataset ordering;
- acquisition ordering;
- manifest normalization;
- row numbering;
- metadata preservation;
- output formatting.

Scientific interpretation of detector data is outside the scope of
Step01 validation.

---

## Inputs

The validation uses a small repository-contained manifest.

Planned location:

    fixtures/input_manifest.csv

The fixture should include multiple datasets and multiple acquisition
orders so that ordering behavior can be verified.

---

## Expected Output

The primary output is

    normalized_manifest.csv

This file becomes the reference input for Step02.

---

## Reference Artifact

The expected normalized manifest will be stored as

    fixtures/expected_normalized_manifest.csv

This artifact defines the expected deterministic output of Step01.

It will be introduced in the next validation milestone.

---

## Validation Procedure

The validation procedure is:

    input_manifest.csv
            │
            ▼
        Step01
            │
            ▼
 normalized_manifest.csv
            │
            ▼
 compare_manifest.py
            │
            ▼
        PASS / FAIL

The comparison tool verifies:

- identical column order;
- identical row count;
- identical dataset order;
- identical acquisition order;
- identical field values.

---

## Acceptance Criteria

Validation succeeds only if the generated normalized manifest is
identical to the reference artifact.

Any difference is treated as a regression until intentionally reviewed.

---

## Current Status

Current milestone:

- validation framework established;
- validation procedure defined.

Reference fixtures will be added in the next milestone.

---

## Future Scientific Validation

Large production manifests, such as IMX811 or IMX455 datasets, may be
used for scientific validation outside the repository.

Those datasets are intended to verify robustness rather than deterministic
repository regression.
