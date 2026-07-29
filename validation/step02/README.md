# Step02 Validation

## Overview

Step02 validates the generation of deterministic image statistics from
the normalized dataset manifest.

These statistics provide the quantitative foundation for subsequent RTS
candidate detection.

---

## Purpose

Step02 computes statistical quantities for every input frame and
produces a reproducible statistics table.

The objective of this validation is to ensure that identical input data
always produce identical statistical results.

---

## Scope

This validation covers:

- image loading;
- frame iteration;
- statistical calculations;
- metadata association;
- output formatting.

RTS classification and state estimation are outside the scope of this
step.

---

## Inputs

Step02 requires:

    normalized_manifest.csv

and the corresponding detector image files referenced by the manifest.

---

## Expected Output

The primary output is

    statistics.csv

This file becomes the input for Step03.

---

## Reference Artifact

The expected statistics table will be stored as

    fixtures/expected_statistics.csv

The reference artifact represents the deterministic output of Step02 for
a controlled validation dataset.

---

## Validation Procedure

The validation procedure is:

    normalized_manifest.csv
              +
         detector images
              │
              ▼
           Step02
              │
              ▼
      statistics.csv
              │
              ▼
   compare_statistics.py
              │
              ▼
         PASS / FAIL

The comparison verifies:

- identical row count;
- identical frame ordering;
- identical statistical quantities;
- identical metadata values.

---

## Acceptance Criteria

Validation succeeds only if the generated statistics table satisfies the
defined comparison criteria.

Any unexpected difference is treated as a regression until reviewed.

---

## Current Status

Current milestone:

- validation procedure defined.

Reference artifacts will be introduced in a future milestone.

---

## Future Scientific Validation

Future validation may compare statistical properties generated from
large detector datasets obtained with multiple CMOS sensors.

