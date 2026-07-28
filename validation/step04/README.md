# Step04 Validation

## Overview

Step04 validates RTS state estimation for every candidate pixel
identified by Step03.

This step produces the RTS dictionary, which is the primary scientific
artifact of RTS-Framework.

---

## Purpose

The objective of Step04 is to estimate the discrete RTS states and
associated parameters for each candidate pixel in a reproducible
manner.

Validation ensures that state estimation remains scientifically
consistent and that algorithmic changes are intentional and
well-documented.

---

## Scope

This validation covers:

- RTS state estimation;
- state center estimation;
- transition statistics;
- parameter calculation;
- dictionary generation;
- output formatting.

Image correction and performance evaluation are outside the scope of
this step.

---

## Inputs

Step04 requires:

    candidate_list.csv

together with the detector image sequence referenced by the normalized
manifest.

---

## Expected Output

The primary output is

    rts_dictionary.csv

Typical dictionary entries include:

- pixel coordinates;
- estimated state centers;
- state amplitudes;
- transition counts;
- transition probabilities;
- quality metrics.

This dictionary becomes the primary input for Step05.

---

## Reference Artifact

The expected RTS dictionary will be stored as

    fixtures/expected_rts_dictionary.csv

Because state estimation algorithms may evolve, the reference artifact
must be updated only after scientific review.

---

## Validation Procedure

The validation procedure is:

      candidate_list.csv
              +
       detector images
              │
              ▼
           Step04
              │
              ▼
      rts_dictionary.csv
              │
              ▼
 compare_dictionary.py
              │
              ▼
         PASS / FAIL

The comparison verifies:

- identical candidate ordering;
- identical state count;
- identical state centers within defined tolerance;
- identical transition statistics within defined tolerance;
- identical output structure.

---

## Acceptance Criteria

Integer-valued quantities shall match exactly.

Floating-point quantities shall satisfy the tolerance defined by the
validation procedure.

Any change exceeding these criteria is treated as a regression until
reviewed.

---

## Current Status

Current milestone:

- validation procedure defined.

Reference artifacts and comparison utilities will be introduced in
future milestones.

---

## Future Scientific Validation

Future validation may compare dictionary quality using independently
validated detector datasets and simulated RTS data.

