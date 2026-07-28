# STEP01 – Dataset Preparation

## Purpose

Step01 normalizes the raw acquisition metadata into a
`FrameManifest` while preserving all information required for
scientific reproducibility.

The output of Step01 serves as the canonical input for all subsequent
pipeline stages.

---

## Design principles

Step01 follows three principles.

1. Preserve scientific equivalence with the legacy pipeline.
2. Never reorder acquired frames.
3. Preserve traceability back to the original manifest.

---

## Acquisition order

Frame order is preserved exactly as recorded in the input manifest.

Frames are never reordered by filename, timestamp, or any other
metadata.

Dataset grouping must not modify acquisition order.

---

## Manifest row numbering

`manifest_row` represents the physical line number of the record in the
source CSV manifest.

Because the CSV header occupies line 1,

```
manifest_row == 2
```

is the first valid data record.

The value is preserved throughout all processing so that every output
record can be traced back to the original manifest.

---

## Validation

Before processing, `FrameManifest.validate_structure()` performs
structural validation.

Current validation includes

- duplicate file paths
- negative exposure times
- inconsistent declared frame counts
- invalid `manifest_row`
- temperature-fraction warnings

Validation is intended to detect malformed manifests while preserving
scientifically valid datasets.

---

## Regression tests

Regression tests verify

- acquisition-order preservation
- zero-second exposure support
- manifest row numbering
- validation rules