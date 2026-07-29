# RTS Framework Validation

## Purpose

The `validation/` directory defines the scientific validation framework
for RTS-Framework.

Automated tests verify that the software behaves as intended.
Validation has a different objective: it verifies that each processing
step produces reproducible, scientifically acceptable, and deterministic
results.

The validation framework is organized by processing step so that each
stage of the pipeline can define its own validation procedure,
reference artifacts, and acceptance criteria.

---

## Validation Philosophy

RTS-Framework distinguishes between software correctness and scientific
correctness.

Software correctness is verified through automated tests under
`tests/`.

Scientific correctness is verified through validation procedures
defined in this directory.

Keeping these two responsibilities separate makes the framework easier
to maintain, review, and extend.

---

## Validation Levels

Validation is organized into four levels.

### Level 0 — Software Tests

Level 0 consists of unit tests and integration tests.

These tests verify:

- software interfaces;
- algorithms;
- error handling;
- command-line behavior;
- interactions between modules.

All Level 0 tests are maintained under the `tests/` directory.

### Level 1 — Reference Artifact Validation

A processing step is executed using a small controlled dataset.

The generated output is compared with a version-controlled reference
artifact.

Typical reference artifacts include:

- normalized manifests;
- statistics tables;
- metadata files;
- RTS dictionaries;
- corrected images;
- evaluation summaries.

Level 1 validation detects unintended changes in deterministic output.

### Level 2 — Scientific Validation

Framework outputs are compared with independently validated reference
results.

Depending on the processing step, validation may compare:

- numerical agreement;
- classification consistency;
- statistical properties;
- scientific conclusions.

Exact byte-for-byte agreement is not always required.

Acceptance criteria must therefore be documented individually for each
processing step.

### Level 3 — End-to-End Validation

The complete processing pipeline is executed using a representative
detector dataset.

Intermediate products and final scientific results are evaluated
together.

Level 3 validation is intended primarily for release qualification
rather than routine development.

---

## Reference Artifacts

A reference artifact is a version-controlled output that defines the
expected result of a validation case.

A reference artifact should:

- originate from documented input data;
- be reproducible;
- have deterministic formatting whenever possible;
- have a documented comparison procedure;
- have clearly defined acceptance criteria.

Reference artifacts are reviewed scientific products.
They should never be updated simply because a validation failed.

---

## Updating Reference Artifacts

A reference artifact may be replaced only when:

- the software behavior has intentionally changed;
- the reason for the change is documented;
- the updated artifact has been scientifically reviewed;
- related documentation has been updated;
- the modification is isolated in a dedicated commit.

Whenever practical, reviewers should examine both the software changes
and the resulting artifact changes.

---

## Repository Organization

Each processing step has its own validation directory.

Typical contents include:

    README.md
    fixtures/
    reference/
    reports/

Not every directory exists initially.
Additional material is introduced only when required by the
corresponding milestone.

Large detector datasets should normally remain outside the Git
repository.

Only small fixtures and deterministic reference artifacts are expected
to be stored within the repository.

---

## Current Status

The validation framework is being introduced incrementally.

The first milestone establishes:

- the validation directory structure;
- validation policies;
- step-specific validation documentation.

Reference artifacts and regression datasets will be added in subsequent
milestones.

---

## Future Extensions

Future milestones will introduce:

- repository-contained validation fixtures;
- automated regression validation;
- detector-specific scientific validation;
- end-to-end release validation.

The overall objective is to ensure that RTS-Framework remains
scientifically reproducible while evolving as a long-term software
project.
