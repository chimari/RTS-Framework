# RTS-Framework Architecture

## Introduction

RTS-Framework is a scientific software framework for detecting,
characterizing, correcting, and evaluating Random Telegraph Signal
behavior in scientific detector images.

The framework is designed for long-term scientific use rather than for
a single detector, experiment, or analysis campaign.

Its architecture emphasizes reproducibility, maintainability,
detector-independent components, explicit processing stages, and
scientific validation.

---

## Design Goals

RTS-Framework is developed according to the following goals.

### Scientific Reproducibility

Identical documented inputs and configurations should produce
reproducible scientific artifacts.

Processing behavior, input assumptions, output formats, and acceptance
criteria should be documented sufficiently for independent review.

### Long-Term Maintainability

The software should remain understandable and modifiable after the
original implementation period.

Responsibilities are therefore divided among small modules, explicit
pipeline steps, tests, validation procedures, and design documents.

### Detector Independence

Reusable software components should not depend unnecessarily on a
specific detector model, camera system, directory layout, or observing
campaign.

Detector-specific behavior should be introduced through explicit
configuration or specialized processing layers.

### Extensibility

New detector formats, RTS detection algorithms, correction methods, and
evaluation metrics should be introducible without restructuring the
entire framework.

### Simplicity

The public interface and internal architecture should remain as small
and direct as practical.

New abstractions should be introduced only when they provide a clear
scientific or maintenance benefit.

---

## Design Principles

### Single Responsibility

Each processing step has one clearly defined responsibility.

A step should not silently perform work belonging to another stage of
the pipeline.

### Explicit Data Products

The output of each processing step is an explicit artifact that can be
inspected, stored, compared, and validated independently.

### Separation of Concerns

RTS-Framework separates:

- reusable libraries from pipeline orchestration;
- software tests from scientific validation;
- public APIs from internal implementation details;
- detector-independent logic from detector-specific configuration;
- permanent repository artifacts from large external datasets.

### Documentation First

Changes affecting architecture, public interfaces, validation policy,
file formats, or processing responsibilities should be documented
before implementation.

Small bug fixes and implementation-only changes do not require an
architecture update unless they alter documented behavior.

### Correctness Before Optimization

Scientific correctness and reproducibility take priority over
performance optimization.

Optimized implementations must preserve the documented behavior and
must be covered by tests and validation.

### Small Reviewable Changes

Development should proceed through small logical changes that can be
reviewed, tested, and validated independently.

---

## Overall Architecture

The framework is organized around a sequence of processing steps
supported by reusable libraries, command-line interfaces, tests,
validation cases, and documentation.

Conceptually, the architecture is:

    User or Automation
            |
            v
       CLI or Public API
            |
            v
      Processing Steps
            |
            +-------------------+
            |                   |
            v                   v
       Common Modules         Tools
            |                   |
            +---------+---------+
                      |
                      v
              Scientific Artifacts
                      |
                      v
              Validation Procedures

The command-line interface and public APIs provide controlled entry
points into the framework.

Processing steps coordinate scientific operations.

Common modules provide reusable detector-independent functionality.

Tools provide comparison, inspection, conversion, and development
support.

Validation procedures verify that generated artifacts remain
scientifically reproducible.

---

## Repository Structure

The repository is organized by responsibility.

    common/
        Reusable detector-independent libraries.

    steps/
        Processing-step implementations and orchestration.

    tests/
        Unit and integration tests for software behavior.

    validation/
        Scientific validation cases, fixtures, reference artifacts,
        and reports.

    docs/
        Architecture, development policy, terminology, APIs, and
        processing-step documentation.

    tools/
        Standalone utilities for comparison, inspection, conversion,
        and development support.

The exact contents may evolve, but the separation of responsibilities
should remain stable.

---

## Pipeline Architecture

RTS-Framework is divided into six primary processing stages.

    Raw detector data
            |
            v
    Step01: Dataset Preparation
            |
            v
    Step02: Statistical Characterization
            |
            v
    Step03: RTS Candidate Detection
            |
            v
    Step04: RTS State Estimation
            |
            v
    Step05: RTS Correction
            |
            v
    Step06: Scientific Evaluation

Each step produces an explicit artifact that serves as an interface to
the next stage.

### Step01: Dataset Preparation

Step01 validates and normalizes dataset registration information.

Primary artifact:

    normalized_manifest.csv

### Step02: Statistical Characterization

Step02 reads the registered image sequence and computes deterministic
statistical quantities required by downstream processing.

Primary artifact:

    statistics.csv

### Step03: RTS Candidate Detection

Step03 applies documented selection criteria to identify pixels that
require RTS state analysis.

Primary artifact:

    candidate_list.csv

### Step04: RTS State Estimation

Step04 estimates discrete RTS states, state centers, transitions, and
quality metrics for candidate pixels.

Primary artifact:

    rts_dictionary.csv

### Step05: RTS Correction

Step05 applies the RTS dictionary to detector images while preserving
unaffected image content and metadata.

Primary artifact:

    corrected_image.fits

### Step06: Scientific Evaluation

Step06 evaluates correction effectiveness and the preservation of
scientifically relevant detector characteristics.

Primary artifact:

    evaluation_report.json

The names and schemas of these artifacts may be refined as the
implementation evolves. Any change to a documented interface must be
reviewed as an architectural change.

---

## Software Layers

The framework follows a layered dependency model.

    Layer 4
    Scientific evaluation and release qualification

    Layer 3
    Processing steps and workflow orchestration

    Layer 2
    Shared scientific algorithms and data handling

    Layer 1
    Image I/O, manifests, formats, errors, and utilities

Lower layers should not depend on higher layers.

In particular:

- common modules should not import processing-step orchestration;
- reusable I/O should not contain detector-specific scientific policy;
- validation code should exercise production code rather than replace
  it;
- command-line interfaces should remain thin wrappers around reusable
  functions.

---

## Public API Policy

The public API of RTS-Framework should remain intentionally small.

Only documented interfaces are considered stable.

Internal functions, private modules, temporary helpers, and
implementation details may change without notice.

A public API should be introduced only when:

- it has a clear reusable purpose;
- its behavior can be documented;
- its inputs and outputs are stable enough to support external use;
- it can be tested independently;
- long-term compatibility is scientifically or operationally useful.

Public APIs should avoid exposing unnecessary internal state.

Where practical, command-line interfaces and pipeline steps should call
the same reusable functions that are made available through the public
API.

The authoritative list of supported public interfaces is maintained in:

    docs/API_REFERENCE.md

---

## Testing Strategy

Software correctness is verified under:

    tests/

Tests include unit tests and integration tests.

They verify matters such as:

- function behavior;
- type and shape validation;
- error handling;
- deterministic ordering;
- command-line behavior;
- module integration;
- serialization and parsing.

Tests should be fast enough for routine development and continuous
integration.

Tests do not by themselves establish scientific validity.

---

## Validation Strategy

Scientific reproducibility is verified under:

    validation/

Each processing step defines:

- its validation scope;
- controlled input fixtures;
- reference artifacts;
- comparison procedures;
- numerical tolerances;
- acceptance criteria.

The validation framework distinguishes four levels:

    Level 0
    Software verification through automated tests.

    Level 1
    Regression comparison against reference artifacts.

    Level 2
    Scientific validation against independently accepted results.

    Level 3
    End-to-end pipeline qualification using representative datasets.

Detailed validation policy is defined in:

    validation/README.md

and will be summarized in:

    docs/VALIDATION_GUIDE.md

---

## Reference Artifacts

Reference artifacts are version-controlled scientific products that
define expected results for controlled validation cases.

They are not ordinary test output.

A reference artifact should be updated only when:

- the behavior change is intentional;
- the scientific reason is documented;
- the changed artifact is reviewed;
- affected acceptance criteria and documentation are updated.

A failed validation is not, by itself, sufficient justification for
replacing a reference artifact.

Large production datasets should normally remain outside the Git
repository.

---

## Development Workflow

Architectural or interface-level changes should follow this sequence:

    Proposal
       |
       v
    Documentation
       |
       v
    Design Review
       |
       v
    Implementation
       |
       v
    Automated Tests
       |
       v
    Scientific Validation
       |
       v
    Code Review
       |
       v
    Merge and Release

Small implementation changes may use a shorter workflow when they do not
alter documented behavior.

Development should normally use:

- one issue or clearly defined objective;
- one focused branch;
- small logical commits;
- tests added with behavioral changes;
- validation updates when scientific artifacts change;
- documentation updates when contracts or interfaces change.

Detailed development procedures will be maintained in:

    docs/DEVELOPMENT_GUIDE.md

---

## Architecture Decisions

The architecture document describes the current design.

The reasons behind major design choices are recorded separately in:

    docs/DECISIONS.md

Examples include:

- separating tests from scientific validation;
- introducing version-controlled reference artifacts;
- preserving acquisition order;
- maintaining a minimal public API;
- keeping common modules detector-independent.

This separation keeps the architecture readable while preserving the
history of important decisions.

---

## Documentation Structure

The documentation set is divided by responsibility.

    docs/ARCHITECTURE.md
        Current project architecture and design principles.

    docs/DECISIONS.md
        Historical record of major architectural decisions.

    docs/DEVELOPMENT_GUIDE.md
        Contribution and development workflow.

    docs/VALIDATION_GUIDE.md
        Scientific validation policy and procedures.

    docs/TERMINOLOGY.md
        Definitions of project-specific terms.

    docs/API_REFERENCE.md
        Supported public interfaces.

    docs/STEP01.md through docs/STEP06.md
        Processing-step specifications and usage.

    validation/stepXX/README.md
        Validation contract for each processing step.

Documentation and implementation should evolve together.

---

## Current Status

The architecture is being established incrementally.

Current work includes:

- reusable image I/O;
- manifest normalization;
- Step01 implementation and documentation;
- Step02 statistical processing;
- software tests;
- comparison utilities;
- the validation documentation structure.

Later pipeline steps remain subject to milestone-driven design and
implementation.

This document defines the intended architecture but does not imply that
every described component is already implemented.

---

## Roadmap

Planned architectural development includes:

- completion of Step02 through Step06;
- repository-contained validation fixtures;
- automated reference-artifact regression;
- detector-specific scientific validation datasets;
- explicit public API documentation;
- release qualification procedures;
- support for additional detector formats and detector models;
- performance optimization after scientific equivalence is established.

Potential future capabilities may include parallel processing,
accelerated numerical implementations, distributed execution, and
additional RTS state-estimation methods.

Such additions should preserve the architectural principles defined in
this document.

---

## Stability of the Architecture

Algorithms, detector models, and implementation techniques may evolve.

The architectural boundaries should remain comparatively stable:

- processing stages have explicit responsibilities;
- common modules remain reusable;
- public APIs remain controlled;
- tests and scientific validation remain separate;
- scientific artifacts remain inspectable;
- major decisions remain documented.

RTS-Framework is intended to evolve without losing scientific
reproducibility or long-term maintainability.
