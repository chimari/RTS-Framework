# RTS-Framework Architecture Decisions

## Purpose

This document records significant architectural decisions made during
the development of RTS-Framework.

The architecture document describes the current design.

This document explains why important design decisions were made.

Only decisions that have long-term architectural impact should be
recorded here.

Minor implementation details, bug fixes, and temporary workarounds are
outside the scope of this document.

---

## Decision Format

Each decision contains:

- Identifier
- Status
- Date
- Context
- Decision
- Consequences

The identifier follows the format:

    ADR-0001

Decision records should never be modified after acceptance except to
correct typographical errors or to update their status.

If a decision is replaced, a new ADR should supersede the previous one.

---

# ADR-0001

Title

    Separate software testing from scientific validation

Status

    Accepted

Context

Software correctness and scientific correctness are different concerns.

Unit tests verify software behavior.

Scientific validation verifies reproducibility of scientific results.

Decision

Software testing and scientific validation are maintained in separate
directories:

    tests/

and

    validation/

Consequences

Software development remains independent from scientific validation.

Validation procedures may evolve without affecting the software testing
framework.

---

# ADR-0002

Title

    Introduce version-controlled reference artifacts

Status

    Accepted

Context

Scientific regressions cannot always be detected by software tests.

Deterministic scientific products require comparison against accepted
reference outputs.

Decision

Each processing step may define version-controlled reference artifacts
used for regression validation.

Consequences

Scientific regressions become detectable during development.

Reference artifacts become part of the documented validation process.

---

# ADR-0003

Title

    Keep common modules detector-independent

Status

    Accepted

Context

Detector-specific implementations reduce reuse and increase maintenance
cost.

Decision

Reusable modules under

    common/

should avoid detector-specific assumptions whenever practical.

Detector-specific behavior should be implemented through explicit
configuration or specialized processing layers.

Consequences

Support for additional detector systems becomes significantly easier.

---

# ADR-0004

Title

    Maintain a minimal public API

Status

    Accepted

Context

Large public interfaces are difficult to maintain over long periods.

Decision

Only documented interfaces are considered public APIs.

Internal implementation details may change without notice.

Consequences

Internal refactoring becomes easier while maintaining external
compatibility.

---

# ADR-0005

Title

    Preserve explicit processing stages

Status

    Accepted

Context

Each processing stage produces a scientifically meaningful artifact.

Independent validation requires explicit boundaries between processing
steps.

Decision

The framework is organized into six primary processing steps.

Each step produces a documented artifact.

Consequences

Validation, testing, and debugging become easier.

Scientific processing remains modular and extensible.


---

# ADR-0006

Title

    Documentation First

Status

    Accepted

Context

Architectural decisions, public interfaces, validation procedures,
repository organization, and processing responsibilities evolve much
more slowly than their implementations.

Historically, scientific software often accumulates undocumented design
changes, making long-term maintenance increasingly difficult.

Decision

Changes that affect the architecture or documented behavior of
RTS-Framework shall be described and reviewed before implementation.

Examples include:

    - repository structure;
    - processing responsibilities;
    - public APIs;
    - validation procedures;
    - file formats;
    - architectural boundaries.

Small implementation improvements, bug fixes, and internal
refactoring that do not alter documented behavior may proceed without
prior architectural documentation.

Consequences

Documentation becomes the authoritative description of the framework.

Implementation follows documented design rather than defining it.

Design discussions become reviewable independently of source code.

Long-term maintenance is improved by reducing divergence between
documentation and implementation.

Architecture documents become a stable reference for future
contributors and scientific collaborators.


---

# ADR-0007

Title

    Treat scientific artifacts as first-class outputs

Status

    Accepted

Context

Many scientific software packages treat intermediate files as temporary
implementation details.

This makes debugging, validation, reproducibility, and scientific review
more difficult.

RTS-Framework is organized as a sequence of explicit processing stages,
each producing a scientifically meaningful artifact.

Decision

Every processing step shall produce a documented output artifact.

Artifacts are considered part of the scientific interface between
processing steps rather than temporary implementation products.

Whenever practical, artifacts should:

    - have documented schemas;
    - be independently inspectable;
    - be reproducible;
    - support validation against reference artifacts;
    - remain usable without knowledge of internal implementation.

Consequences

Processing stages become naturally modular.

Intermediate scientific products can be reviewed independently.

Validation can be performed incrementally at each processing stage.

Future processing algorithms may evolve without changing the conceptual
pipeline.

Scientific reproducibility is improved by making intermediate results
explicit rather than implicit.

