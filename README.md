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

# Development Principles

RTS-Framework is developed as a long-term scientific software project.
The primary goal is not rapid feature addition, but reproducibility,
maintainability, and scientific reliability.

The following principles guide all development.

## 1. Legacy parity before refactoring

The first milestone is to reproduce the behavior of the legacy RTS
pipeline as faithfully as possible.

Refactoring and optimization should only be performed after the legacy
implementation has been reproduced and verified.

## 2. Test first

Every bug fix or behavior change should be accompanied by a regression
test.

Whenever practical:

1. Write a failing test.
2. Implement the smallest possible change.
3. Verify that all tests pass.

## 3. One logical change per commit

Each commit should represent exactly one logical change.

Examples:

- add one regression test
- fix one bug
- refactor one module
- update documentation

Avoid mixing unrelated modifications in a single commit.

## 4. Detector-independent common modules

Modules under `common/` must remain detector-independent.

Detector-specific algorithms belong in detector-specific modules or
future pipeline steps.

## 5. Stable public API

Public APIs should remain stable within each milestone.

Breaking changes should be introduced only at clearly defined milestone
boundaries.

## 6. Readability over cleverness

Scientific software is maintained for many years.

Code should therefore prioritize readability, explicit behavior, and
clear error messages over clever or highly compact implementations.


# Git Workflow

Development follows a lightweight GitHub workflow.

1. Create an Issue.
2. Create a feature branch.
3. Implement one logical change.
4. Run the test suite.
5. Commit.
6. Push.
7. Open a Pull Request.
8. Review and merge.

Recommended commit prefixes:

- feat:
- fix:
- test:
- refactor:
- docs:
- chore:


# Code Review

Every change should be reviewed from two independent viewpoints:

- Scientific correctness
- Software architecture

A change is considered complete only when both are satisfied.


# Project Philosophy

This framework is intended to become a reusable and detector-independent
RTS analysis framework for scientific CMOS sensors.

The design philosophy is:

Scientific correctness first.
Software quality second.
Performance optimization third.

Performance improvements should never compromise reproducibility or
scientific validity.
A
A


