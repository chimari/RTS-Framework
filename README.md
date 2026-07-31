# RTS Framework

> Detector-independent scientific framework for astronomical detector characterization

**Current milestone:** Step02 completed → Step03 in progress


> **A detector-independent scientific software framework for detecting,
> characterizing, correcting, and evaluating Random Telegraph Signal (RTS)
> behavior in astronomical detector images.**

---

## Overview

RTS Framework is an open scientific software project for the analysis of
Random Telegraph Signal (RTS) behavior in astronomical imaging detectors.

Unlike detector-specific analysis scripts, RTS Framework is designed as a
long-term, reusable framework that separates detector-independent
processing from instrument-specific configuration.

The project emphasizes

- Scientific reproducibility
- Detector independence
- Explicit processing pipelines
- Long-term maintainability
- Deterministic outputs
- Scientific validation

The framework is intended to support both detector characterization and
future astronomical instrument development.

RTS Framework provides reusable infrastructure for
scientific detector characterization, including
read-noise analysis, RTS characterization,
correction, and performance evaluation.

---

## Why RTS Framework?

Most existing RTS analysis software is tightly coupled to a particular
detector, instrument, or observing campaign.

RTS Framework aims to provide a reusable scientific framework that can be
applied across different detector systems while maintaining reproducible
processing and well-defined interfaces.

The project is designed as scientific software rather than as a
collection of analysis scripts.

---

## Project Status

**Current development status**

| Stage | Status |
|--------|--------|
| Common libraries | ✅ Stable |
| Step01 – Dataset Preparation | ✅ Implemented |
| Step02 – Statistical Characterization | ✅ Implemented |
| Step03 – RTS Candidate Detection | 🚧 In Progress |
| Step04 – RTS State Estimation | Planned |
| Step05 – RTS Correction | Planned |
| Step06 – Scientific Evaluation | Planned |

Current milestone:

> **Transition from detector-specific prototype software to a fully
> detector-independent scientific framework.**

---

## Processing Pipeline

```
Raw detector images
        │
        ▼
Step01  Dataset Preparation
        │
        ▼
Step02  Statistical Characterization
        │
        ▼
Step03  RTS Candidate Detection
        │
        ▼
Step04  RTS State Estimation
        │
        ▼
Step05  RTS Correction
        │
        ▼
Step06  Scientific Evaluation
```

Each processing stage produces explicit scientific artifacts that become
the documented interface to the following stage.

---

## Main Features

Current capabilities include

- Detector-independent image I/O
- FITS and RAW image support
- Dataset manifest validation
- Deterministic statistical characterization
- Python API
- Command-line interfaces
- Scientific validation framework
- Versioned processing artifacts

Additional functionality will be introduced incrementally while
maintaining backward compatibility where practical.

---

## Repository Structure

```
common/
    Detector-independent core libraries

steps/
    Pipeline implementations

tests/
    Software verification

validation/
    Scientific validation

docs/
    Architecture and design documentation

tools/
    Standalone development utilities
```

---

## Installation

```bash
git clone <repository>

cd RTS-Framework

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

---

## Quick Start

Run the processing pipeline from the command line.

Dataset preparation:

```bash
python -m steps.step01_prepare_dataset ...
```

Statistical characterization:

```bash
python -m steps.step02_prepare_frame_groups ...
```

Detailed usage examples are available in the pipeline documentation.

---

## Documentation

Project documentation is located under `docs/`.

| Document | Description |
|----------|-------------|
| `ARCHITECTURE.md` | Overall software architecture |
| `DECISIONS.md` | Major architectural decisions |
| `STEP01.md` – `STEP06.md` | Pipeline specifications |
| `developer_guide.md` | Development guide |

AGENTS.md
        │
ARCHITECTURE.md
        │
Pipeline Specifications
        │
Implementation
        │
Tests
        │
Scientific Validation

---

## Development Philosophy

RTS Framework follows several guiding principles.

- Correctness before optimization
- Explicit processing stages
- Detector-independent design
- Reproducible scientific outputs
- Small reviewable changes
- Documentation before implementation

Documentation is treated as part of the implementation.

---

## Contributing

Contributors should consult

- `AGENTS.md`
- `docs/ARCHITECTURE.md`
- `docs/DECISIONS.md`

before implementing new functionality.

New features should normally include

- documentation
- automated tests
- scientific validation where appropriate

---

## Acknowledgements

RTS Framework is being developed as a long-term scientific software
project for astronomical detector characterization and future
instrumentation development.

---

## License

(To be determined.)
