# RTS Framework

## Project Goal

The RTS Framework is a detector-independent scientific framework for
detecting, characterizing, correcting, and evaluating Random Telegraph
Signal (RTS) noise in astronomical detector images.

Scientific reproducibility has higher priority than execution speed.

---

## Design Philosophy

- deterministic outputs
- immutable dataclasses
- semantic versioning
- explicit APIs
- lazy loading
- atomic writes

---

## Scientific Principles

Never modify image pixels unless the processing step explicitly performs
a correction.

Every derived product must be reproducible.

All processing decisions should remain traceable.

---

## Pipeline

Step01
Validation and normalization

Step02
Dataset grouping and image statistics

Step03
Read-noise characterization and master bias

Step04
RTS dictionary generation

Step05
RTS correction

Step06
Evaluation

---

## Coding Rules

Always use

    read_image(frame)

Never

    read_image(frame.filepath)

Reuse existing common modules.

Avoid duplicated algorithms.

All public APIs require type hints.

All new features require tests.

---

## Performance

Prefer

correctness
>
reproducibility
>
clarity
>
speed

Speed optimization should never change numerical results.

---

## Before implementing

When requirements are ambiguous,

STOP

and ask for clarification rather than inventing behavior.