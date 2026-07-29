# RTS Framework Architecture

## Purpose

Scientific detector characterization framework.

---

## Design Philosophy

Separate

- validation

- grouping

- statistics

- analysis

into independent pipeline steps.

---

## Pipeline

Input Manifest

↓

Step01

↓

normalized.csv

↓

Step02

↓

group_manifest.csv

↓

...

---

## Repository Structure

README.md

CHANGELOG.md

docs/

steps/

tests/

common/

legacy/

---

## Future Pipeline

Raw Statistics

↓

Candidate Extraction

↓

Time Series

↓

Histogram Analysis

↓

State Assignment

↓

Transition Statistics

↓

RTS Classification

↓

Dictionary

↓

QA Report