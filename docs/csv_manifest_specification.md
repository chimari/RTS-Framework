# CSV Manifest Specification

Version 1.0

---

## Purpose

Defines the standard manifest format used by RTS Framework.

---

## General Rules

- UTF-8

- Comma-separated

- Header required

- Relative file paths

- One image per row

---

## Required Columns

| Column | Type | Required |

...

---

## Image Layout

| Column | FITS | RAW |

image_width

image_height

pixel_dtype

byte_order

---

## FITS

Header is authoritative.

---

## RAW

Manifest is authoritative.

---

## Output

normalized.csv

contains complete image metadata.