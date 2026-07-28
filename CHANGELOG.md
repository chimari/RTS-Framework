# Changelog

All notable changes to RTS-Framework are documented here.

---

## Documentation Milestone 1.0

### Added

- Architecture document
- Architecture Decision Records
- Validation architecture
- Validation documentation for Step01–Step06
- Development policy
- Documentation-first workflow

## [Step06 v6.4.0]

### Added
- Numerical evaluation of RTS correction results.
- Diagnostic PNG generation.
- Automatic multi-page PDF report generation.
- Scientific assessment (Excellent / Good / Acceptable / Warning).
- Automatic concern detection.
- Optional RTS mask-based science metrics.
- Science JSON output.
- Cluster analysis using 8-connectivity.

### Improved
- Deterministic visualization and PDF generation.
- CLI reporting.
- Report quality and diagnostics.

### Compatibility
- Legacy JSON output remains fully backward compatible.

### Testing
- All Step06 integration tests passed.
- Full regression tests passed.

# Step05 v5.12.0 (2026-07-25)

## New features

### v5.10.0
- Batch preflight validation
- validate_rts_correction_batch()
- immutable validation results
- --preflight CLI

### v5.11.0
- Directory batch processing
- --input-dir
- --pattern
- deterministic discovery

### v5.12.0
- Multiprocessing support
- --workers
- deterministic result ordering
- parent-side manifest/provenance aggregation

## Reliability

- Dedicated integration tests added
- Full Step05 regression tests passed
- Existing APIs remain backward compatible
