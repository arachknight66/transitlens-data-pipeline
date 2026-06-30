# HANDOFF.md

# TransitLens Data Pipeline Handoff

## Repository Status

Current Phase

Phase 1 complete - Repository Setup

Completed on

2026-06-30

---

## Public Interfaces

REST API

GET /search

POST /download

POST /process

GET /status

Output Format

ProcessedLightCurve

---

## Contracts

This repository guarantees

- Stable processed output
- Stable metadata schema
- Stable REST endpoints

Breaking these interfaces requires updating both

- transitlens-ml-core
- transitlens-platform

---

## Dependencies

External

- astroquery
- astropy
- lightkurve
- pywavelets
- scipy

Internal

None

---

## Current Deliverables

Completed

- Python 3.11+ project and dependency configuration
- Ruff, Black, pytest, coverage, and pre-commit configuration
- Typed TOML and environment-based runtime settings
- Structured Loguru configuration
- FastAPI application factory (business endpoints intentionally deferred)
- Reproducible dependency lock file
- Phase 1 smoke and configuration tests

Verification

- Ruff passes
- Black passes
- 7 tests pass
- Test coverage: 98.25%
- Source distribution and wheel build successfully
- FastAPI application factory starts successfully

Pending

- MAST integration
- FITS parser
- Wavelet preprocessing
- REST API

---

## Configuration Contract

Default configuration is stored in `configs/default.toml`.

Runtime values may be supplied with environment variables using the
`TRANSITLENS_` prefix. Environment variables take precedence over values from
the configuration file. MAST credentials remain optional and are never stored
in source configuration.

---

## Phase 1 Notes

- No MAST, FITS, preprocessing, feature, exporter, or business endpoint logic
  has been implemented.
- The FastAPI application uses an application factory so settings are injected
  without mutable global application state.
- Phase 2 should begin with MAST authentication only.

---

## Known Risks

- FITS formats differ between missions.
- Large downloads require caching.
- MAST authentication should remain optional for public datasets.
- Wavelet parameters must preserve transit features while reducing noise.

---

## Next Repository

transitlens-ml-core

Expected Input

ProcessedLightCurve

Fields

- time
- normalized_flux
- wavelet_flux
- metadata

The data format produced here is considered the canonical input for all machine learning models.
