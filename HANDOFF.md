# HANDOFF.md

# TransitLens Data Pipeline Handoff

## Repository Status

Current Phase

Not Started

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

None

Pending

- MAST integration
- FITS parser
- Wavelet preprocessing
- REST API

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