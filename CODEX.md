# CODEX.md

# TransitLens Data Pipeline - Codex Implementation Guide

This document defines the implementation rules for AI coding agents working on the `transitlens-data-pipeline` repository.

Read this file before making any code changes.

---

# Repository Mission

The purpose of this repository is to acquire, validate, preprocess, and export astronomical light curve datasets for downstream machine learning.

This repository is responsible for everything between the MAST archive and ML-ready datasets.

This repository must never contain machine learning models.

---

# Scope

## Responsibilities

- MAST integration
- Authentication
- Observation search
- FITS download
- FITS parsing
- Data validation
- Quality filtering
- Flux normalization
- Median filtering
- Wavelet denoising
- Feature generation
- Dataset export
- REST API

---

## Out of Scope

Do NOT implement

- CNNs
- Autoencoders
- Transformers
- Training
- Inference
- Model evaluation
- Frontend
- Authentication systems unrelated to MAST
- Database models

These belong to other repositories.

---

# Technology Stack

Language

- Python 3.11+

Frameworks

- FastAPI

Scientific Libraries

- astroquery
- astropy
- lightkurve
- numpy
- scipy
- pandas
- pywavelets

Utilities

- pydantic
- loguru
- pytest

---

# Coding Principles

Follow these principles strictly.

## Single Responsibility

Each module should have one responsibility.

Bad

reader.py

- Downloads files
- Parses FITS
- Cleans data

Good

download.py

reader.py

validator.py

wavelet.py

---

## Pure Functions

Preprocessing functions should avoid side effects.

Good

normalize_flux()

wavelet_denoise()

remove_invalid_points()

Bad

Functions that modify global state.

---

## Deterministic Processing

The same FITS file must always produce the same processed output.

No randomness.

---

## Type Hints

Every public function must contain type hints.

Example

```python
def normalize_flux(
    flux: np.ndarray
) -> np.ndarray:
```

---

## Documentation

Every public class and function must contain docstrings.

Use Google style.

---

## Logging

Use structured logging.

Never use print().

---

# Directory Structure

```
src/

    mast/

        auth.py

        search.py

        download.py

        cache.py

    fits/

        reader.py

        parser.py

        validator.py

    preprocessing/

        normalize.py

        quality.py

        median_filter.py

        wavelet.py

    features/

        statistics.py

        metadata.py

    exporters/

        numpy_export.py

        parquet_export.py

    api/

        routes.py

tests/

configs/

scripts/
```

Do not introduce additional top-level folders without justification.

---

# Processing Pipeline

The processing pipeline must remain

```
MAST

↓

Search

↓

Download

↓

FITS Reader

↓

Validation

↓

Quality Filtering

↓

Normalization

↓

Median Filtering

↓

Wavelet Denoising

↓

Feature Extraction

↓

Export
```

Do not change the order.

---

# Wavelet Denoising Requirements

Use

- PyWavelets

Recommended default

Wavelet

```
db4
```

Mode

```
soft threshold
```

Threshold

Adaptive thresholding preferred.

Requirements

- Preserve transit depth
- Preserve ingress
- Preserve egress
- Reduce high-frequency detector noise

Never oversmooth the signal.

Transit features are more important than aggressive denoising.

---

# FITS Requirements

Use

Astropy

Required fields

- TIME
- FLUX

Support quality flags when available.

Gracefully handle missing HDUs.

Raise meaningful exceptions.

---

# MAST Integration

Use astroquery.

Support

- Anonymous access

Optional

- API Token
- Session authentication

Never hardcode credentials.

---

# REST API

Only expose

```
GET /search

POST /download

POST /process

GET /status
```

Do not invent additional endpoints.

---

# Performance Goals

Pipeline should process a single FITS observation in under

```
3 seconds
```

excluding download time.

Cache downloaded FITS files.

Avoid repeated parsing.

---

# Error Handling

Never silently ignore failures.

Return descriptive exceptions.

Examples

- Invalid FITS

- Missing TIME column

- Missing FLUX column

- MAST unavailable

- Authentication failed

---

# Testing

Every module requires tests.

Minimum

- Unit tests

- Integration tests

Target coverage

90%

---

# Files You May Modify

Everything inside

```
src/

tests/

configs/
```

---

# Files You Must Not Modify

README.md

ARCHITECTURE.md

TASKS.md

HANDOFF.md

CODEX.md

unless explicitly instructed.

---

# Git Rules

Keep commits small.

Each commit should implement one logical feature.

Example

```
feat(mast): implement observation search

feat(fits): parse primary HDU

feat(preprocessing): add wavelet denoising
```

---

# Completion Criteria

The repository is considered complete when

✓ Searches MAST

✓ Downloads FITS

✓ Reads FITS

✓ Validates observations

✓ Cleans data

✓ Applies median filtering

✓ Applies wavelet denoising

✓ Exports processed light curves

✓ Provides REST API

✓ All tests pass

---

# If Requirements Are Ambiguous

Do NOT guess.

Document the ambiguity.

Choose the simplest scientifically valid implementation.

Maintain compatibility with

- transitlens-ml-core

- transitlens-platform

---

# Primary Objective

Produce a clean, modular, production-quality astronomical preprocessing pipeline that transforms raw MAST FITS observations into deterministic, ML-ready light curves while preserving exoplanet transit signals.