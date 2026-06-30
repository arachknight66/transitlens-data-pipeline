# ARCHITECTURE.md

# TransitLens Data Pipeline Architecture

## Design Principles

- Single responsibility
- Independent modules
- Deterministic preprocessing
- Minimal external state
- Easy testing
- Future extensibility

---

## Directory Structure

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

---

## Module Responsibilities

### mast

Responsible for all communication with MAST.

Functions

- Login
- Search
- Download
- Cache

No FITS parsing occurs here.

---

### fits

Responsible for reading FITS files.

Responsibilities

- Open FITS
- Read HDUs
- Extract columns
- Validate required fields

Output

Structured LightCurve object

---

### preprocessing

Input

Structured LightCurve

Operations

- Remove NaN
- Remove invalid quality flags
- Normalize flux
- Median filter
- Wavelet denoising

Output

Processed LightCurve

---

### features

Compute deterministic features

Examples

- Mean
- Standard deviation
- RMS
- Signal to noise ratio
- Flux variance
- Observation duration
- Cadence

These are exported for future ML models.

---

### exporters

Supported formats

- NumPy
- Parquet

Future

- HDF5

---

### api

Expose endpoints used by platform.

Endpoints

GET /search

POST /download

POST /process

GET /status

---

## Data Flow

MAST

↓

Search

↓

Download FITS

↓

Parse FITS

↓

Validation

↓

Normalization

↓

Median Filter

↓

Wavelet Denoising

↓

Feature Extraction

↓

Export

---

## Dependencies

No dependency on

- ml-core
- platform

This repository must remain completely standalone.

Only platform may call its REST API.