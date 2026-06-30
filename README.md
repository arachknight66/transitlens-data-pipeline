# README.md

# TransitLens Data Pipeline

## Overview

The TransitLens Data Pipeline is responsible for acquiring, validating, preprocessing, and exporting astronomical light curve data for machine learning models.

This repository is the only component responsible for interacting with external astronomical data sources. It converts raw observational data into standardized, ML-ready datasets.

No machine learning models are implemented in this repository.

---

## Responsibilities

- Search astronomical observations using MAST
- Authenticate with MAST when required
- Download FITS files
- Parse FITS files using Astropy
- Extract light curve information
- Validate observations
- Remove invalid measurements
- Normalize light curves
- Apply median filtering
- Apply wavelet denoising
- Generate metadata
- Export processed datasets
- Cache downloaded observations
- Provide REST endpoints for the platform repository

---

## Data Sources

Primary source:

- MAST Archive

Supported missions:

- Kepler
- K2
- TESS

Future support:

- PLATO
- Roman Space Telescope
- Chandrayaan datasets (if applicable)

---

## Technology Stack

Python 3.11+

Libraries

- astroquery
- astropy
- lightkurve
- numpy
- scipy
- pandas
- pywavelets
- fastapi
- uvicorn

---

## Processing Pipeline

MAST Search

↓

Download FITS

↓

Read FITS

↓

Validate

↓

Extract Time + Flux

↓

Quality Filtering

↓

Normalization

↓

Median Filtering

↓

Wavelet Denoising

↓

Feature Generation

↓

Export Processed Dataset

---

## Output

Each processed observation should contain

- Time
- Flux
- Normalized Flux
- Wavelet Denoised Flux
- Quality Flags
- Metadata
- Observation Information

The output format must remain stable because ml-core depends on it.

---

## Repository Goals

- Modular
- Deterministic
- Reproducible
- Well documented
- Unit tested
- Independent from ML implementation

---

## Non Goals

This repository must NOT contain

- Neural networks
- CNNs
- Autoencoders
- Training scripts
- Model inference
- Frontend code

Those belong to ml-core and platform.