# TASKS.md

# TransitLens Data Pipeline Tasks

## Phase 1

Repository Setup

Status

Not Started

Tasks

- Configure Python project
- Configure Ruff
- Configure Black
- Configure pytest
- Configure pre-commit
- Configure FastAPI

Acceptance Criteria

Project runs successfully.

---

## Phase 2

MAST Integration

Tasks

- Implement authentication
- Implement search
- Implement download
- Implement caching

Acceptance Criteria

Search and download a FITS file.

---

## Phase 3

FITS Processing

Tasks

- Read FITS
- Parse HDUs
- Extract Time
- Extract Flux
- Validate columns

Acceptance Criteria

Return structured light curve.

---

## Phase 4

Preprocessing

Tasks

- Remove NaNs
- Remove invalid measurements
- Normalize flux
- Median filtering
- Wavelet denoising

Acceptance Criteria

Generate cleaned light curve.

---

## Phase 5

Feature Generation

Tasks

- Generate metadata
- Generate statistics
- Export features

Acceptance Criteria

Produce deterministic feature file.

---

## Phase 6

REST API

Tasks

Implement

GET /search

POST /download

POST /process

GET /status

Acceptance Criteria

Platform repository can use the API.

---

## Phase 7

Testing

Tasks

- Unit tests
- Integration tests
- Benchmark preprocessing
- Validate output consistency

Target Coverage

Minimum 90%

---

## Deferred

Not part of prototype

- Auto dataset retraining
- Scheduler
- Background workers
- Distributed processing
- Multiple archive providers