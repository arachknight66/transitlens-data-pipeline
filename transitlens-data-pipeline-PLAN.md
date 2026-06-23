# transitlens-data-pipeline — Phased Build Plan

> **Project:** TransitLens — Bharatiya Antariksh Hackathon 2026
> **Problem Statement:** PS7 — AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves
> **Repo Role:** Data ingestion, synthetic generation, real TESS loading, dataset assembly
> **Document Type:** Engineering build plan — no code, phases only
> **Last Updated:** 2026

---

## Table of Contents

1. [Repo Purpose and Boundaries](#1-repo-purpose-and-boundaries)
2. [Folder Structure Reference](#2-folder-structure-reference)
3. [Output Contract](#3-output-contract)
4. [Phase Overview](#4-phase-overview)
5. [Phase 1 — Foundation and Synthetic Core](#5-phase-1--foundation-and-synthetic-core)
6. [Phase 2 — Dataset Assembly and Ground Truth](#6-phase-2--dataset-assembly-and-ground-truth)
7. [Phase 3 — Real TESS Integration](#7-phase-3--real-tess-integration)
8. [Phase 4 — Interface, Tests, and Polish](#8-phase-4--interface-tests-and-polish)
9. [Phase 5 — Stretch Goals](#9-phase-5--stretch-goals)
10. [File-by-File Responsibility Matrix](#10-file-by-file-responsibility-matrix)
11. [Dependencies and Install Plan](#11-dependencies-and-install-plan)
12. [Configuration Reference](#12-configuration-reference)
13. [Data Schema Specification](#13-data-schema-specification)
14. [Risk Register](#14-risk-register)
15. [Hackathon Priority Tiers](#15-hackathon-priority-tiers)
16. [Definition of Done](#16-definition-of-done)

---

## 1. Repo Purpose and Boundaries

### What this repo does

`transitlens-data-pipeline` is the **data layer** of the TransitLens system. Its only responsibility is producing clean, correctly-shaped, well-labelled light curve data and delivering it to `transitlens-ml-core` through a single interface function.

It handles:

- Generating synthetic TESS-like light curves with injected transit signals
- Applying realistic noise models to synthetic data
- Loading real TESS light curves from MAST (stretch goal)
- Assembling labelled datasets for model training and evaluation
- Exposing a unified `load_light_curve()` interface consumed by ml-core

### What this repo does NOT do

This repo must never:

- Run BLS detection or any signal processing algorithm
- Classify signals or assign confidence scores
- Render plots or produce visualisations
- Serve HTTP endpoints
- Import anything from `transitlens-ml-core` or `transitlens-platform`

The boundary is absolute. If a function feels like it belongs in ml-core, it does.

### Position in the tri-repo system

```
transitlens-data-pipeline   →   transitlens-ml-core   →   transitlens-platform
        (feeds)                       (analyses)                 (displays)
```

ml-core calls `load_light_curve()` from this repo. That is the only integration point.

---

## 2. Folder Structure Reference

```
transitlens-data-pipeline/
│
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── .gitignore
├── interface.py                        ← single public entry point
│
├── synthetic/
│   ├── __init__.py
│   ├── generator.py                    ← time array + base flux builder
│   ├── noise_models.py                 ← Gaussian + red noise
│   ├── transit_injector.py             ← injects box/trapezoid dips
│   ├── config.yaml                     ← all generation parameters
│   └── cases/
│       ├── README.md
│       ├── candidate_a.csv             ← exoplanet_like
│       ├── candidate_b.csv             ← eclipsing_binary_like
│       └── candidate_c.csv             ← noise_or_other
│
├── real_tess/
│   ├── __init__.py
│   ├── mast_loader.py                  ← fetch by TIC ID via Lightkurve
│   ├── sector_selector.py              ← pick highest-coverage sector
│   ├── flux_normaliser.py              ← PDC SAP → median normalised
│   ├── cache/
│   │   └── .gitkeep
│   └── README.md
│
├── datasets/
│   ├── __init__.py
│   ├── build_dataset.py                ← assembles final labeled CSV
│   ├── labeled_dataset.csv             ← ground truth dataset
│   ├── metadata.json                   ← per-target metadata
│   ├── schema.md                       ← column definitions
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_generator.py
│   ├── test_schema.py
│   ├── test_loader.py
│   └── test_interface.py
│
└── notebooks/
    ├── exploration.ipynb
    └── synthetic_visualisation.ipynb
```

---

## 3. Output Contract

Every call to `load_light_curve()` — regardless of source — must return exactly this structure. ml-core depends on this shape unconditionally.

### Return dict specification

```
{
    "time":        float[]     # BTJD timestamps, evenly spaced, no gaps
    "flux":        float[]     # normalised flux, median ≈ 1.0
    "target_id":   str         # unique identifier for this light curve
    "source":      str         # "synthetic" | "tess" | "csv"
    "n_points":    int         # length of time and flux arrays (must match)

    "metadata": {
        "cadence_min":       float        # time between points in minutes
        "time_span_days":    float        # total observation window
        "sector":            int | None   # TESS sector number (None for synthetic)
        "label":             str | None   # "exoplanet_like" | "eclipsing_binary_like"
                                          # | "noise_or_other" | None
        "true_period":       float | None # ground truth period in days (None if unknown)
        "true_depth":        float | None # ground truth fractional depth (None if unknown)
        "true_duration":     float | None # ground truth duration in days (None if unknown)
    }
}
```

### Invariants that must always hold

- `len(time) == len(flux) == n_points`
- `time` is monotonically increasing with no repeated values
- `flux` is normalised such that the median baseline is 1.0 ± 0.001
- `source` is always one of the three allowed string values
- `metadata.label` is `None` for unlabelled real TESS targets only
- `true_period`, `true_depth`, `true_duration` are `None` for noise cases and unlabelled real data

### Validation requirement

`interface.py` must validate all invariants before returning. If any invariant fails, raise a descriptive `DataPipelineError` rather than returning malformed data.

---

## 4. Phase Overview

| Phase | Name | Priority | Estimated Effort | Hackathon Tier |
|-------|------|----------|-----------------|----------------|
| 1 | Foundation and Synthetic Core | Critical | 3–4 hours | Must-have |
| 2 | Dataset Assembly and Ground Truth | High | 2–3 hours | Should-have |
| 3 | Real TESS Integration | Medium | 3–5 hours | Nice-to-have |
| 4 | Interface, Tests, and Polish | High | 2–3 hours | Must-have |
| 5 | Stretch Goals | Low | Open-ended | Future |

Phases 1 and 4 are non-negotiable for a working hackathon demo. Phase 2 is required to tell a credible evaluation story to judges. Phase 3 is the wow-factor upgrade. Phase 5 is post-hackathon.

Build order: **Phase 1 → Phase 4 (partial) → Phase 2 → Phase 3 → Phase 4 (complete) → Phase 5**

The reason Phase 4 is split: write `interface.py` early (after Phase 1) so ml-core can start consuming data immediately, then complete tests and documentation last.

---

## 5. Phase 1 — Foundation and Synthetic Core

### Goal

Produce three synthetic light curve CSVs — one for each target class — that ml-core can immediately consume. By the end of this phase, the entire pipeline demo must be runnable offline without any external dependencies beyond numpy, scipy, and pandas.

### Deliverables

- `synthetic/config.yaml` — all generation parameters defined
- `synthetic/generator.py` — time array and base flux builder working
- `synthetic/noise_models.py` — at least Gaussian noise working
- `synthetic/transit_injector.py` — box dip injection working
- `synthetic/cases/candidate_a.csv` — exoplanet_like case generated
- `synthetic/cases/candidate_b.csv` — eclipsing_binary_like case generated
- `synthetic/cases/candidate_c.csv` — noise_or_other case generated
- `interface.py` — basic version returning the correct dict shape

### Step 1.1 — Design `synthetic/config.yaml`

This file is the single source of truth for all synthetic data generation. No hardcoded parameters anywhere in the code — everything reads from here.

The config must define three sections:

**`generation` section** — global parameters that apply to all cases:
- `n_points`: number of data points per light curve (target: 18000, matching real TESS 2-minute cadence over 27 days)
- `time_span_days`: total observation window in days (27.0, one TESS sector)
- `cadence_minutes`: time between observations (2.0 minutes)
- `baseline_flux`: normalised flux value for a quiet star (1.0)

**`cases` section** — one sub-entry per synthetic case, each defining:
- `label`: the class string for this case
- `period_days`: true orbital or binary period (null for noise case)
- `depth`: fractional flux drop at transit centre (null for noise case)
- `duration_days`: transit duration in days (null for noise case)
- `noise_level`: standard deviation of Gaussian noise
- `noise_type`: "gaussian" or "red" or "combined"
- `v_shape`: boolean — whether to inject a V-shaped dip (for eclipsing binary)
- `secondary_eclipse`: boolean — whether to add a secondary dip at phase 0.5 (for eclipsing binary)
- `seed`: integer random seed for full reproducibility

**Concrete values to use:**

Candidate A (exoplanet_like):
- period = 3.42 days, depth = 0.013 (1.3%), duration = 0.16 days, noise = 0.002, gaussian, seed = 42

Candidate B (eclipsing_binary_like):
- period = 1.87 days, depth = 0.18 (18%), duration = 0.08 days, noise = 0.003, v_shape = true, secondary_eclipse = true, seed = 43

Candidate C (noise_or_other):
- no transit, noise = 0.015, noise_type = "red", seed = 44

### Step 1.2 — Build `synthetic/generator.py`

**Purpose:** Creates the time array and an initial flat flux array. This is the foundation that all other synthetic modules build on top of.

**What it must do:**

1. Read `n_points`, `time_span_days`, and `cadence_minutes` from config
2. Generate a uniformly spaced time array starting at BTJD 1325.3 (a realistic TESS sector start time)
3. Generate a flat flux array of all 1.0 values with length `n_points`
4. Accept an optional `seed` parameter for reproducibility
5. Return `(time_array, flux_array)` as a tuple of numpy arrays

**Important design note:** The generator must NOT add noise or inject transits. Those are the responsibility of `noise_models.py` and `transit_injector.py` respectively. Separation of concerns is critical here because it allows testing each stage independently.

**Batch generation function:** A second function `generate_batch(config_path)` reads the config, loops over all cases, calls the individual generation pipeline for each, and saves the resulting CSVs to `synthetic/cases/`. This is the single command that regenerates all demo data from scratch.

### Step 1.3 — Build `synthetic/transit_injector.py`

**Purpose:** Injects a physically plausible transit signal into an existing flux array.

**What it must do:**

1. Accept `(time, flux, period, depth, duration, t0, v_shape, secondary_eclipse)` as parameters
2. Compute the phase of each timestamp relative to `t0` and `period`
3. Identify timestamps that fall within the transit window (phase within ±duration/2 of zero)
4. Apply the transit signal:
   - For box dip (exoplanet): multiply flux by `(1 - depth)` uniformly within the window
   - For V-shape dip (eclipsing binary): apply a parabolic depth profile within the window, deepest at phase zero
5. If `secondary_eclipse` is true, add a second shallower dip at phase 0.5 (typically half the primary depth)
6. Set `t0` automatically to `period * 0.3` so the first transit appears early in the light curve

**Physical realism notes:**

The box dip model is a simplification of the true Mandel-Agol transit model, but it is scientifically appropriate for a hackathon BLS detector because BLS itself assumes a box-shaped transit. The key parameters that make the signal realistic:

- Transit duration should satisfy the approximate relation: `duration ≈ period × (R_star / (π × a))` where typical values give durations of 2–6 hours for close-in planets
- Depth of 1–3% is physically consistent with a Jupiter-sized planet transiting a Sun-like star
- A depth of 10–20% immediately signals an eclipsing binary — no planet is large enough to block that fraction of stellar light

**Eclipsing binary signal characteristics:**

For Candidate B, the V-shape and secondary eclipse are the two distinguishing features that the ml-core classifier will use. The secondary eclipse at phase 0.5 represents the secondary star passing behind the primary. The V-shape arises because the secondary star has finite size relative to the primary, creating a curved ingress and egress.

### Step 1.4 — Build `synthetic/noise_models.py`

**Purpose:** Adds realistic noise to a clean flux array. Two noise regimes are required.

**Gaussian white noise:**

Adds independent random noise drawn from a normal distribution with mean zero and standard deviation equal to `noise_level`. This simulates detector read noise and photon noise. For TESS 2-minute cadence, a quiet star at TESS magnitude 10 has a typical photometric precision of around 200–500 ppm (0.0002–0.0005). The config value of `noise_level = 0.002` (2000 ppm) represents a noisier star or a deliberate challenge case.

**Red (correlated) noise:**

Also called systematic noise or 1/f noise. Unlike white noise, consecutive points are correlated. This simulates instrumental systematics and stellar variability. Implementation approach: generate an AR(1) autoregressive process with correlation coefficient `rho ≈ 0.7` and scale it to the desired amplitude. Alternatively, use a simple sinusoidal trend with period 0.5–2 days and amplitude matching `noise_level`.

Candidate C uses red noise exclusively because the goal is a light curve that has genuine structure (making it harder to classify as noise) but no true periodic transit signal.

**Combined noise:**

For production realism, add both Gaussian and a small red noise component to Candidates A and B. Keep the Gaussian component dominant so the transit signal remains detectable.

### Step 1.5 — Generate `synthetic/cases/` CSVs

Each CSV must have exactly these columns:

| Column | Type | Description |
|--------|------|-------------|
| `time` | float | BTJD timestamp |
| `flux` | float | normalised flux value |
| `flux_err` | float | estimated flux uncertainty (set to noise_level for synthetic) |

No other columns. No index column. Header row required. UTF-8 encoding.

The `cases/README.md` must document:
- What each candidate represents physically
- The true parameters used for injection
- How to regenerate using `generate_batch()`

### Step 1.6 — Write initial `interface.py`

**Function signature:**

```
load_light_curve(source, target_id, config=None) → dict
```

**Initial behaviour (Phase 1 version):**

- If `source == "synthetic"` and `target_id` is one of `"candidate_a"`, `"candidate_b"`, `"candidate_c"`: load the corresponding CSV, read the config, populate metadata, return the full dict
- All other source values: raise `NotImplementedError` with a clear message
- Validate the output dict before returning (length consistency, source string, flux normalisation)

### Phase 1 Completion Checklist

- [ ] `config.yaml` defines all three cases with correct physical parameters
- [ ] `generator.py` produces a time array matching TESS 2-minute cadence
- [ ] `transit_injector.py` produces a visually clear dip when parameters are printed
- [ ] `noise_models.py` adds Gaussian noise without corrupting the flux baseline
- [ ] `candidate_a.csv` exists with 18000 rows, three columns, header
- [ ] `candidate_b.csv` exists with V-shape signal and secondary eclipse
- [ ] `candidate_c.csv` exists with red noise and no injected transit
- [ ] `interface.py` returns a valid dict for all three candidate IDs
- [ ] No external dependencies beyond numpy, scipy, pandas, pyyaml

---

## 6. Phase 2 — Dataset Assembly and Ground Truth

### Goal

Build a labelled dataset CSV and accompanying metadata that tells a credible evaluation story to judges and provides training data for the optional RF/XGBoost classifier in ml-core.

### Deliverables

- `datasets/schema.md` — column definitions documented
- `datasets/metadata.json` — per-target metadata for all cases
- `datasets/labeled_dataset.csv` — assembled ground truth dataset
- `datasets/build_dataset.py` — reproducible assembly script
- `datasets/splits/train.csv`, `val.csv`, `test.csv` — train/val/test split

### Step 2.1 — Define `datasets/schema.md`

This document is what judges read when they ask "how did you label your data?" It must be clear, honest, and scientifically precise.

**Required column definitions:**

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `target_id` | str | No | Unique identifier for this light curve |
| `time` | float | No | BTJD timestamp |
| `flux` | float | No | Normalised flux, median baseline ≈ 1.0 |
| `flux_err` | float | Yes | Per-point flux uncertainty estimate |
| `source` | str | No | "synthetic" or "tess" |
| `label` | str | Yes | Class string or null for unlabelled |
| `true_period` | float | Yes | Ground truth orbital period in days |
| `true_depth` | float | Yes | Ground truth fractional flux drop |
| `true_duration` | float | Yes | Ground truth transit duration in days |
| `cadence_min` | float | No | Observation cadence in minutes |
| `sector` | int | Yes | TESS sector number (null for synthetic) |

**Label vocabulary:**

The `label` column must be exactly one of these strings or null:

- `exoplanet_like` — shallow, flat-bottomed, periodic dip consistent with a transiting planet
- `eclipsing_binary_like` — deep, V-shaped, often with secondary eclipse, consistent with a stellar binary
- `noise_or_other` — no detectable periodic transit signal

**Data provenance note (important for judges):**

The initial labeled dataset for the hackathon consists entirely of synthetic cases with known ground truth. The labels are therefore exact — not probabilistic estimates. When real TESS data is added (Phase 3), labels are assigned based on published TOI dispositions from the TESS mission page, not inferred by the model.

### Step 2.2 — Design `datasets/metadata.json`

One JSON object per target, keyed by `target_id`. Each entry contains:

- `target_id`: string
- `source`: "synthetic" or "tess"
- `label`: class string or null
- `true_period`: float or null
- `true_depth`: float or null
- `true_duration`: float or null
- `n_points`: integer
- `cadence_min`: float
- `time_span_days`: float
- `sector`: integer or null
- `notes`: free-text string for any special information about this case

For synthetic cases, `notes` should explain the generation parameters and what physical scenario was intended. For real TESS cases, `notes` should reference the TIC ID, the TOI number if applicable, and the published disposition.

### Step 2.3 — Build `datasets/build_dataset.py`

**Purpose:** Reads all available cases, attaches labels from `metadata.json`, and assembles a single `labeled_dataset.csv`.

**What it must do:**

1. Load `metadata.json` to get the list of all available targets and their labels
2. For each target, call `interface.load_light_curve()` to get the standardised dict
3. Convert to a flat DataFrame by repeating the metadata columns for each time point
4. Concatenate all targets into a single DataFrame
5. Write to `datasets/labeled_dataset.csv`
6. Print a summary: number of targets per class, total number of data points, label distribution

**Train/val/test split logic:**

Split at the target level (not the row level). With only three synthetic cases, the split is trivial — each case goes into a different set. When real data is added, use stratified splitting: 70% train, 15% val, 15% test, with equal class representation in each split.

**Reproducibility requirement:**

`build_dataset.py` must be a pure function of its inputs. Running it twice on the same data must produce identical output. Use a fixed random seed for any shuffling operations.

### Step 2.4 — What the labeled dataset enables in ml-core

1. **Rule-based classifier calibration:** The `true_period`, `true_depth`, and `true_duration` values are used to verify that the BLS detector recovers the correct parameters. If the detected period is within 1% of `true_period`, the detection is considered successful.

2. **Period recovery rate:** A key evaluation metric. After running ml-core on the labeled dataset, compute what fraction of exoplanet_like cases had their period recovered to within 1% of the true value. For synthetic data with clean injections, this should be above 90%.

3. **Classification accuracy:** Precision, recall, and F1 per class. For a well-tuned rule-based classifier on synthetic data, expect near-perfect scores.

4. **RF/XGBoost training (optional):** The feature vectors extracted by ml-core can be used to train a supervised classifier. The labeled dataset is the training data for this.

### Phase 2 Completion Checklist

- [ ] `schema.md` defines all columns with types, nullability, and descriptions
- [ ] `metadata.json` has an entry for all three synthetic candidates
- [ ] `labeled_dataset.csv` contains all three candidates with labels attached
- [ ] `build_dataset.py` runs without errors and prints a class distribution summary
- [ ] `splits/train.csv`, `splits/val.csv`, `splits/test.csv` are written
- [ ] The combined dataset has no missing values in required columns

---

## 7. Phase 3 — Real TESS Integration

### Goal

Load at least one real TESS light curve from MAST, normalise it to the standard output format, and return it from `interface.py`. A single successful real TESS detection is a major judge impression moment.

### Important caveat

Phase 3 requires internet access and the `lightkurve` library. The hackathon demo must still work completely offline using only synthetic data. The real TESS path is an enhancement, not a dependency.

### Recommended TIC IDs for the demo

| TIC ID | Planet | Period (days) | Depth | Why good for demo |
|--------|--------|--------------|-------|-------------------|
| 25155310 | WASP-126b | 3.29 | 0.011 | Bright star, clean signal, extensively studied |
| 279741377 | TOI-270b | 3.36 | 0.005 | Multi-planet system, community favourite |
| 149603524 | LHS 3844b | 0.46 | 0.004 | Ultra-short period, many transits per sector |

Hardcode these TIC IDs in `real_tess/README.md` as the "verified demo targets."

### Step 3.1 — Build `real_tess/mast_loader.py`

**Purpose:** Fetch a TESS light curve for a given TIC ID and sector from MAST using the `lightkurve` library.

**What it must do:**

1. Accept `(tic_id, sector=None, use_cache=True)` as parameters
2. Check `real_tess/cache/` for a pre-downloaded file before making a network request
3. Use `lightkurve.search_lightcurve()` to find available observations
4. Download the PDC-SAP flux product (not SAP flux — PDC-SAP has systematics corrections applied)
5. Save to `real_tess/cache/` as a `.fits` file
6. Return the raw `LightCurve` object

**Cache file naming convention:**

`real_tess/cache/TIC{tic_id}_sector{sector:03d}.fits`

**Error handling requirements:**

- Network unreachable and no cache: raise `TessDataUnavailableError`
- No TESS observations for TIC ID: raise `TessDataUnavailableError`
- Download timeout: retry once, then raise

### Step 3.2 — Build `real_tess/sector_selector.py`

**Purpose:** When multiple TESS sectors are available, choose the best one.

**Selection criteria (in priority order):**

1. Highest number of data points (fewest gaps)
2. PDC-SAP flux available
3. 2-minute cadence (not 30-minute FFI cadence)
4. Most recent sector (in case of tie)

### Step 3.3 — Build `real_tess/flux_normaliser.py`

**Purpose:** Convert a raw `lightkurve` LightCurve object to the standard `(time, flux)` arrays.

**What it must do:**

1. Accept a `LightCurve` object as input
2. Remove NaN values (TESS data contains NaN for flagged cadences)
3. Remove quality-flagged cadences (sigma=5 outlier removal)
4. Divide the flux by its median to normalise to a baseline of 1.0
5. Convert the time array to BTJD
6. Return `(time_array, flux_array, flux_err_array)` as numpy arrays

**Flux normalisation detail:**

TESS PDC-SAP flux is in electrons per second (e⁻/s). Dividing by the median converts it to a dimensionless normalised flux where baseline = 1.0. A transit appears as a dip below 1.0.

**Gap handling:**

Real TESS sectors have a data download gap of approximately 1 day near the midpoint of each 27-day sector. The normaliser does not fill this gap — it returns the time array as-is. ml-core's BLS implementation must handle gapped time arrays correctly.

### Step 3.4 — Extend `interface.py` for real TESS

Update `load_light_curve()` to handle `source == "tess"`:

1. Accept a `target_id` that is a TIC ID string (e.g., `"25155310"`)
2. Call `sector_selector.py` to determine the best sector
3. Call `mast_loader.py` to fetch or load from cache
4. Call `flux_normaliser.py` to convert to standard arrays
5. Populate metadata: set `sector` from selected sector, set `label` to None, set all `true_*` fields to None
6. Return the standard dict

**Optional label lookup:**

For the three recommended demo TIC IDs, hardcode a lookup table in `interface.py` that provides known `label` and `true_period` values for evaluation purposes.

### Phase 3 Completion Checklist

- [ ] `mast_loader.py` successfully downloads WASP-126b (TIC 25155310) when internet is available
- [ ] Downloaded file is saved to `cache/` and reloaded on subsequent calls without network access
- [ ] `flux_normaliser.py` converts PDC-SAP flux to normalised float array
- [ ] Normalised flux has median ≈ 1.0 ± 0.001
- [ ] `interface.py` returns a valid dict for `source="tess"`, `target_id="25155310"`
- [ ] Error handling works: graceful failure when network is unavailable
- [ ] `cache/*.fits` is in `.gitignore`

---

## 8. Phase 4 — Interface, Tests, and Polish

### Goal

Harden `interface.py` with full validation, write the test suite, and complete all documentation.

### Step 4.1 — Complete `interface.py`

**Validation checks:**

1. `len(time) == len(flux)` — raise `DataShapeError` if not
2. `time` is monotonically increasing — raise `DataQualityError` if not
3. `abs(median(flux) - 1.0) < 0.001` — raise `DataNormalisationError` if not
4. `source` is one of the three allowed values — raise `InvalidSourceError` if not
5. If `label` is not None, it must be one of the three class strings — raise `InvalidLabelError` if not
6. `n_points == len(time)` — must always be consistent

**Exception hierarchy:**

All custom exceptions must inherit from a single `DataPipelineError` base class.

**The `config` parameter:**

Allow an optional `config` dict to override any parameter from `config.yaml`. Used in testing and in ml-core when requesting a specific synthetic case variant.

### Step 4.2 — Write `tests/conftest.py`

Define shared pytest fixtures:

- `tiny_lc`: a minimal valid light curve dict with 100 points — for fast unit tests
- `candidate_a_lc`: the full loaded dict for candidate_a
- `candidate_b_lc`: the full loaded dict for candidate_b
- `candidate_c_lc`: the full loaded dict for candidate_c
- `synthetic_config`: the parsed config.yaml as a Python dict
- `mock_mast_response`: a mock LightCurve object for testing without network access

### Step 4.3 — Write `tests/test_generator.py`

Tests to include:

- Output arrays have the correct length matching `n_points` from config
- Time array is monotonically increasing
- Time array spans `time_span_days` ± 0.01 days
- Flux array is all 1.0 before noise is added
- Two calls with the same seed produce identical output
- Two calls with different seeds produce different output

### Step 4.4 — Write `tests/test_schema.py`

Tests to include:

- `candidate_a.csv` has exactly three columns: time, flux, flux_err
- `candidate_a.csv` has the correct number of rows (18000)
- No NaN values in any CSV
- Flux values are within a physically plausible range (0.5 to 1.5)
- Time values are within a plausible BTJD range (1300 to 2000)
- Header row is present and column names match the schema exactly
- All three candidate CSVs pass identical checks

### Step 4.5 — Write `tests/test_interface.py`

Tests to include:

- `load_light_curve("synthetic", "candidate_a")` returns a dict with all required keys
- Returned dict passes all invariant checks
- `n_points` matches `len(time)` and `len(flux)`
- `source` field equals "synthetic"
- `metadata.label` equals "exoplanet_like" for candidate_a
- `metadata.true_period` equals the config value for candidate_a
- Calling with an unknown `target_id` raises a descriptive exception
- Calling with an invalid `source` raises a descriptive exception
- The `config` override parameter correctly changes generation parameters

### Step 4.6 — Complete all documentation

**`README.md` must contain:**

1. One-sentence description of what this repo does
2. Where it fits in the tri-repo system (ASCII diagram)
3. Quick start: how to install and generate all synthetic cases in under 3 commands
4. How to run the tests
5. How to add a new synthetic case
6. How to fetch real TESS data (Phase 3)
7. Output contract description
8. Links to the other two repos

**`CONTRIBUTING.md` must contain:**

1. Coding conventions (function names snake_case, class names PascalCase)
2. How to add a new noise model
3. How to add a new target class
4. Testing requirements (all new functions need a test)
5. The boundary rule: what belongs here vs ml-core

### Phase 4 Completion Checklist

- [ ] `interface.py` validates all invariants before returning
- [ ] Custom exception hierarchy defined and used consistently
- [ ] `conftest.py` defines all five fixtures
- [ ] `test_generator.py` has at least 6 passing tests
- [ ] `test_schema.py` has at least 6 passing tests
- [ ] `test_interface.py` has at least 8 passing tests
- [ ] `pytest` runs with zero failures from the repo root
- [ ] `README.md` has quick-start instructions that work in under 3 commands
- [ ] `CONTRIBUTING.md` explains the boundary rule

---

## 9. Phase 5 — Stretch Goals

These are post-hackathon enhancements that significantly increase scientific depth and startup potential.

### 5.1 — Expanded synthetic dataset

Generate 50–100 synthetic cases per class with randomised parameters drawn from physically realistic distributions. This gives ml-core enough labeled data to train and properly evaluate a Random Forest classifier.

Parameter distributions to use:
- Exoplanet periods: log-uniform between 1.0 and 15.0 days
- Exoplanet depths: uniform between 0.001 and 0.02
- Eclipsing binary periods: uniform between 0.5 and 5.0 days
- Eclipsing binary depths: uniform between 0.05 and 0.40
- Noise levels: uniform between 0.001 and 0.020

### 5.2 — Lightkurve integration with TOI catalogue

Fetch light curves for 20+ confirmed TOI targets with known dispositions (PC = Planet Candidate, EB = Eclipsing Binary, FP = False Positive). Label using published TOI dispositions. Creates a semi-real labeled dataset that demonstrates the system works on genuine observational data.

### 5.3 — Flux contamination simulator

Real TESS pixels are large (21 arcseconds). A background eclipsing binary in the same pixel can mimic a planetary transit. Add a contamination model to `transit_injector.py` that blends two synthetic light curves (target + contaminant). Enables testing the classifier's ability to identify "blend" candidates.

### 5.4 — Starspot / stellar variability model

Add a periodic sinusoidal modulation to `noise_models.py` to simulate starspots rotating in and out of view. Creates a fourth class — `stellar_variability` — that looks superficially transit-like but is much smoother and does not have the sharp ingress/egress of a true transit.

### 5.5 — Data versioning with DVC

Integrate DVC (Data Version Control) to track the `synthetic/cases/` and `datasets/` directories. Allows exact reproduction of any evaluation run from a specific DVC commit.

---

## 10. File-by-File Responsibility Matrix

| File | Owned by | Input | Output | Used by |
|------|----------|-------|--------|---------|
| `interface.py` | data-pipeline | source, target_id, config | light curve dict | ml-core |
| `synthetic/config.yaml` | data-pipeline | — | generation parameters | generator, injector, interface |
| `synthetic/generator.py` | data-pipeline | config params | (time[], flux[]) | interface, injector |
| `synthetic/transit_injector.py` | data-pipeline | (time, flux), transit params | flux[] with dip | interface |
| `synthetic/noise_models.py` | data-pipeline | flux[], noise params | flux[] with noise | interface |
| `synthetic/cases/*.csv` | data-pipeline (generated) | — | time, flux, flux_err | interface, datasets |
| `real_tess/mast_loader.py` | data-pipeline | TIC ID, sector | LightCurve object | interface |
| `real_tess/sector_selector.py` | data-pipeline | search results | sector int | interface |
| `real_tess/flux_normaliser.py` | data-pipeline | LightCurve object | (time[], flux[], err[]) | interface |
| `datasets/build_dataset.py` | data-pipeline | all case CSVs, metadata.json | labeled_dataset.csv | ml-core eval |
| `datasets/labeled_dataset.csv` | data-pipeline (generated) | — | labeled rows | ml-core |
| `datasets/metadata.json` | data-pipeline | — | per-target metadata | build_dataset, interface |
| `datasets/schema.md` | data-pipeline | — | — | judges, CONTRIBUTING |
| `tests/` | data-pipeline | test fixtures | pass/fail | CI, developers |

---

## 11. Dependencies and Install Plan

### Production dependencies

| Package | Version | Why needed | Required for |
|---------|---------|-----------|-------------|
| `numpy` | ≥ 1.24 | Array operations, time/flux arrays | All phases |
| `pandas` | ≥ 2.0 | CSV reading/writing, dataset assembly | All phases |
| `scipy` | ≥ 1.11 | AR(1) noise generation, signal utilities | Phase 1 |
| `pyyaml` | ≥ 6.0 | Reading config.yaml | All phases |
| `lightkurve` | ≥ 2.4 | TESS data download from MAST | Phase 3 only |
| `astroquery` | ≥ 0.4.6 | MAST catalogue queries | Phase 3 only |

### Development dependencies

| Package | Version | Why needed |
|---------|---------|-----------|
| `pytest` | ≥ 7.4 | Test runner |
| `pytest-cov` | ≥ 4.1 | Coverage reporting |
| `black` | ≥ 23.0 | Code formatting |

### Install strategy for hackathon

**Offline demo install (Phase 1 + 2 only, fast):**

```
pip install numpy pandas scipy pyyaml
```

Installs in under 60 seconds. No internet required after install.

**Full install with real TESS support (Phase 3):**

```
pip install numpy pandas scipy pyyaml lightkurve astroquery
```

Takes 3–5 minutes. Must be done before the hackathon if internet is unreliable at the venue.

### Python version requirement

Python 3.10 or higher.

---

## 12. Configuration Reference

### `synthetic/config.yaml` — complete specification

```yaml
# Global generation parameters
generation:
  n_points: 18000
  time_span_days: 27.0
  cadence_minutes: 2.0
  baseline_flux: 1.0
  btjd_start: 1325.3

cases:

  candidate_a:
    label: "exoplanet_like"
    period_days: 3.42
    depth: 0.013
    duration_days: 0.16
    t0_offset: 0.3
    noise_level: 0.002
    noise_type: "gaussian"
    v_shape: false
    secondary_eclipse: false
    seed: 42

  candidate_b:
    label: "eclipsing_binary_like"
    period_days: 1.87
    depth: 0.18
    duration_days: 0.08
    t0_offset: 0.15
    noise_level: 0.003
    noise_type: "gaussian"
    v_shape: true
    secondary_eclipse: true
    secondary_depth_ratio: 0.6
    seed: 43

  candidate_c:
    label: "noise_or_other"
    period_days: null
    depth: null
    duration_days: null
    t0_offset: null
    noise_level: 0.015
    noise_type: "red"
    red_noise_rho: 0.7
    v_shape: false
    secondary_eclipse: false
    seed: 44
```

### Environment variables

No environment variables required for Phase 1 and 2. Phase 3 uses no API keys — MAST is publicly accessible without authentication.

---

## 13. Data Schema Specification

### CSV output format (all synthetic cases)

- Columns (in order): `time`, `flux`, `flux_err`
- Separator: comma
- Decimal: period
- Encoding: UTF-8
- Header: first row, lowercase column names
- Index: none
- Line endings: Unix (LF)
- Missing values: none allowed in Phase 1 output

### Value ranges (synthetic data)

| Column | Minimum | Maximum | Typical |
|--------|---------|---------|---------|
| `time` | 1300.0 | 1500.0 | 1325.3 to 1352.3 |
| `flux` | 0.7 | 1.1 | 0.98 to 1.02 (baseline) |
| `flux_err` | 0.0001 | 0.05 | equals noise_level |

### Flux value interpretation

A `flux` value of 1.0 means the star is at its median brightness. Values below 1.0 mean the star appears dimmer than normal (as during a transit). Values above 1.0 occur due to noise only in the synthetic data.

A transit with `depth = 0.013` means the star's flux drops to `1.0 - 0.013 = 0.987` at transit centre — a 1.3% dimming consistent with a Jupiter-sized planet blocking 1.3% of the stellar disc.

---

## 14. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MAST network unavailable at hackathon venue | High | Medium | Phase 3 is optional; pre-download cache files before event |
| `lightkurve` install fails due to dependency conflict | Medium | Medium | Test install on fresh environment before event; document exact working versions |
| Synthetic transit signal too weak for BLS to detect | Low | High | Verify BLS recovers all three cases manually; use config to increase depth if needed |
| `interface.py` output shape changes breaking ml-core | Medium | High | Define contract in writing (Section 3) and validate in tests; never change top-level key names |
| Red noise in candidate_c accidentally looks periodic | Low | Medium | Run BLS on candidate_c and confirm no significant peak above threshold |
| Time array has floating point precision issues | Low | Low | Use `np.linspace` rather than `np.arange` to avoid cumulative FP errors |
| Mismatch between synthetic cadence and real TESS cadence | Low | Medium | Parameterise cadence in config; real TESS normaliser converts to the same time units |

---

## 15. Hackathon Priority Tiers

### Tier 1 — Must-have (complete before anything else)

Everything required for the three-candidate demo to run end-to-end:

- `synthetic/config.yaml` with all three cases defined
- `synthetic/generator.py` producing valid time and flux arrays
- `synthetic/transit_injector.py` injecting box dips correctly
- `synthetic/noise_models.py` adding Gaussian noise
- `synthetic/cases/candidate_a.csv` — 18000 rows, 3 columns
- `synthetic/cases/candidate_b.csv` — with V-shape and secondary eclipse
- `synthetic/cases/candidate_c.csv` — red noise, no transit
- `interface.py` — returns valid dict for all three cases
- `datasets/schema.md` — basic column definitions

**Estimated effort: 3–4 hours. Must be complete before ml-core can start.**

### Tier 2 — Should-have (complete if time allows)

Makes the evaluation story credible to judges:

- `datasets/metadata.json` with entries for all three cases
- `datasets/build_dataset.py` assembling the labeled CSV
- `datasets/labeled_dataset.csv` with correct labels
- `datasets/splits/` with train/val/test CSVs
- `tests/test_schema.py` with basic shape checks
- `tests/test_interface.py` with return dict validation
- `README.md` with quick-start instructions

**Estimated effort: 2–3 hours.**

### Tier 3 — Nice-to-have (strong judge impression)

Provides the "does it work on real data?" wow moment:

- `real_tess/mast_loader.py` fetching WASP-126b
- `real_tess/flux_normaliser.py` converting PDC-SAP flux
- Cache pre-populated with at least one FITS file
- `interface.py` extended for `source="tess"`

**Estimated effort: 3–5 hours. Requires internet during development.**

### Tier 4 — Stretch (post-hackathon)

Everything in Phase 5. Do not attempt during the hackathon.

---

## 16. Definition of Done

The `transitlens-data-pipeline` repo is considered complete for hackathon submission when:

1. Calling `load_light_curve("synthetic", "candidate_a")` returns a valid dict within 1 second
2. All three synthetic CSV files exist in `synthetic/cases/` with 18000 rows and 3 columns each
3. `pytest tests/` runs with zero failures
4. `README.md` contains a working quick-start in 3 or fewer commands
5. The output dict passes all invariant checks defined in Section 3
6. No import from `transitlens-ml-core` or `transitlens-platform` exists anywhere in the repo
7. The repo installs from scratch using only `pip install numpy pandas scipy pyyaml` for the offline demo
8. `datasets/schema.md` is complete and matches the actual CSV columns
9. `synthetic/cases/README.md` documents the physical parameters used for each case
10. `.gitignore` excludes `*.fits`, `*.pkl`, `__pycache__`, `.DS_Store`, and `*.pyc`

---

*This document covers the complete engineering plan for `transitlens-data-pipeline`. No code is included. All implementation decisions, physical parameters, validation requirements, and inter-repo contracts are documented here for use during the hackathon build.*

*Next document: `transitlens-ml-core-PLAN.md`*