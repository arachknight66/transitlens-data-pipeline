# transitlens-data-pipeline — Implementation Plan

> **Project:** Bharatiya Antariksh Hackathon 2026 — PS7  
> **Repo role:** Feeds cleaned, labelled light curve data to `transitlens-ml-core`  
> **Status:** Scaffolded (empty files) — ready for implementation  
> **Last updated:** 2026-06-21

---

## Table of Contents

1. [Repo purpose and boundaries](#1-repo-purpose-and-boundaries)
2. [Folder structure reference](#2-folder-structure-reference)
3. [Output contract](#3-output-contract)
4. [Phase 0 — Environment setup](#phase-0--environment-setup)
5. [Phase 1 — Synthetic data generation (MVP)](#phase-1--synthetic-data-generation-mvp)
6. [Phase 2 — Dataset assembly](#phase-2--dataset-assembly)
7. [Phase 3 — Interface layer](#phase-3--interface-layer)
8. [Phase 4 — Tests](#phase-4--tests)
9. [Phase 5 — Real TESS data (stretch)](#phase-5--real-tess-data-stretch)
10. [Phase 6 — Notebooks](#phase-6--notebooks)
11. [File-by-file implementation guide](#file-by-file-implementation-guide)
12. [Config reference](#config-reference)
13. [Schema reference](#schema-reference)
14. [Priority matrix](#priority-matrix)
15. [What this repo must never do](#what-this-repo-must-never-do)
16. [Handoff checklist to ml-core](#handoff-checklist-to-ml-core)

---

## 1. Repo purpose and boundaries

### What this repo does

`transitlens-data-pipeline` has exactly one job: **produce light curve data in a standard shape that `transitlens-ml-core` can consume.**

It handles:
- Generating synthetic TESS-like light curves (MVP — offline, no internet required)
- Loading and caching real TESS data from MAST via Lightkurve (stretch)
- Assembling labelled datasets for training and evaluation
- Exposing a single entry-point function `load_light_curve()` via `interface.py`

### What this repo does NOT do

| Responsibility | Belongs to |
|---|---|
| BLS period search | `transitlens-ml-core` |
| Feature extraction | `transitlens-ml-core` |
| Classification | `transitlens-ml-core` |
| Plot generation | `transitlens-ml-core` |
| Dashboard / UI | `transitlens-platform` |
| API endpoints | `transitlens-ml-core` |

The boundary is strict. If a file in this repo is doing signal processing, it is in the wrong repo.

### Why this separation matters

- A judge can open `ml-core` and see only science. Clean.
- Data sources (synthetic vs real TESS) can be swapped without touching the ML logic.
- The pipeline can be tested independently of the classifier.

---

## 2. Folder structure reference

```
transitlens-data-pipeline/
│
├── README.md                          ← project overview, quick start
├── CONTRIBUTING.md                    ← how to add new synthetic cases
├── requirements.txt                   ← numpy, pandas, scipy, pyyaml (+ optional lightkurve)
├── .gitignore                         ← ignore cache/, *.fits, *.pyc, __pycache__
├── interface.py                       ← load_light_curve() — only file ml-core imports
│
├── synthetic/
│   ├── __init__.py
│   ├── generator.py                   ← Phase 1A: time array + base flux
│   ├── noise_models.py                ← Phase 1B: Gaussian + red noise
│   ├── transit_injector.py            ← Phase 1C: periodic dip injection
│   ├── config.yaml                    ← Phase 1D: all case parameters
│   └── cases/
│       ├── README.md
│       ├── candidate_a.csv            ← exoplanet_like output
│       ├── candidate_b.csv            ← eclipsing_binary_like output
│       └── candidate_c.csv            ← noise_or_other output
│
├── real_tess/
│   ├── __init__.py
│   ├── mast_loader.py                 ← Phase 5A: Lightkurve fetch by TIC ID
│   ├── sector_selector.py             ← Phase 5B: pick best sector
│   ├── flux_normaliser.py             ← Phase 5C: PDC SAP → normalised flux
│   └── cache/
│       └── .gitkeep                   ← local .fits store (git-ignored)
│
├── datasets/
│   ├── __init__.py
│   ├── build_dataset.py               ← Phase 2A: assembles labeled_dataset.csv
│   ├── labeled_dataset.csv            ← Phase 2B: ground truth CSV
│   ├── metadata.json                  ← Phase 2C: per-case metadata
│   ├── schema.md                      ← Phase 2D: column definitions
│   └── splits/
│       ├── train.csv                  ← Phase 2E: 70% split
│       ├── val.csv                    ← Phase 2E: 15% split
│       └── test.csv                   ← Phase 2E: 15% split
│
├── notebooks/
│   ├── exploration.ipynb              ← Phase 6A: visual inspection of cases
│   └── synthetic_visualisation.ipynb  ← Phase 6B: noise and transit parameter sweep
│
└── tests/
    ├── __init__.py
    ├── conftest.py                    ← Phase 4A: shared fixtures
    ├── test_generator.py              ← Phase 4B
    ├── test_noise_models.py           ← Phase 4B
    ├── test_transit_injector.py       ← Phase 4B
    ├── test_schema.py                 ← Phase 4C
    ├── test_loader.py                 ← Phase 4D
    └── test_interface.py              ← Phase 4E
```

---

## 3. Output contract

Every call to `load_light_curve()` must return this exact shape regardless of whether the source is synthetic or real TESS. `transitlens-ml-core` depends on this shape and must never be changed without updating both repos simultaneously.

```python
{
    # Arrays — always present
    "time":  List[float],        # BTJD timestamps, length N
    "flux":  List[float],        # normalised flux, median ~1.0, length N

    # Identity — always present
    "target_id": str,            # e.g. "candidate_a" or "TIC-25155310"
    "source":    str,            # "synthetic" | "tess" | "csv"

    # Diagnostics — always present
    "n_points": int,             # len(time) == len(flux)

    # Metadata dict — always present, some keys may be None
    "metadata": {
        "cadence_min":    float,         # 2.0 for TESS 2-min cadence
        "time_span_days": float,         # total observation window
        "sector":         int | None,    # TESS sector number, None for synthetic
        "label":          str | None,    # ground truth class, None if unknown
        "true_period":    float | None,  # days, None for noise/unknown
        "true_depth":     float | None,  # fractional flux drop, None if N/A
        "true_duration":  float | None,  # days, None if N/A
    }
}
```

### Contract rules

- `time` and `flux` are always plain Python lists of floats (not numpy arrays) so they are JSON-serialisable.
- `flux` is always normalised: median = 1.0, values typically in range [0.97, 1.03].
- `label` is one of: `"exoplanet_like"`, `"eclipsing_binary_like"`, `"noise_or_other"`, or `None`.
- If `true_period` is `None`, it means either noise case (no period) or real data with unknown period.
- `n_points` is always `len(time)`. Never trust a cached value — always recompute.

---

## Phase 0 — Environment setup

**Goal:** Repository is installable and importable in under 60 seconds.  
**Time estimate:** 15 minutes  
**Files touched:** `requirements.txt`, `.gitignore`, `README.md`

### Step 0.1 — Write `requirements.txt`

```
# Core — always required
numpy>=1.24
pandas>=2.0
scipy>=1.11
pyyaml>=6.0

# Testing
pytest>=7.4

# Real TESS — comment out for offline / hackathon demo mode
# lightkurve>=2.4
# astroquery>=0.4.6
```

Keep `lightkurve` and `astroquery` commented out. They pull in dozens of transitive dependencies and add 2–3 minutes to install time. Only uncomment when actively working on Phase 5.

### Step 0.2 — Write `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.venv/
env/

# TESS data cache
real_tess/cache/*.fits
real_tess/cache/*.fits.gz

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Large model artefacts (if any land here)
*.pkl
*.h5
```

### Step 0.3 — Verify install

```bash
cd transitlens-data-pipeline
pip install -r requirements.txt
python -c "import numpy, pandas, scipy, yaml; print('OK')"
```

Expected output: `OK`

---

## Phase 1 — Synthetic data generation (MVP)

**Goal:** Three synthetic TESS-like light curves written to `synthetic/cases/` as CSVs.  
**Time estimate:** 3–4 hours  
**Priority:** CRITICAL — everything in ml-core depends on this working first.

### Phase 1A — `synthetic/config.yaml`

Implement this first. It is the single source of truth for all synthetic parameters. Every other file in Phase 1 reads from it.

```yaml
# ─────────────────────────────────────────────
# Global generation settings
# ─────────────────────────────────────────────
generation:
  n_points: 18000          # ~27 days at 2-min TESS cadence
  time_span_days: 27.0
  cadence_minutes: 2.0

# ─────────────────────────────────────────────
# Synthetic cases
# ─────────────────────────────────────────────
cases:

  candidate_a:
    label: exoplanet_like
    description: "Shallow periodic transit, flat bottom, consistent depth"
    period_days: 3.42
    depth: 0.013           # 1.3% flux drop
    duration_days: 0.16    # ~3.8 hours
    v_shape: false
    secondary_eclipse: false
    noise_level: 0.002
    noise_type: gaussian
    seed: 42

  candidate_b:
    label: eclipsing_binary_like
    description: "Deep V-shaped primary + secondary eclipse at half period"
    period_days: 1.87
    depth: 0.18            # 18% — much deeper than a planet
    duration_days: 0.08
    v_shape: true
    secondary_eclipse: true
    secondary_depth: 0.09  # secondary eclipse at half period
    noise_level: 0.003
    noise_type: gaussian
    seed: 43

  candidate_c:
    label: noise_or_other
    description: "No transit signal — red noise + stellar variability"
    period_days: null
    depth: null
    duration_days: null
    v_shape: false
    secondary_eclipse: false
    noise_level: 0.015
    noise_type: red
    seed: 44
```

**Why these values?**
- Candidate A mimics a hot Jupiter or super-Earth around a quiet star. Depth 1.3% is realistic for a Jupiter-sized planet around a Sun-like star. 8 transits visible in 27 days.
- Candidate B mimics a contact eclipsing binary. The 18% depth and V-shape are dead giveaways the classifier must catch.
- Candidate C mimics a star with stellar variability and detector red noise — the hardest "nothing here" case.

### Phase 1B — `synthetic/generator.py`

Generates the base time array and flat normalised flux. No noise, no transit — just the skeleton.

**Key functions to implement:**

```python
def make_time_array(n_points, time_span_days, cadence_minutes):
    """
    Returns a 1D numpy array of timestamps in BTJD.
    Starts at 0.0, spaced by cadence_minutes/1440 days.
    Includes small random gaps (~2%) to simulate TESS momentum dumps.
    """

def make_base_flux(n_points):
    """
    Returns a 1D numpy array of ones with length n_points.
    This is the flat, noiseless, transitless baseline.
    """

def generate_from_config(config_path, case_name):
    """
    Reads config.yaml, calls make_time_array + make_base_flux,
    then hands off to noise_models and transit_injector.
    Returns (time, flux, metadata_dict).
    """

def generate_all_cases(config_path, output_dir):
    """
    Loops over all cases in config.yaml.
    Calls generate_from_config for each.
    Writes CSV to output_dir/candidate_{name}.csv.
    Prints a summary line per case.
    """
```

**Gap simulation detail:** Real TESS data has 0.5–2 day gaps every ~13 days for data downlink and momentum wheel desaturation. Simulate by randomly removing ~2% of points from the time array. This makes BLS detection slightly more realistic.

### Phase 1C — `synthetic/noise_models.py`

Adds realistic photometric noise to the base flux array.

**Key functions to implement:**

```python
def add_gaussian_noise(flux, sigma, seed=None):
    """
    Adds i.i.d. Gaussian noise with standard deviation sigma.
    sigma ~ 0.002 means 2000 ppm noise floor — realistic for a V=11 star.
    Returns noisy flux array (same length as input).
    """

def add_red_noise(flux, sigma, correlation=0.3, seed=None):
    """
    Adds correlated (red) noise via an AR(1) process.
    AR(1): noise[i] = correlation * noise[i-1] + white_noise[i]
    Mimics instrumental systematics and stellar variability trends.
    sigma controls the overall amplitude.
    Returns noisy flux array.
    """

def add_stellar_variability(flux, time, amplitude=0.005, period_days=12.0, seed=None):
    """
    Adds a sinusoidal stellar rotation signal.
    Amplitude ~0.5% is typical for a moderately active star.
    Used only for candidate_c to make it look like real variable star data.
    Returns flux array with variability superimposed.
    """
```

**Noise level reference:**

| Star magnitude | Typical TESS noise (ppm/hour) | Equivalent sigma |
|---|---|---|
| V = 8 | 150 | 0.00015 |
| V = 10 | 400 | 0.0004 |
| V = 12 | 1500 | 0.0015 |
| V = 14 | 5000 | 0.005 |

Use `sigma=0.002` for candidates A and B (quiet, bright-ish stars). Use `sigma=0.015` for candidate C (faint or variable star).

### Phase 1D — `synthetic/transit_injector.py`

Injects the actual transit signal into the noisy flux array.

**Key functions to implement:**

```python
def inject_transit(flux, time, period_days, depth, duration_days,
                   v_shape=False, t0=None):
    """
    Injects a box-shaped (or V-shaped) transit at every phase where
    the planet crosses the stellar disk.

    For each timestamp t:
      phase = ((t - t0) % period_days) / period_days
      if phase < duration_days / period_days:
          if v_shape:
              flux[i] *= (1 - depth * triangle_function(phase))
          else:
              flux[i] *= (1 - depth)

    t0: time of first transit midpoint (default: period_days / 4)
    Returns modified flux array (in-place modification is fine).
    """

def inject_secondary_eclipse(flux, time, period_days,
                               secondary_depth, duration_days, t0=None):
    """
    Injects a secondary eclipse at phase 0.5 (half period from primary).
    Used only for eclipsing binary simulation (candidate_b).
    Secondary depth is typically 40–60% of primary depth.
    """

def compute_transit_count(time, period_days, duration_days, t0=None):
    """
    Returns the integer number of full transits visible in the time array.
    Used to populate metadata['transit_count'] for ml-core feature extraction.
    """
```

**Box vs V-shape geometry:**

```
Box transit (exoplanet):        V-shape transit (eclipsing binary):
flux                            flux
1.0 ─────┐     ┌─────           1.0 ─────\     /─────
         │     │                           \   /
1-depth  └─────┘                1-depth     \ /
         ←dur→                              ←dur→
```

The V-shape is implemented as a triangle function: depth varies linearly from 0 at ingress to full depth at midpoint, then back to 0 at egress. This is the key physical feature that `feature_extractor.py` in ml-core will measure with the v-shape score.

### Phase 1E — Run generation, verify outputs

After implementing Phases 1A–1D, run the generation script:

```bash
cd transitlens-data-pipeline
python -c "
from synthetic.generator import generate_all_cases
generate_all_cases('synthetic/config.yaml', 'synthetic/cases')
"
```

Expected outputs:

```
synthetic/cases/candidate_a.csv  — 18000 rows, columns: time, flux
synthetic/cases/candidate_b.csv  — 18000 rows, columns: time, flux
synthetic/cases/candidate_c.csv  — 18000 rows, columns: time, flux
```

Quick sanity checks to run manually:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("synthetic/cases/candidate_a.csv")
assert len(df) > 15000             # some rows removed for gap simulation
assert abs(df["flux"].median() - 1.0) < 0.01   # normalised
assert df["flux"].min() < 0.99     # transits are present
assert df["flux"].std() > 0.001    # not completely flat
print("candidate_a OK")
```

---

## Phase 2 — Dataset assembly

**Goal:** A single `labeled_dataset.csv` with ground truth labels, plus train/val/test splits.  
**Time estimate:** 1–2 hours  
**Priority:** HIGH — needed by ml-core evaluation module.

### Phase 2A — `datasets/schema.md`

Write this first — it defines what every other dataset file must contain.

```markdown
# Dataset schema

## labeled_dataset.csv columns

| Column | Type | Description | Nullable |
|---|---|---|---|
| target_id | str | Unique identifier for the light curve | No |
| time | float | Timestamp in BTJD | No |
| flux | float | Normalised flux (median ~1.0) | No |
| source | str | "synthetic" or "tess" | No |
| label | str | Ground truth class | Yes (real unlabelled data) |
| true_period | float | Known orbital period in days | Yes |
| true_depth | float | Known transit depth (fractional) | Yes |
| true_duration | float | Known transit duration in days | Yes |
| cadence_min | float | Observation cadence in minutes | No |
| sector | int | TESS sector number | Yes (None for synthetic) |

## Valid label values

- `exoplanet_like`
- `eclipsing_binary_like`
- `noise_or_other`
- `null` — real data with unknown classification

## Notes

- One row per time step. A 27-day light curve at 2-min cadence produces ~18,000 rows per target.
- true_period, true_depth, true_duration are None for noise cases and unlabelled real data.
- sector is None for all synthetic cases.
```

### Phase 2B — `datasets/build_dataset.py`

```python
def build_from_synthetic(cases_dir, config_path, output_path):
    """
    Reads all CSV files from synthetic/cases/.
    Reads labels and parameters from config.yaml.
    Concatenates into labeled_dataset.csv with all schema columns.
    """

def build_from_tess(tess_dir, labels_path, output_path):
    """
    Reads normalised TESS CSVs from real_tess/cache/.
    Merges with manual labels from a labels JSON file.
    Appends to (or creates) labeled_dataset.csv.
    Stretch goal — only implement after Phase 5 is complete.
    """

def split_dataset(dataset_path, splits_dir,
                  train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Reads labeled_dataset.csv.
    Splits by target_id (not by row) to avoid data leakage.
    Writes train.csv, val.csv, test.csv to splits/.
    Prints class distribution per split.
    """
```

**Critical detail — split by target_id, not by row:**
If you split by row, the same light curve ends up in both train and test. Always group by `target_id` first, then assign groups to splits. With only three synthetic cases the splits will be trivial (one case per split), but the logic must be correct for when real TESS data is added later.

### Phase 2C — `datasets/metadata.json`

One JSON object per case. Written automatically by `build_from_synthetic`.

```json
{
  "candidate_a": {
    "target_id": "candidate_a",
    "source": "synthetic",
    "label": "exoplanet_like",
    "true_period": 3.42,
    "true_depth": 0.013,
    "true_duration": 0.16,
    "n_points": 17640,
    "cadence_min": 2.0,
    "time_span_days": 27.0,
    "sector": null,
    "seed": 42,
    "generated_at": "2026-06-21T00:00:00"
  },
  "candidate_b": { "..." : "..." },
  "candidate_c": { "..." : "..." }
}
```

---

## Phase 3 — Interface layer

**Goal:** `interface.py` is complete, tested, and the only file ml-core ever imports.  
**Time estimate:** 1 hour  
**Priority:** CRITICAL — ml-core cannot run without this.

### `interface.py` — complete specification

```python
def load_light_curve(
    source: str,
    target_id: str,
    config: dict = None
) -> dict:
    """
    Single entry point consumed by transitlens-ml-core.

    Parameters
    ----------
    source : str
        One of "synthetic", "tess", or "csv".
        "synthetic" — loads from synthetic/cases/{target_id}.csv
        "tess"      — loads from real_tess/cache/ via mast_loader
        "csv"       — loads from an arbitrary file path passed in config["path"]

    target_id : str
        For synthetic: one of "candidate_a", "candidate_b", "candidate_c"
        For tess: TIC ID string, e.g. "TIC-25155310"
        For csv: descriptive name chosen by caller

    config : dict, optional
        Override any default parameters.
        Valid keys:
            "path"         — used when source="csv"
            "sector"       — override sector selection for TESS
            "cadence_min"  — override cadence assumption
            "generate"     — if True and synthetic case CSV missing, generate it

    Returns
    -------
    dict matching the output contract defined in schema above.

    Raises
    ------
    FileNotFoundError  — source file or cache entry not found
    ValueError         — unknown source type or malformed CSV
    ImportError        — real_tess source requested but lightkurve not installed
    """
```

**Fallback behaviour:**
If `source="synthetic"` and the CSV does not exist in `cases/`, and `config["generate"]=True`, auto-generate it by calling `generate_from_config()`. This makes the interface self-healing during development.

**Loading a synthetic case step by step:**

```
1. Resolve path: synthetic/cases/{target_id}.csv
2. Load CSV with pandas
3. Extract time and flux columns as lists
4. Load metadata from datasets/metadata.json for this target_id
5. Build and return the output dict
```

---

## Phase 4 — Tests

**Goal:** All core modules covered by unit tests. `pytest` passes with zero failures.  
**Time estimate:** 2 hours  
**Priority:** HIGH — proves the pipeline works before ml-core tries to use it.

### Phase 4A — `tests/conftest.py`

Shared fixtures used by all test files.

```python
import pytest
import numpy as np

@pytest.fixture
def synthetic_time():
    """Returns a small 500-point time array for fast tests."""
    return list(np.linspace(0, 5.0, 500))

@pytest.fixture
def synthetic_flux():
    """Returns a flat normalised flux array with light Gaussian noise."""
    rng = np.random.default_rng(seed=0)
    return list(1.0 + rng.normal(0, 0.002, 500))

@pytest.fixture
def config_path(tmp_path):
    """Copies the real config.yaml to a tmp location for test isolation."""
    import shutil
    src = "synthetic/config.yaml"
    dst = tmp_path / "config.yaml"
    shutil.copy(src, dst)
    return str(dst)
```

### Phase 4B — `tests/test_generator.py`

```python
def test_time_array_length():
    # n_points=1000 → array of length ~980-1000 (gaps reduce it slightly)

def test_time_array_monotonic():
    # time must always be strictly increasing

def test_base_flux_all_ones():
    # base flux before noise/transit must be exactly 1.0

def test_generate_candidate_a(tmp_path, config_path):
    # runs full generation for candidate_a
    # checks output CSV exists and has correct columns

def test_all_cases_generate(tmp_path, config_path):
    # loops over all three cases, verifies each CSV is written
```

### Phase 4C — `tests/test_noise_models.py`

```python
def test_gaussian_noise_mean(synthetic_flux):
    # after adding Gaussian noise, mean should still be ~1.0

def test_gaussian_noise_std(synthetic_flux):
    # std of (noisy - original) should be close to sigma parameter

def test_red_noise_correlated():
    # consecutive residuals should have positive autocorrelation at lag 1

def test_seed_reproducibility(synthetic_flux):
    # same seed → identical output; different seed → different output
```

### Phase 4D — `tests/test_transit_injector.py`

```python
def test_transit_creates_dip(synthetic_time, synthetic_flux):
    # after injection, min(flux) must be significantly below 1.0

def test_box_transit_depth(synthetic_time, synthetic_flux):
    # flux inside transit window should be approximately 1.0 - depth

def test_v_shape_deeper_at_center():
    # for v_shape=True, the minimum should be at the midpoint of the transit

def test_transit_count():
    # with period=3.42 days and span=27 days → expect 7-8 transits

def test_secondary_eclipse_at_half_phase(synthetic_time, synthetic_flux):
    # secondary eclipse minimum should be at phase 0.5
```

### Phase 4E — `tests/test_schema.py`

```python
def test_labeled_dataset_columns():
    # labeled_dataset.csv must have all required columns from schema.md

def test_no_nulls_in_required_columns():
    # time, flux, target_id, source, cadence_min must never be null

def test_flux_normalised():
    # median of flux column must be between 0.99 and 1.01 per target

def test_label_values_valid():
    # all non-null labels must be in the valid label set
```

### Phase 4F — `tests/test_interface.py`

```python
def test_load_synthetic_candidate_a():
    # returns dict with correct shape
    # time and flux are lists, not numpy arrays
    # n_points == len(time) == len(flux)

def test_output_contract_all_keys_present():
    # result must contain: time, flux, target_id, source, n_points, metadata

def test_metadata_keys_present():
    # metadata must contain all required keys

def test_flux_values_normalised():
    # abs(median(flux) - 1.0) < 0.01

def test_unknown_source_raises_value_error():
    # load_light_curve("unknown_source", "x") must raise ValueError

def test_missing_case_raises_file_not_found():
    # load_light_curve("synthetic", "nonexistent") must raise FileNotFoundError
```

**Run all tests:**

```bash
cd transitlens-data-pipeline
pytest tests/ -v
```

Expected output: all tests green before handing off to ml-core.

---

## Phase 5 — Real TESS data (stretch)

**Goal:** At least one confirmed exoplanet TIC ID loadable from MAST.  
**Time estimate:** 3–5 hours including MAST download time  
**Priority:** NICE-TO-HAVE — adds massive credibility but not required for demo.

### Recommended TIC IDs for demo

| TIC ID | Common name | Period (days) | Depth | Why it's good |
|---|---|---|---|---|
| TIC 25155310 | WASP-126b | 3.29 | 0.008 | Very clean signal, bright host star |
| TIC 279741377 | TOI-270 b/c/d | 3.36 / 5.66 / 11.38 | multi | Multi-planet, impressive |
| TIC 149603524 | LHS 3844b | 0.46 | 0.004 | Ultra-short period, dramatic |

### Phase 5A — `real_tess/mast_loader.py`

```python
def fetch_light_curve(tic_id: str, sector: int = None,
                      cache_dir: str = "real_tess/cache") -> dict:
    """
    Downloads PDC-SAP flux for a given TIC ID using Lightkurve.

    Steps:
    1. Check cache_dir for {tic_id}_s{sector}.fits — return if found
    2. Call lightkurve.search_lightcurve(f"TIC {tic_id}", mission="TESS")
    3. Download the best result (highest coverage sector)
    4. Save .fits to cache_dir
    5. Extract time and pdcsap_flux columns
    6. Return (time_array, flux_array, sector_int)

    Raises ImportError if lightkurve is not installed.
    """
```

### Phase 5B — `real_tess/sector_selector.py`

```python
def select_best_sector(search_results) -> int:
    """
    Given Lightkurve SearchResult, returns the sector with:
    - Maximum number of data points
    - Fewest NaN values
    - 2-minute cadence preferred over 10-minute
    """
```

### Phase 5C — `real_tess/flux_normaliser.py`

```python
def normalise_pdcsap(flux_raw: np.ndarray,
                     quality_flags: np.ndarray = None) -> np.ndarray:
    """
    Normalises PDC-SAP flux to median = 1.0.

    Steps:
    1. Remove flagged cadences (quality_flags != 0) by setting to NaN
    2. Compute median of unflagged values
    3. Divide entire array by median
    4. Clip extreme outliers beyond 5-sigma
    5. Return normalised array
    """
```

### Phase 5D — Cache strategy

```
real_tess/cache/
    TIC-25155310_s007.fits       ← sector 7 for WASP-126
    TIC-279741377_s004.fits      ← sector 4 for TOI-270
    TIC-149603524_s001.fits      ← sector 1 for LHS 3844b
    .gitkeep
```

Cache files are git-ignored (too large for version control). The `README.md` inside `real_tess/` documents which TIC IDs have been pre-downloaded and how to re-download them.

---

## Phase 6 — Notebooks

**Goal:** Visual verification of all synthetic cases and parameter exploration.  
**Time estimate:** 1–2 hours  
**Priority:** MEDIUM — useful for the hackathon presentation and for debugging.

### Phase 6A — `notebooks/exploration.ipynb`

Cells to include:

```
Cell 1: Load all three synthetic cases via interface.py
Cell 2: Plot raw flux for each case (matplotlib subplots, 3×1)
Cell 3: Plot phase-folded flux for candidates A and B using true_period
Cell 4: Show noise level comparison across cases
Cell 5: Print summary statistics table
```

### Phase 6B — `notebooks/synthetic_visualisation.ipynb`

Parameter sweep to understand how changing config values affects the output:

```
Cell 1: Depth sweep (0.001 to 0.2) — plot transit visibility vs SNR
Cell 2: Period sweep (1.0 to 15.0 days) — plot number of visible transits
Cell 3: Noise level sweep — plot at what sigma the transit becomes undetectable
Cell 4: V-shape vs box — side-by-side comparison of same depth, different shape
```

This notebook is the "science explainer" for the judging panel Q&A. Print it to PDF and keep it ready.

---

## File-by-file implementation guide

This table gives the exact implementation order and estimated time for every file in the repo.

| Order | File | Phase | Est. time | Depends on |
|---|---|---|---|---|
| 1 | `requirements.txt` | 0 | 5 min | — |
| 2 | `.gitignore` | 0 | 5 min | — |
| 3 | `synthetic/config.yaml` | 1A | 15 min | — |
| 4 | `synthetic/generator.py` | 1B | 45 min | config.yaml |
| 5 | `synthetic/noise_models.py` | 1C | 30 min | — |
| 6 | `synthetic/transit_injector.py` | 1D | 45 min | — |
| 7 | Run generation → `cases/*.csv` | 1E | 10 min | 4, 5, 6 |
| 8 | `datasets/schema.md` | 2A | 15 min | — |
| 9 | `datasets/build_dataset.py` | 2B | 45 min | 7, 8 |
| 10 | `datasets/metadata.json` | 2C | auto | 9 |
| 11 | `datasets/labeled_dataset.csv` | 2B | auto | 9 |
| 12 | `datasets/splits/*.csv` | 2E | auto | 9 |
| 13 | `interface.py` | 3 | 60 min | 7, 10 |
| 14 | `tests/conftest.py` | 4A | 20 min | — |
| 15 | `tests/test_generator.py` | 4B | 30 min | 4, 14 |
| 16 | `tests/test_noise_models.py` | 4B | 20 min | 5, 14 |
| 17 | `tests/test_transit_injector.py` | 4B | 20 min | 6, 14 |
| 18 | `tests/test_schema.py` | 4C | 20 min | 11 |
| 19 | `tests/test_interface.py` | 4E | 30 min | 13 |
| 20 | `README.md` | — | 20 min | all above |
| 21 | `notebooks/exploration.ipynb` | 6A | 45 min | 13 |
| 22 | `real_tess/mast_loader.py` | 5A | 60 min | lightkurve |
| 23 | `real_tess/sector_selector.py` | 5B | 30 min | 22 |
| 24 | `real_tess/flux_normaliser.py` | 5C | 30 min | 22 |

**Total hackathon-critical path (items 1–19):** ~7.5 hours  
**With real TESS (items 22–24):** additional ~2 hours

---

## Config reference

### `synthetic/config.yaml` — complete field reference

| Field | Type | Description |
|---|---|---|
| `generation.n_points` | int | Total data points per light curve |
| `generation.time_span_days` | float | Total observation window in days |
| `generation.cadence_minutes` | float | Sampling interval in minutes |
| `cases.{name}.label` | str | Ground truth class |
| `cases.{name}.period_days` | float\|null | Orbital/binary period |
| `cases.{name}.depth` | float\|null | Transit depth as fraction of flux |
| `cases.{name}.duration_days` | float\|null | Transit duration in days |
| `cases.{name}.v_shape` | bool | True for eclipsing binary shape |
| `cases.{name}.secondary_eclipse` | bool | True for EB secondary at phase 0.5 |
| `cases.{name}.secondary_depth` | float\|null | Depth of secondary eclipse |
| `cases.{name}.noise_level` | float | Sigma of primary noise component |
| `cases.{name}.noise_type` | str | "gaussian" or "red" |
| `cases.{name}.seed` | int | Random seed for reproducibility |

---

## Schema reference

### `datasets/labeled_dataset.csv` — column definitions

| Column | Type | Example | Required |
|---|---|---|---|
| `target_id` | str | `candidate_a` | Yes |
| `time` | float | `1.2048` | Yes |
| `flux` | float | `0.9987` | Yes |
| `source` | str | `synthetic` | Yes |
| `label` | str\|null | `exoplanet_like` | No |
| `true_period` | float\|null | `3.42` | No |
| `true_depth` | float\|null | `0.013` | No |
| `true_duration` | float\|null | `0.16` | No |
| `cadence_min` | float | `2.0` | Yes |
| `sector` | int\|null | `null` | No |

---

## Priority matrix

| Phase | Files | Hackathon critical | Demo impact | Time |
|---|---|---|---|---|
| 0 | `requirements.txt`, `.gitignore` | Yes | Low | 15 min |
| 1A | `config.yaml` | Yes | Low | 15 min |
| 1B | `generator.py` | Yes | Low | 45 min |
| 1C | `noise_models.py` | Yes | Low | 30 min |
| 1D | `transit_injector.py` | Yes | High | 45 min |
| 1E | Run generation, verify CSVs | Yes | High | 10 min |
| 2 | `build_dataset.py`, `labeled_dataset.csv` | Yes | Medium | 1 h |
| 3 | `interface.py` | Yes | High | 1 h |
| 4 | All tests | Yes | Low | 2 h |
| 6A | `exploration.ipynb` | No | Medium | 45 min |
| 5 | Real TESS modules | No | Very high | 2–3 h |
| 6B | `synthetic_visualisation.ipynb` | No | Medium | 1 h |

---

## What this repo must never do

These are hard rules. Violating them breaks the tri-repo separation.

1. **Never run BLS.** Period search belongs in `ml-core/core/bls_detector.py`.
2. **Never compute SNR or classify.** That is `ml-core/core/classifier.py`.
3. **Never generate plots for the dashboard.** That is `transitlens-platform`.
4. **Never import from `transitlens-ml-core`.** Data flows one way: pipeline → ml-core.
5. **Never put model weights or trained classifiers here.** They belong in `ml-core/models/`.
6. **Never hard-code file paths outside of `config.yaml`.** All paths must be configurable.

---

## Handoff checklist to ml-core

Before `transitlens-ml-core` can start consuming data from this repo, verify all items below.

### Functional checklist

- [ ] `pytest tests/ -v` passes with zero failures
- [ ] `synthetic/cases/candidate_a.csv` exists and has `time` and `flux` columns
- [ ] `synthetic/cases/candidate_b.csv` exists and has `time` and `flux` columns
- [ ] `synthetic/cases/candidate_c.csv` exists and has `time` and `flux` columns
- [ ] `datasets/labeled_dataset.csv` exists with all schema columns
- [ ] `datasets/metadata.json` contains entries for all three cases
- [ ] `interface.py` is importable: `from interface import load_light_curve`
- [ ] `load_light_curve("synthetic", "candidate_a")` returns a valid dict
- [ ] `load_light_curve("synthetic", "candidate_b")` returns a valid dict
- [ ] `load_light_curve("synthetic", "candidate_c")` returns a valid dict

### Output contract checklist

- [ ] `result["time"]` is a Python list of floats
- [ ] `result["flux"]` is a Python list of floats
- [ ] `len(result["time"]) == len(result["flux"]) == result["n_points"]`
- [ ] `abs(statistics.median(result["flux"]) - 1.0) < 0.01`
- [ ] `result["metadata"]` contains all required keys
- [ ] `result["metadata"]["label"]` is one of the valid label strings or `None`

### Stretch checklist (Phase 5)

- [ ] `load_light_curve("tess", "TIC-25155310")` returns a valid dict
- [ ] Real TESS flux is normalised to median = 1.0
- [ ] `.fits` file is cached in `real_tess/cache/`

---

*End of plan — `transitlens-data-pipeline` v0.1*  
*Next: implement Phase 0 → Phase 1 → Phase 3 in order.*  
*Do not begin Phase 5 until the ml-core MVP is demo-ready.*
