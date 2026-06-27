# Dataset Data Card: Phase 1 Evaluation Benchmark

This Data Card documents the design, scope, provenance, validation, and limitations of the TransitLens exoplanet transit detection benchmark dataset.

---

## 1. Dataset Purpose
The purpose of the Phase 1 benchmark dataset is to establish a rigorous, target-disjoint, and class-stratified evaluation environment for machine learning classifiers. It transitions evaluation from point-by-point tabular rows containing toy synthetic data to structured, light-curve time-series arrays representing both simulated and real observations.

---

## 2. Included Sources & Evidence Levels
Every target in the processed dataset is classified under a strict evidence-level taxonomy:

| Evidence Level | Description |
| :--- | :--- |
| `real_tess` | Cleaned and normalized timeseries arrays extracted from official NASA TESS FITS files (Level 3/4). |
| `real_kepler` | Normalized timeseries arrays from the Kepler observatory. |
| `synthetic` | Simulated light curves with injected transit profiles and noise models (Level 1). |
| `injected` | Real light curves containing injected synthetic planet signals (Level 2). |
| `curated_catalog_only` | Rows in a metadata table containing target parameters without timeseries light-curve data. |
| `unknown` | Undocumented data source. |

### Available Targets in Phase 1
- **Synthetic Cases**: `candidate_a` (exoplanet transit), `candidate_b` (eclipsing binary), `candidate_c` (stellar variability or other noise).
- **Real TESS Cases**: `TIC-237913194`, `TIC-25155310`, `TIC-261136679`, `TIC-307210830` (all confirmed or candidate exoplanet transit signals).

---

## 3. Dataset Counts

### Summary of Counts

- **Evaluable Targets (Actual Time-Series Light Curves)**: 36
- **Catalog-Only Targets (Metadata Rows)**: 7,892

### Counts by Source (Evaluable)
- `real_tess`: 33 targets
- `synthetic`: 3 targets
- `total`: 36 targets

### Counts by Class (Evaluable)
- `exoplanet_transit`: 13 targets
- `eclipsing_binary`: 1 target
- `blend_contamination`: 0 targets (available in training manifest but no raw cached FITS data)
- `stellar_variability_or_other`: 22 targets

### Counts by Split
- **Train Split**: 22 targets (synthetic `candidate_c` + 21 real TESS sectors)
- **Val Split**: 12 targets (synthetic `candidate_b` + 11 real TESS sectors)
- **Test Split**: 2 targets (synthetic `candidate_a` + 1 real TESS sector)

---

## 4. Time/Flux vs. Catalog-Only Data
- **Actual Time/Flux**: Stored in compressed `.npz` files under `datasets/processed/lightcurves/` with corresponding sorted, deduplicated `time` and `flux` arrays ($N \ge 100$).
- **Catalog-Only Data**: Large metadata lists (such as `train_targets.csv` containing ~11,000 targets) contain only catalog parameters (period, depth, etc.) and class labels. They **cannot** be parsed into timeseries light curves for evaluation until physical timeseries data is acquired.

---

## 5. Leakage Prevention
To prevent scientific leakage, the following invariants are enforced:
1. **Target-Disjoint Splits**: Target IDs are mapped to splits such that no target ID is present in more than one split manifest. Evaluation is done strictly on targets that the model was not trained on.
2. **Deterministic Preprocessing**: Any normalization, cleaning, and sorting are applied target-by-target without using aggregate properties of the splits.

---

## 6. Reproducibility & Build Instructions
To rebuild and validate the dataset, run the following commands from the `transitlens-data-pipeline` repository root:

```powershell
# 1. Parse raw FITS cache and synthetic files to write .npz targets and manifests
python datasets/build_real_evaluation_dataset.py

# 2. Run the validator to verify constraints and write reports
python datasets/validate_dataset.py
```

---

## 7. Limitations & Path to 95+ Scoring

> [!WARNING]
> **Dataset Status: Partially Complete / Framework Ready**
> 
> The current evaluable dataset is extremely small (7 targets) and lacks representation for the `blend_contamination` class. This is insufficient for training deep learning classifiers or reporting statistically significant validation/test metrics.

### Next Required Data Expansion
To reach a strong hackathon-ready or production grade scoring path (95+), the following actions must be taken in Phase 2:
1. **FITS Cache Ingestion**: Automatically download and parse the raw FITS files corresponding to the ~15,000 targets listed in the catalog targets files (`train_targets.csv`, etc.).
2. **Injectors for Blend/Contamination**: Programmatically generate simulated and injected cases of `blend_contamination` and `eclipsing_binary` to resolve class imbalance.
3. **Target Ingestion Goal**:
   - At least 700 actual evaluable light curves.
   - All 4 canonical classes represented with at least 50 targets each.
   - A test split containing at least 100 actual light curves.
