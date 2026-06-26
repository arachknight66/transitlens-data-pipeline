# Processed Dataset Schema (Phase 1 Benchmark)

This document defines the schema contract for the processed exoplanet detection benchmark dataset used in TransitLens. All evaluation split manifests, data pipelines, and validation scripts must conform to these definitions.

---

## 1. Directory Structure

Processed data is stored in the following directory layout under the `transitlens-data-pipeline` repository root:

```
datasets/
├── processed/
│   ├── lightcurves/
│   │   ├── README.md               # Processing and file guide
│   │   ├── manifest.csv            # Central register of all evaluable targets
│   │   ├── <target_id>.npz         # Individual light-curve data arrays
│   │   └── splits/
│   │       ├── train_manifest.csv  # Manifest for the training split
│   │       ├── val_manifest.csv    # Manifest for the validation split
│   │       └── test_manifest.csv   # Manifest for the test split
│   └── validation/
│       ├── dataset_validation.json # Machine-readable validation status
│       └── dataset_validation_report.md # Human-readable validation report
└── schema.md                       # This schema definition
```

---

## 2. Processed Light-Curve File Format (`.npz`)

Each target is saved as an individual compressed NumPy array file: `datasets/processed/lightcurves/<target_id>.npz`.

Each `.npz` file contains the following arrays:

| Array Key | Data Type | Description |
| :--- | :--- | :--- |
| `time` | `float64` | 1D array of time values in BTJD. Must be sorted and strictly increasing. |
| `flux` | `float64` | 1D array of normalized flux values (median ≈ 1.0). Length must match `time`. |
| `flux_err` | `float64` (Optional) | 1D array of flux measurement errors. |
| `centroid_x` | `float64` (Optional) | 1D array of aperture centroid X positions. |
| `centroid_y` | `float64` (Optional) | 1D array of aperture centroid Y positions. |
| `quality` | `int64` (Optional) | 1D array of quality flags (e.g., TESS SPOC quality flags). |

---

## 3. Central Manifest Schema (`manifest.csv`)

`manifest.csv` acts as the single source of truth registering all evaluable targets.

| Column | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| **target_id** | `str` | No | Unique identifier for the light curve (e.g., `TIC-25155310`). |
| **source** | `str` | No | Data origin: `synthetic` or `real_tess`. |
| **evidence_level** | `str` | No | Credibility/reproducibility level: `synthetic`, `real_tess`, `real_kepler`, `curated_catalog_only`, `injected`, or `unknown`. |
| **class_label** | `str` | No | Ground truth classification: must be one of the four canonical classes. |
| **lightcurve_path** | `str` | No | Relative path from `datasets/processed/lightcurves/` to the `.npz` file. |
| **n_points** | `int` | No | Total number of valid time-series data points in the `.npz` file. |
| **time_span_days** | `float` | No | Difference between maximum and minimum timestamps ($t_{max} - t_{min}$). |
| **cadence_min_median** | `float` | No | Median spacing between consecutive timestamps in minutes. |
| **true_period_days** | `float` | Yes | Catalog period in days (if periodic). |
| **true_depth** | `float` | Yes | Catalog transit depth (fractional flux drop). |
| **true_duration_days**| `float` | Yes | Catalog transit duration in days. |
| **true_epoch_btjd** | `float` | Yes | Catalog transit epoch (time of center of transit). |
| **ground_truth_source**| `str` | Yes | Origin catalog for the ground truth parameters (e.g., `TESS_TOI`, `Kepler_KOI`). |
| **sector** | `int` | Yes | TESS sector number. |
| **mission** | `str` | Yes | Observatory name: `TESS`, `Kepler`, etc. |
| **has_flux_err** | `bool` | No | True if the `.npz` contains non-zero `flux_err` values. |
| **has_centroid** | `bool` | No | True if the `.npz` contains valid `centroid_x` and `centroid_y` arrays. |
| **has_quality_flags** | `bool` | No | True if the `.npz` contains valid `quality` arrays. |
| **contamination_available**| `bool` | No | True if spatial blend crowding metadata is defined. |
| **created_at** | `str` | No | ISO 8601 creation timestamp. |
| **notes** | `str` | Yes | Developer notes or extra context. |

---

## 4. Split Manifests Schema (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`)

These manifests define the target partitions for model training and evaluation.

| Column | Type | Nullable | Description |
| :--- | :--- | :--- | :--- |
| **target_id** | `str` | No | Unique identifier for the light curve. |
| **class_label** | `str` | No | Canonical class label. |
| **source** | `str` | No | Data origin. |
| **evidence_level** | `str` | No | Evidence level. |
| **lightcurve_path** | `str` | No | Relative path from `datasets/processed/lightcurves/` to the `.npz` file. |
| **true_period_days** | `float` | Yes | True orbital period in days. |
| **true_depth** | `float` | Yes | True transit depth (fractional). |
| **true_duration_days**| `float` | Yes | True transit duration in days. |

---

## 5. Invariants and Validation Rules
1. **Target Disjointness**: A `target_id` must appear in at most one split manifest.
2. **Canonical Labels Only**: The `class_label` column must strictly contain canonical labels: `exoplanet_transit`, `eclipsing_binary`, `blend_contamination`, or `stellar_variability_or_other`.
3. **No Row Duplication**: The `target_id` must be unique inside `manifest.csv`.
4. **Time Monotonicity**: Time values within each `.npz` file must be sorted and strictly increasing.
5. **No Infinite Flux**: Flux values must be finite numbers (no `NaN` or `inf` permitted after cleaning).
6. **Data Shape Consistency**: The lengths of the `time` and `flux` arrays in any `.npz` file must be equal and have a size of at least 100 points.