# Dataset Schema

> **Purpose:** Defines the exact column contract for all dataset files in `transitlens-data-pipeline`.
> Every CSV in `datasets/` and `datasets/splits/` must conform to this schema.
> `transitlens-ml-core` depends on these column names and types — do not rename without updating both repos.

---

## `labeled_dataset.csv` columns

| Column | Type | Description | Nullable |
|---|---|---|---|
| `target_id` | str | Unique identifier for the light curve (e.g. `candidate_a`, `TIC-25155310`) | No |
| `time` | float | Timestamp in BTJD (Barycentric TESS Julian Date) | No |
| `flux` | float | Normalised flux (median ≈ 1.0) | No |
| `source` | str | Data origin: `"synthetic"` or `"tess"` | No |
| `label` | str | Ground truth classification class | Yes (real unlabelled data) |
| `true_period` | float | Known orbital period in days | Yes |
| `true_depth` | float | Known transit depth (fractional flux drop) | Yes |
| `true_duration` | float | Known transit duration in days | Yes |
| `cadence_min` | float | Observation cadence in minutes | No |
| `sector` | int | TESS sector number | Yes (`None` for synthetic) |

---

## Valid `label` values

| Value | Meaning |
|---|---|
| `exoplanet_like` | Periodic shallow transits consistent with a planetary companion |
| `eclipsing_binary_like` | Deep and/or V-shaped eclipses typical of a stellar binary system |
| `noise_or_other` | No astrophysical transit signal detected; noise-dominated |
| `null` / empty | Real data with unknown or unverified classification |

---

## Notes

- **One row per time step.** A 27-day light curve at 2-min cadence produces ~17,600 rows per target (after gap simulation).
- **`true_period`, `true_depth`, `true_duration`** are `None` for `noise_or_other` cases and unlabelled real data.
- **`sector`** is `None` for all synthetic cases.
- **Split files** (`train.csv`, `val.csv`, `test.csv`) follow the same schema as `labeled_dataset.csv`.
- **Splits are by `target_id`**, not by row, to prevent data leakage (no light curve appears in more than one split).

---

## Validation rules

1. `time` must be strictly increasing within each `target_id` group.
2. `flux` median per `target_id` must be in `[0.99, 1.01]`.
3. `len(time)` must equal `len(flux)` per `target_id` (enforced by row structure).
4. No nulls allowed in: `target_id`, `time`, `flux`, `source`, `cadence_min`.
5. All non-null `label` values must be one of the valid label values above.