# Processed Light Curves Directory

This folder contains the Phase 1 benchmark dataset for TransitLens.

## Storage Layout
- `manifest.csv`: Central registry of all processed targets.
- `<target_id>.npz`: Compressed NumPy file containing target light curve time series.
- `splits/`: Folder containing target-disjoint manifests:
  - `train_manifest.csv`
  - `val_manifest.csv`
  - `test_manifest.csv`

## Structure of .npz Files
Each target's `.npz` file contains the following arrays:
- `time`: Barycentric TESS Julian Date (BTJD) timestamps.
- `flux`: Normalized flux values (median ≈ 1.0).
- `flux_err` (Optional): Flux errors.
- `quality` (Optional): Quality flags.

## Rerun Dataset Build
To regenerate this dataset, run:
```bash
python datasets/build_real_evaluation_dataset.py
```
To validate the generated dataset, run:
```bash
python datasets/validate_dataset.py
```
