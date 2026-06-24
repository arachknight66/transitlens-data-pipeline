"""
datasets/build_dataset.py
──────────────────────────
Assembles labelled light curve datasets from synthetic cases (and
optionally real TESS data) into a single labeled_dataset.csv, then
creates train/val/test splits.

Usage (from repo root):
    python -m datasets.build_dataset

This will:
  1. Read all CSVs in synthetic/cases/
  2. Merge with labels & parameters from synthetic/config.yaml
  3. Write datasets/labeled_dataset.csv
  4. Write datasets/metadata.json
  5. Split into datasets/splits/{train,val,test}.csv

Performance notes
------------------
- String columns that repeat the same value across every row of a
  case (target_id, source, label) are stored as pandas `category`
  dtype before concatenation. This costs nothing for 3 synthetic
  cases but meaningfully reduces memory and speeds up groupby /
  isin operations once real TESS data (many more targets, many more
  rows) is appended in Phase 5.
- `pd.concat` is called once on a list of DataFrames (not in a loop
  with repeated concatenation), which avoids the O(n^2) copy cost of
  growing a DataFrame incrementally.
- `split_dataset` shuffles target_ids with `rng.permutation` on a
  numpy array instead of `rng.shuffle` on a Python list, avoiding an
  extra Python-list round trip.
- CSV reads explicitly pin `time`/`flux` to float64 via `dtype=` so
  pandas doesn't spend time inferring column types.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

_CASE_COLUMN_DTYPE = {'time': 'float64', 'flux': 'float64'}


def build_from_synthetic(cases_dir, config_path, output_path):
    """
    Reads all CSV files from synthetic/cases/ and merges them with
    labels and parameters from config.yaml to produce a single
    labeled_dataset.csv that conforms to datasets/schema.md.

    Also writes datasets/metadata.json with per-case metadata.

    Parameters
    ----------
    cases_dir : str
        Path to directory containing per-case CSVs (e.g. 'synthetic/cases').
    config_path : str
        Path to config.yaml with generation parameters.
    output_path : str
        Path for the output labeled_dataset.csv.

    Returns
    -------
    pd.DataFrame
        The assembled dataset.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    gen = config['generation']
    cases = config['cases']
    all_frames = []
    metadata_dict = {}

    for case_name, case_params in cases.items():
        csv_path = os.path.join(cases_dir, f'{case_name}.csv')

        if not os.path.exists(csv_path):
            print(f"  [SKIP] {case_name}: CSV not found at {csv_path}")
            continue

        # Read the raw time/flux CSV. Pinning dtype avoids pandas'
        # type-inference pass over every column.
        df = pd.read_csv(csv_path, dtype=_CASE_COLUMN_DTYPE)

        if 'time' not in df.columns or 'flux' not in df.columns:
            raise ValueError(
                f"{csv_path} is missing required columns 'time' and/or 'flux'. "
                f"Found columns: {list(df.columns)}"
            )

        # Attach schema columns from config.
        # target_id / source / label are identical for every row in
        # this frame, so storing them as 'category' keeps the
        # eventual concatenated dataset small and keeps later
        # groupby/isin operations (used in split_dataset and in
        # ml-core) fast even as more targets are added.
        n_rows = len(df)
        df['target_id'] = pd.Series([case_name] * n_rows, dtype='category')
        df['source'] = pd.Series(['synthetic'] * n_rows, dtype='category')
        df['label'] = pd.Series([case_params.get('label')] * n_rows, dtype='category')
        df['true_period'] = case_params.get('period_days')
        df['true_depth'] = case_params.get('depth')
        df['true_duration'] = case_params.get('duration_days')
        df['cadence_min'] = gen['cadence_minutes']
        df['sector'] = None  # always None for synthetic

        all_frames.append(df)

        # Build metadata entry for this case
        metadata_dict[case_name] = {
            'target_id': case_name,
            'source': 'synthetic',
            'label': case_params.get('label'),
            'true_period': case_params.get('period_days'),
            'true_depth': case_params.get('depth'),
            'true_duration': case_params.get('duration_days'),
            'n_points': n_rows,
            'cadence_min': gen['cadence_minutes'],
            'time_span_days': gen['time_span_days'],
            'sector': None,
            'seed': case_params.get('seed'),
            'generated_at': datetime.now(timezone.utc).isoformat(),
        }

        print(f"  [OK] {case_name}: {n_rows} rows, label={case_params.get('label')}")

    if not all_frames:
        raise FileNotFoundError(
            f"No case CSVs found in {cases_dir}. "
            "Run synthetic generation first (Phase 1)."
        )

    # Concatenate all cases into one dataset in a single pass.
    dataset = pd.concat(all_frames, ignore_index=True)

    # Enforce column order matching schema.md
    column_order = [
        'target_id', 'time', 'flux', 'source', 'label',
        'true_period', 'true_depth', 'true_duration',
        'cadence_min', 'sector',
    ]
    dataset = dataset[column_order]

    # Write labeled_dataset.csv
    out_dir = os.path.dirname(output_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"\n  Wrote {len(dataset)} total rows to {output_path}")

    # Write metadata.json alongside the dataset
    metadata_path = os.path.join(out_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"  Wrote metadata to {metadata_path}")

    return dataset


def build_from_tess(tess_dir, labels_path, output_path):
    """
    Reads normalised TESS CSVs from real_tess/cache/ and merges with
    manual labels from a labels JSON file.

    Stretch goal — only implement after Phase 5 is complete.

    Parameters
    ----------
    tess_dir : str
        Path to real_tess/cache/ directory with normalised CSVs.
    labels_path : str
        Path to a JSON file mapping TIC IDs to manual labels.
    output_path : str
        Path for the output labeled_dataset.csv (appends if exists).

    Raises
    ------
    NotImplementedError
        Always — this is a stretch goal placeholder.
    """
    raise NotImplementedError(
        "build_from_tess() is a Phase 5 stretch goal. "
        "Use build_from_synthetic() for the MVP."
    )


def split_dataset(dataset_path, splits_dir,
                  train_frac=0.70, val_frac=0.15, test_frac=0.15,
                  seed=42):
    """
    Reads labeled_dataset.csv and creates train/val/test splits.

    CRITICAL: Splits by target_id (not by row) to avoid data leakage.
    All rows for a given target_id go into the same split.

    Parameters
    ----------
    dataset_path : str
        Path to labeled_dataset.csv.
    splits_dir : str
        Directory to write train.csv, val.csv, test.csv.
    train_frac : float
        Fraction of target_ids assigned to training set.
    val_frac : float
        Fraction of target_ids assigned to validation set.
    test_frac : float
        Fraction of target_ids assigned to test set.
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    dict
        Keys 'train', 'val', 'test' mapping to DataFrames.
    """
    # Validate fractions
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total:.4f} "
            f"(train={train_frac}, val={val_frac}, test={test_frac})"
        )

    dataset = pd.read_csv(dataset_path)

    # Get unique target_ids and shuffle as a numpy array directly
    # (rng.permutation avoids the list round-trip / dtype warning
    # that rng.shuffle on a pandas StringArray-backed list triggers).
    target_ids = dataset['target_id'].unique()
    rng = np.random.default_rng(seed)
    target_ids = rng.permutation(target_ids)

    n_targets = len(target_ids)
    n_train = max(1, int(np.round(n_targets * train_frac)))
    n_val = max(1, int(np.round(n_targets * val_frac)))
    # test gets the remainder to guarantee all targets are assigned
    n_test = n_targets - n_train - n_val

    # Handle edge case: fewer targets than 3 splits
    if n_test < 1:
        # With very few targets, put at least 1 in each split
        if n_targets >= 3:
            n_train = n_targets - 2
            n_val = 1
            n_test = 1
        elif n_targets == 2:
            n_train = 1
            n_val = 1
            n_test = 0
            print("  [WARN] Only 2 targets — test split will be empty.")
        else:
            n_train = 1
            n_val = 0
            n_test = 0
            print("  [WARN] Only 1 target — val and test splits will be empty.")

    train_ids = set(target_ids[:n_train])
    val_ids = set(target_ids[n_train:n_train + n_val])
    test_ids = set(target_ids[n_train + n_val:])

    # Using Python sets for membership + .isin gives O(1) lookups per
    # row during the boolean mask construction below, which matters
    # once the dataset grows beyond a handful of synthetic targets.
    splits = {
        'train': dataset[dataset['target_id'].isin(train_ids)],
        'val': dataset[dataset['target_id'].isin(val_ids)],
        'test': dataset[dataset['target_id'].isin(test_ids)],
    }

    # Write split files
    os.makedirs(splits_dir, exist_ok=True)

    for split_name, split_df in splits.items():
        split_path = os.path.join(splits_dir, f'{split_name}.csv')
        split_df.to_csv(split_path, index=False)

    # Print summary
    print(f"\n  Split summary ({n_targets} targets, seed={seed}):")
    print(f"  {'Split':<8} {'Targets':<10} {'Rows':<10} {'Classes'}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*30}")

    for split_name, split_df in splits.items():
        split_targets = split_df['target_id'].unique()
        if len(split_df) > 0:
            class_dist = split_df.groupby('target_id')['label'].first().value_counts()
            class_str = ', '.join(f'{k}={v}' for k, v in class_dist.items())
        else:
            class_str = '(empty)'
        print(f"  {split_name:<8} {len(split_targets):<10} {len(split_df):<10} {class_str}")

    return splits


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    """
    Runs the full dataset build pipeline:
    1. Build labeled_dataset.csv from synthetic cases
    2. Create train/val/test splits
    """
    print("=" * 60)
    print("  transitlens-data-pipeline — Dataset Assembly (Phase 2)")
    print("=" * 60)

    # Resolve paths relative to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cases_dir = os.path.join(repo_root, 'synthetic', 'cases')
    config_path = os.path.join(repo_root, 'synthetic', 'config.yaml')
    output_path = os.path.join(repo_root, 'datasets', 'labeled_dataset.csv')
    splits_dir = os.path.join(repo_root, 'datasets', 'splits')

    print("\n[Step 1] Building labeled_dataset.csv from synthetic cases...")
    dataset = build_from_synthetic(cases_dir, config_path, output_path)

    print("\n[Step 2] Creating train/val/test splits...")
    split_dataset(output_path, splits_dir)

    # Quick validation summary
    print("\n[Step 3] Validation checks...")
    targets = dataset['target_id'].unique()
    print(f"  Unique targets: {len(targets)} — {list(targets)}")
    print(f"  Total rows:     {len(dataset)}")

    for tid in targets:
        subset = dataset[dataset['target_id'] == tid]
        median_flux = subset['flux'].median()
        ok = 'OK' if abs(median_flux - 1.0) < 0.01 else 'FAIL'
        print(f"  {ok} {tid}: {len(subset)} rows, "
              f"flux median={median_flux:.4f}, "
              f"label={subset['label'].iloc[0]}")

    print("\n" + "=" * 60)
    print("  Dataset assembly complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()