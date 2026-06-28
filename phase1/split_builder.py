import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def build_splits(config):
    """
    Groups observations by TIC ID, excludes review_required/unlabeled from supervised training,
    stratifies by canonical label, and outputs deterministic splits.
    """
    config.ensure_dirs()
    manifests_dir = config.manifests_dir
    
    resolved_labels_path = manifests_dir / "resolved_labels.parquet"
    if not resolved_labels_path.exists():
        raise FileNotFoundError(f"Resolved labels manifest not found: {resolved_labels_path}. Run label resolution first.")
        
    df_labels = pd.read_parquet(resolved_labels_path)
    
    # We also check which TICs are actually successfully processed/verified
    # by loading the download manifest to see which ones exist
    download_manifest_path = manifests_dir / "download_manifest.parquet"
    if not download_manifest_path.exists():
        raise FileNotFoundError(f"Download manifest not found: {download_manifest_path}")
    df_dl = pd.read_parquet(download_manifest_path)
    
    # Keep only TICs that have at least one successfully downloaded/verified product
    eligible_downloads = df_dl[df_dl["final_status"].isin(["verified", "processed"])]
    if "parse_status" in eligible_downloads.columns:
        eligible_downloads = eligible_downloads[eligible_downloads["parse_status"] == "success"]
    verified_tics = set(eligible_downloads["tic_id"].unique())
    df_labels = df_labels[df_labels["tic_id"].isin(verified_tics)].copy()
    
    # Separate supervised, unlabeled, and review_required cohorts
    mask_review = df_labels["resolved_label"] == "review_required"
    mask_unlabeled = df_labels["resolved_label"] == "unlabeled"
    mask_supervised = (~mask_review) & (~mask_unlabeled)
    
    df_review = df_labels[mask_review].copy()
    df_unlabeled = df_labels[mask_unlabeled].copy()
    df_supervised = df_labels[mask_supervised].copy()
    
    # Supervised split mapping
    # Stratify by resolved_label
    split_assignments = []
    previous = {}
    previous_path = manifests_dir / "split_manifest.parquet"
    if previous_path.exists():
        old = pd.read_parquet(previous_path)
        if {"tic_id", "split"}.issubset(old.columns):
            previous = dict(zip(old["tic_id"].astype(int), old["split"]))
    
    # For each class in supervised cohort, perform deterministic shuffle and partition
    for label, group in df_supervised.groupby("resolved_label"):
        tics_arr = group["tic_id"].astype(int).unique()
        retained = [tic for tic in tics_arr if previous.get(tic) in ("train", "val", "test")]
        for tic in retained:
            split_assignments.append((tic, previous[tic]))
        tics_arr = np.array([tic for tic in tics_arr if tic not in set(retained)], dtype=np.int64)
        # Stable pseudorandom ordering. This is independent of input row order.
        import hashlib
        tics_arr = np.array(sorted(
            tics_arr,
            key=lambda tic: hashlib.sha256(f"{config.random_seed}:{label}:{int(tic)}".encode()).hexdigest(),
        ), dtype=np.int64)
        
        n = len(tics_arr)
        if n == 0:
            continue
        elif n == 1:
            split_assignments.append((tics_arr[0], "train"))
        elif n == 2:
            split_assignments.append((tics_arr[0], "train"))
            split_assignments.append((tics_arr[1], "val"))
        else:
            n_val = max(1, int(np.round(config.split_ratios["val"] * n)))
            n_test = max(1, int(np.round(config.split_ratios["test"] * n)))
            n_train = n - n_val - n_test
            
            if n_train <= 0:
                n_train = 1
                n_val = 1
                n_test = n - 2
                
            for tid in tics_arr[:n_train]:
                split_assignments.append((tid, "train"))
            for tid in tics_arr[n_train:n_train+n_val]:
                split_assignments.append((tid, "val"))
            for tid in tics_arr[n_train+n_val:n_train+n_val+n_test]:
                split_assignments.append((tid, "test"))
                
            # If any remaining
            if n_train + n_val + n_test < n:
                for tid in tics_arr[n_train+n_val+n_test:]:
                    split_assignments.append((tid, "train"))
                    
    # Map back to a splits dataframe
    df_split_map = pd.DataFrame(split_assignments, columns=["tic_id", "split"])
    
    # Unlabeled & Review target splits
    df_review["split"] = "review"
    df_unlabeled["split"] = "screening"
    
    # Combine everything into split_manifest
    df_supervised_merged = pd.merge(df_supervised, df_split_map, on="tic_id", how="left")
    df_split_manifest = pd.concat([
        df_supervised_merged[["tic_id", "split", "resolved_label", "label_subtype"]],
        df_review[["tic_id", "split", "resolved_label", "label_subtype"]],
        df_unlabeled[["tic_id", "split", "resolved_label", "label_subtype"]]
    ], ignore_index=True)
    
    split_manifest_path = manifests_dir / "split_manifest.parquet"
    df_split_manifest.to_parquet(split_manifest_path, index=False)
    logger.info(f"Wrote split manifest with {len(df_split_manifest)} rows to {split_manifest_path}")
    
    # Save individual target lists
    # Train
    train_targets = df_split_manifest[df_split_manifest["split"] == "train"][["tic_id", "resolved_label"]].copy()
    train_targets.to_parquet(manifests_dir / "train_targets.parquet", index=False)
    
    # Val
    val_targets = df_split_manifest[df_split_manifest["split"] == "val"][["tic_id", "resolved_label"]].copy()
    val_targets.to_parquet(manifests_dir / "validation_targets.parquet", index=False)
    
    # Test
    test_targets = df_split_manifest[df_split_manifest["split"] == "test"][["tic_id", "resolved_label"]].copy()
    test_targets.to_parquet(manifests_dir / "test_targets.parquet", index=False)
    
    # Unlabeled screening
    unlabeled_targets = df_split_manifest[df_split_manifest["split"] == "screening"][["tic_id"]].copy()
    unlabeled_targets.to_parquet(manifests_dir / "unlabeled_screening_targets.parquet", index=False)
    
    # Review Required
    review_targets = df_split_manifest[df_split_manifest["split"] == "review"][["tic_id"]].copy()
    review_targets.to_parquet(manifests_dir / "review_required_targets.parquet", index=False)
    
    # ----------------------------------------------------
    # Leakage & Integrity Diagnostics
    # ----------------------------------------------------
    train_set = set(train_targets["tic_id"].unique())
    val_set = set(val_targets["tic_id"].unique())
    test_set = set(test_targets["tic_id"].unique())
    screening_set = set(unlabeled_targets["tic_id"].unique())
    review_set = set(review_targets["tic_id"].unique())
    
    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)
    overlap_supervised_unlabeled = (train_set | val_set | test_set).intersection(screening_set | review_set)
    
    leakage_ok = (len(overlap_train_val) == 0) and (len(overlap_train_test) == 0) and (len(overlap_val_test) == 0) and (len(overlap_supervised_unlabeled) == 0)
    
    # Count classes by split
    def get_class_counts(df_split):
        counts = df_split["resolved_label"].value_counts().to_dict()
        return {
            "planets": counts.get("exoplanet_transit", 0),
            "ebs": counts.get("eclipsing_binary", 0),
            "blends": counts.get("blend_contamination", 0),
            "stellar_var": counts.get("stellar_variability_or_other", 0)
        }
        
    train_counts = get_class_counts(df_split_manifest[df_split_manifest["split"] == "train"])
    val_counts = get_class_counts(df_split_manifest[df_split_manifest["split"] == "val"])
    test_counts = get_class_counts(df_split_manifest[df_split_manifest["split"] == "test"])
    
    # Compute Shortfalls
    def compute_shortfall(actual, desired):
        sf = {}
        for k, v in desired.items():
            act = actual.get(k, 0)
            if act < v:
                sf[k] = int(v - act)
            else:
                sf[k] = 0
        return sf
        
    train_sf = compute_shortfall(train_counts, config.min_class_counts.get("train", {}))
    val_sf = compute_shortfall(val_counts, config.min_class_counts.get("validation", {}))
    test_sf = compute_shortfall(test_counts, config.min_class_counts.get("test", {}))
    
    integrity = {
        "leakage_detected": not leakage_ok,
        "overlap_train_validation_count": len(overlap_train_val),
        "overlap_train_test_count": len(overlap_train_test),
        "overlap_validation_test_count": len(overlap_val_test),
        "overlap_supervised_unlabeled_count": len(overlap_supervised_unlabeled),
        "total_train_tics": len(train_set),
        "total_val_tics": len(val_set),
        "total_test_tics": len(test_set),
        "total_screening_tics": len(screening_set),
        "total_review_tics": len(review_set),
        "class_counts": {
            "train": train_counts,
            "validation": val_counts,
            "test": test_counts
        },
        "class_shortfalls": {
            "train": train_sf,
            "validation": val_sf,
            "test": test_sf
        }
    }
    
    integrity_path = manifests_dir / "split_integrity_report.json"
    with open(integrity_path, "w", encoding="utf-8") as f:
        json.dump(integrity, f, indent=2)
    with open(manifests_dir / "leakage_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "leakage_detected": not leakage_ok,
            "overlap_train_validation_count": len(overlap_train_val),
            "overlap_train_test_count": len(overlap_train_test),
            "overlap_validation_test_count": len(overlap_val_test),
            "overlap_supervised_nontraining_count": len(overlap_supervised_unlabeled),
        }, f, indent=2)
    logger.info(f"Wrote split integrity report to {integrity_path}")
    
    # Markdown Report
    report_path = config.report_split_methodology
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_md = f"""# Phase 1 Target-Disjoint Split Methodology & Distribution Report

This document reports the target splitting configuration, leakage checks, and class distributions of the Phase 1 dataset.

## 1. Split Strategy & Leakage Prevention

To ensure a scientifically valid evaluation, we enforce the following constraints:
1. **Target Grouping**: Splitting is executed strictly on unique **TIC IDs**. All observations of a star across multiple TESS sectors are grouped and assigned to the same split.
2. **Disjoint Datasets**: Zero TIC overlap is allowed between splits.
3. **Supervised Isolation**: Unlabeled targets (used for screening) and targets requiring manual review (due to label conflicts) are completely excluded from supervised splits.
4. **Reproducibility**: Splits are built deterministically using a fixed random seed (`{config.random_seed}`).

### Leakage Diagnostic Status: **{"PASS (No Leakage)" if leakage_ok else "FAIL (Leakage Detected!)"}**
* Overlap Train/Validation: {len(overlap_train_val)}
* Overlap Train/Test: {len(overlap_train_test)}
* Overlap Validation/Test: {len(overlap_val_test)}
* Overlap Supervised/Unlabeled: {len(overlap_supervised_unlabeled)}

---

## 2. Dataset Partition Distributions

| Split | Target Count | Exoplanets | Eclipsing Binaries | Blends | Stellar Var / Other |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | {len(train_set)} | {train_counts['planets']} | {train_counts['ebs']} | {train_counts['blends']} | {train_counts['stellar_var']} |
| **Validation** | {len(val_set)} | {val_counts['planets']} | {val_counts['ebs']} | {val_counts['blends']} | {val_counts['stellar_var']} |
| **Test** | {len(test_set)} | {test_counts['planets']} | {test_counts['ebs']} | {test_counts['blends']} | {test_counts['stellar_var']} |
| **Unlabeled (Screening)** | {len(screening_set)} | N/A | N/A | N/A | N/A |
| **Review Required** | {len(review_set)} | N/A | N/A | N/A | N/A |

---

## 3. Class Shortfall Report

Authoritative catalog listings are limited in real space observations. The shortfalls relative to the desired counts are detailed below:

### Train Shortfall (Desired: ≥1000 Planets, ≥2000 EBs, ≥1000 Blends)
* Planets Shortfall: {train_sf['planets']}
* EBs Shortfall: {train_sf['ebs']}
* Blends Shortfall: {train_sf['blends']}

### Validation Shortfall (Desired: ≥250 Planets, ≥400 EBs, ≥250 Blends)
* Planets Shortfall: {val_sf['planets']}
* EBs Shortfall: {val_sf['ebs']}
* Blends Shortfall: {val_sf['blends']}

### Test Shortfall (Desired: ≥250 Planets, ≥400 EBs, ≥250 Blends)
* Planets Shortfall: {test_sf['planets']}
* EBs Shortfall: {test_sf['ebs']}
* Blends Shortfall: {test_sf['blends']}

> [!WARNING]
> Class count shortfalls are a physical limitation of the current TESS catalogs for the selected short-cadence sector population. Under strict Phase 1 rules, we report these shortfalls exactly rather than fabricating dummy classes or padding observations.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    logger.info(f"Wrote split methodology report to {report_path}")
    return df_split_manifest
