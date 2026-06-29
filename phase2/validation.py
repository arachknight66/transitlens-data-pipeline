# validation.py
# -------------
# Structural validation checks for Phase 2 feature parquets and benchmark manifests.

from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def run_phase2_validation(config) -> dict:
    """Runs checks on the materialized feature tables and benchmark splits."""
    m = config.manifests_dir
    errors = []
    warnings = []
    
    # 1. Presence of materialized parquets
    files = ["phase2_features_train.parquet", "phase2_features_validation.parquet", "phase2_features_test.parquet"]
    for f in files:
        fpath = m / f
        if not fpath.exists():
            errors.append(f"Materialized feature table absent: {f}")
            
    # 2. Check overlap between train/val/test feature rows
    train_path = m / "phase2_features_train.parquet"
    val_path = m / "phase2_features_validation.parquet"
    test_path = m / "phase2_features_test.parquet"
    
    if all(p.exists() for p in [train_path, val_path, test_path]):
        df_train = pd.read_parquet(train_path)
        df_val = pd.read_parquet(val_path)
        df_test = pd.read_parquet(test_path)
        
        train_set = set(df_train["tic_id"].unique()) if "tic_id" in df_train.columns else set()
        val_set = set(df_val["tic_id"].unique()) if "tic_id" in df_val.columns else set()
        test_set = set(df_test["tic_id"].unique()) if "tic_id" in df_test.columns else set()
        
        overlap_tv = train_set.intersection(val_set)
        overlap_tt = train_set.intersection(test_set)
        overlap_vt = val_set.intersection(test_set)
        
        if overlap_tv:
            errors.append(f"Overlap detected: {len(overlap_tv)} targets appear in both train and validation feature sets.")
        if overlap_tt:
            errors.append(f"Overlap detected: {len(overlap_tt)} targets appear in both train and test feature sets.")
        if overlap_vt:
            errors.append(f"Overlap detected: {len(overlap_vt)} targets appear in both validation and test feature sets.")
            
    status = "FAIL" if errors else "PASS"
    if warnings and not errors:
        status = "PARTIAL"
        
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
    }
