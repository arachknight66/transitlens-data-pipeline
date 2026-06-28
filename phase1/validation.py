import json
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def run_release_validation(config):
    """
    Enforces all strict Phase 1 release gate checks.
    Generates validation reports and logs.
    """
    config.ensure_dirs()
    manifests_dir = config.manifests_dir
    processed_dir = config.processed_dir
    
    errors = []
    warnings = []
    infos = []
    
    # 1. Required Manifests presence check
    required_manifests = [
        "discovery_manifest.parquet",
        "download_manifest.parquet",
        "observation_manifest.parquet",
        "label_evidence.parquet",
        "resolved_labels.parquet",
        "split_manifest.parquet",
        "train_targets.parquet",
        "validation_targets.parquet",
        "test_targets.parquet",
        "unlabeled_screening_targets.parquet",
        "failures.parquet",
        "exclusions.parquet",
        "duplicate_groups.parquet",
        "contradictions.parquet",
        "review_required_targets.parquet",
        "checksums.sha256"
    ]
    
    for m in required_manifests:
        mpath = manifests_dir / m
        if not mpath.exists():
            errors.append(f"Required manifest absent: {m}")
            
    # Stop early if core manifest is missing
    obs_manifest_path = manifests_dir / "observation_manifest.parquet"
    if not obs_manifest_path.exists():
        errors.append("Core observation manifest is missing. Cannot perform deep checks.")
        return _build_report_and_exit(config, errors, warnings, infos, "FAIL")
        
    df_obs = pd.read_parquet(obs_manifest_path)
    from phase1.schemas import OBSERVATION_REQUIRED_COLUMNS, missing_columns
    schema_missing = missing_columns(df_obs, OBSERVATION_REQUIRED_COLUMNS)
    if schema_missing:
        errors.append(f"Observation manifest missing required columns: {schema_missing}")
    
    # 2. Gate Count Check (≥20,000 successful parsed observations)
    parsed_obs = df_obs[df_obs["parse_status"] == "success"]
    n_parsed = len(parsed_obs)
    infos.append(f"Successfully parsed observations count: {n_parsed}")
    
    unique_tics_parsed = parsed_obs["tic_id"].nunique()
    infos.append(f"Unique TICs successfully parsed: {unique_tics_parsed}")
    
    # The scientific release floor is immutable. Configuration may raise it,
    # but a development fixture/flag may never lower it into a false PASS.
    required_count = max(20_000, config.minimum_successful_observations)
    if n_parsed < required_count:
        errors.append(f"Release gate unmet: successfully parsed observations count {n_parsed} is below required {required_count:,}.")
        
    # 3. Source Checksum validation
    no_checksum = parsed_obs[parsed_obs["raw_sha256"] == ""]
    if len(no_checksum) > 0:
        errors.append(f"Source checksum missing for {len(no_checksum)} parsed observations.")
        
    # 4. Target Leakage (Disjoint Splits Check)
    split_manifest_path = manifests_dir / "split_manifest.parquet"
    if split_manifest_path.exists():
        df_split = pd.read_parquet(split_manifest_path)
        if len(df_split) > 0 and "split" in df_split.columns and "tic_id" in df_split.columns:
            train_set = set(df_split[df_split["split"] == "train"]["tic_id"].unique())
            val_set = set(df_split[df_split["split"] == "val"]["tic_id"].unique())
            test_set = set(df_split[df_split["split"] == "test"]["tic_id"].unique())
            screening_set = set(df_split[df_split["split"] == "screening"]["tic_id"].unique())
            review_set = set(df_split[df_split["split"] == "review"]["tic_id"].unique())
            
            overlap_tv = train_set.intersection(val_set)
            overlap_tt = train_set.intersection(test_set)
            overlap_vt = val_set.intersection(test_set)
            
            if len(overlap_tv) > 0:
                errors.append(f"Leakage detected: {len(overlap_tv)} TICs appear in both Train and Validation splits.")
            if len(overlap_tt) > 0:
                errors.append(f"Leakage detected: {len(overlap_tt)} TICs appear in both Train and Test splits.")
            if len(overlap_vt) > 0:
                errors.append(f"Leakage detected: {len(overlap_vt)} TICs appear in both Validation and Test splits.")
                
            # 5. Supervised Split Labels constraint
            # review_required or unlabeled targets must not enter supervised splits
            if "resolved_label" in df_split.columns:
                df_sup_split = df_split[df_split["split"].isin(["train", "val", "test"])]
                invalid_sup_labels = df_sup_split[df_sup_split["resolved_label"].isin(["unlabeled", "review_required"])]
                if len(invalid_sup_labels) > 0:
                    errors.append(f"{len(invalid_sup_labels)} targets with unlabeled or review_required labels found in supervised splits.")
        else:
            if len(df_split) > 0:
                errors.append("split_manifest.parquet columns are missing or invalid.")
            
    # 6. Readability and Time/Flux array integrity checks (on a subset or summary)
    corrupt_files = 0
    checksum_mismatch = 0
    missing_paths = 0
    non_monotonic = 0
    out_of_bounds_cadence = 0
    missing_sector = 0
    
    # We check fields compiled in the manifest from metadata sidecars
    for idx, row in parsed_obs.iterrows():
        if row["n_points_raw"] <= 0:
            corrupt_files += 1
        if row["n_points_usable"] < config.minimum_points:
            errors.append(f"Observation {row['observation_id']} has usable points count {row['n_points_usable']} which is below minimum {config.minimum_points}.")
        if not (config.min_cadence_seconds <= row["cadence_seconds"] <= config.max_cadence_seconds):
            out_of_bounds_cadence += 1
        if pd.isnull(row["sector"]) or row["sector"] <= 0:
            missing_sector += 1

        raw_path = Path(row["raw_path"])
        processed_path = Path(row["processed_path"])
        if not raw_path.is_file() or not processed_path.is_file():
            missing_paths += 1
            continue
        try:
            with np.load(processed_path, allow_pickle=False) as arrays:
                time_values = arrays["time"]
                flux_values = arrays["flux"]
                required_arrays = {
                    "time_btjd", "sap_flux", "sap_flux_err", "pdcsap_flux",
                    "pdcsap_flux_err", "quality_raw", "finite_mask",
                    "archive_quality_mask", "usable_mask", "normalization_mask",
                }
                if required_arrays - set(arrays.files):
                    corrupt_files += 1
                if len(time_values) != len(flux_values):
                    corrupt_files += 1
                if not np.all(np.isfinite(time_values)) or np.any(np.diff(time_values) <= 0):
                    non_monotonic += 1
                usable_flux = flux_values[np.isfinite(flux_values)]
                if len(usable_flux) < config.minimum_points:
                    corrupt_files += 1
        except Exception:
            corrupt_files += 1
            continue

        # Source and processed hashes are the final provenance link. Stream both.
        from phase1.checksums import file_sha256
        if file_sha256(raw_path) != row["raw_sha256"] or file_sha256(processed_path) != row["processed_sha256"]:
            checksum_mismatch += 1
            
    if out_of_bounds_cadence > 0:
        errors.append(f"{out_of_bounds_cadence} observations have cadence outside configured bounds ({config.min_cadence_seconds}s - {config.max_cadence_seconds}s).")
    if missing_sector > 0:
        errors.append(f"{missing_sector} observations lack valid TESS sector metadata.")
    if missing_paths > 0:
        errors.append(f"{missing_paths} parsed observations point to missing raw or processed files.")
    if corrupt_files > 0:
        errors.append(f"{corrupt_files} processed observations failed NPZ array integrity checks.")
    if non_monotonic > 0:
        errors.append(f"{non_monotonic} processed observations have non-finite or non-monotonic time arrays.")
    if checksum_mismatch > 0:
        errors.append(f"{checksum_mismatch} parsed observations have raw or processed checksum mismatches.")
        
    # 7. Check if synthetic data was counted as real TESS
    # Synthetic targets shouldn't have raw TESS spoc download path
    synthetic_in_real = parsed_obs[parsed_obs["mission"] == "synthetic"]
    if len(synthetic_in_real) > 0:
        errors.append(f"Leakage: {len(synthetic_in_real)} synthetic targets counted as real observations.")
        
    # 8. Check if catalogue-only rows were counted as observations
    catalog_only = parsed_obs[parsed_obs["processed_path"] == ""]
    if len(catalog_only) > 0:
        errors.append(f"Schema violation: {len(catalog_only)} catalogue-only rows counted as successfully parsed light curves.")
        
    # 9. Supervised label evidence validation
    # All supervised targets must have evidence rows
    label_evidence_path = manifests_dir / "label_evidence.parquet"
    if label_evidence_path.exists():
        df_evidence = pd.read_parquet(label_evidence_path)
        if len(df_evidence) > 0 and "tic_id" in df_evidence.columns:
            supervised_rows = parsed_obs[parsed_obs["is_supervised_eligible"]]
            supervised_tics = supervised_rows["tic_id"].unique()
            evidence_tics = set(df_evidence["tic_id"].unique())
            
            missing_evidence = [t for t in supervised_tics if t not in evidence_tics]
            if len(missing_evidence) > 0:
                errors.append(f"Label integrity violation: {len(missing_evidence)} supervised targets lack evidence records.")
            weak_supervised = supervised_rows[
                (supervised_rows["evidence_level"] != "catalog_authoritative") |
                (~supervised_rows["label_strength"].isin(["strong", "medium"]))
            ]
            if len(weak_supervised) > 0:
                errors.append(f"Label integrity violation: {len(weak_supervised)} supervised observations rely only on weak/non-authoritative evidence.")
        else:
            supervised_tics = parsed_obs[parsed_obs["is_supervised_eligible"]]["tic_id"].unique()
            if len(supervised_tics) > 0:
                errors.append("Label integrity violation: supervised targets exist but label_evidence.parquet is empty or missing columns.")
            
    # 10. Check if placeholder coordinate or TIC IDs exist
    placeholders = parsed_obs[(parsed_obs["tic_id"] <= 0) | (parsed_obs["ra"] == 0.0) | (parsed_obs["dec"] == 0.0)]
    if len(placeholders) > 0:
        errors.append(f"{len(placeholders)} observations contain placeholder TIC IDs or coordinates (0.0).")
        
    # 11. Class shortfalls check (WARNING only, does not fail the release unless total count fails)
    # We log shortfalls as WARNINGS since they represent physical database limitations
    split_integrity_path = manifests_dir / "split_integrity_report.json"
    if split_integrity_path.exists():
        with open(split_integrity_path, "r", encoding="utf-8") as f:
            integrity = json.load(f)
            
        shortfalls = integrity.get("class_shortfalls", {})
        for split_name, sf in shortfalls.items():
            for cls, val in sf.items():
                if val > 0:
                    warnings.append(f"Split {split_name} has shortfall of {val} in class '{cls}'.")
                    
    # 12. Duplicate representatives ambiguity check
    dup_manifest_path = manifests_dir / "duplicate_groups.parquet"
    if dup_manifest_path.exists():
        df_dups = pd.read_parquet(dup_manifest_path)
        if len(df_dups) > 0 and "excluded_obs_id" in df_dups.columns and "selected_obs_id" in df_dups.columns:
            excl_obs = set(df_dups["excluded_obs_id"].unique())
            sel_obs = set(df_dups["selected_obs_id"].unique())
            
            ambiguous = excl_obs.intersection(sel_obs)
            if len(ambiguous) > 0:
                errors.append(f"Duplicate resolution is ambiguous: {len(ambiguous)} observation IDs are marked both as representative and excluded duplicates.")
            
    # Determine overall status
    status = "PASS"
    if len(errors) > 0:
        status = "FAIL"
    elif n_parsed < required_count:
        status = "FAIL"
    elif len(warnings) > 0:
        status = "PARTIAL"
        
    # 13. Write Checksum Verification report
    import phase1.checksums as csums
    chk_ok, chk_failed, chk_missing = csums.verify_checksums_file(config)
    
    checksum_report = {
        "verified": chk_ok,
        "failed_files": chk_failed,
        "missing_files": chk_missing
    }
    
    with open(manifests_dir / "checksum_report.json", "w", encoding="utf-8") as f:
        json.dump(checksum_report, f, indent=2)
        
    if not chk_ok:
        errors.append(f"Checksum verification failed. Corrupted manifest files: {chk_failed}. Missing files: {chk_missing}")
        status = "FAIL"
        
    return _build_report_and_exit(config, errors, warnings, infos, status)

def _build_report_and_exit(config, errors, warnings, infos, status):
    """Saves reports and returns validation dictionary."""
    manifests_dir = config.manifests_dir
    
    # Sector distribution
    obs_manifest_path = manifests_dir / "observation_manifest.parquet"
    if obs_manifest_path.exists():
        df_obs = pd.read_parquet(obs_manifest_path)
        parsed = df_obs[df_obs["parse_status"] == "success"]
        
        sector_dist = parsed["sector"].value_counts().to_frame("count")
        sector_dist.to_csv(manifests_dir / "sector_distribution.csv")
        
        class_dist = parsed["canonical_label"].value_counts().to_frame("count")
        class_dist.to_csv(manifests_dir / "class_distribution.csv")
        
        cadence_dist = parsed["cadence_seconds"].value_counts().to_frame("count")
        cadence_dist.to_csv(manifests_dir / "cadence_distribution.csv")
        
        provenance_comp = parsed.groupby("source_catalog")["observation_id"].count().to_frame("count") if "source_catalog" in parsed.columns else pd.DataFrame(columns=["count"])
        provenance_comp.to_csv(manifests_dir / "provenance_completeness.csv")
    else:
        pd.DataFrame(columns=["count"]).to_csv(manifests_dir / "sector_distribution.csv")
        pd.DataFrame(columns=["count"]).to_csv(manifests_dir / "class_distribution.csv")
        pd.DataFrame(columns=["count"]).to_csv(manifests_dir / "cadence_distribution.csv")
        pd.DataFrame(columns=["count"]).to_csv(manifests_dir / "provenance_completeness.csv")

    validation_json = {
        "status": status,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "infos": infos
    }
    
    # Save JSON report
    report_json_path = manifests_dir / "validation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(validation_json, f, indent=2)
        
    # Save Markdown report
    report_md_path = manifests_dir / "validation_report.md"
    
    err_list = "\n".join([f"* [ERROR] {e}" for e in errors]) if errors else "* No release-blocking errors detected."
    warn_list = "\n".join([f"* [WARNING] {w}" for w in warnings]) if warnings else "* No validation warnings detected."
    info_list = "\n".join([f"* [INFO] {i}" for i in infos])
    
    report_md = f"""# Phase 1 Dataset Release Validation Report
Generated: {datetime.now(timezone.utc).isoformat()}
Overall Status: **{status}**

---

## 1. Summary of Release Logs

### Errors ({len(errors)}):
{err_list}

### Warnings ({len(warnings)}):
{warn_list}

### Descriptive Info:
{info_list}

---

## 2. Ingestion Statistics & Completeness

* Checksum Verification Status: **{"PASS" if status != "FAIL" or "Checksum" not in str(errors) else "FAIL"}**
* Supervised Split Separation Status: **{"PASS" if "Leakage" not in str(errors) else "FAIL"}**
"""
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    logger.info(f"Wrote release validation reports to {report_json_path} and {report_md_path}")
    return validation_json
