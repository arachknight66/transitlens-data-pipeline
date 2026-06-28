import json
import os
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def build_observation_manifest(config, selected_obs_ids, run_id):
    """
    Compiles the canonical observation manifest containing all metadata, 
    source references, parsing metrics, resolved labels, and status logs.
    """
    config.ensure_dirs()
    manifests_dir = config.manifests_dir
    processed_dir = config.processed_dir
    
    discovery_path = manifests_dir / "discovery_manifest.parquet"
    download_manifest_path = manifests_dir / "download_manifest.parquet"
    resolved_labels_path = manifests_dir / "resolved_labels.parquet"
    split_manifest_path = manifests_dir / "split_manifest.parquet"
    
    if not discovery_path.exists():
        raise FileNotFoundError(f"Discovery manifest not found: {discovery_path}")
        
    df_disc = pd.read_parquet(discovery_path)
    df_dl = pd.read_parquet(download_manifest_path) if download_manifest_path.exists() else pd.DataFrame()
    df_labels = pd.read_parquet(resolved_labels_path) if resolved_labels_path.exists() else pd.DataFrame()
    df_splits = pd.read_parquet(split_manifest_path) if split_manifest_path.exists() else pd.DataFrame()
    
    # Create fast lookups
    dl_lookup = {}
    if len(df_dl) > 0:
        for _, row in df_dl.iterrows():
            dl_lookup[row["obs_id"]] = row
            
    labels_lookup = {}
    if len(df_labels) > 0:
        for _, row in df_labels.iterrows():
            labels_lookup[row["tic_id"]] = row
            
    splits_lookup = {}
    if len(df_splits) > 0:
        for _, row in df_splits.iterrows():
            splits_lookup[row["tic_id"]] = row
            
    rows = []
    
    # Track exclusions/failures
    exclusions_list = []
    failures_list = []
    
    build_time = datetime.now(timezone.utc).isoformat()
    
    for idx, row in df_disc.iterrows():
        obs_id = row["obs_id"]
        tic_id = int(row["tic_id"])
        
        # 1. Resolve statuses
        disc_status = "discovered"
        dl_status = "pending"
        parse_status = "pending"
        validation_status = "pending"
        
        raw_path = ""
        raw_sha256 = ""
        processed_path = ""
        processed_sha256 = ""
        
        err_msg = ""
        quar_reason = ""
        excl_reason = ""
        
        # Download status
        if obs_id in dl_lookup:
            dl_row = dl_lookup[obs_id]
            terminal_status = dl_row["final_status"]
            dl_status = "verified" if terminal_status == "processed" else terminal_status
            raw_path = dl_row["local_path"]
            raw_sha256 = dl_row["sha256"]
            err_msg = dl_row["failure_message"]
            
        # Parse status / processed path
        # Look for sidecar JSON
        meta_filename = f"TIC-{tic_id:012d}_sector-{int(row['sector']):04d}_lc_meta.json"
        meta_path = processed_dir / "metadata" / meta_filename
        
        # Load parsing metadata if successful
        parsed_meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    parsed_meta = json.load(f)
                parse_status = "success"
                processed_path = parsed_meta.get("processed_path", "")
                processed_sha256 = parsed_meta.get("processed_sha256", "")
            except Exception as parse_err:
                parse_status = "failed"
                err_msg = f"Failed to load sidecar metadata: {parse_err}"
        elif dl_status == "verified":
            # If download succeeded but metadata doesn't exist, it means parsing failed or was skipped
            parse_status = "failed"
            err_msg = "No processed metadata sidecar found. Parse might have failed or skipped."
            
        # Deduplication check
        is_duplicate = False
        if obs_id not in selected_obs_ids and dl_status == "verified":
            is_duplicate = True
            excl_reason = "Excluded as duplicate observation. Deterministic preference policy resolved to another product for this sector."
            
        # Class and split resolution
        label_info = labels_lookup.get(tic_id, {})
        split_info = splits_lookup.get(tic_id, {})
        
        canonical_label = label_info.get("resolved_label", "unlabeled")
        label_subtype = label_info.get("label_subtype", "unlabeled")
        evidence_level = label_info.get("evidence_level", "none")
        label_strength = label_info.get("label_strength", "none")
        requires_review = bool(label_info.get("requires_review", False))
        source_catalogs = label_info.get("source_catalogs", [])
        source_catalog_versions = label_info.get("source_catalog_versions", [])
        source_record_identifiers = label_info.get("source_record_identifiers", [])
        catalogue_checksums = label_info.get("catalogue_checksums", [])
        
        split = split_info.get("split", "none")
        split_group_id = str(tic_id)
        
        # Decide if eligible for supervised training
        # Must be parsed successfully, not duplicate, and not unlabeled/review_required
        is_supervised_eligible = (
            parse_status == "success" and
            not is_duplicate and
            canonical_label not in ("unlabeled", "review_required")
            and evidence_level == "catalog_authoritative"
            and label_strength in ("strong", "medium")
        )
        
        if parse_status == "failed" or dl_status in ("network_failed", "archive_missing", "parse_failed"):
            validation_status = "failed"
            quar_reason = err_msg
            failures_list.append({
                "observation_id": obs_id,
                "tic_id": tic_id,
                "phase": "download" if dl_status != "verified" else "parse",
                "failure_reason": err_msg,
                "created_at": build_time
            })
        elif is_duplicate:
            validation_status = "excluded"
            exclusions_list.append({
                "observation_id": obs_id,
                "tic_id": tic_id,
                "sector": int(row["sector"]),
                "reason": excl_reason,
                "created_at": build_time
            })
        elif canonical_label == "review_required":
            validation_status = "quarantined"
            quar_reason = "Label conflicts route to review_required"
        else:
            validation_status = "passed"
            
        # Arrays stats from sidecar metadata
        n_points_raw = parsed_meta.get("n_points_raw", 0)
        n_points_finite = parsed_meta.get("n_points_finite", 0)
        n_points_usable = parsed_meta.get("n_points_usable", 0)
        usable_fraction = parsed_meta.get("usable_fraction", 0.0)
        time_span_days = parsed_meta.get("time_span_days", 0.0)
        median_cadence_seconds = parsed_meta.get("median_cadence_seconds", 0.0)
        gap_count = parsed_meta.get("gap_count", 0)
        selected_flux_column = parsed_meta.get("selected_flux_column", "")
        
        # Spatial from sidecar or discovery row
        ra = parsed_meta.get("ra") if parsed_meta.get("ra") is not None else row.get("ra")
        dec = parsed_meta.get("dec") if parsed_meta.get("dec") is not None else row.get("dec")
        tess_magnitude = parsed_meta.get("tess_magnitude") if parsed_meta.get("tess_magnitude") is not None else None
        crowding_metric = parsed_meta.get("crowding_metric")
        flux_fraction = parsed_meta.get("flux_fraction")
        centroid_available = bool(parsed_meta.get("centroid_available", False))
        
        rows.append({
            # Identity
            "observation_id": obs_id,
            "tic_id": tic_id,
            "target_id": f"TIC-{tic_id}",
            "sector": int(row["sector"]),
            "camera": int(parsed_meta.get("camera")) if parsed_meta.get("camera") is not None else -1,
            "ccd": int(parsed_meta.get("ccd")) if parsed_meta.get("ccd") is not None else -1,
            "cadence_seconds": round(median_cadence_seconds, 2),
            
            # Source
            "mission": "TESS",
            "archive": "MAST",
            "author": str(row.get("product_author", "SPOC")),
            "pipeline_name": "SPOC",
            "pipeline_version": parsed_meta.get("pipeline_version", ""),
            "data_release": int(parsed_meta.get("data_release_number", 0)),
            "product_type": "lightcurve",
            "product_uri": row["product_uri"],
            "raw_path": str(raw_path) if raw_path else "",
            "processed_path": str(processed_path) if processed_path else "",
            "raw_sha256": str(raw_sha256) if raw_sha256 else "",
            "processed_sha256": str(processed_sha256) if processed_sha256 else "",
            
            # Time series
            "n_points_raw": int(n_points_raw),
            "n_points_finite": int(n_points_finite),
            "n_points_usable": int(n_points_usable),
            "usable_fraction": float(usable_fraction),
            "time_span_days": float(time_span_days),
            "median_cadence_seconds": float(median_cadence_seconds),
            "gap_count": int(gap_count),
            "selected_flux_column": selected_flux_column,
            "normalization_method": config.normalization_method,
            
            # Spatial
            "ra": float(ra) if ra is not None else None,
            "dec": float(dec) if dec is not None else None,
            "tess_magnitude": float(tess_magnitude) if tess_magnitude is not None else None,
            "crowding_metric": float(crowding_metric) if crowding_metric is not None else None,
            "flux_fraction": float(flux_fraction) if flux_fraction is not None else None,
            "centroid_available": centroid_available,
            "target_pixel_file_available": False,
            
            # Labels
            "canonical_label": canonical_label,
            "label_subtype": label_subtype,
            "evidence_level": evidence_level,
            "label_strength": label_strength,
            "label_policy_version": config.label_policy_version,
            "requires_review": requires_review,
            "is_supervised_eligible": is_supervised_eligible,
            "source_catalogs": source_catalogs,
            "source_catalog_versions": source_catalog_versions,
            "source_record_identifiers": source_record_identifiers,
            "catalogue_checksums": catalogue_checksums,
            
            # Split
            "split": split,
            "split_group_id": split_group_id,
            "split_seed": config.random_seed,
            "split_version": "1.0.0",
            
            # Status
            "discovery_status": disc_status,
            "download_status": dl_status,
            "parse_status": parse_status,
            "validation_status": validation_status,
            "exclusion_reason": excl_reason,
            "quarantine_reason": quar_reason,
            
            # Provenance
            "discovery_run_id": run_id,
            "download_run_id": run_id,
            "processing_run_id": run_id,
            "code_version": "1.0.0",
            "created_at": build_time,
            "updated_at": build_time
        })
        
    df_manifest = pd.DataFrame(rows)
    output_path = manifests_dir / "observation_manifest.parquet"
    df_manifest.to_parquet(output_path, index=False)
    logger.info(f"Wrote canonical observation manifest with {len(df_manifest)} entries to {output_path}")
    
    # Save failures
    df_fail = pd.DataFrame(failures_list)
    if len(df_fail) == 0:
        df_fail = pd.DataFrame(columns=["observation_id", "tic_id", "phase", "failure_reason", "created_at"])
    df_fail.to_parquet(manifests_dir / "failures.parquet", index=False)
    
    # Save exclusions
    df_excl = pd.DataFrame(exclusions_list)
    if len(df_excl) == 0:
        df_excl = pd.DataFrame(columns=["observation_id", "tic_id", "sector", "reason", "created_at"])
    df_excl.to_parquet(manifests_dir / "exclusions.parquet", index=False)
    
    return df_manifest
