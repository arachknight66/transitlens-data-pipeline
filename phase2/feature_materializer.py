# feature_materializer.py
# --------------------
# Materializes diagnostics features into split parquets for Phase 3 ML training.

from __future__ import annotations
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Canonical feature list for materialization
FEATURE_COLUMNS = [
    "bls_power", "snr", "period_days", "duration_days", "depth", "transit_count",
    "odd_even_depth_delta", "odd_even_significance", "odd_even_p_value",
    "v_shape_score", "grazing_probability_proxy", "morphology_fit_quality",
    "ellipsoidal_amplitude", "ellipsoidal_significance",
    "reflection_amplitude", "reflection_significance",
    "beaming_amplitude", "beaming_significance",
    "centroid_shift_pixels", "centroid_shift_arcsec", "centroid_shift_significance",
    "centroid_mahalanobis_distance", "centroid_permutation_p_value",
    "source_target_offset_pixels", "source_target_offset_arcsec", "source_target_offset_significance",
    "summed_neighbor_flux_ratio", "aperture_weighted_neighbor_flux_ratio", "gaia_neighbor_count",
    "crowdsap", "flfrcsap", "contamination_fraction",
    "dilution_corrected_depth", "dilution_corrected_depth_uncertainty",
    "aperture_depth_slope", "aperture_depth_consistency_chi2", "aperture_depth_consistency_p_value",
    "eb_risk_score", "blend_risk_score"
]

def materialize_features(config, limit: int | None = None) -> dict:
    """
    Computes diagnostics for targets in the manifest and writes train/val/test splits.
    """
    m = config.manifests_dir
    split_manifest = pd.read_parquet(m / "split_manifest.parquet")
    obs_manifest = pd.read_parquet(m / "observation_manifest.parquet")
    
    # Merge splits and observations
    df_merged = pd.merge(
        obs_manifest.drop(columns=["split"], errors="ignore"),
        split_manifest[["tic_id", "split", "resolved_label"]],
        on="tic_id", how="inner"
    )
    
    parsed = df_merged[df_merged["parse_status"] == "success"].copy()
    if limit is not None:
        parsed = parsed.head(limit)
        
    logger.info(f"Materializing features for {len(parsed)} observations...")
    
    # Set up imports from ml-core
    import sys
    sys.path.insert(0, str(config.REPO_ROOT / "transitlens-ml-core"))
    from diagnostics import run_diagnostics
    
    all_features = []
    
    for idx, row in parsed.iterrows():
        # Load light curve arrays
        lc_path = Path(row["processed_path"])
        if not lc_path.exists():
            continue
            
        try:
            with np.load(lc_path) as data:
                time = data["time"]
                flux = data["flux"]
                centroid_x = data.get("centroid_column")
                centroid_y = data.get("centroid_row")
                quality = data.get("quality")
                
            # Perform diagnostic run
            # Use nominal initial parameters from manifest or defaults
            # (In production we use the BLS detected period, etc.)
            p_val = float(row.get("period", 1.0))
            t0_val = float(row.get("epoch", time[0]))
            dur_val = float(row.get("duration", 0.05))
            dep_val = float(row.get("depth", 0.01))
            
            meta = {
                "target_id": f"TIC-{row['tic_id']}",
                "tic_id": int(row["tic_id"]),
                "sector": int(row["sector"]),
                "ra": float(row.get("ra", 0.0)),
                "dec": float(row.get("dec", 0.0)),
                "crowding_metric": float(row.get("crowding_metric", 1.0)) if pd.notnull(row.get("crowding_metric")) else None,
                "flux_fraction": float(row.get("flux_fraction", 1.0)) if pd.notnull(row.get("flux_fraction")) else None,
            }
            
            diag = run_diagnostics(
                time, flux, period=p_val, epoch_btjd=t0_val, duration_days=dur_val, depth=dep_val,
                centroid_x=centroid_x, centroid_y=centroid_y, quality=quality,
                metadata=meta
            )
            
            # Map diagnostic dict to flat features dict
            row_features = {
                "tic_id": int(row["tic_id"]),
                "observation_id": str(row["observation_id"]),
                "split": str(row["split"]),
                "label": str(row["resolved_label"]),
            }
            
            for col in FEATURE_COLUMNS:
                row_features[col] = diag.get(col)
                
            all_features.append(row_features)
        except Exception as e:
            logger.warning(f"Failed to process {row['observation_id']}: {e}")
            
    df_features = pd.DataFrame(all_features)
    if df_features.empty:
        df_features = pd.DataFrame(columns=["tic_id", "observation_id", "split", "label"] + FEATURE_COLUMNS)
        
    # Write splits
    for split_name in ["train", "val", "test"]:
        split_df = df_features[df_features["split"] == (split_name if split_name != "val" else "val")]
        # Rename val split mapping
        if split_name == "val":
            split_df = df_features[df_features["split"] == "val"]
            out_name = "phase2_features_validation.parquet"
        else:
            out_name = f"phase2_features_{split_name}.parquet"
            
        split_df.to_parquet(m / out_name, index=False)
        logger.info(f"Saved {len(split_df)} features to {m / out_name}")
        
    # Write metadata
    df_features[["tic_id", "observation_id", "split", "label"]].to_parquet(m / "phase2_feature_metadata.parquet", index=False)
    
    # Save feature ordering and schema jsons
    with open(m / "phase2_feature_order.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
        
    schema = {col: "float64" for col in FEATURE_COLUMNS}
    with open(m / "phase2_feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
        
    # Split integrity report
    split_integrity = {
        "train_count": int((df_features["split"] == "train").sum()),
        "val_count": int((df_features["split"] == "val").sum()),
        "test_count": int((df_features["split"] == "test").sum()),
    }
    with open(m / "phase2_split_integrity.json", "w") as f:
        json.dump(split_integrity, f, indent=2)
        
    # Feature card Markdown
    card_md = f"""# Phase 2 Feature Card
Features count: {len(FEATURE_COLUMNS)}
"""
    (m / "phase2_feature_card.md").write_text(card_md)
    
    return {
        "features_count": len(df_features),
        "split_integrity": split_integrity,
    }
