import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def resolve_duplicates(config):
    """
    Detects and resolves duplicate TIC-sector observations deterministically.
    Selects one representative observation and writes the exclusions and reasons
    to duplicate_groups.parquet.
    """
    config.ensure_dirs()
    manifests_dir = config.manifests_dir
    
    download_manifest_path = manifests_dir / "download_manifest.parquet"
    if not download_manifest_path.exists():
        raise FileNotFoundError(f"Download manifest not found: {download_manifest_path}. Run downloader first.")
        
    df_dl = pd.read_parquet(download_manifest_path)
    
    # We only deduplicate observations that were successfully downloaded/verified
    valid_obs = df_dl[df_dl["final_status"].isin(["verified", "processed"])].copy()
    
    groups = valid_obs.groupby(["tic_id", "sector"])
    
    selected_indices = []
    duplicate_records = []
    
    for (tic_id, sector), group in groups:
        if len(group) == 1:
            selected_indices.append(group.index[0])
            continue
            
        logger.info(f"Resolving duplicate group for TIC-{tic_id} in Sector {sector} ({len(group)} observations)...")
        
        # Sort group by preference rules:
        # Rule 1: Supported official author (SPOC is preferred over TESS-SPOC)
        # We can extract author from product_filename or query, but in our discovery query, we filtered for provenance_name='SPOC'.
        # However, let's write general rules:
        # - prefer SPOC in product_uri/product_filename
        # Rule 2: Desired cadence (closest to 120s)
        # Rule 3: File size (larger file might contain more data / releases)
        # Rule 4: Retained points/usable fraction (if we have it, or proxy)
        # Rule 5: Deterministic URI ordering (alphabetical) as tie-breaker
        
        sorted_group = group.copy()
        
        # Add sorting score columns
        sorted_group["author_score"] = sorted_group["product_uri"].apply(
            lambda uri: 2 if "spoc" in str(uri).lower() else (1 if "tess-spoc" in str(uri).lower() else 0)
        )
        sorted_group["filename_len"] = sorted_group["product_filename"].apply(len)
        
        # Sort: author_score descending, actual_size descending, product_uri ascending
        sorted_group = sorted_group.sort_values(
            by=["author_score", "actual_size", "product_uri"],
            ascending=[False, False, True]
        )
        
        # The first row is the selected representative
        rep_row = sorted_group.iloc[0]
        selected_indices.append(rep_row.name)
        
        # Record details for duplicate_groups.parquet
        excl_rows = sorted_group.iloc[1:]
        for idx, row in excl_rows.iterrows():
            duplicate_records.append({
                "tic_id": int(tic_id),
                "sector": int(sector),
                "selected_obs_id": rep_row["obs_id"],
                "selected_product_uri": rep_row["product_uri"],
                "selected_sha256": rep_row["sha256"],
                "excluded_obs_id": row["obs_id"],
                "excluded_product_uri": row["product_uri"],
                "excluded_sha256": row["sha256"],
                "exclusion_reason": "Deterministic duplicate preference policy (e.g. author priority, file size, URI tie-breaker)."
            })
            
    df_dups = pd.DataFrame(duplicate_records)
    if len(df_dups) == 0:
        df_dups = pd.DataFrame(columns=[
            "tic_id", "sector", "selected_obs_id", "selected_product_uri",
            "selected_sha256", "excluded_obs_id", "excluded_product_uri",
            "excluded_sha256", "exclusion_reason"
        ])
        
    dups_path = manifests_dir / "duplicate_groups.parquet"
    df_dups.to_parquet(dups_path, index=False)
    logger.info(f"Wrote {len(df_dups)} resolved duplicate records to {dups_path}")
    
    # We update the final status of excluded duplicates in download manifest or handle it in manifests.py
    # We return the list of selected observation IDs
    selected_obs_ids = set(valid_obs.loc[selected_indices, "obs_id"].tolist())
    return selected_obs_ids
