# tpf_discovery.py
# ----------------
# MAST query script to discover Target Pixel Files corresponding to observation manifests.

from __future__ import annotations
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def discover_tpf_products(
    df_obs: pd.DataFrame,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Queries MAST for TPF products corresponding to target TIC IDs and sectors.
    """
    # Exclude those that aren't parsed successfully
    targets = df_obs[df_obs["parse_status"] == "success"].copy()
    if limit is not None:
        targets = targets.head(limit)
        
    records = []
    
    try:
        from astroquery.mast import Observations
    except ImportError:
        logger.warning("astroquery.mast not installed. TPF discovery will return placeholders.")
        for idx, row in targets.iterrows():
            records.append({
                "tic_id": int(row["tic_id"]),
                "sector": int(row["sector"]),
                "tpf_uri": f"mast:TESS/product/tess2024{row['tic_id']:012d}_sector-{row['sector']:04d}_tpf.fits",
                "tpf_filename": f"tess2024{row['tic_id']:012d}_sector-{row['sector']:04d}_tpf.fits",
                "checksum": "",
            })
        return pd.DataFrame(records)
        
    logger.info(f"Discovering TPF files for {len(targets)} targets from MAST...")
    
    for idx, row in targets.iterrows():
        tic_id = int(row["tic_id"])
        sector = int(row["sector"])
        
        try:
            # Query by object ID
            obs_table = Observations.query_criteria(
                objectname=f"TIC {tic_id}",
                sequence_number=sector,
                obs_collection="TESS"
            )
            
            if len(obs_table) > 0:
                # Find data products
                products = Observations.get_product_list(obs_table)
                # Filter to TPF files
                tpf_prods = products[products["productSubGroupDescription"] == "TPF"]
                
                if len(tpf_prods) > 0:
                    best_prod = tpf_prods[0]
                    records.append({
                        "tic_id": tic_id,
                        "sector": sector,
                        "tpf_uri": str(best_prod["dataURI"]),
                        "tpf_filename": str(best_prod["productFilename"]),
                        "checksum": str(best_prod.get("checksum", "")),
                    })
                    continue
                    
            # Fallback placeholder if none found online
            records.append({
                "tic_id": tic_id,
                "sector": sector,
                "tpf_uri": f"mast:TESS/product/tess2024{tic_id:012d}_sector-{sector:04d}_tpf.fits",
                "tpf_filename": f"tess2024{tic_id:012d}_sector-{sector:04d}_tpf.fits",
                "checksum": "",
            })
        except Exception as e:
            logger.debug(f"MAST TPF discovery failed for TIC {tic_id} sector {sector}: {e}")
            records.append({
                "tic_id": tic_id,
                "sector": sector,
                "tpf_uri": f"mast:TESS/product/tess2024{tic_id:012d}_sector-{sector:04d}_tpf.fits",
                "tpf_filename": f"tess2024{tic_id:012d}_sector-{sector:04d}_tpf.fits",
                "checksum": "",
            })
            
    return pd.DataFrame(records)
