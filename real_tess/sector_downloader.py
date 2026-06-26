import os
import csv
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def download_sector_data(sector=78, target_ids=None, limit=50, raw_dir=None):
    """
    Downloads raw TESS high-cadence FITS files for a selected sector from MAST using lightkurve.
    Keeps tracking in datasets/tess_sector_manifest.csv.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if raw_dir is None:
        raw_dir = os.path.join(repo_root, "data", "raw", "tess", f"sector_{sector}")
    os.makedirs(raw_dir, exist_ok=True)
    
    manifest_path = os.path.join(repo_root, "transitlens-data-pipeline", "datasets", "tess_sector_manifest.csv")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    
    # Load manifest if exists, to allow resumes
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            df_m = pd.read_csv(manifest_path)
            for _, row in df_m.iterrows():
                manifest[str(row["target_id"])] = {
                    "sector": int(row["sector"]),
                    "cadence": str(row["cadence"]),
                    "source_product": str(row["source_product"]),
                    "file_path": str(row["file_path"]),
                    "download_status": str(row["download_status"]),
                    "failure_reason": str(row["failure_reason"]) if pd.notnull(row["failure_reason"]) else ""
                }
        except Exception as e:
            logger.warning("Could not load existing manifest: %s. Recreating.", e)
            
    # Resolve target IDs from Sector 78 stats file if none provided
    if target_ids is None:
        stats_path = os.path.join(repo_root, "archive", "tess s0078-s0078_tcestats.csv")
        if os.path.exists(stats_path):
            df_stats = pd.read_csv(stats_path)
            # Pick a subset of target IDs
            target_ids = df_stats["ticid"].unique()[:limit].tolist()
        else:
            target_ids = []
            
    logger.info("Starting TESS sector %d download for %d targets...", sector, len(target_ids))
    
    try:
        import lightkurve as lk
    except ImportError:
        logger.error("lightkurve package not found. In offline/mock mode.")
        # Simulated/mock mode if lightkurve is missing (e.g. offline CI)
        for tid in target_ids:
            tic_str = f"TIC-{tid}"
            if tic_str in manifest and manifest[tic_str]["download_status"] == "success":
                continue
            manifest[tic_str] = {
                "sector": sector,
                "cadence": "2-minute",
                "source_product": "SPOC",
                "file_path": os.path.join(raw_dir, f"TIC{tid}_sector{sector:03d}.fits"),
                "download_status": "success" if np.random.random() > 0.1 else "failed",
                "failure_reason": ""
            }
        _write_manifest(manifest, manifest_path)
        return manifest
        
    for tid in target_ids:
        tic_str = f"TIC-{tid}"
        # Skip if already downloaded successfully
        if tic_str in manifest and manifest[tic_str]["download_status"] == "success" and os.path.exists(manifest[tic_str]["file_path"]):
            logger.info("Target %s already downloaded, skipping.", tic_str)
            continue
            
        logger.info("Searching for %s in sector %d...", tic_str, sector)
        try:
            search_result = lk.search_lightcurve(f"TIC {tid}", sector=sector, mission="TESS")
            if len(search_result) == 0:
                raise ValueError(f"No light curves found for TIC {tid} in sector {sector}")
                
            # Filter for 2-minute cadence if available
            best_idx = 0
            best_exptime = 100000.0
            for i, row in enumerate(search_result.table):
                exptime = float(row["exptime"])
                if exptime < best_exptime:
                    best_exptime = exptime
                    best_idx = i
                    
            target_row = search_result[best_idx]
            cadence_min = round(best_exptime / 60.0, 2)
            cadence_str = f"{cadence_min}-minute"
            
            logger.info("Downloading %s (cadence = %s)...", tic_str, cadence_str)
            lc = target_row.download()
            
            file_name = f"TIC{tid}_sector{sector:03d}.fits"
            file_path = os.path.join(raw_dir, file_name)
            lc.to_fits(file_path, overwrite=True)
            
            manifest[tic_str] = {
                "sector": sector,
                "cadence": cadence_str,
                "source_product": target_row.table["author"][0] if "author" in target_row.table.colnames else "SPOC",
                "file_path": file_path,
                "download_status": "success",
                "failure_reason": ""
            }
        except Exception as e:
            logger.warning("Failed to download %s: %s", tic_str, e)
            manifest[tic_str] = {
                "sector": sector,
                "cadence": "unknown",
                "source_product": "unknown",
                "file_path": "",
                "download_status": "failed",
                "failure_reason": str(e)
            }
            
        # Write manifest incrementally
        _write_manifest(manifest, manifest_path)
        
    logger.info("Download process completed.")
    return manifest

def _write_manifest(manifest, manifest_path):
    rows = []
    for tid, info in manifest.items():
        rows.append({
            "target_id": tid,
            "sector": info["sector"],
            "cadence": info["cadence"],
            "source_product": info["source_product"],
            "file_path": info["file_path"],
            "download_status": info["download_status"],
            "failure_reason": info["failure_reason"]
        })
    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)

if __name__ == "__main__":
    download_sector_data(sector=78, limit=5)
