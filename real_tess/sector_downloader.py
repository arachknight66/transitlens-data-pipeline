import os
import logging
import pandas as pd
from real_tess.sector_manifest import update_manifest_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def download_manifest_targets(manifest_path, limit=None, cache_dir="real_tess/cache"):
    """
    Downloads raw TESS high-cadence FITS files for pending targets in the manifest.
    Updates status in manifest_path.
    """
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest file missing: {manifest_path}")
        return
        
    df = pd.read_csv(manifest_path)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get pending targets
    pending_targets = df[df["status"] == "pending"]
    if len(pending_targets) == 0:
        logger.info("No pending downloads in manifest.")
        return
        
    logger.info(f"Starting downloads for {len(pending_targets)} pending targets (limit={limit})...")
    
    try:
        import lightkurve as lk
    except ImportError:
        logger.error("lightkurve package not found. Skipping downloads (offline/fallback mode).")
        # Mark all pending targets as failed
        for _, row in pending_targets.iterrows():
            update_manifest_status(manifest_path, row["target_id"], "failed", "lightkurve package missing")
        return

    download_count = 0
    for _, row in pending_targets.iterrows():
        if limit and download_count >= limit:
            logger.info("Reached limit of downloads. Stopping.")
            break
            
        target_id = row["target_id"]
        tic_id = row["tic_id"]
        sector = int(row["sector"])
        
        # Verify cached file doesn't already exist
        local_filename = f"TIC{tic_id}_sector{sector:03d}.fits"
        local_path = os.path.join(cache_dir, local_filename)
        
        if os.path.exists(local_path):
            logger.info(f"Target {target_id} already cached locally. Updating manifest.")
            df.loc[df["target_id"] == target_id, "status"] = "cached"
            df.loc[df["target_id"] == target_id, "local_fits_path"] = local_path
            df.loc[df["target_id"] == target_id, "failure_reason"] = ""
            df.to_csv(manifest_path, index=False)
            continue
            
        logger.info(f"Searching MAST for {target_id} in sector {sector}...")
        try:
            search_result = lk.search_lightcurve(f"TIC {tic_id}", sector=sector, mission="TESS")
            if len(search_result) == 0:
                raise ValueError(f"No light curves found for TIC {tic_id} in sector {sector}")
                
            # Prefer SPOC 2-minute cadence (exptime = 120.0s)
            best_idx = 0
            best_exptime = 120.0
            found_preferred = False
            for i, r in enumerate(search_result.table):
                exptime = float(r["exptime"]) if pd.notna(r["exptime"]) else 120.0
                if abs(exptime - 120.0) < 1.0:
                    best_idx = i
                    best_exptime = exptime
                    found_preferred = True
                    break
            if not found_preferred:
                # Fall back to any cadence closest to 120.0s
                best_diff = 100000.0
                for i, r in enumerate(search_result.table):
                    exptime = float(r["exptime"]) if pd.notna(r["exptime"]) else 120.0
                    diff = abs(exptime - 120.0)
                    if diff < best_diff:
                        best_diff = diff
                        best_exptime = exptime
                        best_idx = i
                        
            target_row = search_result[best_idx]
            logger.info(f"Downloading {target_id} (exptime={best_exptime}s)...")
            
            # Retry download on timeout/failure
            lc = None
            for attempt in range(2):
                try:
                    lc = target_row.download()
                    break
                except Exception as exc:
                    if attempt == 1:
                        raise exc
                    logger.warning(f"Download attempt {attempt+1} failed for {target_id}, retrying...")
                    
            if lc is None:
                raise ValueError("Download returned None")
                
            lc.to_fits(local_path, overwrite=True)
            logger.info(f"Successfully saved {target_id} FITS to {local_path}")
            
            df.loc[df["target_id"] == target_id, "status"] = "downloaded"
            df.loc[df["target_id"] == target_id, "local_fits_path"] = local_path
            df.loc[df["target_id"] == target_id, "failure_reason"] = ""
            df.to_csv(manifest_path, index=False)
            download_count += 1
            
        except Exception as e:
            logger.warning(f"Failed to download {target_id}: {e}")
            df.loc[df["target_id"] == target_id, "status"] = "failed"
            df.loc[df["target_id"] == target_id, "failure_reason"] = str(e)
            df.to_csv(manifest_path, index=False)
            
    logger.info("Download sequence completed.")

if __name__ == "__main__":
    download_manifest_targets("sector_manifest.csv", limit=5)
