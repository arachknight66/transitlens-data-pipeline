import os
import pandas as pd
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Constants
MANIFEST_COLS = [
    "target_id", "tic_id", "sector", "camera", "ccd", "ra", "dec", "tess_mag",
    "cadence", "observation_id", "product_filename", "download_url",
    "local_fits_path", "status", "failure_reason", "created_at"
]

def load_sector_manifest(path):
    """Loads a sector manifest from a CSV file."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=MANIFEST_COLS)

def update_manifest_status(path, target_id, status, failure_reason=None):
    """Safely updates target status and failure reason in the manifest CSV."""
    if not os.path.exists(path):
        logger.warning(f"Manifest not found for update: {path}")
        return
        
    df = pd.read_csv(path)
    idx = df[df["target_id"] == target_id].index
    if len(idx) > 0:
        df.loc[idx, "status"] = status
        df.loc[idx, "failure_reason"] = failure_reason if failure_reason else ""
        df.to_csv(path, index=False)
        logger.debug(f"Updated target {target_id} to status '{status}'")
    else:
        logger.warning(f"Target ID {target_id} not found in manifest: {path}")

def build_sector_manifest(sector, output_path, limit=None, cadence="short", cache_dir="real_tess/cache"):
    """
    Builds a target manifest for a given sector. 
    Attempts online query to MAST first. If offline/fails, falls back to parsing cache_dir FITS files.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if FITS cache contains files for this sector to mark as cached immediately
    cached_targets = {}
    if os.path.exists(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.startswith("TIC") and fname.endswith(".fits"):
                parts = fname.replace("TIC", "").split("_sector")
                if len(parts) == 2:
                    try:
                        tic_id = parts[0]
                        file_sector = int(parts[1].split(".")[0])
                        if file_sector == sector:
                            target_id = f"TIC-{tic_id}"
                            cached_targets[target_id] = os.path.join(cache_dir, fname)
                    except ValueError:
                        continue
                        
    rows = []
    online_success = False
    
    try:
        from astroquery.mast import Observations
        logger.info(f"Querying MAST for TESS Sector {sector} observations...")
        
        # Use pagination via pagesize if limit is specified
        pagesize = limit if limit else 5000
        
        # Query MAST Observations for TESS timeseries data in the sector
        obs = Observations.query_criteria(
            obs_collection="TESS",
            sequence_number=sector,
            provenance_name="SPOC",
            dataproduct_type="timeseries",
            pagesize=pagesize,
            page=1
        )
        
        if len(obs) > 0:
            logger.info(f"Successfully resolved {len(obs)} observations from MAST.")
            online_success = True
            
            for row in obs:
                # Exptime represents cadence
                exptime = float(row["t_exptime"]) if pd.notna(row["t_exptime"]) else 120.0
                # Filter out 20-second cadence to speed up downloading & processing
                if cadence == "short" and exptime != 120.0:
                    continue
                    
                target_name = str(row["target_name"]).strip()
                tic_id = "".join(c for c in target_name if c.isdigit())
                if not tic_id:
                    continue
                target_id = f"TIC-{tic_id}"
                
                # Check cache status
                local_path = cached_targets.get(target_id, "")
                status = "cached" if local_path else "pending"
                
                # Retrieve metadata fields
                ra = float(row["s_ra"]) if pd.notna(row["s_ra"]) else None
                dec = float(row["s_dec"]) if pd.notna(row["s_dec"]) else None
                obs_id = str(row["obs_id"])
                download_url = str(row["dataURL"]) if pd.notna(row["dataURL"]) else ""
                
                # Exptime represents cadence
                exptime = float(row["t_exptime"]) if pd.notna(row["t_exptime"]) else 120.0
                cadence_min = round(exptime / 60.0, 2)
                cadence_str = f"{cadence_min}-minute"
                
                # Standard TESS filename format
                product_filename = f"TIC{tic_id}_sector{sector:03d}.fits"
                
                rows.append({
                    "target_id": target_id,
                    "tic_id": tic_id,
                    "sector": sector,
                    "camera": "", # Filled during FITS parse
                    "ccd": "",    # Filled during FITS parse
                    "ra": ra,
                    "dec": dec,
                    "tess_mag": None, # Filled during FITS parse
                    "cadence": cadence_str,
                    "observation_id": obs_id,
                    "product_filename": product_filename,
                    "download_url": download_url,
                    "local_fits_path": local_path,
                    "status": status,
                    "failure_reason": "",
                    "created_at": datetime.now(timezone.utc).isoformat()
                })
                
                if limit and len(rows) >= limit:
                    break
    except Exception as e:
        logger.warning(f"Online MAST query failed: {e}. Falling back to cached-only manifest building.")
        
    # Fallback to local cache files if query was unsuccessful or returned empty
    if not online_success:
        logger.info(f"Building manifest from local cache for Sector {sector}...")
        for target_id, local_path in cached_targets.items():
            tic_id = target_id.replace("TIC-", "")
            product_filename = os.path.basename(local_path)
            rows.append({
                "target_id": target_id,
                "tic_id": tic_id,
                "sector": sector,
                "camera": "",
                "ccd": "",
                "ra": None,
                "dec": None,
                "tess_mag": None,
                "cadence": "2-minute",
                "observation_id": "",
                "product_filename": product_filename,
                "download_url": "",
                "local_fits_path": local_path,
                "status": "cached",
                "failure_reason": "",
                "created_at": datetime.now(timezone.utc).isoformat()
            })
            
            if limit and len(rows) >= limit:
                break
                
    manifest_df = pd.DataFrame(rows, columns=MANIFEST_COLS)
    manifest_df.to_csv(output_path, index=False)
    logger.info(f"Wrote sector manifest with {len(manifest_df)} rows to {output_path}")
    return manifest_df

if __name__ == "__main__":
    build_sector_manifest(98, "sector_manifest.csv", limit=5)
