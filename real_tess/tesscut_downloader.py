import os
import sys
import time
import argparse
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.mast import Tesscut
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def compute_checksum(filepath):
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()

def validate_fits(filepath):
    """Validate that the downloaded file is a readable FITS target-pixel cube."""
    try:
        with fits.open(filepath, memmap=False) as hdul:
            if len(hdul) < 2:
                return False, "FITS file has fewer than 2 HDUs"
            
            data = hdul[1].data
            if data is None:
                return False, "HDU 1 data is None"
            
            colnames = [c.name.upper() for c in hdul[1].columns]
            if "TIME" not in colnames:
                return False, "Missing TIME column in HDU 1"
            if "FLUX" not in colnames:
                return False, "Missing FLUX column in HDU 1"
            
            time_arr = np.array(data["TIME"])
            if len(time_arr) == 0:
                return False, "TIME column is empty"
            if not np.any(np.isfinite(time_arr)):
                return False, "No finite observation times found"
                
            flux_arr = np.array(data["FLUX"])
            shape = flux_arr.shape
            if len(shape) != 3:
                return False, f"FLUX column is not 3D (cadence, height, width). Shape is {shape}"
                
            n_cadence, height, width = shape
            if n_cadence == 0:
                return False, "FLUX column has zero cadences"
            if height <= 1 or width <= 1:
                return False, f"FLUX column has invalid 2D spatial dimensions: {height}x{width}"
                
        return True, ""
    except Exception as e:
        return False, str(e)

def save_manifest_atomically(df, dest_path):
    dest_path = Path(dest_path)
    dest_dir = dest_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    temp_path = dest_dir / f"temp_{dest_path.name}"
    df.to_parquet(temp_path, index=False)
    try:
        pd.read_parquet(temp_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Manifest validation failed after write: {e}")
    os.replace(str(temp_path), str(dest_path))

def worker_download_task(row_dict, cutout_size, cache_dir, sector_policy, explicit_sector, attempt_limit=3):
    tic_id = int(row_dict["tic_id"])
    ra = float(row_dict["ra"])
    dec = float(row_dict["dec"])
    manifest_sector = row_dict.get("sector")
    target_id = row_dict["target_id"]
    
    # Coordinate validation
    if not np.isfinite(ra) or not np.isfinite(dec) or ra < 0.0 or ra > 360.0 or dec < -90.0 or dec > 90.0:
        return {
            "tic_id": tic_id,
            "status": "failed",
            "sector": None,
            "fits_path": "",
            "reason": "invalid_coordinates",
            "checksum": ""
        }
    if ra == 0.0 and dec == 0.0:
        return {
            "tic_id": tic_id,
            "status": "failed",
            "sector": None,
            "fits_path": "",
            "reason": "invalid_coordinates (placeholder 0,0)",
            "checksum": ""
        }
        
    # Sector discovery
    sectors_to_download = []
    if pd.notnull(manifest_sector):
        sectors_to_download = [int(manifest_sector)]
    else:
        try:
            coord = SkyCoord(ra, dec, unit="deg")
            sector_table = Tesscut.get_sectors(coordinates=coord)
            if len(sector_table) > 0:
                discovered = list(sector_table["sector"].astype(int))
                discovered.sort()
                
                if explicit_sector is not None:
                    if explicit_sector in discovered:
                        sectors_to_download = [explicit_sector]
                    else:
                        return {
                            "tic_id": tic_id,
                            "status": "no_coverage",
                            "sector": None,
                            "fits_path": "",
                            "reason": f"Explicit sector {explicit_sector} not covered",
                            "checksum": ""
                        }
                elif sector_policy == "latest":
                    sectors_to_download = [discovered[-1]]
                else: # default "first"
                    sectors_to_download = [discovered[0]]
            else:
                return {
                    "tic_id": tic_id,
                    "status": "no_coverage",
                    "sector": None,
                    "fits_path": "",
                    "reason": "No TESS coverage found on MAST",
                    "checksum": ""
                }
        except Exception as e:
            return {
                "tic_id": tic_id,
                "status": "failed",
                "sector": None,
                "fits_path": "",
                "reason": f"Sector discovery failed: {e}",
                "checksum": ""
            }
            
    if not sectors_to_download:
        return {
            "tic_id": tic_id,
            "status": "no_coverage",
            "sector": None,
            "fits_path": "",
            "reason": "No sectors available",
            "checksum": ""
        }
        
    chosen_sector = sectors_to_download[0]
    coord = SkyCoord(ra, dec, unit="deg")
    local_filename = f"TIC{tic_id}_sector{chosen_sector:04d}_{cutout_size}x{cutout_size}.fits"
    local_path = os.path.join(cache_dir, local_filename)
    
    # Check cache
    if os.path.exists(local_path):
        is_valid, msg = validate_fits(local_path)
        if is_valid:
            try:
                csum = compute_checksum(local_path)
            except Exception:
                csum = ""
            return {
                "tic_id": tic_id,
                "status": "cached",
                "sector": chosen_sector,
                "fits_path": local_path,
                "reason": "",
                "checksum": csum
            }
        else:
            try:
                os.remove(local_path)
            except Exception:
                pass
                
    attempt = 0
    backoff = 2.0
    err_msg = ""
    
    while attempt < attempt_limit:
        try:
            manifest_table = Tesscut.download_cutouts(coordinates=coord, size=cutout_size, sector=chosen_sector, path=cache_dir)
            if len(manifest_table) == 0:
                raise ValueError("No cutouts downloaded")
                
            downloaded_path = manifest_table[0]["Local Path"]
            if os.path.exists(downloaded_path):
                if os.path.exists(local_path):
                    os.remove(local_path)
                os.rename(downloaded_path, local_path)
                
                is_valid, msg = validate_fits(local_path)
                if is_valid:
                    try:
                        csum = compute_checksum(local_path)
                    except Exception:
                        csum = ""
                    return {
                        "tic_id": tic_id,
                        "status": "downloaded",
                        "sector": chosen_sector,
                        "fits_path": local_path,
                        "reason": "",
                        "checksum": csum
                    }
                else:
                    try:
                        os.remove(local_path)
                    except Exception:
                        pass
                    raise ValueError(f"Corrupt FITS: {msg}")
            else:
                raise FileNotFoundError(f"Downloaded file not found at {downloaded_path}")
                
        except Exception as e:
            attempt += 1
            err_msg = str(e)
            if attempt < attempt_limit:
                time.sleep(backoff ** attempt)
                
    status = "failed"
    reason = err_msg
    if "no_coverage" in err_msg.lower() or "not covered" in err_msg.lower() or "no sector found" in err_msg.lower():
        status = "no_coverage"
    elif "invalid" in err_msg.lower():
        status = "failed"
        reason = "invalid_coordinates"
    elif "network" in err_msg.lower() or "connection" in err_msg.lower() or "timeout" in err_msg.lower():
        reason = f"network_error: {err_msg}"
    elif "rate limit" in err_msg.lower() or "503" in err_msg.lower() or "429" in err_msg.lower():
        reason = f"rate_limited: {err_msg}"
    elif "corrupt" in err_msg.lower() or "fits" in err_msg.lower():
        reason = f"corrupt_fits: {err_msg}"
    else:
        reason = f"unexpected_error: {err_msg}"
        
    return {
        "tic_id": tic_id,
        "status": status,
        "sector": chosen_sector,
        "fits_path": "",
        "reason": reason,
        "checksum": ""
    }

def main():
    parser = argparse.ArgumentParser(description="TESSCut Target Pixel Cutout Downloader")
    parser.add_argument("--manifest", required=True, help="Path to targets manifest Parquet file")
    parser.add_argument("--cache-dir", default="data/raw/tesscut", help="Directory to cache downloaded FITS")
    parser.add_argument("--cutout-size", type=int, default=15, help="Size of cutout in pixels (default 15)")
    parser.add_argument("--sector-policy", default="first", choices=["first", "latest"], help="Sector discovery policy")
    parser.add_argument("--explicit-sector", type=int, default=None, help="Download explicit sector if available")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent workers (default 2)")
    parser.add_argument("--resume", action="store_true", help="Skip already downloaded targets")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of downloads")
    parser.add_argument("--confirm-large-download", action="store_true", help="Confirm large downloads without prompting")
    args = parser.parse_args()
    
    os.makedirs(args.cache_dir, exist_ok=True)
    
    if not os.path.exists(args.manifest):
        logger.error(f"Manifest path does not exist: {args.manifest}")
        sys.exit(1)
        
    df = pd.read_parquet(args.manifest)
    
    # Resume validation (check actual file validity instead of just status string)
    logger.info("Validating existing cached files...")
    for idx, row in df.iterrows():
        fits_path = row.get("raw_fits_path")
        if pd.notnull(fits_path) and fits_path != "":
            p = Path(fits_path)
            if p.exists():
                is_valid, msg = validate_fits(p)
                if is_valid:
                    stored_csum = row.get("checksum")
                    if pd.notnull(stored_csum) and stored_csum != "":
                        try:
                            actual_csum = compute_checksum(p)
                            if actual_csum != stored_csum:
                                df.loc[idx, "download_status"] = "pending"
                                df.loc[idx, "raw_fits_path"] = ""
                                df.loc[idx, "checksum"] = ""
                            else:
                                df.loc[idx, "download_status"] = "cached"
                        except Exception:
                            df.loc[idx, "download_status"] = "cached"
                    else:
                        df.loc[idx, "download_status"] = "cached"
                else:
                    df.loc[idx, "download_status"] = "pending"
                    df.loc[idx, "raw_fits_path"] = ""
                    df.loc[idx, "checksum"] = ""
            else:
                df.loc[idx, "download_status"] = "pending"
                df.loc[idx, "raw_fits_path"] = ""
                df.loc[idx, "checksum"] = ""
        else:
            df.loc[idx, "download_status"] = "pending"
            
    # Filter out already successfully processed targets if resuming
    if args.resume:
        eligible_mask = ~df["download_status"].isin(["downloaded", "cached"])
    else:
        eligible_mask = pd.Series(True, index=df.index)
        
    df_eligible = df[eligible_mask]
    eligible_count = len(df_eligible)
    class_balance = df["class_label"].value_counts().to_dict()
    approx_size_mb = eligible_count * 2.5
    
    print("\n" + "=" * 60)
    print("  TESSCut Downloader Audit Summary")
    print("=" * 60)
    print(f"Destination cache directory: {args.cache_dir}")
    print(f"Cutout size: {args.cutout_size}x{args.cutout_size} pixels")
    print(f"Eligible target count to process: {eligible_count}")
    print(f"Total manifest class balance: {class_balance}")
    print(f"Estimated number of TIC-sector cutouts: {eligible_count}")
    print(f"Approximate download size: {approx_size_mb:.1f} MB")
    print("=" * 60 + "\n")
    
    if eligible_count > 10 and not args.limit and not args.confirm_large_download:
        logger.error("Bulk download exceeds 10 targets. Please run with --limit or --confirm-large-download flag.")
        sys.exit(1)
        
    limit_count = args.limit if args.limit is not None else eligible_count
    
    # Group eligible targets by tic_id to process each TIC only once
    unique_eligible_tics = df_eligible["tic_id"].unique()
    limit_tics = unique_eligible_tics[:limit_count] if limit_count < len(unique_eligible_tics) else unique_eligible_tics
    
    # Build list of representative row dicts for each unique eligible TIC
    tasks_to_run = []
    for tic in limit_tics:
        row_dict = df_eligible[df_eligible["tic_id"] == tic].iloc[0].to_dict()
        tasks_to_run.append(row_dict)
        
    # Cap workers at conservative value
    max_workers = min(args.workers, 4)
    logger.info(f"Starting downloads using ThreadPoolExecutor with {max_workers} workers. Total TICs: {len(tasks_to_run)}")
    
    success_count = 0
    failed_count = 0
    
    # Process concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                worker_download_task, task, args.cutout_size, args.cache_dir, args.sector_policy, args.explicit_sector
            ): task["tic_id"] for task in tasks_to_run
        }
        
        for future in as_completed(futures):
            tic_id = futures[future]
            try:
                result = future.result()
                
                # Main thread updates the manifest for all candidate rows matching this tic_id
                target_rows = df[df["tic_id"] == result["tic_id"]]
                for idx in target_rows.index:
                    df.loc[idx, "download_status"] = result["status"]
                    df.loc[idx, "failure_reason"] = result["reason"]
                    df.loc[idx, "attempt_count"] += 1
                    
                    if result["status"] in ["downloaded", "cached"]:
                        df.loc[idx, "raw_fits_path"] = result["fits_path"]
                        df.loc[idx, "checksum"] = result["checksum"]
                        df.loc[idx, "download_timestamp"] = pd.Timestamp.now().isoformat()
                        if result["sector"] is not None:
                            df.loc[idx, "sector"] = int(result["sector"])
                            
                # Save atomically
                save_manifest_atomically(df, args.manifest)
                
                if result["status"] in ["downloaded", "cached"]:
                    success_count += 1
                    logger.info(f"TIC {result['tic_id']} status: {result['status']}. Sector: {result['sector']}")
                else:
                    failed_count += 1
                    logger.warning(f"TIC {result['tic_id']} status: {result['status']}. Reason: {result['reason']}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Worker task raised exception for TIC {tic_id}: {e}")
                
    logger.info(f"Acquisition completed. Success: {success_count}, Failed: {failed_count}.")

if __name__ == "__main__":
    main()
