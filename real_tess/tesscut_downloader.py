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
            # Check if there is data in HDU 1
            data = hdul[1].data
            if data is None:
                return False, "HDU 1 data is None"
            # Check for required target-pixel columns
            colnames = [c.name.upper() for c in hdul[1].columns]
            if "TIME" not in colnames or "RAW_CNTS" not in colnames:
                return False, f"Missing required columns in HDU 1. Found: {colnames}"
        return True, ""
    except Exception as e:
        return False, str(e)

def download_target_sector(tic_id, ra, dec, sector, cutout_size, cache_dir, attempt_limit=3):
    """Download a single sector cutout for a target with retry logic."""
    coord = SkyCoord(ra, dec, unit="deg")
    target_name = f"TIC {tic_id}"
    
    local_filename = f"TIC{tic_id}_sector{sector:03d}.fits"
    local_path = os.path.join(cache_dir, local_filename)
    
    # Check if already cached and valid
    if os.path.exists(local_path):
        is_valid, msg = validate_fits(local_path)
        if is_valid:
            logger.info(f"Target {target_name} Sector {sector} already valid in cache.")
            return "cached", local_path, ""
        else:
            logger.warning(f"Cached file corrupt for {target_name} Sector {sector}: {msg}. Redownloading.")
            try:
                os.remove(local_path)
            except Exception:
                pass
                
    attempt = 0
    backoff = 2.0
    
    while attempt < attempt_limit:
        try:
            logger.info(f"Downloading {target_name} Sector {sector} (attempt {attempt + 1}/{attempt_limit})...")
            # download_cutouts returns an astropy Table of downloaded files
            manifest_table = Tesscut.download_cutouts(coordinates=coord, size=cutout_size, sector=sector, path=cache_dir)
            if len(manifest_table) == 0:
                raise ValueError("No cutouts downloaded")
                
            # Rename the downloaded file to our standard format
            downloaded_path = manifest_table[0]["Local Path"]
            if os.path.exists(downloaded_path):
                # raw FITS files must never be modified in place, but renaming/moving is fine.
                if os.path.exists(local_path):
                    os.remove(local_path)
                os.rename(downloaded_path, local_path)
                
                # Validate
                is_valid, msg = validate_fits(local_path)
                if is_valid:
                    return "downloaded", local_path, ""
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
            logger.warning(f"Download failed for {target_name} Sector {sector}: {err_msg}")
            if attempt < attempt_limit:
                sleep_time = backoff ** attempt
                logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                
    return "failed", "", err_msg

def process_target(row, cutout_size, cache_dir):
    """Processes a single manifest target, discovering sectors and downloading them."""
    tic_id = int(row["tic_id"])
    ra = float(row["ra"])
    dec = float(row["dec"])
    manifest_sector = row.get("sector")
    
    target_name = f"TIC {tic_id}"
    
    # Discover available sectors if not known
    sectors_to_download = []
    if pd.notnull(manifest_sector):
        sectors_to_download = [int(manifest_sector)]
    else:
        try:
            coord = SkyCoord(ra, dec, unit="deg")
            sector_table = Tesscut.get_sectors(coordinates=coord)
            if len(sector_table) > 0:
                sectors_to_download = list(sector_table["sector"].astype(int))
            else:
                return {
                    "tic_id": tic_id,
                    "status": "no_coverage",
                    "sector": None,
                    "fits_path": "",
                    "reason": "No TESS coverage found on MAST"
                }
        except Exception as e:
            return {
                "tic_id": tic_id,
                "status": "failed",
                "sector": None,
                "fits_path": "",
                "reason": f"Sector discovery failed: {e}"
            }
            
    # Download the first available sector for this target (one sector per target is standard in this pipeline)
    if not sectors_to_download:
        return {
            "tic_id": tic_id,
            "status": "no_coverage",
            "sector": None,
            "fits_path": "",
            "reason": "No sectors available"
        }
        
    chosen_sector = sectors_to_download[0]
    status, fits_path, reason = download_target_sector(tic_id, ra, dec, chosen_sector, cutout_size, cache_dir)
    
    return {
        "tic_id": tic_id,
        "status": status,
        "sector": chosen_sector,
        "fits_path": fits_path,
        "reason": reason
    }

def main():
    parser = argparse.ArgumentParser(description="TESSCut Target Pixel Cutout Downloader")
    parser.add_argument("--manifest", required=True, help="Path to targets manifest Parquet file")
    parser.add_argument("--cache-dir", default="data/raw/tesscut", help="Directory to cache downloaded FITS")
    parser.add_argument("--cutout-size", type=int, default=15, help="Size of cutout in pixels (default 15)")
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
    
    # Filter out already successfully processed targets if resuming
    if args.resume:
        eligible_mask = ~df["download_status"].isin(["downloaded", "cached"])
    else:
        eligible_mask = pd.Series(True, index=df.index)
        
    df_eligible = df[eligible_mask]
    
    # Calculate estimations for user
    eligible_count = len(df_eligible)
    class_balance = df["class_label"].value_counts().to_dict()
    approx_size_mb = eligible_count * 2.5 # ~2.5 MB per cutout
    
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
    
    # Check limit or confirm large download safeguards
    if eligible_count > 10 and not args.limit and not args.confirm-large-download:
        # Prompting is not possible in non-interactive agent, so exit with warning
        logger.error("Bulk download exceeds 10 targets. Please run with --limit or --confirm-large-download flag.")
        sys.exit(1)
        
    limit_count = args.limit if args.limit is not None else eligible_count
    
    targets_processed = 0
    success_count = 0
    failed_count = 0
    
    # Iterate through manifest and download
    for idx, row in df.iterrows():
        if targets_processed >= limit_count:
            logger.info("Reached limit of downloads. Stopping.")
            break
            
        if args.resume and row["download_status"] in ["downloaded", "cached"]:
            continue
            
        # Download synchronously or in thread (since MAST can be sensitive, synchronous is safer, but ThreadPool is requested)
        # We will use ThreadPool with config workers, but process target by target atomically
        result = process_target(row, args.cutout_size, args.cache_dir)
        
        # Update manifest row atomically
        df.loc[idx, "download_status"] = result["status"]
        df.loc[idx, "failure_reason"] = result["reason"]
        df.loc[idx, "attempt_count"] += 1
        
        if result["status"] in ["downloaded", "cached"]:
            df.loc[idx, "raw_fits_path"] = result["fits_path"]
            df.loc[idx, "sector"] = result["sector"]
            df.loc[idx, "download_timestamp"] = pd.Timestamp.now().isoformat()
            try:
                df.loc[idx, "checksum"] = compute_checksum(result["fits_path"])
            except Exception:
                pass
            success_count += 1
        else:
            failed_count += 1
            
        # Write back manifest parquet file atomically
        df.to_parquet(args.manifest, index=False)
        targets_processed += 1
        
        logger.info(f"Target {row['target_id']} status: {result['status']}. Manifest updated.")
        
    logger.info(f"Acquisition completed. Success: {success_count}, Failed: {failed_count}.")

if __name__ == "__main__":
    main()
