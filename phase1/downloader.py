import os
import time
import random
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from astropy.io import fits
import pandas as pd
import requests
from datetime import datetime, timezone
import shutil

logger = logging.getLogger(__name__)


def ensure_download_manifest_contract(frame, run_id=""):
    """Add nullable provenance fields when importing a legacy manifest.

    Empty/``legacy_unknown`` values are explicit unknowns; no acquisition facts
    are fabricated during migration.
    """
    frame = frame.copy()
    defaults = {
        "expected_size": pd.NA,
        "failure_type": "",
        "first_attempt": "",
        "last_attempt": "",
        "download_run_id": run_id or "legacy_manifest_import",
        "code_version": "1.1.0" if run_id else "legacy_unknown",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    return frame

def compute_sha256(filepath):
    """Computes SHA-256 checksum of a file in chunks."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def verify_fits_readable(filepath):
    """Verifies that a FITS file can be opened and is not corrupt."""
    try:
        with fits.open(filepath, memmap=False) as hdul:
            if len(hdul) < 2:
                return False, "FITS file has fewer than 2 HDUs"
            # Attempt to touch table data structure
            _ = hdul[1].data
            # Try to read OBJECT header
            _ = hdul[0].header.get("OBJECT") or hdul[1].header.get("OBJECT")
        return True, ""
    except Exception as e:
        return False, f"FITS open error: {e}"

def download_file_with_retry(download_url, dest_path, retries=3, backoff=1.5, timeout=45, session=None):
    """Downloads a file with exponential backoff and jitter."""
    dest_path = Path(dest_path)
    part_path = dest_path.with_suffix(".part")
    
    last_err = ""
    for attempt in range(retries + 1):
        if attempt > 0:
            sleep_time = (backoff ** attempt) + random.uniform(0, 1)
            logger.info(f"Retrying download for {dest_path.name} in {sleep_time:.2f}s (attempt {attempt}/{retries})...")
            time.sleep(sleep_time)
            
        try:
            getter = session.get if session is not None else requests.get
            resume_from = part_path.stat().st_size if part_path.exists() else 0
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
            r = getter(download_url, stream=True, timeout=timeout, headers=headers)
            if r.status_code == 404:
                return False, "archive_missing", "MAST returned HTTP 404", ""
            elif r.status_code not in (200, 206):
                last_err = f"HTTP {r.status_code}"
                continue

            append = resume_from > 0 and r.status_code == 206
            response_size = int(r.headers.get("content-length", 0))
            expected_size = resume_from + response_size if append else response_size
            
            # Write to part file
            with open(part_path, "ab" if append else "wb") as f:
                for chunk in r.iter_content(chunk_size=131072):
                    if chunk:
                        f.write(chunk)
                        
            # Verify file size
            actual_size = part_path.stat().st_size
            if expected_size > 0 and actual_size != expected_size:
                last_err = f"Size mismatch: expected {expected_size}, got {actual_size}"
                continue
                
            # Verify basic readability as FITS
            is_valid, fits_err = verify_fits_readable(part_path)
            if not is_valid:
                last_err = f"FITS corruption check failed: {fits_err}"
                if part_path.exists():
                    part_path.unlink()
                return False, "parse_failed", last_err, ""
                
            # Compute SHA-256
            sha256 = compute_sha256(part_path)
            
            # Atomic rename
            os.replace(part_path, dest_path)
            
            return True, "verified", "", sha256
            
        except Exception as e:
            last_err = str(e)
            # Preserve a partial response so the next attempt/run can resume.
                
    return False, "network_failed", f"Failed after {retries} retries. Last error: {last_err}", ""

def run_download(config, limit=None, sector=None, resume=True, retry_failures=False, dry_run=False, verify_only=False):
    """
    Orchestrates the concurrent downloading and verification process.
    """
    config.ensure_dirs()
    manifests_dir = config.manifests_dir
    
    discovery_path = manifests_dir / "discovery_manifest.parquet"
    download_manifest_path = manifests_dir / "download_manifest.parquet"
    
    if not discovery_path.exists():
        raise FileNotFoundError(f"Discovery manifest not found: {discovery_path}. Run discovery first.")
        
    df_disc = pd.read_parquet(discovery_path)
    if len(df_disc) == 0:
        logger.warning("Discovery manifest is empty.")
        return
        
    # Filter by sector if requested
    if sector is not None:
        df_disc = df_disc[df_disc["sector"] == int(sector)].copy()
        
    logger.info(f"Loaded {len(df_disc)} discovered observations.")
    
    # Load or initialize download manifest
    if resume and download_manifest_path.exists():
        logger.info("Resuming from existing download manifest...")
        df_dl = pd.read_parquet(download_manifest_path)
        # Outer join to include new discovered items if any
        merged = pd.merge(df_disc, df_dl[["obs_id", "local_path", "actual_size", "sha256", "attempt_count", "final_status", "failure_message"]], on="obs_id", how="left")
        # Fill NaNs
        merged["final_status"] = merged["final_status"].fillna("pending")
        merged["attempt_count"] = merged["attempt_count"].fillna(0).astype(int)
        merged["actual_size"] = merged["actual_size"].fillna(0).astype(int)
        merged["sha256"] = merged["sha256"].fillna("")
        df_manifest = merged
    else:
        logger.info("Initializing new download manifest...")
        df_manifest = df_disc.copy()
        df_manifest["local_path"] = ""
        df_manifest["actual_size"] = 0
        df_manifest["sha256"] = ""
        df_manifest["attempt_count"] = 0
        df_manifest["final_status"] = "pending"
        df_manifest["download_status"] = "pending"
        df_manifest["parse_status"] = "pending"
        df_manifest["failure_message"] = ""
        df_manifest["failure_type"] = ""
        df_manifest["first_attempt"] = ""
        df_manifest["last_attempt"] = ""

    # Forward-compatible migration for manifests written by earlier versions.
    defaults = {
        "download_status": "pending", "parse_status": "pending",
        "failure_type": "", "first_attempt": "", "last_attempt": "",
    }
    for column, default in defaults.items():
        if column not in df_manifest.columns:
            if column == "download_status":
                df_manifest[column] = df_manifest["final_status"].replace({"processed": "verified"})
            elif column == "parse_status":
                df_manifest[column] = df_manifest["final_status"].map({"processed": "success"}).fillna(default)
            else:
                df_manifest[column] = default
    df_manifest = ensure_download_manifest_contract(df_manifest)

    # Filter out failures if we are NOT retrying failures
    mask_to_download = df_manifest["final_status"] == "pending"
    if retry_failures:
        mask_to_download = mask_to_download | df_manifest["final_status"].isin(["network_failed", "checksum_failed"])
        
    # If verify_only, we don't download, we just check already downloaded files
    if verify_only:
        logger.info("Running in VERIFY-ONLY mode. Checking files on disk...")
        verified_count = 0
        corrupt_count = 0
        
        for idx, row in df_manifest.iterrows():
            lpath = row["local_path"]
            if lpath and os.path.exists(lpath):
                is_ok, err = verify_fits_readable(lpath)
                if is_ok:
                    df_manifest.at[idx, "final_status"] = "verified"
                    df_manifest.at[idx, "download_status"] = "verified"
                    df_manifest.at[idx, "sha256"] = compute_sha256(lpath)
                    df_manifest.at[idx, "actual_size"] = os.path.getsize(lpath)
                    verified_count += 1
                else:
                    df_manifest.at[idx, "final_status"] = "parse_failed"
                    df_manifest.at[idx, "download_status"] = "parse_failed"
                    df_manifest.at[idx, "failure_message"] = f"Corrupted file: {err}"
                    corrupt_count += 1
                    
        df_manifest.to_parquet(download_manifest_path, index=False)
        logger.info(f"Verify-only check complete: {verified_count} verified, {corrupt_count} corrupt.")
        return
        
    targets = df_manifest[mask_to_download].copy()
    if limit is not None:
        targets = targets.head(int(limit))
        
    if len(targets) == 0:
        logger.info("No targets require downloading.")
        df_manifest.to_parquet(download_manifest_path, index=False)
        return

    # Estimate from archive sizes when available, otherwise from verified files
    # in this exact frozen cohort. Never silently launch into insufficient space.
    if "expected_size" in targets.columns and targets["expected_size"].fillna(0).sum() > 0:
        estimated_bytes = int(targets["expected_size"].fillna(0).sum())
        estimate_method = "archive expected_size"
    else:
        known = df_manifest[df_manifest["actual_size"].fillna(0) > 0]["actual_size"]
        mean_size = float(known.mean()) if len(known) else 1_500_000.0
        estimated_bytes = int(mean_size * len(targets))
        estimate_method = "mean verified product size" if len(known) else "conservative fallback"
    free_bytes = shutil.disk_usage(config.raw_dir).free
    logger.info(
        f"Storage preflight: estimated {estimated_bytes} bytes by {estimate_method}; "
        f"{free_bytes} bytes free."
    )
    if estimated_bytes > free_bytes:
        raise RuntimeError(
            f"Insufficient storage: estimated {estimated_bytes} bytes required, "
            f"but only {free_bytes} bytes are free."
        )
        
    logger.info(f"Need to download/verify {len(targets)} targets (concurrency={config.download_concurrency}).")
    
    if dry_run:
        logger.info(f"[Dry Run] Would download {len(targets)} targets.")
        return
        
    # Start thread pool
    concurrency = config.download_concurrency
    
    results = []
    
    # Save the manifest initially
    df_manifest.to_parquet(download_manifest_path, index=False)
    
    def worker(row_tuple):
        idx, row = row_tuple
        sec = int(row["sector"])
        
        # Build local target directory
        sec_dir = config.raw_dir / f"sector_{sec:04d}" / "lightcurves"
        sec_dir.mkdir(parents=True, exist_ok=True)
        
        dest_file = sec_dir / row["product_filename"]
        dl_url = row["download_url"]
        
        # Check if already exists and is readable
        if dest_file.exists():
            is_valid, fits_err = verify_fits_readable(dest_file)
            if is_valid:
                sha = compute_sha256(dest_file)
                size = dest_file.stat().st_size
                return idx, True, "verified", "", sha, size, int(row["attempt_count"])
            else:
                dest_file.unlink()
                
        # Download
        t0 = time.time()
        success, status, err_msg, sha = download_file_with_retry(
            dl_url, dest_file, 
            retries=config.download_retries, 
            backoff=config.download_backoff_factor,
            timeout=config.download_timeout,
            session=session
        )
        
        actual_size = dest_file.stat().st_size if success and dest_file.exists() else 0
        attempts = int(row["attempt_count"]) + 1
        
        return idx, success, status, err_msg, sha, actual_size, attempts

    # Lock-free safe incremental updates
    completed_count = 0
    t_start = time.time()
    
    # Initialize connection pool Session
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=concurrency, pool_maxsize=concurrency)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Process concurrent pool
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(worker, item): item for item in targets.iterrows()}
        
        for future in as_completed(futures):
            idx, success, status, err_msg, sha, actual_size, attempts = future.result()
            
            # Update manifest row
            df_manifest.at[idx, "final_status"] = status
            df_manifest.at[idx, "download_status"] = status
            df_manifest.at[idx, "failure_message"] = err_msg
            df_manifest.at[idx, "failure_type"] = "" if success else status
            now = datetime.now(timezone.utc).isoformat()
            if not str(df_manifest.at[idx, "first_attempt"]):
                df_manifest.at[idx, "first_attempt"] = now
            df_manifest.at[idx, "last_attempt"] = now
            df_manifest.at[idx, "sha256"] = sha
            df_manifest.at[idx, "actual_size"] = actual_size
            df_manifest.at[idx, "attempt_count"] = attempts
            df_manifest.at[idx, "local_path"] = str(config.raw_dir / f"sector_{int(df_manifest.at[idx, 'sector']):04d}" / "lightcurves" / df_manifest.at[idx, 'product_filename']) if success else ""
            
            completed_count += 1
            if completed_count % 100 == 0 or completed_count == len(targets):
                elapsed = time.time() - t_start
                rate = completed_count / elapsed if elapsed > 0 else 0
                logger.info(f"Downloaded/Verified {completed_count}/{len(targets)} files ({rate:.2f} files/sec). Elapsed: {elapsed:.1f}s")
                # Incremental write
                df_manifest.to_parquet(download_manifest_path, index=False)
    finally:
        session.close()

    df_manifest.to_parquet(download_manifest_path, index=False)
    logger.info("Download job finished.")
