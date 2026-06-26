import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import logging
from astropy.io import fits
from real_tess.flux_normaliser import normalise_pdcsap
from real_tess.sector_manifest import update_manifest_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def process_fits_file(fits_path):
    """
    Parses a single TESS FITS file.
    Extracts time, flux, errors, quality, and centroids.
    Returns a dict with processed arrays and header metadata.
    """
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"FITS file not found: {fits_path}")
        
    with fits.open(fits_path, memmap=False) as hdul:
        primary_hdr = hdul[0].header
        hdu = hdul[1]
        colnames = [c.name.upper() for c in hdu.columns]
        data = hdu.data
        
        # Determine TIC ID, Sector, Camera, CCD, RA, Dec
        target_id = primary_hdr.get("OBJECT") or primary_hdr.get("TICID")
        sector = primary_hdr.get("SECTOR")
        camera = primary_hdr.get("CAMERA")
        ccd = primary_hdr.get("CCD")
        ra = primary_hdr.get("RA_OBJ")
        dec = primary_hdr.get("DEC_OBJ")
        tess_mag = primary_hdr.get("TESSMAG")
        
        # If TESSMAG is not in primary, try extension 1
        if tess_mag is None:
            tess_mag = hdu.header.get("TESSMAG")
            
        time = np.array(data["TIME"], dtype=np.float64)
        
        # Resolve flux column
        flux_col = None
        for col in ["PDCSAP_FLUX", "KSPSAP_FLUX", "FLUX", "SAP_FLUX"]:
            if col in colnames:
                flux_col = col
                break
        if not flux_col:
            raise ValueError(f"No flux column found in FITS: {fits_path}")
            
        flux_raw = np.array(data[flux_col], dtype=np.float64)
        
        # Resolve quality
        quality = None
        for col in ["QUALITY", "SAP_QUALITY", "FLAGS"]:
            if col in colnames:
                quality = np.array(data[col], dtype=np.int64)
                break
                
        # Normalize flux
        flux_norm = normalise_pdcsap(flux_raw, quality_flags=quality)
        
        # Clean: keep only finite values of time and flux
        valid = np.isfinite(time) & np.isfinite(flux_norm)
        
        # Also clean by quality flags if present (masking non-zero flags)
        if quality is not None:
            # Mask out non-zero quality flags if it leaves at least 100 points
            clean_quality_mask = (quality == 0) & valid
            if np.sum(clean_quality_mask) >= 100:
                valid = clean_quality_mask
                
        time_clean = time[valid]
        flux_clean = flux_norm[valid]
        
        if len(time_clean) < 100:
            raise ValueError(f"FITS file has too few valid data points ({len(time_clean)}). Min is 100.")
            
        # Sort and deduplicate
        sort_idx = np.argsort(time_clean)
        time_sorted = time_clean[sort_idx]
        flux_sorted = flux_clean[sort_idx]
        
        _, unique_idx = np.unique(time_sorted, return_index=True)
        time_unique = time_sorted[unique_idx]
        flux_unique = flux_sorted[unique_idx]
        
        # Clean up optional fields
        valid_indices = np.where(valid)[0]
        sorted_indices = valid_indices[sort_idx]
        final_indices = sorted_indices[unique_idx]
        
        result = {
            "time": time_unique,
            "flux": flux_unique,
            "metadata": {
                "tic_id": str(target_id) if target_id else "",
                "sector": int(sector) if sector else None,
                "camera": int(camera) if camera else "",
                "ccd": int(ccd) if ccd else "",
                "ra": float(ra) if ra is not None else None,
                "dec": float(dec) if dec is not None else None,
                "tess_mag": float(tess_mag) if tess_mag is not None else None,
            }
        }
        
        # Optional fields
        flux_err_col = flux_col + "_ERR" if (flux_col + "_ERR") in colnames else None
        if flux_err_col:
            flux_err_raw = np.array(data[flux_err_col], dtype=np.float64)
            median_flux = np.nanmedian(flux_raw)
            if median_flux > 0:
                result["flux_err"] = (flux_err_raw / median_flux)[final_indices]
                
        # Centroids
        centroid_x = None
        centroid_y = None
        for col in ["MOM_CENTR1", "CENTROID_X", "POS_CORR1"]:
            if col in colnames:
                centroid_x = np.array(data[col], dtype=np.float64)
                break
        for col in ["MOM_CENTR2", "CENTROID_Y", "POS_CORR2"]:
            if col in colnames:
                centroid_y = np.array(data[col], dtype=np.float64)
                break
                
        if centroid_x is not None:
            result["centroid_x"] = centroid_x[final_indices]
        if centroid_y is not None:
            result["centroid_y"] = centroid_y[final_indices]
        if quality is not None:
            result["quality"] = quality[final_indices]
        if "CADENCENO" in colnames:
            result["cadenceno"] = np.array(data["CADENCENO"], dtype=np.int64)[final_indices]
            
        return result

def process_sector_manifest(manifest_path, output_dir):
    """
    Reads sector_manifest.csv, parses downloaded/cached FITS files,
    writes .npz files and updates the status of the targets.
    """
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest missing: {manifest_path}")
        return
        
    df = pd.read_csv(manifest_path)
    
    lightcurves_dir = os.path.join(output_dir, "lightcurves")
    os.makedirs(lightcurves_dir, exist_ok=True)
    
    targets_to_process = df[df["status"].isin(["downloaded", "cached"])]
    if len(targets_to_process) == 0:
        logger.info("No targets downloaded or cached to process.")
        return
        
    logger.info(f"Processing {len(targets_to_process)} targets...")
    
    processed_count = 0
    parsed_count = 0
    
    for idx, row in targets_to_process.iterrows():
        target_id = row["target_id"]
        fits_path = row["local_fits_path"]
        
        if not fits_path or pd.isna(fits_path) or not os.path.exists(fits_path):
            logger.warning(f"Target {target_id} FITS path missing or invalid: {fits_path}")
            update_manifest_status(manifest_path, target_id, "failed", f"Local FITS missing: {fits_path}")
            continue
            
        logger.info(f"Processing target {target_id} from {fits_path}...")
        try:
            data = process_fits_file(fits_path)
            meta = data["metadata"]
            
            # Save to NPZ
            npz_filename = f"{target_id}.npz"
            npz_path = os.path.join(lightcurves_dir, npz_filename)
            
            save_args = {
                "time": data["time"],
                "flux": data["flux"]
            }
            for opt_key in ["flux_err", "quality", "centroid_x", "centroid_y", "cadenceno"]:
                if opt_key in data:
                    save_args[opt_key] = data[opt_key]
                    
            np.savez_compressed(npz_path, **save_args)
            parsed_count += 1
            
            # Calculate stats
            n_points = len(data["time"])
            time_span = float(data["time"][-1] - data["time"][0])
            
            # Update manifest columns
            df.loc[df["target_id"] == target_id, "status"] = "processed"
            df.loc[df["target_id"] == target_id, "camera"] = meta["camera"]
            df.loc[df["target_id"] == target_id, "ccd"] = meta["ccd"]
            df.loc[df["target_id"] == target_id, "ra"] = meta["ra"]
            df.loc[df["target_id"] == target_id, "dec"] = meta["dec"]
            df.loc[df["target_id"] == target_id, "tess_mag"] = meta["tess_mag"]
            df.loc[df["target_id"] == target_id, "lightcurve_path"] = npz_filename
            
            # Save manifest incrementally
            df.to_csv(manifest_path, index=False)
            processed_count += 1
            logger.info(f"Saved processed lightcurve to {npz_path}")
            
        except Exception as e:
            logger.warning(f"Failed to process FITS for {target_id}: {e}")
            update_manifest_status(manifest_path, target_id, "failed", f"Parse failed: {e}")
            # Reload to keep manifest synced
            df = pd.read_csv(manifest_path)
            
    logger.info(f"Processing completed. Successfully processed {processed_count} targets.")

if __name__ == "__main__":
    process_sector_manifest("sector_manifest.csv", "./output")
