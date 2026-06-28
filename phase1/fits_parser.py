import os
import json
import logging
import hashlib
import os
from pathlib import Path
import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

def compute_sha256(filepath):
    """Computes SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

class ParsingError(ValueError):
    """Exception raised for FITS parsing issues."""
    pass

def parse_single_fits(fits_path, checksum, config):
    """
    Parses a single TESS SPOC FITS file.
    Extracts time series arrays, quality flags, centroid information, 
    and extensive header metadata. Normalizes flux, computes masks, and
    performs quality checks.
    """
    fits_path = Path(fits_path)
    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")
        
    try:
        hdul = fits.open(fits_path, memmap=False)
    except Exception as e:
        raise ParsingError(f"Failed to open FITS file: {e}")
        
    try:
        if len(hdul) < 2:
            raise ParsingError("FITS has fewer than 2 HDUs.")
            
        primary_hdr = hdul[0].header
        hdu1 = hdul[1]
        
        if not isinstance(hdu1, (fits.BinTableHDU, fits.TableHDU)):
            raise ParsingError("HDU 1 is not a binary table.")
            
        colnames = [c.name.upper() for c in hdu1.columns]
        data = hdu1.data
        
        # 1. Identity & Core Metadata
        tic_id_raw = primary_hdr.get("TICID") or primary_hdr.get("OBJECT")
        if tic_id_raw is None:
            raise ParsingError("Header missing TICID/OBJECT.")
            
        tic_id = int("".join(c for c in str(tic_id_raw) if c.isdigit()))
        sector = int(primary_hdr.get("SECTOR")) if primary_hdr.get("SECTOR") is not None else None
        camera = int(primary_hdr.get("CAMERA")) if primary_hdr.get("CAMERA") is not None else None
        ccd = int(primary_hdr.get("CCD")) if primary_hdr.get("CCD") is not None else None
        
        # 2. Extract arrays
        if "TIME" not in colnames:
            raise ParsingError("TIME column missing in binary table.")
        time_btjd = np.array(data["TIME"], dtype=np.float64)
        
        sap_flux = np.array(data["SAP_FLUX"], dtype=np.float64) if "SAP_FLUX" in colnames else np.full_like(time_btjd, np.nan)
        sap_flux_err = np.array(data["SAP_FLUX_ERR"], dtype=np.float64) if "SAP_FLUX_ERR" in colnames else np.full_like(time_btjd, np.nan)
        
        pdcsap_flux = np.array(data["PDCSAP_FLUX"], dtype=np.float64) if "PDCSAP_FLUX" in colnames else np.full_like(time_btjd, np.nan)
        pdcsap_flux_err = np.array(data["PDCSAP_FLUX_ERR"], dtype=np.float64) if "PDCSAP_FLUX_ERR" in colnames else np.full_like(time_btjd, np.nan)
        
        quality = np.array(data["QUALITY"], dtype=np.int64) if "QUALITY" in colnames else np.zeros_like(time_btjd, dtype=np.int64)
        
        # Centroids
        centroid_col = np.array(data["MOM_CENTR1"], dtype=np.float64) if "MOM_CENTR1" in colnames else (
            np.array(data["CENTROID_X"], dtype=np.float64) if "CENTROID_X" in colnames else np.full_like(time_btjd, np.nan)
        )
        centroid_row = np.array(data["MOM_CENTR2"], dtype=np.float64) if "MOM_CENTR2" in colnames else (
            np.array(data["CENTROID_Y"], dtype=np.float64) if "CENTROID_Y" in colnames else np.full_like(time_btjd, np.nan)
        )
        centroid_col_err = np.array(data["MOM_CENTR1_ERR"], dtype=np.float64) if "MOM_CENTR1_ERR" in colnames else np.full_like(time_btjd, np.nan)
        centroid_row_err = np.array(data["MOM_CENTR2_ERR"], dtype=np.float64) if "MOM_CENTR2_ERR" in colnames else np.full_like(time_btjd, np.nan)
        
        # Background
        background = np.array(data["SAP_BKG"], dtype=np.float64) if "SAP_BKG" in colnames else np.full_like(time_btjd, np.nan)
        background_err = np.array(data["SAP_BKG_ERR"], dtype=np.float64) if "SAP_BKG_ERR" in colnames else np.full_like(time_btjd, np.nan)
        
        raw_cadence_number = np.array(data["CADENCENO"], dtype=np.int64) if "CADENCENO" in colnames else np.arange(len(time_btjd), dtype=np.int64)
        
        # 3. Create Masks
        # finite mask: points where time is finite and at least one flux column is finite
        finite_mask = np.isfinite(time_btjd) & (np.isfinite(pdcsap_flux) | np.isfinite(sap_flux))
        
        # ``0`` is the strict archive policy: only cadences with QUALITY == 0.
        # A non-zero value means reject cadences containing any configured bit.
        q_val = config.quality_bitmask
        archive_quality_mask = quality == 0 if q_val == 0 else (quality & q_val) == 0
        
        usable_mask = finite_mask & archive_quality_mask
        normalization_mask = usable_mask.copy()
        
        # 4. Flux Selection and Normalization
        # Compute medians using only usable points to avoid outlier distortion
        sap_usable = sap_flux[usable_mask]
        pdcsap_usable = pdcsap_flux[usable_mask]
        
        sap_median = np.nanmedian(sap_usable) if len(sap_usable) > 0 else np.nanmedian(sap_flux[finite_mask])
        pdcsap_median = np.nanmedian(pdcsap_usable) if len(pdcsap_usable) > 0 else np.nanmedian(pdcsap_flux[finite_mask])
        
        # Normalize
        sap_flux_norm = sap_flux / sap_median if (not np.isnan(sap_median) and sap_median > 0) else np.full_like(sap_flux, np.nan)
        pdcsap_flux_norm = pdcsap_flux / pdcsap_median if (not np.isnan(pdcsap_median) and pdcsap_median > 0) else np.full_like(pdcsap_flux, np.nan)
        
        sap_flux_err_norm = sap_flux_err / sap_median if (not np.isnan(sap_median) and sap_median > 0) else np.full_like(sap_flux_err, np.nan)
        pdcsap_flux_err_norm = pdcsap_flux_err / pdcsap_median if (not np.isnan(pdcsap_median) and pdcsap_median > 0) else np.full_like(pdcsap_flux_err, np.nan)
        
        # Downstream Default Choice (PDCSAP when valid, fallback to SAP)
        pdcsap_valid_count = np.sum(np.isfinite(pdcsap_flux_norm) & usable_mask)
        sap_valid_count = np.sum(np.isfinite(sap_flux_norm) & usable_mask)
        
        if pdcsap_valid_count >= config.minimum_points:
            flux_norm = pdcsap_flux_norm
            flux_err_norm = pdcsap_flux_err_norm
            selected_flux_column = "PDCSAP_FLUX"
            fallback_reason = ""
        elif sap_valid_count >= config.minimum_points:
            flux_norm = sap_flux_norm
            flux_err_norm = sap_flux_err_norm
            selected_flux_column = "SAP_FLUX"
            fallback_reason = "PDCSAP_FLUX lacked sufficient valid points. Fallback to SAP_FLUX."
        else:
            raise ParsingError(f"Insufficient finite points for either PDCSAP ({pdcsap_valid_count}) or SAP ({sap_valid_count}). Min is {config.minimum_points}.")
            
        # Cleaned Arrays (Only keep usable points for processed timeseries)
        # Enforce sorted and strictly monotonic time
        time_usable = time_btjd[usable_mask]
        flux_usable = flux_norm[usable_mask]
        flux_err_usable = flux_err_norm[usable_mask]
        quality_usable = quality[usable_mask]
        
        centroid_col_usable = centroid_col[usable_mask]
        centroid_row_usable = centroid_row[usable_mask]
        
        if len(time_usable) < config.minimum_points:
            raise ParsingError(f"Usable point count {len(time_usable)} is below minimum {config.minimum_points}.")
            
        # Sort and deduplicate timestamps
        sort_idx = np.argsort(time_usable)
        time_sorted = time_usable[sort_idx]
        flux_sorted = flux_usable[sort_idx]
        flux_err_sorted = flux_err_usable[sort_idx]
        quality_sorted = quality_usable[sort_idx]
        centroid_col_sorted = centroid_col_usable[sort_idx]
        centroid_row_sorted = centroid_row_usable[sort_idx]
        
        _, unique_idx = np.unique(time_sorted, return_index=True)
        time_final = time_sorted[unique_idx]
        flux_final = flux_sorted[unique_idx]
        flux_err_final = flux_err_sorted[unique_idx]
        quality_final = quality_sorted[unique_idx]
        centroid_col_final = centroid_col_sorted[unique_idx]
        centroid_row_final = centroid_row_sorted[unique_idx]
        
        if len(time_final) < config.minimum_points:
            raise ParsingError(f"Point count after duplicate removal ({len(time_final)}) is below minimum {config.minimum_points}.")
            
        # Time span and cadence
        time_span = float(time_final[-1] - time_final[0])
        if time_span < config.minimum_time_span_days:
            raise ParsingError(f"Total time span {time_span:.2f} days is below minimum {config.minimum_time_span_days} days.")
            
        diffs = np.diff(time_final)
        median_cadence_days = float(np.median(diffs)) if len(diffs) > 0 else 0.0
        median_cadence_sec = median_cadence_days * 86400.0
        cadence_scatter_sec = float(np.std(diffs)) * 86400.0 if len(diffs) > 0 else 0.0
        
        # Cadence check
        if not (config.min_cadence_seconds <= median_cadence_sec <= config.max_cadence_seconds):
            raise ParsingError(f"Observation cadence {median_cadence_sec:.1f}s is outside bounds ({config.min_cadence_seconds}s - {config.max_cadence_seconds}s).")
            
        # Gap analysis (Gaps > 5x median cadence)
        gap_threshold = 5 * median_cadence_days
        gaps = diffs[diffs > gap_threshold]
        gap_count = int(len(gaps))
        
        # Crowding and flux fraction (CROWDSAP, FLFRCSAP)
        # Search all extensions for crowding headers
        crowdsap = None
        flfrcsap = None
        for hdu in hdul:
            if hasattr(hdu, "header"):
                if crowdsap is None and "CROWDSAP" in hdu.header:
                    crowdsap = float(hdu.header["CROWDSAP"])
                if flfrcsap is None and "FLFRCSAP" in hdu.header:
                    flfrcsap = float(hdu.header["FLFRCSAP"])
                    
        # Coordinates
        ra = float(primary_hdr.get("RA_OBJ")) if primary_hdr.get("RA_OBJ") is not None else None
        dec = float(primary_hdr.get("DEC_OBJ")) if primary_hdr.get("DEC_OBJ") is not None else None
        tessmag = float(primary_hdr.get("TESSMAG")) if primary_hdr.get("TESSMAG") is not None else (
            float(hdu1.header.get("TESSMAG")) if hdu1.header.get("TESSMAG") is not None else None
        )
        
        # 5. Populate Metadata sidecar
        metadata = {
            "tic_id": tic_id,
            "target_id": f"TIC-{tic_id}",
            "sector": sector,
            "camera": camera,
            "ccd": ccd,
            "object_name": str(primary_hdr.get("OBJECT", "")).strip(),
            "ra": ra,
            "dec": dec,
            "tess_magnitude": tessmag,
            "cadence_seconds": round(median_cadence_sec, 2),
            "exposure_time": float(primary_hdr.get("EXPTIME", 120.0)),
            "time_system": str(primary_hdr.get("TIMESYS", "TDB")),
            "time_reference": str(primary_hdr.get("TIMEREF", "SOLARSYSTEM")),
            "pipeline_author": str(primary_hdr.get("ORIGIN", "SPOC")),
            "pipeline_version": str(primary_hdr.get("PROCVER", "unknown")),
            "data_release_number": int(primary_hdr.get("DATA_REL", 0)) if primary_hdr.get("DATA_REL") is not None else 0,
            "processing_date": str(primary_hdr.get("DATE", "")),
            "observation_id": str(primary_hdr.get("OBSID", "")),
            "product_uri": row_obs_uri_fallback(fits_path.name),
            "fits_filename": fits_path.name,
            "crowding_metric": crowdsap,
            "flux_fraction": flfrcsap,
            "n_points_raw": int(len(time_btjd)),
            "n_points_finite": int(np.sum(finite_mask)),
            "n_points_usable": int(len(time_final)),
            "usable_fraction": float(len(time_final) / len(time_btjd)),
            "median_cadence_seconds": round(median_cadence_sec, 2),
            "cadence_scatter_seconds": round(cadence_scatter_sec, 4),
            "time_span_days": round(time_span, 4),
            "gap_count": gap_count,
            "gap_durations_days": gaps.astype(float).tolist(),
            "selected_flux_column": selected_flux_column,
            "fallback_reason": fallback_reason,
            "normalization_method": config.normalization_method,
            "normalization_factors": {"sap_median": float(sap_median), "pdcsap_median": float(pdcsap_median)},
            "quality_mask_policy": "QUALITY == 0" if q_val == 0 else f"reject_bits={q_val}",
            "parser_version": "1.1.0",
            "source_checksum": checksum,
            "centroid_available": bool(np.any(np.isfinite(centroid_col_final))),
            "target_pixel_file_available": False
        }
        
        arrays = {
            "time": time_final,
            "flux": flux_final,
            "flux_err": flux_err_final,
            "quality": quality_final,
            "centroid_column": centroid_col_final,
            "centroid_row": centroid_row_final,
            "sap_flux_norm": sap_flux_norm[usable_mask][sort_idx][unique_idx],
            "pdcsap_flux_norm": pdcsap_flux_norm[usable_mask][sort_idx][unique_idx],
            # Preserve all archive columns and masks in the processed product.
            "time_btjd": time_btjd,
            "sap_flux": sap_flux,
            "sap_flux_err": sap_flux_err,
            "pdcsap_flux": pdcsap_flux,
            "pdcsap_flux_err": pdcsap_flux_err,
            "quality_raw": quality,
            "centroid_column_raw": centroid_col,
            "centroid_row_raw": centroid_row,
            "centroid_column_error": centroid_col_err,
            "centroid_row_error": centroid_row_err,
            "background": background,
            "background_error": background_err,
            "raw_cadence_number": raw_cadence_number,
            "finite_mask": finite_mask,
            "archive_quality_mask": archive_quality_mask,
            "usable_mask": usable_mask,
            "normalization_mask": normalization_mask,
        }
        
        return arrays, metadata
        
    finally:
        hdul.close()

def row_obs_uri_fallback(filename):
    """Guesses dataURL from filename when not in download manifest."""
    return f"mast:TESS/product/{filename}"

def row_obs_uri(data_url):
    return data_url

def process_and_save(row_idx, row, config, manifests_dir, download_manifest_path):
    """
    Parses a single FITS file, saves the processed arrays (.npz) and metadata (.json),
    or copy/moves to quarantine if failed. Returns updated manifest state dict.
    """
    fits_path = row["local_path"]
    checksum = row["sha256"]
    
    # Target processed filenames
    obs_filename = f"TIC-{row['tic_id']:012d}_sector-{row['sector']:04d}_lc.npz"
    processed_lc_path = config.processed_dir / "lightcurves" / obs_filename
    processed_meta_path = config.processed_dir / "metadata" / obs_filename.replace(".npz", "_meta.json")
    
    try:
        arrays, metadata = parse_single_fits(fits_path, checksum, config)
        
        # Save preprocessed NPZ
        temporary_lc_path = processed_lc_path.with_name(processed_lc_path.stem + ".tmp.npz")
        np.savez_compressed(
            temporary_lc_path,
            time=arrays["time"],
            flux=arrays["flux"],
            flux_err=arrays["flux_err"],
            quality=arrays["quality"],
            centroid_column=arrays["centroid_column"],
            centroid_row=arrays["centroid_row"],
            sap_flux_norm=arrays["sap_flux_norm"],
            pdcsap_flux_norm=arrays["pdcsap_flux_norm"],
            time_btjd=arrays["time_btjd"],
            sap_flux=arrays["sap_flux"],
            sap_flux_err=arrays["sap_flux_err"],
            pdcsap_flux=arrays["pdcsap_flux"],
            pdcsap_flux_err=arrays["pdcsap_flux_err"],
            quality_raw=arrays["quality_raw"],
            centroid_column_raw=arrays["centroid_column_raw"],
            centroid_row_raw=arrays["centroid_row_raw"],
            centroid_column_error=arrays["centroid_column_error"],
            centroid_row_error=arrays["centroid_row_error"],
            background=arrays["background"],
            background_error=arrays["background_error"],
            raw_cadence_number=arrays["raw_cadence_number"],
            finite_mask=arrays["finite_mask"],
            archive_quality_mask=arrays["archive_quality_mask"],
            usable_mask=arrays["usable_mask"],
            normalization_mask=arrays["normalization_mask"],
        )
        os.replace(temporary_lc_path, processed_lc_path)
        
        # Save metadata sidecar
        metadata["processed_path"] = str(processed_lc_path)
        metadata["processed_sha256"] = compute_sha256(processed_lc_path)
        
        temporary_meta_path = processed_meta_path.with_suffix(".json.tmp")
        with open(temporary_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        os.replace(temporary_meta_path, processed_meta_path)
            
        return {
            "status": "processed",
            "parse_status": "success",
            "processed_path": str(processed_lc_path),
            "processed_sha256": metadata["processed_sha256"],
            "metadata": metadata,
            "error_msg": ""
        }
        
    except Exception as e:
        logger.warning(f"FITS Parse failed for {Path(fits_path).name}: {e}")
        # Quarantine the file
        quarantine_dir = config.processed_dir / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        quar_dest = quarantine_dir / Path(fits_path).name
        try:
            if Path(fits_path).exists():
                # Copy instead of move to preserve raw download cache
                import shutil
                shutil.copy2(fits_path, quar_dest)
        except Exception as copy_err:
            logger.error(f"Failed to copy to quarantine: {copy_err}")
            
        return {
            "status": "quarantined",
            "parse_status": "failed",
            "processed_path": "",
            "processed_sha256": "",
            "metadata": {},
            "error_msg": str(e)
        }
