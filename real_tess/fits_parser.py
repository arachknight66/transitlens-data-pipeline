import os
import numpy as np
from astropy.io import fits
from real_tess.flux_normaliser import normalise_pdcsap

class InvalidFITSStructureError(ValueError):
    """Raised when a FITS file does not have the expected structure or columns."""
    pass

def read_fits_lightcurve(path):
    """
    Read time, flux, and quality arrays from a TESS FITS file.
    Supports both standard TESS SPOC/QLP light curves, lightkurve-exported FITS,
    and Target Pixel Files (TPFs).
    
    Parameters
    ----------
    path : str
        Path to the local FITS file.
        
    Returns
    -------
    dict
        {
            "time": np.ndarray (float64),
            "flux_raw": np.ndarray (float64),
            "quality": np.ndarray (int64) or None,
            "metadata": dict
        }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"FITS file not found: {path}")
        
    # Attempt parsing with lightkurve first (enables TPF auto-extraction)
    try:
        import lightkurve as lk
        # Read file with memmap=False to avoid Windows process locking
        obj = lk.read(path, memmap=False)
        
        # Check if it is a Target Pixel File or light curve
        if hasattr(obj, "to_lightcurve"):
            lc = obj.to_lightcurve()
        else:
            lc = obj
            
        time = np.array(lc.time.value, copy=True, dtype=np.float64)
        flux_raw = np.array(lc.flux.value, copy=True, dtype=np.float64)
        
        quality = None
        if hasattr(lc, "quality"):
            quality = np.array(lc.quality, copy=True, dtype=np.int64)
            
        # Resolve target ID and sector
        target_id = getattr(lc, "targetid", None) or getattr(obj, "targetid", None)
        if target_id is not None:
            target_id = str(target_id)
        else:
            target_id = getattr(lc, "label", None) or getattr(obj, "label", None)
            if target_id:
                target_id = str(target_id)
                
        sector = getattr(lc, "sector", None) or getattr(obj, "sector", None)
        if sector is not None:
            sector = int(sector)
            
        metadata = {
            "target_id": target_id,
            "sector": sector,
            "flux_type_used": "PDCSAP_FLUX" if hasattr(lc, "pdcsap_flux") else "SAP_FLUX",
            "camera": getattr(lc, "camera", None) or getattr(obj, "camera", None),
            "ccd": getattr(lc, "ccd", None) or getattr(obj, "ccd", None),
        }
        
        if time.ndim == 1 and flux_raw.ndim == 1:
            return {
                "time": time,
                "flux_raw": flux_raw,
                "quality": quality,
                "metadata": metadata
            }
    except Exception:
        # Fall back to manual astropy parsing on any error
        pass

    # ── Fallback: Manual astropy parsing ─────────────────────────────────
    try:
        hdul = fits.open(path, memmap=False)
    except Exception as e:
        raise InvalidFITSStructureError(f"Corrupted FITS file or invalid format: {e}")
        
    try:
        if len(hdul) < 2:
            raise InvalidFITSStructureError("FITS file has fewer than 2 HDUs. Expected a binary table HDU at index 1.")
            
        hdu = hdul[1]
        if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            found = False
            for h in hdul:
                if isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
                    hdu = h
                    found = True
                    break
            if not found:
                raise InvalidFITSStructureError("No binary table HDU found in FITS file.")
                
        colnames = [c.name.upper() for c in hdu.columns]
        data = hdu.data
        
        if "TIME" not in colnames:
            raise InvalidFITSStructureError("FITS table missing required 'TIME' column.")
        time = np.array(data["TIME"], copy=True, dtype=np.float64)
        
        flux_col = None
        for col in ["PDCSAP_FLUX", "KSPSAP_FLUX", "FLUX", "SAP_FLUX", "LC_DETREND", "LC_WHITE", "LC_INIT"]:
            if col in colnames:
                flux_col = col
                break
                
        if flux_col is None:
            raise InvalidFITSStructureError(
                f"FITS table missing recognized flux column. Expected one of: PDCSAP_FLUX, KSPSAP_FLUX, FLUX, SAP_FLUX, LC_DETREND, LC_WHITE, LC_INIT. Found: {colnames}"
            )
            
        flux_raw = np.array(data[flux_col], copy=True, dtype=np.float64)
        
        if time.ndim != 1:
            raise InvalidFITSStructureError(
                f"Time column must be 1-dimensional, but found shape {time.shape}."
            )
            
        if flux_raw.ndim != 1:
            raise InvalidFITSStructureError(
                f"Flux column '{flux_col}' must be 1-dimensional, but found shape {flux_raw.shape}. "
                "This file appears to be a Target Pixel File (TPF) or contain multidimensional images. "
                "TransitLens only supports 1D Light Curve (LC) FITS files."
            )
        
        quality = None
        for col in ["QUALITY", "SAP_QUALITY", "FLAGS"]:
            if col in colnames:
                quality = np.array(data[col], copy=True, dtype=np.int64)
                break
                
        header = hdu.header
        primary_header = hdul[0].header
        
        target_id = (
            primary_header.get("OBJECT") or 
            primary_header.get("TICID") or 
            header.get("OBJECT") or 
            header.get("TICID")
        )
        if target_id is not None:
            target_id = str(target_id)
            
        sector = primary_header.get("SECTOR") or header.get("SECTOR")
        if sector is not None:
            sector = int(sector)
            
        metadata = {
            "target_id": target_id,
            "sector": sector,
            "flux_type_used": flux_col,
            "camera": primary_header.get("CAMERA") or header.get("CAMERA"),
            "ccd": primary_header.get("CCD") or header.get("CCD"),
        }
        
        return {
            "time": time,
            "flux_raw": flux_raw,
            "quality": quality,
            "metadata": metadata
        }
        
    finally:
        hdul.close()

def load_fits_and_normalize(path, config=None):
    """
    Load a FITS file, extract columns, normalise flux, clean NaNs/outliers, 
    sort, deduplicate, and perform final checks.
    """
    config = config or {}
    parsed = read_fits_lightcurve(path)
    
    time = parsed["time"]
    flux_raw = parsed["flux_raw"]
    quality = parsed["quality"]
    fits_metadata = parsed["metadata"]
    
    # Normalise
    flux_norm = normalise_pdcsap(flux_raw, quality_flags=quality)
    
    # Clean: Drop NaNs (and non-finite points)
    valid = np.isfinite(time) & np.isfinite(flux_norm)
    time_clean = time[valid]
    flux_clean = flux_norm[valid]
    
    if len(time_clean) < 100:
        raise ValueError(
            f"FITS file has too few valid data points ({len(time_clean)}). "
            f"Minimum required is 100."
        )
        
    # Enforce strictly monotonic time.
    sort_idx = np.argsort(time_clean)
    time_sorted = time_clean[sort_idx]
    flux_sorted = flux_clean[sort_idx]
    
    # Handle duplicates by keeping the first occurrence (or unique times)
    _, unique_idx = np.unique(time_sorted, return_index=True)
    time_unique = time_sorted[unique_idx]
    flux_unique = flux_sorted[unique_idx]
    
    if len(time_unique) < 100:
        raise ValueError(
            f"After removing duplicate timestamps, only {len(time_unique)} valid points remain. "
            f"Minimum required is 100."
        )
        
    cadence_min = config.get("cadence_min")
    if cadence_min is None and len(time_unique) > 1:
        cadence_min = float(np.median(np.diff(time_unique)) * 1440.0)
        
    time_span_days = float(time_unique[-1] - time_unique[0]) if len(time_unique) > 1 else 0.0
    
    metadata = {
        "cadence_min": cadence_min,
        "time_span_days": time_span_days,
        "sector": fits_metadata.get("sector") or config.get("sector"),
        "label": config.get("label"),
        "true_period": config.get("true_period"),
        "true_depth": config.get("true_depth"),
        "true_duration": config.get("true_duration"),
        "flux_type_used": fits_metadata["flux_type_used"],
        "camera": fits_metadata.get("camera"),
        "ccd": fits_metadata.get("ccd"),
    }
    
    return {
        "time": time_unique.tolist(),
        "flux": flux_unique.tolist(),
        "target_id": fits_metadata.get("target_id"),
        "metadata": metadata
    }
