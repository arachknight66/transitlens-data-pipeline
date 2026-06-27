import os
import sys
import argparse
import logging
import json
import time as _time
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path

# Add project paths to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "transitlens-ml-core"))

from core.preprocess import clean
from core.bls_detector import detect
from core.utils import phase_fold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def check_suspicious(catalog_period, detected_period, catalog_depth_fractional, detected_depth, fraction_retained, aperture_size, background_size, cutout_size, centroid_x, centroid_y, time):
    reasons = []
    
    if not np.isfinite(detected_period) or detected_period <= 0:
        reasons.append("detected period is nonfinite or <= 0")
    if not np.isfinite(detected_depth) or detected_depth <= 0:
        reasons.append("detected depth is nonfinite or <= 0")
        
    if fraction_retained < 0.8:
        reasons.append(f"retained cadence fraction {fraction_retained:.2f} < 0.8")
        
    total_pixels = cutout_size * cutout_size
    if aperture_size >= total_pixels:
        reasons.append(f"aperture size {aperture_size} is full cutout")
        
    if background_size < 10:
        reasons.append(f"background mask size {background_size} is too small (< 10)")
        
    if len(centroid_x) > 1:
        cx_std = np.std(centroid_x)
        cy_std = np.std(centroid_y)
        if cx_std > 1.5 or cy_std > 1.5:
            reasons.append(f"large centroid shift standard deviation (x_std={cx_std:.3f}, y_std={cy_std:.3f})")
            
    if np.isfinite(detected_period) and catalog_period > 0:
        time_span = time[-1] - time[0] if len(time) > 1 else 27.0
        aliases = [
            catalog_period,
            catalog_period / 2.0,
            catalog_period * 2.0,
            time_span
        ]
        for sign in [-1.0, 1.0]:
            freq = 1.0 / catalog_period + sign
            if freq > 0:
                aliases.append(1.0 / freq)
                
        matched_alias = False
        for al in aliases:
            if abs(detected_period - al) / al < 0.05:
                matched_alias = True
                break
        if not matched_alias:
            reasons.append(f"detected period {detected_period:.4f}d differs from catalog period {catalog_period:.4f}d and its aliases")
            
    if np.isfinite(detected_depth) and catalog_depth_fractional > 0:
        ratio = detected_depth / catalog_depth_fractional
        if ratio > 10.0 or ratio < 0.1:
            reasons.append(f"detected depth {detected_depth:.6f} differs from catalog depth {catalog_depth_fractional:.6f} by > order of magnitude (ratio {ratio:.2f})")
            
    return len(reasons) > 0, reasons

def perform_aperture_photometry(tpf_path, cutout_size=15):
    """
    Perform aperture photometry on a TESS Target Pixel File (FITS).
    
    Returns a dict containing:
        time, raw_flux, background_flux, corrected_flux, flux, flux_err, quality, centroid_x, centroid_y, aperture_mask, background_mask, metadata
    """
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from scipy.ndimage import median_filter

    with fits.open(tpf_path, memmap=False) as hdul:
        tpf_table = hdul[1].data
        time = np.array(tpf_table["TIME"], dtype=np.float64)
        flux_cube = np.array(tpf_table["FLUX"], dtype=np.float64)
        flux_err_cube = np.array(tpf_table["FLUX_ERR"], dtype=np.float64)
        quality = np.array(tpf_table["QUALITY"], dtype=np.int64)
        
        # Original cadence count
        original_cadence_count = len(time)
        
        # TESS default quality flag filtering (bitmask 17087)
        quality_bitmask = 17087
        quality_mask = (quality & quality_bitmask) == 0
        cadence_count_after_quality = int(np.sum(quality_mask))
        quality_rejected_count = original_cadence_count - cadence_count_after_quality
        
        # Clean bad quality points and non-finite times first
        valid_mask = quality_mask & np.isfinite(time)
        time = time[valid_mask]
        flux_cube = flux_cube[valid_mask]
        flux_err_cube = flux_err_cube[valid_mask]
        quality = quality[valid_mask]

        # Estimate median image from quality-filtered frames
        median_img = np.nanmedian(flux_cube, axis=0)
        height, width = median_img.shape
        cy, cx = height // 2, width // 2

        # 4. Target localization using WCS coordinates
        try:
            # WCS coordinates are typically defined in HDU 2 (APERTURE) or HDU 1
            wcs_hdu_idx = 2 if len(hdul) > 2 and isinstance(hdul[2], fits.ImageHDU) else 1
            wcs = WCS(hdul[wcs_hdu_idx].header)
            ra_obj = hdul[0].header.get("RA_OBJ") or hdul[wcs_hdu_idx].header.get("RA_OBJ")
            dec_obj = hdul[0].header.get("DEC_OBJ") or hdul[wcs_hdu_idx].header.get("DEC_OBJ")
            if ra_obj is not None and dec_obj is not None:
                px_x, px_y = wcs.world_to_pixel(SkyCoord(ra_obj, dec_obj, unit="deg"))
                cx_targ = int(np.round(px_x))
                cy_targ = int(np.round(px_y))
                if 0 <= cx_targ < width and 0 <= cy_targ < height:
                    cx, cy = cx_targ, cy_targ
                    logger.info(f"Target localized using WCS at pixel (x={cx}, y={cy}) from RA={ra_obj}, Dec={dec_obj}")
                else:
                    logger.warning(f"Target WCS coordinates (x={cx_targ}, y={cy_targ}) out of bounds, using center.")
        except Exception as e:
            logger.warning(f"Target WCS localization failed: {e}. Falling back to central pixel.")

        # Extract pipeline aperture mask if present (Bit 2/value 2 is pipeline mask)
        pipeline_mask = np.zeros((height, width), dtype=bool)
        try:
            if len(hdul) > 2 and isinstance(hdul[2], fits.ImageHDU):
                pipeline_mask = (hdul[2].data & 2) > 0
        except Exception as e:
            logger.warning(f"Failed to read pipeline aperture mask: {e}")

    img_median = np.nanmedian(median_img)
    img_std = np.nanstd(median_img)
    threshold = img_median + 1.5 * img_std
    
    # Strategy A: Connected Threshold (BFS mask around target pixel)
    connected_threshold_mask = np.zeros((height, width), dtype=bool)
    queue = [(cy, cx)]
    visited = set([(cy, cx)])
    while queue:
        y, x = queue.pop(0)
        if median_img[y, x] >= threshold or (y == cy and x == cx):
            connected_threshold_mask[y, x] = True
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append((ny, nx))
                        
    # Fallback if connected threshold aperture is too small
    is_fallback = False
    if np.sum(connected_threshold_mask) < 3:
        connected_threshold_mask = np.zeros((height, width), dtype=bool)
        connected_threshold_mask[max(0, cy-1):min(height, cy+2), max(0, cx-1):min(width, cx+2)] = True
        is_fallback = True
        
    # Strategy B: Circular Aperture (radius of 2.5 pixels around cx, cy)
    circular_mask = np.zeros((height, width), dtype=bool)
    for y_px in range(height):
        for x_px in range(width):
            if (x_px - cx) ** 2 + (y_px - cy) ** 2 <= 2.5 ** 2:
                circular_mask[y_px, x_px] = True
                
    # Strategy C: Threshold Aperture (all pixels > threshold)
    threshold_mask = median_img >= threshold
    if np.sum(threshold_mask) < 3:
        threshold_mask = connected_threshold_mask.copy()

    # Fallback for pipeline mask if empty
    if np.sum(pipeline_mask) < 3:
        pipeline_mask = connected_threshold_mask.copy()

    # Use connected_threshold_mask as default aperture_mask
    aperture_mask = connected_threshold_mask
    
    # Exclude bright source pixels from the background mask
    background_mask = (~aperture_mask) & (median_img < img_median + 1.0 * img_std)
    if np.sum(background_mask) < 10:
        # fallback background mask
        background_mask = (~aperture_mask)
        
    background_pixel_count = int(np.sum(background_mask))
    aperture_pixel_count = int(np.sum(aperture_mask))
    
    raw_flux = []
    background_flux = []
    corrected_flux = []
    flux_err = []
    centroid_x = []
    centroid_y = []
    
    y_coords, x_coords = np.where(aperture_mask)
    
    for t in range(len(time)):
        frame = flux_cube[t]
        frame_err = flux_err_cube[t]
        
        # Median background level per cadence, ignoring bright source pixels
        bg_level = np.nanmedian(frame[background_mask])
        if not np.isfinite(bg_level):
            bg_level = 0.0
            
        frame_ap = frame[aperture_mask]
        frame_err_ap = frame_err[aperture_mask]
        
        r_val = np.nansum(frame_ap)
        bg_val = bg_level * aperture_pixel_count
        c_val = np.nansum(frame_ap - bg_level)
        err_val = np.sqrt(np.nansum(frame_err_ap ** 2))
        
        # Centroid
        weights = frame_ap - bg_level
        weights = np.clip(weights, 0.0, None)
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            cx_val = np.sum(x_coords * weights) / sum_weights
            cy_val = np.sum(y_coords * weights) / sum_weights
        else:
            cx_val = float(cx)
            cy_val = float(cy)
            
        raw_flux.append(r_val)
        background_flux.append(bg_val)
        corrected_flux.append(c_val)
        flux_err.append(err_val)
        centroid_x.append(cx_val)
        centroid_y.append(cy_val)
        
    raw_flux = np.array(raw_flux)
    background_flux = np.array(background_flux)
    corrected_flux = np.array(corrected_flux)
    flux_err = np.array(flux_err)
    centroid_x = np.array(centroid_x)
    centroid_y = np.array(centroid_y)
    
    # Filter non-finite and <= 0 flux values
    valid_points = np.isfinite(corrected_flux) & (corrected_flux > 0)
    nan_outlier_rejected_count = len(corrected_flux) - int(np.sum(valid_points))
    
    time = time[valid_points]
    raw_flux = raw_flux[valid_points]
    background_flux = background_flux[valid_points]
    corrected_flux = corrected_flux[valid_points]
    flux_err = flux_err[valid_points]
    centroid_x = centroid_x[valid_points]
    centroid_y = centroid_y[valid_points]
    quality = quality[valid_points]
    
    if len(time) < 100:
        raise ValueError(f"Too few valid points after photometry: {len(time)} < 100")
        
    median_val = np.median(corrected_flux)
    normalized_flux = corrected_flux / median_val
    normalized_err = flux_err / median_val
    
    fraction_retained = len(time) / original_cadence_count

    # Helper function to compute robust RMS (from MAD)
    def compute_robust_rms(arr):
        if len(arr) == 0:
            return 999.0
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        return float(1.4826 * mad / med) if med > 0 else 999.0

    # Helper function to estimate CDPP-like noise
    def estimate_cdpp(arr):
        if len(arr) < 13:
            return 999.0
        detrended = arr - median_filter(arr, size=13)
        return float(np.std(detrended))

    # Evaluate multiple aperture strategies
    aperture_comparison = {}
    for name, mask in [
        ("connected_threshold", connected_threshold_mask),
        ("circular", circular_mask),
        ("threshold", threshold_mask),
        ("pipeline", pipeline_mask)
    ]:
        if np.sum(mask) > 0:
            strat_pixel_count = int(np.sum(mask))
            strat_c_flux = []
            for t in range(len(flux_cube)):
                if not valid_mask[t]:
                    continue
                frame = flux_cube[t]
                bg_level = np.nanmedian(frame[background_mask])
                if not np.isfinite(bg_level):
                    bg_level = 0.0
                strat_c_flux.append(np.nansum(frame[mask] - bg_level))
            strat_c_flux = np.array(strat_c_flux)[valid_points]
            aperture_comparison[name] = {
                "pixel_count": strat_pixel_count,
                "robust_rms": compute_robust_rms(strat_c_flux),
                "cdpp_ppm": estimate_cdpp(strat_c_flux) * 1e6
            }
        else:
            aperture_comparison[name] = {
                "pixel_count": 0,
                "robust_rms": 999.0,
                "cdpp_ppm": 999999.0
            }

    # Centroid stability
    centroid_stability = {
        "std_x": float(np.std(centroid_x)),
        "std_y": float(np.std(centroid_y))
    }

    # Background contamination ratio (median bg flux divided by median raw target flux)
    med_bg = np.median(background_flux)
    med_raw = np.median(raw_flux)
    contamination_ratio = float(med_bg / med_raw) if med_raw > 0 else 0.0

    metadata = {
        "aperture_pixels": aperture_pixel_count,
        "background_pixels": background_pixel_count,
        "is_fallback": is_fallback,
        "median_raw_flux": float(median_val),
        "quality_bitmask": quality_bitmask,
        "cadence_counts": {
            "original": original_cadence_count,
            "after_quality": cadence_count_after_quality,
            "retained": len(time),
            "quality_rejected": quality_rejected_count,
            "nan_outlier_rejected": nan_outlier_rejected_count
        },
        "fraction_retained": float(fraction_retained),
        "robust_rms": compute_robust_rms(corrected_flux),
        "cdpp_ppm": estimate_cdpp(corrected_flux) * 1e6,
        "centroid_stability": centroid_stability,
        "contamination_ratio": contamination_ratio,
        "aperture_comparison": aperture_comparison
    }
    
    return {
        "time": time,
        "raw_flux": raw_flux,
        "background_flux": background_flux,
        "corrected_flux": corrected_flux,
        "flux": normalized_flux,
        "flux_err": normalized_err,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "quality": quality,
        "aperture_mask": aperture_mask,
        "background_mask": background_mask,
        "median_img": median_img,
        "metadata": metadata,
        "aperture_masks": {
            "connected_threshold": connected_threshold_mask,
            "circular": circular_mask,
            "threshold": threshold_mask,
            "pipeline": pipeline_mask
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Process TESS Target Pixel Files via Aperture Photometry")
    parser.add_argument("--manifest", required=True, help="Path to manifest Parquet file")
    parser.add_argument("--output-dir", default="transitlens-data-pipeline/datasets/processed/lightcurves", help="Directory to save processed NPZ light curves")
    parser.add_argument("--resume", action="store_true", help="Resume processing from manifest status")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    qa_dir = os.path.join(args.output_dir, "qa")
    os.makedirs(qa_dir, exist_ok=True)
    
    if not os.path.exists(args.manifest):
        logger.error(f"Manifest path not found: {args.manifest}")
        sys.exit(1)
        
    df = pd.read_parquet(args.manifest)
    
    success_count = 0
    failed_count = 0
    suspicious_count = 0
    
    # Track the temporary list of processed files for QA limits
    processed_targets_for_qa = []
    
    for idx, row in df.iterrows():
        target_id = row["target_id"]
        status = row["download_status"]
        raw_fits = row["raw_fits_path"]
        sector = row["sector"]
        tic_id = row["tic_id"]
        
        if status not in ["downloaded", "cached"] or not raw_fits or not os.path.exists(raw_fits):
            continue
            
        # Standard sector-specific filename
        npz_filename = f"TIC-{tic_id}_sector{int(sector):04d}.npz"
        npz_path = os.path.join(args.output_dir, npz_filename)
        meta_path = os.path.join(args.output_dir, f"TIC-{tic_id}_sector{int(sector):04d}_meta.json")
        
        # Check resume condition
        if args.resume and os.path.exists(npz_path) and os.path.exists(meta_path):
            df.loc[idx, "processed_path"] = npz_path
            df.loc[idx, "processing_status"] = "success"
            success_count += 1
            continue
            
        logger.info(f"Extracting photometry for target {target_id} sector {sector} from {raw_fits}...")
        try:
            res = perform_aperture_photometry(raw_fits, cutout_size=int(row.get("cutout_size", 15)))
            
            # Save compressed npz with ALL required arrays and alternative masks
            np.savez_compressed(
                npz_path,
                time=res["time"],
                raw_flux=res["raw_flux"],
                background_flux=res["background_flux"],
                corrected_flux=res["corrected_flux"],
                flux=res["flux"],
                flux_err=res["flux_err"],
                centroid_x=res["centroid_x"],
                centroid_y=res["centroid_y"],
                quality=res["quality"],
                aperture_mask=res["aperture_mask"],
                background_mask=res["background_mask"],
                median_img=res["median_img"],
                connected_threshold_mask=res["aperture_masks"]["connected_threshold"],
                circular_mask=res["aperture_masks"]["circular"],
                threshold_mask=res["aperture_masks"]["threshold"],
                pipeline_mask=res["aperture_masks"]["pipeline"]
            )
            
            # Run clean + BLS check for scientific comparison
            preprocess_res = clean(res["time"], res["flux"])
            clean_time = preprocess_res.time
            clean_flux = preprocess_res.flux
            bls_result = detect(clean_time, clean_flux)
            
            detected_period = bls_result.best_period if bls_result.candidate_detected else 0.0
            detected_depth = bls_result.best_depth if bls_result.candidate_detected else 0.0
            
            # Check suspicious conditions
            catalog_period = float(row.get("period_days", 0.0))
            catalog_depth_frac = float(row.get("depth_ppm", 0.0)) / 1e6
            
            is_susp, susp_reasons = check_suspicious(
                catalog_period=catalog_period,
                detected_period=detected_period,
                catalog_depth_fractional=catalog_depth_frac,
                detected_depth=detected_depth,
                fraction_retained=res["metadata"]["fraction_retained"],
                aperture_size=res["metadata"]["aperture_pixels"],
                background_size=res["metadata"]["background_pixels"],
                cutout_size=int(row.get("cutout_size", 15)),
                centroid_x=res["centroid_x"],
                centroid_y=res["centroid_y"],
                time=res["time"]
            )
            
            # Save JSON metadata sidecar
            meta_payload = {
                "tic_id": int(tic_id),
                "sector": int(sector),
                "source_fits_path": str(raw_fits),
                "checksum": str(row.get("checksum", "")),
                "cutout_size": int(row.get("cutout_size", 15)),
                "aperture_version": "connected_threshold_v1.0",
                "aperture_pixels": res["metadata"]["aperture_pixels"],
                "background_pixels": res["metadata"]["background_pixels"],
                "is_fallback": res["metadata"]["is_fallback"],
                "quality_bitmask": res["metadata"]["quality_bitmask"],
                "cadence_counts": res["metadata"]["cadence_counts"],
                "median_flux": res["metadata"]["median_raw_flux"],
                "normalization_method": "median_division",
                "processing_timestamp": pd.Timestamp.now().isoformat(),
                "robust_rms": res["metadata"]["robust_rms"],
                "cdpp_ppm": res["metadata"]["cdpp_ppm"],
                "centroid_stability": res["metadata"]["centroid_stability"],
                "contamination_ratio": res["metadata"]["contamination_ratio"],
                "aperture_comparison": res["metadata"]["aperture_comparison"],
                "scientific_check": {
                    "catalog_period": catalog_period,
                    "detected_period": detected_period,
                    "catalog_depth_fractional": catalog_depth_frac,
                    "detected_depth": detected_depth,
                    "is_suspicious": is_susp,
                    "suspicious_reasons": susp_reasons
                }
            }
            with open(meta_path, "w") as f:
                json.dump(meta_payload, f, indent=2)
                
            df.loc[idx, "processed_path"] = npz_path
            df.loc[idx, "aperture_version"] = "connected_threshold_v1.0"
            
            if is_susp:
                df.loc[idx, "processing_status"] = "suspicious"
                df.loc[idx, "failure_reason"] = "; ".join(susp_reasons)
                suspicious_count += 1
                logger.warning(f"Target {target_id} sector {sector} is suspicious: {susp_reasons}")
            else:
                df.loc[idx, "processing_status"] = "success"
                df.loc[idx, "failure_reason"] = ""
                success_count += 1
                logger.info(f"Photometry completed for target {target_id} -> {npz_path}")
                
            # Generate Phase-folded QA Plots for first 10 targets or failed/suspicious targets
            if len(processed_targets_for_qa) < 10 or is_susp:
                processed_targets_for_qa.append(target_id)
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(12, 8))
                    
                    # Subplot 1: Median image
                    plt.subplot(2, 2, 1)
                    plt.imshow(res["median_img"], origin="lower", cmap="viridis")
                    plt.title(f"{target_id} Median Image")
                    y_coords, x_coords = np.where(res["aperture_mask"])
                    plt.plot(x_coords, y_coords, "r+", markersize=6, label="Aperture")
                    bg_y, bg_x = np.where(res["background_mask"])
                    plt.plot(bg_x, bg_y, "b.", markersize=2, label="Background")
                    plt.legend()
                    
                    # Subplot 2: Raw vs Corrected Light curve
                    plt.subplot(2, 2, 2)
                    plt.plot(res["time"], res["raw_flux"]/np.median(res["raw_flux"]), "r.", markersize=2, label="Raw")
                    plt.plot(res["time"], res["flux"], "k.", markersize=2, label="Corrected")
                    plt.title("Light Curves")
                    plt.legend()
                    
                    # Subplot 3: Phase-folded using detected period
                    plt.subplot(2, 2, 3)
                    if bls_result.candidate_detected:
                        folded = phase_fold(clean_time, bls_result.best_period, bls_result.best_t0)
                        plt.plot(folded, clean_flux, "k.", markersize=2)
                        plt.title(f"Folded Detected P={bls_result.best_period:.4f}d")
                    else:
                        plt.title("No Period Detected")
                        
                    # Subplot 4: Phase-folded using catalog period
                    plt.subplot(2, 2, 4)
                    if catalog_period > 0:
                        folded_cat = phase_fold(clean_time, catalog_period, bls_result.best_t0 or clean_time[0])
                        plt.plot(folded_cat, clean_flux, "b.", markersize=2)
                        plt.title(f"Folded Catalog P={catalog_period:.4f}d")
                    else:
                        plt.title("No Catalog Period")
                        
                    plt.tight_layout()
                    plot_path = os.path.join(qa_dir, f"{target_id}_sector{int(sector):04d}_qa.png")
                    plt.savefig(plot_path)
                    plt.close()
                except Exception as e:
                    logger.warning(f"Could not save QA plot for {target_id}: {e}")
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Failed to process photometry for {target_id}: {e}")
            df.loc[idx, "processing_status"] = "failed"
            df.loc[idx, "failure_reason"] = f"Photometry error: {e}"
            failed_count += 1
            
        # Write back manifest Parquet atomically after each target
        from build_tess_training_manifest import save_manifest_atomically
        save_manifest_atomically(df, args.manifest)
        
    logger.info(f"Photometry processing completed. Success: {success_count}, Failed: {failed_count}, Suspicious: {suspicious_count}.")

if __name__ == "__main__":
    main()
