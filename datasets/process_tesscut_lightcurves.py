import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def perform_aperture_photometry(tpf_path, qa_dir=None, target_id=""):
    """
    Perform aperture photometry on a TESS Target Pixel File (FITS).
    
    Returns a dict containing:
        time, flux, flux_err, centroid_x, centroid_y, quality, metadata
    """
    with fits.open(tpf_path, memmap=False) as hdul:
        # TPF HDU 1 contains the binary table with arrays
        tpf_table = hdul[1].data
        time = np.array(tpf_table["TIME"], dtype=np.float64)
        flux_cube = np.array(tpf_table["FLUX"], dtype=np.float64)
        flux_err_cube = np.array(tpf_table["FLUX_ERR"], dtype=np.float64)
        quality = np.array(tpf_table["QUALITY"], dtype=np.int64)
        
        header = hdul[1].header
        primary_header = hdul[0].header
        
        # Clean bad quality points or NaNs from the time array first
        valid_time = np.isfinite(time)
        time = time[valid_time]
        flux_cube = flux_cube[valid_time]
        flux_err_cube = flux_err_cube[valid_time]
        quality = quality[valid_time]
        
    # Estimate median image
    median_img = np.nanmedian(flux_cube, axis=0)
    height, width = median_img.shape
    cy, cx = height // 2, width // 2
    
    # 1. Connected threshold aperture
    # Threshold: median of image + 1.5 * standard deviation
    img_median = np.nanmedian(median_img)
    img_std = np.nanstd(median_img)
    threshold = img_median + 1.5 * img_std
    
    aperture_mask = np.zeros((height, width), dtype=bool)
    
    # BFS to find connected pixels
    queue = [(cy, cx)]
    visited = set([(cy, cx)])
    
    while queue:
        y, x = queue.pop(0)
        if median_img[y, x] >= threshold or (y == cy and x == cx):
            aperture_mask[y, x] = True
            # Check 4-connected neighbors
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append((ny, nx))
                        
    # Fallback to 3x3 centered window if aperture is too small (e.g. < 3 pixels)
    is_fallback = False
    if np.sum(aperture_mask) < 3:
        aperture_mask = np.zeros((height, width), dtype=bool)
        aperture_mask[max(0, cy-1):min(height, cy+2), max(0, cx-1):min(width, cx+2)] = True
        is_fallback = True
        
    # Background subtraction: median of pixels outside the aperture
    background_mask = ~aperture_mask
    
    raw_flux = []
    corrected_flux = []
    flux_err = []
    centroid_x = []
    centroid_y = []
    
    y_coords, x_coords = np.where(aperture_mask)
    
    for t in range(len(time)):
        frame = flux_cube[t]
        frame_err = flux_err_cube[t]
        
        # Background level for this frame
        bg_level = np.nanmedian(frame[background_mask])
        if not np.isfinite(bg_level):
            bg_level = 0.0
            
        # Sum flux in aperture
        frame_ap = frame[aperture_mask]
        frame_err_ap = frame_err[aperture_mask]
        
        raw_val = np.nansum(frame_ap)
        corrected_val = np.nansum(frame_ap - bg_level)
        err_val = np.sqrt(np.nansum(frame_err_ap ** 2))
        
        # Centroid
        weights = frame_ap - bg_level
        weights = np.clip(weights, 0.0, None) # only positive weights
        sum_weights = np.sum(weights)
        if sum_weights > 0:
            cx_val = np.sum(x_coords * weights) / sum_weights
            cy_val = np.sum(y_coords * weights) / sum_weights
        else:
            cx_val = float(cx)
            cy_val = float(cy)
            
        raw_flux.append(raw_val)
        corrected_flux.append(corrected_val)
        flux_err.append(err_val)
        centroid_x.append(cx_val)
        centroid_y.append(cy_val)
        
    raw_flux = np.array(raw_flux)
    corrected_flux = np.array(corrected_flux)
    flux_err = np.array(flux_err)
    centroid_x = np.array(centroid_x)
    centroid_y = np.array(centroid_y)
    
    # Clean NaNs and gross outliers (e.g. flux values <= 0)
    valid_points = np.isfinite(corrected_flux) & (corrected_flux > 0)
    time = time[valid_points]
    corrected_flux = corrected_flux[valid_points]
    flux_err = flux_err[valid_points]
    centroid_x = centroid_x[valid_points]
    centroid_y = centroid_y[valid_points]
    quality = quality[valid_points]
    
    if len(time) < 100:
        raise ValueError(f"Too few valid points after photometry: {len(time)} < 100")
        
    # Normalize flux using training-independent deterministic logic (divide by median)
    median_val = np.median(corrected_flux)
    normalized_flux = corrected_flux / median_val
    normalized_err = flux_err / median_val
    
    # Save diagnostic plot for QA
    if qa_dir is not None:
        try:
            import matplotlib.pyplot as plt
            os.makedirs(qa_dir, exist_ok=True)
            plt.figure(figsize=(10, 4))
            
            # Subplot 1: Median Image with Aperture
            plt.subplot(1, 2, 1)
            plt.imshow(median_img, origin="lower", cmap="viridis")
            plt.title(f"{target_id} Median TPF")
            # Overlay aperture boundary
            for y in range(height):
                for x in range(width):
                    if aperture_mask[y, x]:
                        plt.plot(x, y, "r+", markersize=8)
            plt.colorbar(label="ADU/s")
            
            # Subplot 2: Extracted light curve
            plt.subplot(1, 2, 2)
            plt.plot(time, normalized_flux, "k.", markersize=2)
            plt.title(f"Extracted LC (Fallback={is_fallback})")
            plt.xlabel("Time (BTJD)")
            plt.ylabel("Normalized Flux")
            
            plt.tight_layout()
            plot_path = os.path.join(qa_dir, f"{target_id}_qa.png")
            plt.savefig(plot_path)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not save QA plot: {e}")
            
    return {
        "time": time,
        "flux": normalized_flux,
        "flux_err": normalized_err,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "quality": quality,
        "metadata": {
            "aperture_pixels": int(np.sum(aperture_mask)),
            "is_fallback": is_fallback,
            "median_raw_flux": float(median_val)
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
    
    # Process each target in the manifest that has been successfully downloaded/cached
    success_count = 0
    failed_count = 0
    
    for idx, row in df.iterrows():
        target_id = row["target_id"]
        status = row["download_status"]
        raw_fits = row["raw_fits_path"]
        
        if status not in ["downloaded", "cached"] or not raw_fits or not os.path.exists(raw_fits):
            continue
            
        npz_filename = f"{target_id}.npz"
        npz_path = os.path.join(args.output_dir, npz_filename)
        
        if args.resume and os.path.exists(npz_path):
            df.loc[idx, "processed_path"] = npz_path
            success_count += 1
            continue
            
        logger.info(f"Extracting photometry for target {target_id} from {raw_fits}...")
        try:
            # Generate QA plots for first 10 targets as a sample
            target_qa_dir = qa_dir if success_count < 10 else None
            
            res = perform_aperture_photometry(raw_fits, target_qa_dir, target_id)
            
            # Save compressed npz
            np.savez_compressed(
                npz_path,
                time=res["time"],
                flux=res["flux"],
                flux_err=res["flux_err"],
                centroid_x=res["centroid_x"],
                centroid_y=res["centroid_y"],
                quality=res["quality"]
            )
            
            df.loc[idx, "processed_path"] = npz_path
            df.loc[idx, "failure_reason"] = ""
            success_count += 1
            logger.info(f"Photometry completed for target {target_id} -> {npz_path}")
        except Exception as e:
            logger.error(f"Failed to process photometry for {target_id}: {e}")
            df.loc[idx, "failure_reason"] = f"Photometry error: {e}"
            failed_count += 1
            
        # Write back manifest Parquet atomically after processing each target
        df.to_parquet(args.manifest, index=False)
        
    logger.info(f"Photometry processing completed. Success: {success_count}, Failed: {failed_count}.")

if __name__ == "__main__":
    main()
