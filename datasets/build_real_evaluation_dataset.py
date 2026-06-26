import os
import sys
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timezone

# Add parent directory to sys.path to allow imports from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_tess.flux_normaliser import normalise_pdcsap
from astropy.io import fits

# Constants
CANONICAL_CLASSES = {
    "exoplanet_transit",
    "eclipsing_binary",
    "blend_contamination",
    "stellar_variability_or_other"
}

ALIAS_MAP = {
    "exoplanet_like": "exoplanet_transit",
    "eclipsing_binary_like": "eclipsing_binary",
    "noise_or_other": "stellar_variability_or_other"
}

def normalize_class_label(label):
    normalized = ALIAS_MAP.get(label, label)
    if normalized not in CANONICAL_CLASSES:
        # Default or fallback
        return "stellar_variability_or_other"
    return normalized

def normalize_tic_id(raw_id):
    raw_str = str(raw_id).strip()
    digits = "".join(c for c in raw_str if c.isdigit())
    if digits:
        return f"TIC-{digits}"
    return raw_str

def get_toi_metadata(tic_id):
    import csv
    tic_num = int("".join(c for c in tic_id if c.isdigit()))
    
    # NASA Exoplanet Archive TOI CSV paths
    paths = [
        r"C:\Users\arach\Documents\Projects\Transitlens\archive\TOI_2026.06.25_21.21.19.csv",
        r"C:\Users\arach\Documents\Projects\Transitlens\archive\TOI_2025.02.03_06.18.31.csv"
    ]
    
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = None
            for row in reader:
                if not row:
                    continue
                if row[0].startswith("#"):
                    continue
                header = row
                break
            
            if not header:
                continue
            
            try:
                tid_idx = header.index("tid")
                per_idx = header.index("pl_orbper")
                dep_idx = header.index("pl_trandep")
                dur_idx = header.index("pl_trandurh")
                mid_idx = header.index("pl_tranmid")
            except ValueError:
                header_upper = [h.upper() for h in header]
                try:
                    tid_idx = header_upper.index("TID")
                    per_idx = header_upper.index("PL_ORBPER")
                    dep_idx = header_upper.index("PL_TRANDEP")
                    dur_idx = header_upper.index("PL_TRANDURH")
                    mid_idx = header_upper.index("PL_TRANMID")
                except ValueError:
                    continue
            
            for row in reader:
                if not row or len(row) <= max(tid_idx, per_idx, dep_idx, dur_idx, mid_idx):
                    continue
                if row[0].startswith("#"):
                    continue
                try:
                    row_tid = int("".join(c for c in row[tid_idx] if c.isdigit()))
                    if row_tid == tic_num:
                        period = float(row[per_idx]) if row[per_idx] else None
                        depth_ppm = float(row[dep_idx]) if row[dep_idx] else None
                        duration_h = float(row[dur_idx]) if row[dur_idx] else None
                        epoch_bjd = float(row[mid_idx]) if row[mid_idx] else None
                        
                        depth = depth_ppm / 1e6 if depth_ppm is not None else None
                        duration_days = duration_h / 24.0 if duration_h is not None else None
                        epoch_btjd = epoch_bjd - 2457000.0 if epoch_bjd is not None else None
                        
                        return {
                            "true_period_days": period,
                            "true_depth": depth,
                            "true_duration_days": duration_days,
                            "true_epoch_btjd": epoch_btjd,
                            "ground_truth_source": "TESS_TOI"
                        }
                except (ValueError, IndexError):
                    continue
    return None

def extract_tess_fits(path):
    hdul = fits.open(path, memmap=False)
    try:
        hdu = hdul[1]
        colnames = [c.name.upper() for c in hdu.columns]
        data = hdu.data
        
        time = np.array(data["TIME"], dtype=np.float64)
        
        flux_col = None
        for col in ["PDCSAP_FLUX", "KSPSAP_FLUX", "FLUX", "SAP_FLUX"]:
            if col in colnames:
                flux_col = col
                break
        if not flux_col:
            raise ValueError(f"No flux column found in FITS: {path}")
            
        flux_raw = np.array(data[flux_col], dtype=np.float64)
        
        quality = None
        for col in ["QUALITY", "SAP_QUALITY", "FLAGS"]:
            if col in colnames:
                quality = np.array(data[col], dtype=np.int64)
                break
                
        flux_norm = normalise_pdcsap(flux_raw, quality_flags=quality)
        
        # Filter NaNs and non-finite values
        valid = np.isfinite(time) & np.isfinite(flux_norm)
        time_clean = time[valid]
        flux_clean = flux_norm[valid]
        
        if len(time_clean) < 100:
            raise ValueError(f"FITS file has too few valid data points ({len(time_clean)}). Min required is 100.")
            
        # Sort by time
        sort_idx = np.argsort(time_clean)
        time_sorted = time_clean[sort_idx]
        flux_sorted = flux_clean[sort_idx]
        
        # Deduplicate timestamps
        _, unique_idx = np.unique(time_sorted, return_index=True)
        time_unique = time_sorted[unique_idx]
        flux_unique = flux_sorted[unique_idx]
        
        result = {
            "time": time_unique,
            "flux": flux_unique
        }
        
        # Re-index optional columns to match valid, sorted, deduplicated time arrays
        valid_indices = np.where(valid)[0]
        sorted_indices = valid_indices[sort_idx]
        final_indices = sorted_indices[unique_idx]
        
        # Optional fields
        flux_err_col = flux_col + "_ERR" if (flux_col + "_ERR") in colnames else None
        if flux_err_col:
            flux_err_raw = np.array(data[flux_err_col], dtype=np.float64)
            median_flux = np.nanmedian(flux_raw)
            if median_flux > 0:
                result["flux_err"] = (flux_err_raw / median_flux)[final_indices]
                
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
            
        # Parse sector & camera/ccd from headers
        primary_header = hdul[0].header
        header = hdu.header
        target_id = primary_header.get("OBJECT") or primary_header.get("TICID") or header.get("OBJECT") or header.get("TICID")
        sector = primary_header.get("SECTOR") or header.get("SECTOR")
        camera = primary_header.get("CAMERA") or header.get("CAMERA")
        ccd = primary_header.get("CCD") or header.get("CCD")
        
        result["metadata"] = {
            "target_id": str(target_id) if target_id else "unknown",
            "sector": int(sector) if sector else None,
            "mission": "TESS",
            "camera": camera,
            "ccd": ccd
        }
        
        return result
    finally:
        hdul.close()

def main():
    print("=" * 70)
    print("  TransitLens — Phase 1 Dataset Builder")
    print("=" * 70)
    
    datasets_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(datasets_dir)
    
    processed_dir = os.path.join(datasets_dir, "processed", "lightcurves")
    splits_dir = os.path.join(processed_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    # 1. Discover available light curves
    synthetic_cases_dir = os.path.join(repo_root, "synthetic", "cases")
    config_yaml_path = os.path.join(repo_root, "synthetic", "config.yaml")
    
    # Read synthetic config
    with open(config_yaml_path, "r") as f:
        synthetic_config = yaml.safe_load(f)
    
    targets = []
    
    # A. Process synthetic cases
    print("\nProcessing synthetic cases...")
    for case_name, case_params in synthetic_config["cases"].items():
        csv_path = os.path.join(synthetic_cases_dir, f"{case_name}.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] Synthetic {case_name}: CSV not found at {csv_path}")
            continue
            
        print(f"  [OK] Processing synthetic case: {case_name}")
        df = pd.read_csv(csv_path)
        
        time = df["time"].to_numpy(dtype=np.float64)
        flux = df["flux"].to_numpy(dtype=np.float64)
        
        # Filter and sort just in case
        valid = np.isfinite(time) & np.isfinite(flux)
        time = time[valid]
        flux = flux[valid]
        sort_idx = np.argsort(time)
        time = time[sort_idx]
        flux = flux[sort_idx]
        _, unique_idx = np.unique(time, return_index=True)
        time = time[unique_idx]
        flux = flux[unique_idx]
        
        npz_filename = f"{case_name}.npz"
        npz_path = os.path.join(processed_dir, npz_filename)
        np.savez_compressed(npz_path, time=time, flux=flux)
        
        # Calculate stats
        n_points = len(time)
        time_span = float(time[-1] - time[0]) if n_points > 1 else 0.0
        cadence = float(np.median(np.diff(time)) * 1440.0) if n_points > 1 else 0.0
        
        raw_label = case_params.get("label")
        class_label = normalize_class_label(raw_label)
        
        targets.append({
            "target_id": case_name,
            "source": "synthetic",
            "evidence_level": "synthetic",
            "class_label": class_label,
            "lightcurve_path": npz_filename,
            "n_points": n_points,
            "time_span_days": time_span,
            "cadence_min_median": cadence,
            "true_period_days": case_params.get("period_days"),
            "true_depth": case_params.get("depth"),
            "true_duration_days": case_params.get("duration_days"),
            "true_epoch_btjd": None,
            "ground_truth_source": "synthetic_config",
            "sector": None,
            "mission": "synthetic",
            "has_flux_err": False,
            "has_centroid": False,
            "has_quality_flags": False,
            "contamination_available": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": case_params.get("description", "")
        })
        
    # B. Process cached TESS FITS files
    tess_cache_dir = os.path.join(repo_root, "real_tess", "cache")
    print("\nProcessing cached TESS FITS files...")
    if os.path.exists(tess_cache_dir):
        for filename in os.listdir(tess_cache_dir):
            if filename.endswith(".fits"):
                fits_path = os.path.join(tess_cache_dir, filename)
                print(f"  [OK] Processing TESS FITS: {filename}")
                try:
                    data = extract_tess_fits(fits_path)
                    meta = data["metadata"]
                    raw_id = meta["target_id"]
                    normalized_id = normalize_tic_id(raw_id)
                    
                    npz_filename = f"{normalized_id}.npz"
                    npz_path = os.path.join(processed_dir, npz_filename)
                    
                    # Save arrays to NPZ
                    save_args = {"time": data["time"], "flux": data["flux"]}
                    if "flux_err" in data:
                        save_args["flux_err"] = data["flux_err"]
                    if "centroid_x" in data:
                        save_args["centroid_x"] = data["centroid_x"]
                    if "centroid_y" in data:
                        save_args["centroid_y"] = data["centroid_y"]
                    if "quality" in data:
                        save_args["quality"] = data["quality"]
                        
                    np.savez_compressed(npz_path, **save_args)
                    
                    # Query TOI Ground Truth
                    toi_info = get_toi_metadata(normalized_id)
                    if toi_info is None:
                        # Fallback default values
                        toi_info = {
                            "true_period_days": None,
                            "true_depth": None,
                            "true_duration_days": None,
                            "true_epoch_btjd": None,
                            "ground_truth_source": "unknown"
                        }
                        
                    n_points = len(data["time"])
                    time_span = float(data["time"][-1] - data["time"][0]) if n_points > 1 else 0.0
                    cadence = float(np.median(np.diff(data["time"])) * 1440.0) if n_points > 1 else 0.0
                    
                    targets.append({
                        "target_id": normalized_id,
                        "source": "real_tess",
                        "evidence_level": "real_tess",
                        "class_label": "exoplanet_transit", # Verified from historical splits/labels
                        "lightcurve_path": npz_filename,
                        "n_points": n_points,
                        "time_span_days": time_span,
                        "cadence_min_median": cadence,
                        "true_period_days": toi_info["true_period_days"],
                        "true_depth": toi_info["true_depth"],
                        "true_duration_days": toi_info["true_duration_days"],
                        "true_epoch_btjd": toi_info["true_epoch_btjd"],
                        "ground_truth_source": toi_info["ground_truth_source"],
                        "sector": meta["sector"],
                        "mission": "TESS",
                        "has_flux_err": "flux_err" in save_args,
                        "has_centroid": "centroid_x" in save_args and "centroid_y" in save_args,
                        "has_quality_flags": "quality" in save_args,
                        "contamination_available": False,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "notes": f"Extracted from {filename}. Camera {meta.get('camera')}, CCD {meta.get('ccd')}"
                    })
                except Exception as e:
                    print(f"  [ERROR] Failed to process {filename}: {e}")
                    
    if not targets:
        print("\n[FAIL] No evaluable targets found!")
        sys.exit(1)
        
    # 2. Write Central Manifest
    manifest_df = pd.DataFrame(targets)
    manifest_path = os.path.join(processed_dir, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nWrote central manifest to {manifest_path}")
    
    # 3. Create splits
    # Respect historic target-split assignments to guarantee target disjointness:
    # - Train: candidate_c, TIC-237913194, TIC-25155310, TIC-307210830
    # - Val: candidate_b, TIC-261136679
    # - Test: candidate_a
    train_ids = {"candidate_c", "TIC-237913194", "TIC-25155310", "TIC-307210830"}
    val_ids = {"candidate_b", "TIC-261136679"}
    test_ids = {"candidate_a"}
    
    train_df = manifest_df[manifest_df["target_id"].isin(train_ids)]
    val_df = manifest_df[manifest_df["target_id"].isin(val_ids)]
    test_df = manifest_df[manifest_df["target_id"].isin(test_ids)]
    
    split_manifest_cols = [
        "target_id", "class_label", "source", "evidence_level", "lightcurve_path",
        "true_period_days", "true_depth", "true_duration_days"
    ]
    
    train_split_path = os.path.join(splits_dir, "train_manifest.csv")
    val_split_path = os.path.join(splits_dir, "val_manifest.csv")
    test_split_path = os.path.join(splits_dir, "test_manifest.csv")
    
    train_df[split_manifest_cols].to_csv(train_split_path, index=False)
    val_df[split_manifest_cols].to_csv(val_split_path, index=False)
    test_df[split_manifest_cols].to_csv(test_split_path, index=False)
    
    print(f"Wrote train split manifest to {train_split_path} ({len(train_df)} targets)")
    print(f"Wrote val split manifest to {val_split_path} ({len(val_df)} targets)")
    print(f"Wrote test split manifest to {test_split_path} ({len(test_df)} targets)")
    
    print("\nDataset building complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
