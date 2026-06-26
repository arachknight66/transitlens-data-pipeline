import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import logging
from real_tess.sector_manifest import update_manifest_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Add paths to sys.path to resolve imports from ml-core and pipeline
pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pipeline_dir)

ml_core_dir = os.path.join(os.path.dirname(pipeline_dir), "transitlens-ml-core")
sys.path.append(ml_core_dir)

def screen_sector_targets(manifest_path, output_dir):
    """
    Screens processed sector targets using ml-core analyze_light_curve.
    Writes results, rankings, failure logs, and screening reports.
    """
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest missing for screening: {manifest_path}")
        return
        
    df = pd.read_csv(manifest_path)
    processed_targets = df[df["status"] == "processed"]
    
    if len(processed_targets) == 0:
        logger.warning("No processed targets found in manifest to screen.")
        # Write empty or skeleton tables so that outputs still exist
        write_empty_outputs(output_dir)
        write_processing_summary(df, output_dir)
        write_screening_report(df, [], output_dir)
        return
        
    try:
        from pipeline import analyze_light_curve
    except ImportError as e:
        logger.error(f"Failed to import ML core pipeline: {e}")
        return
        
    logger.info(f"Screening {len(processed_targets)} targets...")
    
    results = []
    failed_rows = []
    
    for idx, row in processed_targets.iterrows():
        target_id = row["target_id"]
        tic_id = row["tic_id"]
        sector = int(row["sector"])
        lc_path_rel = row["lightcurve_path"]
        
        npz_path = os.path.join(output_dir, "lightcurves", lc_path_rel)
        if not os.path.exists(npz_path):
            logger.warning(f"Processed lightcurve file missing for {target_id}: {npz_path}")
            update_manifest_status(manifest_path, target_id, "failed", f"Processed NPZ missing: {npz_path}")
            failed_rows.append({
                "target_id": target_id,
                "phase": "pipeline",
                "failure_reason": f"Processed NPZ missing: {npz_path}",
                "created_at": datetime.now(timezone.utc).isoformat()
            })
            continue
            
        logger.info(f"Analyzing lightcurve for {target_id}...")
        try:
            # Load arrays
            npz = np.load(npz_path)
            time = npz["time"]
            flux = npz["flux"]
            
            # Construct metadata
            metadata = {
                "target_id": str(target_id),
                "sector": sector,
                "ra": row["ra"] if pd.notna(row["ra"]) else None,
                "dec": row["dec"] if pd.notna(row["dec"]) else None
            }
            
            # Call ML core pipeline
            res = analyze_light_curve(time=time, flux=flux, metadata=metadata)
            
            candidate_detected = res.get("candidate_detected", False)
            predicted_class = res.get("predicted_class", "stellar_variability_or_other")
            confidence = res.get("confidence", 0.0)
            
            results.append({
                "target_id": target_id,
                "tic_id": tic_id,
                "sector": sector,
                "candidate_detected": candidate_detected,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "period_days": res.get("period_days"),
                "period_uncertainty_days": res.get("period_uncertainty_days"),
                "duration_days": res.get("duration_days"),
                "duration_uncertainty_days": res.get("duration_uncertainty_days"),
                "depth": res.get("depth"),
                "depth_uncertainty": res.get("depth_uncertainty"),
                "epoch_btjd": res.get("epoch_btjd"),
                "snr": res.get("snr", 0.0),
                "bootstrap_fap": res.get("bootstrap_fap"),
                "fit_quality": res.get("fit_quality"),
                "transit_count": res.get("transit_count", 0),
                "processing_time_ms": res.get("processing_time_ms", 0.0),
                "failure_reason": "",
                "evidence_level": "real_tess",
                "lightcurve_path": lc_path_rel
            })
            
            # Update manifest status to screened
            update_manifest_status(manifest_path, target_id, "screened")
            
        except Exception as e:
            logger.warning(f"Pipeline screening failed for {target_id}: {e}")
            update_manifest_status(manifest_path, target_id, "failed", f"Pipeline failed: {e}")
            failed_rows.append({
                "target_id": target_id,
                "phase": "pipeline",
                "failure_reason": f"Pipeline failure: {e}",
                "created_at": datetime.now(timezone.utc).isoformat()
            })
            
    # Save screening results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "screening_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Wrote screening results to {results_path}")
    
    # Reload manifest to capture final statuses
    df = pd.read_csv(manifest_path)
    
    # Save failed targets
    if len(failed_rows) > 0:
        failed_targets_df = pd.DataFrame(failed_rows)
    else:
        failed_targets_df = pd.DataFrame(columns=["target_id", "phase", "failure_reason", "created_at"])
        
    # Combine with manifest failures
    manifest_failures = df[df["status"] == "failed"]
    for _, row in manifest_failures.iterrows():
        phase = "download" if "download" in str(row["failure_reason"]).lower() or "lightkurve" in str(row["failure_reason"]).lower() else "parse"
        if row["target_id"] not in failed_targets_df["target_id"].values:
            failed_targets_df = pd.concat([failed_targets_df, pd.DataFrame([{
                "target_id": row["target_id"],
                "phase": phase,
                "failure_reason": row["failure_reason"],
                "created_at": row["created_at"]
            }])], ignore_index=True)
            
    failed_path = os.path.join(output_dir, "failed_targets.csv")
    failed_targets_df.to_csv(failed_path, index=False)
    logger.info(f"Wrote failed targets log to {failed_path}")
    
    # Rank and save Top Candidates
    if len(results_df) > 0:
        candidates = results_df[results_df["candidate_detected"] == True]
        # Rank by confidence descending, snr descending, bootstrap_fap ascending
        candidates_sorted = candidates.sort_values(
            by=["confidence", "snr", "bootstrap_fap"], 
            ascending=[False, False, True]
        )
        
        top_candidates = []
        for idx, (_, r) in enumerate(candidates_sorted.iterrows()):
            top_candidates.append({
                "rank": idx + 1,
                "target_id": r["target_id"],
                "predicted_class": r["predicted_class"],
                "confidence": r["confidence"],
                "snr": r["snr"],
                "period_days": r["period_days"],
                "depth": r["depth"],
                "duration_days": r["duration_days"],
                "bootstrap_fap": r["bootstrap_fap"],
                "notes": f"Detected with SNR={r['snr']:.2f}, F1-confidence={r['confidence'] * 100:.1f}%"
            })
            
        if len(top_candidates) > 0:
            top_df = pd.DataFrame(top_candidates)
        else:
            top_df = pd.DataFrame(columns=["rank", "target_id", "predicted_class", "confidence", "snr", "period_days", "depth", "duration_days", "bootstrap_fap", "notes"])
        top_path = os.path.join(output_dir, "top_candidates.csv")
        top_df.to_csv(top_path, index=False)
        logger.info(f"Wrote top candidates list to {top_path}")
    else:
        # Write empty skeleton top candidates
        pd.DataFrame(columns=["rank", "target_id", "predicted_class", "confidence", "snr", "period_days", "depth", "duration_days", "bootstrap_fap", "notes"]).to_csv(os.path.join(output_dir, "top_candidates.csv"), index=False)
        
    # Write processing summary
    write_processing_summary(df, output_dir)
    
    # Generate sector screening report
    write_screening_report(df, results, output_dir)

def write_empty_outputs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(columns=["target_id", "tic_id", "sector", "candidate_detected", "predicted_class", "confidence"]).to_csv(os.path.join(output_dir, "screening_results.csv"), index=False)
    pd.DataFrame(columns=["rank", "target_id", "predicted_class", "confidence", "snr", "period_days", "depth", "duration_days", "bootstrap_fap", "notes"]).to_csv(os.path.join(output_dir, "top_candidates.csv"), index=False)
    pd.DataFrame(columns=["target_id", "phase", "failure_reason", "created_at"]).to_csv(os.path.join(output_dir, "failed_targets.csv"), index=False)

def write_processing_summary(df, output_dir):
    # Calculate stats
    sector = int(df["sector"].iloc[0]) if len(df) > 0 else 0
    n_targets_requested = len(df)
    n_targets_with_products = len(df[~df["local_fits_path"].isna() & (df["local_fits_path"] != "")])
    
    n_downloaded = len(df[df["status"] == "downloaded"])
    n_cached = len(df[df["status"] == "cached"])
    
    n_processed = len(df[df["status"].isin(["processed", "screened"])])
    n_screened = len(df[df["status"] == "screened"])
    
    n_failed_download = len(df[(df["status"] == "failed") & df["failure_reason"].astype(str).str.contains("download|lightkurve|timeout", case=False, na=False)])
    n_failed_parse = len(df[(df["status"] == "failed") & df["failure_reason"].astype(str).str.contains("parse|fits|corrupt", case=False, na=False)])
    n_failed_pipeline = len(df[(df["status"] == "failed") & df["failure_reason"].astype(str).str.contains("pipeline|analysis|classifier", case=False, na=False)])
    
    # Calculate NPZ time/flux array statistics
    points = []
    spans = []
    lightcurves_dir = os.path.join(output_dir, "lightcurves")
    
    if os.path.exists(lightcurves_dir):
        for fname in os.listdir(lightcurves_dir):
            if fname.endswith(".npz"):
                try:
                    npz = np.load(os.path.join(lightcurves_dir, fname))
                    points.append(len(npz["time"]))
                    spans.append(float(npz["time"][-1] - npz["time"][0]))
                except Exception:
                    pass
                    
    mean_points = float(np.mean(points)) if points else 0.0
    median_points = float(np.median(points)) if points else 0.0
    mean_span = float(np.mean(spans)) if spans else 0.0
    median_span = float(np.median(spans)) if spans else 0.0
    
    summary = [{
        "sector": sector,
        "n_targets_requested": n_targets_requested,
        "n_targets_with_products": n_targets_with_products,
        "n_downloaded": n_downloaded,
        "n_cached": n_cached,
        "n_parsed": n_processed, # successfully parsed and processed
        "n_processed": n_processed,
        "n_screened": n_screened,
        "n_failed_download": n_failed_download,
        "n_failed_parse": n_failed_parse,
        "n_failed_pipeline": n_failed_pipeline,
        "mean_points_per_lightcurve": mean_points,
        "median_points_per_lightcurve": median_points,
        "mean_time_span_days": mean_span,
        "median_time_span_days": median_span,
        "created_at": datetime.now(timezone.utc).isoformat()
    }]
    
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, "sector_processing_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Wrote processing summary to {summary_path}")

def write_screening_report(df, results, output_dir):
    sector = int(df["sector"].iloc[0]) if len(df) > 0 else 0
    n_targets_requested = len(df)
    n_targets_with_products = len(df[~df["local_fits_path"].isna() & (df["local_fits_path"] != "")])
    n_processed = len(df[df["status"].isin(["processed", "screened"])])
    n_screened = len(df[df["status"] == "screened"])
    
    n_failed_download = len(df[(df["status"] == "failed") & df["failure_reason"].astype(str).str.contains("download|lightkurve|timeout", case=False, na=False)])
    n_failed_parse = len(df[(df["status"] == "failed") & df["failure_reason"].astype(str).str.contains("parse|fits|corrupt", case=False, na=False)])
    n_failed_pipeline = len(df[(df["status"] == "failed") & df["failure_reason"].astype(str).str.contains("pipeline|analysis|classifier", case=False, na=False)])
    
    # Class distribution
    class_dist = {"exoplanet_transit": 0, "eclipsing_binary": 0, "blend_contamination": 0, "stellar_variability_or_other": 0}
    candidates = []
    
    for r in results:
        if r["candidate_detected"]:
            candidates.append(r)
        class_dist[r["predicted_class"]] = class_dist.get(r["predicted_class"], 0) + 1
        
    # Evidence level claim determination
    evidence_level_claim = "Level 3 - Cached Known-Object Demo / Framework Ready"
    if n_screened >= 100:
        evidence_level_claim = "Level 4 - Astronomical Sector-Scale Screening"
        
    # Read top candidates for report
    top_candidates = []
    top_path = os.path.join(output_dir, "top_candidates.csv")
    if os.path.exists(top_path):
        top_candidates_df = pd.read_csv(top_path)
        top_candidates = top_candidates_df.to_dict(orient="records")
        
    report_md = f"""# TESS Sector Screening Report

Generated on: {datetime.now(timezone.utc).isoformat()}
Sector Selected: Sector {sector}
Evidence Level Claim: **{evidence_level_claim}**

---

## 1. Summary Statistics

| Metric | Target Count |
| :--- | :--- |
| **Total Requested Targets** | {n_targets_requested} |
| **Targets with Data Products** | {n_targets_with_products} |
| **Successfully Processed (Parsed)** | {n_processed} |
| **Screened by ML Core** | {n_screened} |

### Failure Breakdown
* **Failed Downloads**: {n_failed_download}
* **Failed Parses**: {n_failed_parse}
* **Failed Pipeline Screening**: {n_failed_pipeline}

---

## 2. Detected Candidate Class Distribution

Among the processed and screened observations, the class labels are distributed as follows:

| Class | Count |
| :--- | :--- |
| `exoplanet_transit` | {class_dist.get("exoplanet_transit", 0)} |
| `eclipsing_binary` | {class_dist.get("eclipsing_binary", 0)} |
| `blend_contamination` | {class_dist.get("blend_contamination", 0)} |
| `stellar_variability_or_other` | {class_dist.get("stellar_variability_or_other", 0)} |

---

## 3. Top Candidates Ranked List

Candidates are ranked by confidence, SNR, and FAP.

{"| Rank | Target ID | Class | Confidence | SNR | Period (days) | Depth | Duration (days) | Notes |" if top_candidates else "No candidates detected in this run."}
{"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |" if top_candidates else ""}
{chr(10).join(f"| {c['rank']} | `{c['target_id']}` | `{c['predicted_class']}` | {c['confidence'] * 100:.1f}% | {c['snr']:.2f} | {c['period_days'] if pd.notna(c['period_days']) else 'N/A'} | {c['depth'] if pd.notna(c['depth']) else 'N/A'} | {c['duration_days'] if pd.notna(c['duration_days']) else 'N/A'} | {c['notes']} |" for c in top_candidates[:20])}

---

## 4. Evaluator's Notes & Limitations

> [!WARNING]
> This run qualifies as **{evidence_level_claim}**. 
> {"The total number of screened targets is under the 100-target threshold for full Sector-Scale evidence. This should be treated as a verification pipeline run." if n_screened < 100 else "The pipeline successfully screened over 100 targets, representing sector-scale astronomical verification."}
> 
> **Key Limitations**:
> 1. SPOC timeseries products represent a pre-selected subset of high-interest targets.
> 2. Noise levels and data downlink gaps (mid-sector gaps) affect period recovery.
> 3. Visual checks on top candidates are recommended to rule out background eclipsing binaries or stellar active regions.
"""
    report_path = os.path.join(output_dir, "sector_screening_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Wrote screening markdown report to {report_path}")

if __name__ == "__main__":
    screen_sector_targets("sector_manifest.csv", "./output")
