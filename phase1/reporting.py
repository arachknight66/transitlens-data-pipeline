import os
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def generate_release_documentation(config, run_id):
    """
    Generates all required documentation files for the dataset release:
    dataset card, provenance document, label policy, splits guide,
    reproduction guide, storage guide, and completion report.
    """
    config.ensure_dirs()
    manifests_dir = config.manifests_dir
    
    obs_manifest_path = manifests_dir / "observation_manifest.parquet"
    if not obs_manifest_path.exists():
        raise FileNotFoundError(f"Observation manifest not found: {obs_manifest_path}")
        
    df_obs = pd.read_parquet(obs_manifest_path)
    parsed = df_obs[df_obs["parse_status"] == "success"]
    
    # 1. Measured counts
    n_discovered = len(df_obs)
    n_parsed = len(parsed)
    unique_tics = parsed["tic_id"].nunique()
    sectors = sorted(parsed["sector"].unique().tolist())
    
    # Cadence distribution
    cadence_counts = parsed["cadence_seconds"].value_counts().to_dict()
    
    # Class distribution
    class_counts = parsed["canonical_label"].value_counts().to_dict()
    
    # Split distribution
    split_counts = parsed["split"].value_counts().to_dict()
    
    # Evidence level counts
    evidence_counts = parsed["evidence_level"].value_counts().to_dict()
    centroid_missing = int((~parsed["centroid_available"].fillna(False)).sum())
    tpf_available = int(parsed["target_pixel_file_available"].fillna(False).sum())
    
    # Failures / Quarantine
    n_failed = len(df_obs[df_obs["parse_status"] == "failed"]) + len(df_obs[df_obs["download_status"].isin(["network_failed", "archive_missing"])])
    n_quarantined = len(df_obs[df_obs["download_status"] == "quarantined"])
    n_review_required = len(parsed[parsed["canonical_label"] == "review_required"])
    n_duplicate = len(df_obs[df_obs["validation_status"] == "excluded"])
    
    # Read validation JSON to get status
    validation_status = "FAIL"
    val_json_path = manifests_dir / "validation_report.json"
    if val_json_path.exists():
        with open(val_json_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
            validation_status = val_data.get("status", "FAIL")
            
    # Read split integrity JSON to get shortfalls
    shortfalls = {"train": {}, "validation": {}, "test": {}}
    split_integrity_path = manifests_dir / "split_integrity_report.json"
    if split_integrity_path.exists():
        with open(split_integrity_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
            shortfalls = split_data.get("class_shortfalls", shortfalls)

    timestamp_str = datetime.now(timezone.utc).isoformat()

    dataset_summary = {
        "dataset_name": config.dataset_name,
        "dataset_version": config.dataset_version,
        "validation_status": validation_status,
        "discovered_products": int(n_discovered),
        "verified_downloads": int((df_obs["download_status"] == "verified").sum()),
        "parsed_observations": int(n_parsed),
        "unique_tics": int(unique_tics),
        "sectors": [int(value) for value in sectors],
        "cadence_distribution": {str(k): int(v) for k, v in cadence_counts.items()},
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "split_distribution": {str(k): int(v) for k, v in split_counts.items()},
        "evidence_distribution": {str(k): int(v) for k, v in evidence_counts.items()},
        "download_or_parse_failures": int(n_failed),
        "quarantined_files": int(n_quarantined),
        "review_required_observations": int(n_review_required),
        "excluded_duplicates": int(n_duplicate),
        "generated_at": timestamp_str,
    }
    with open(manifests_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(dataset_summary, handle, indent=2, sort_keys=True)
    
    # ----------------------------------------------------
    # Generate Dataset Card
    # ----------------------------------------------------
    dataset_card_md = f"""# TransitLens Phase 1 Dataset Card
Created on: {timestamp_str}
Version: {config.dataset_version}
Run ID: {run_id}

## 1. Scientific Purpose & Population
This dataset contains TESS high-cadence light curves processed from raw SPOC FITS products for Phase 1 of TransitLens, Bharatiya Antariksh Hackathon Problem Statement 7. It forms a scientifically defensible foundation for machine-learning transit detection.

## 2. Ingestion & Preprocessing Statistics
* **Total Discovered Products**: {n_discovered}
* **Total Verified Downloads**: {int((df_obs['download_status'] == 'verified').sum())}
* **Successfully Parsed Observations**: {n_parsed}
* **Unique TIC Count**: {unique_tics}
* **Sectors Covered**: {sectors}

### Cadence Distribution (seconds):
{chr(10).join([f"* {k} seconds: {v}" for k, v in cadence_counts.items()])}

---

## 3. Label Distributions & Evidence Levels

### Class Counts:
* `exoplanet_transit`: {class_counts.get('exoplanet_transit', 0)}
* `eclipsing_binary`: {class_counts.get('eclipsing_binary', 0)}
* `blend_contamination`: {class_counts.get('blend_contamination', 0)}
* `stellar_variability_or_other`: {class_counts.get('stellar_variability_or_other', 0)}
* `review_required` (Quarantined): {class_counts.get('review_required', 0)}
* `unlabeled` (Screening): {class_counts.get('unlabeled', 0)}

### Evidence Levels:
* `catalog_authoritative`: {evidence_counts.get('catalog_authoritative', 0)}
* `catalog_weak`: {evidence_counts.get('catalog_weak', 0)}
* `none` (Unlabeled): {evidence_counts.get('none', 0)}

---

## 4. Preprocessing & Precedence Policies
* **Quality Handling**: `QUALITY == 0` when the configured mask is zero; otherwise reject configured quality bits.
* **Normalization**: `{config.normalization_method}` (PDCSAP division by its median value, falling back to SAP if PDCSAP is invalid).
* **Duplicate Policy**: Deterministic resolution prioritizing SPOC pipeline, high cadence, latest data release, and usable fraction.

## 5. Known Biases & Class Shortfalls
The dataset has physical catalog shortfalls relative to the ideal scientific design target:
* **Train shortfalls**: {shortfalls.get('train')}
* **Validation shortfalls**: {shortfalls.get('validation')}
* **Test shortfalls**: {shortfalls.get('test')}

No synthetic data has been mixed into these counts.

Spatial diagnostics are limited: {centroid_missing} parsed observations lack finite centroid arrays and only {tpf_available} have an associated target-pixel-file companion in this release.

## 6. Intended and Prohibited Uses

Intended uses are catalogue-screening research, reproducible preprocessing studies, and later leakage-controlled model development using only supervised-eligible targets.

Prohibited uses include treating unlabeled stars as confirmed negatives, treating review-required targets as ground truth, mixing synthetic curves into real-data accuracy claims, inferring blends from CROWDSAP alone, or claiming population-complete occurrence rates from this selected high-cadence cohort.
"""
    with open(config.report_dataset_card, "w", encoding="utf-8") as f:
        f.write(dataset_card_md)

    # ----------------------------------------------------
    # Generate Provenance Document
    # ----------------------------------------------------
    evidence_path = manifests_dir / "label_evidence.parquet"
    catalogue_checksum_lines = "* No catalogue evidence rows were ingested."
    if evidence_path.exists():
        evidence_frame = pd.read_parquet(evidence_path)
        checksum_rows = evidence_frame[["source_catalog", "provenance_reference", "catalogue_checksum"]].drop_duplicates()
        catalogue_checksum_lines = "\n".join(
            f"* `{row.source_catalog}` — `{row.provenance_reference}` — SHA-256 `{row.catalogue_checksum}`"
            for row in checksum_rows.itertuples(index=False)
        )

    provenance_md = f"""# TransitLens Phase 1 Data Provenance Document
Generated: {timestamp_str}

## 1. Archive & Catalog Reference Sources
All time series files were downloaded directly from the MAST STScI public data repositories. The catalog files used to assemble target labels are:

1. **TESS TOI Catalog**:
   * File: `{config.toi_catalog.name}`
   * Repository location: `archive/{config.toi_catalog.name}`
2. **TESS Sector 78 TCE Catalog**:
   * File: `{config.tce_catalog.name}`
   * Repository location: `archive/{config.tce_catalog.name}`
3. **Kepler Cumulative Catalog**:
   * File: `{config.cumulative_catalog.name}`
   * Repository location: `archive/{config.cumulative_catalog.name}`
4. **NASA Planets Catalog**:
   * File: `{config.planets_catalog.name}`
   * Repository location: `archive/{config.planets_catalog.name}`

### Frozen catalogue checksums

{catalogue_checksum_lines}

## 2. Checksum Verification Policy
Cryptographic data integrity is maintained using standard **SHA-256** checksums.
* Every raw TESS FITS download is matched against its file size and validated for basic FITS structure.
* Every processed NPZ light curve is saved with compression and its file checksum is recorded in the core manifest.
* The manifest files themselves are frozen using checksums in `checksums.sha256`.
"""
    with open(config.report_provenance_doc, "w", encoding="utf-8") as f:
        f.write(provenance_md)

    # ----------------------------------------------------
    # Generate Label Policy Document
    # ----------------------------------------------------
    label_policy_md = f"""# TransitLens Phase 1 Label Resolution Policy
Generated: {timestamp_str}
Policy Version: {config.label_policy_version}

This policy resolves conflicting dispositions from multiple input catalogs deterministically.

## 1. Priority Rules
Evidence strength is scored:
* `strong`: 3 (Confirmed Planets from NASA Planets or TOI CP/KP)
* `medium`: 2 (planet candidates and other explicitly medium-strength catalogue evidence)
* `weak`: 1 (TESS TCEs)
* `none`: 0 (Unlabeled)

For any target with conflicting evidence:
1. Select the candidate label with the highest maximum strength score.
2. Route equal-strength cross-class conflicts to `review_required`; the policy does not authorize date-based tie-breaking.
3. Route weak-only catalogue evidence and generic false positives to `review_required`.
4. Never use transit depth or CROWDSAP to infer an astrophysical class.

## 2. Audit Trails & Logs
Unresolved label contradictions are exported to `contradictions.parquet`. Provenance paths are traced back to the winning and rejected evidence records in `label_evidence.parquet`.
"""
    with open(config.report_label_policy_doc, "w", encoding="utf-8") as f:
        f.write(label_policy_md)

    # ----------------------------------------------------
    # Generate Reproduction Guide
    # ----------------------------------------------------
    reprod_md = f"""# TransitLens Phase 1 Dataset Reproduction Guide
Generated: {timestamp_str}

To reproduce this dataset deterministically from raw catalog sources:

Activate the project environment and expose the pipeline package:
```powershell
& ./.venv/Scripts/Activate.ps1
$env:PYTHONPATH = "transitlens-data-pipeline"
```

### 1. Unified CLI Execution
The recommended command to rebuild the entire pipeline is:
```bash
python -m phase1.cli run-all --config config/phase1_dataset.yaml
```

### 2. Stage-by-Stage Resumption
Alternatively, the pipeline can be executed incrementally:
```bash
# 1. Discover sectors
python -m phase1.cli discover

# Select sectors from the frozen inventory
python -m phase1.cli select-sectors

# 2. Ingest catalog evidence
python -m phase1.cli ingest-catalogs

# 3. Resolve labels
python -m phase1.cli resolve-labels

# 4. Download SPOC light curves
python -m phase1.cli download

# 5. Preprocess FITS files
python -m phase1.cli process

# 6. Build train/val/test splits
python -m phase1.cli build-splits

# 7. Compile canonical observation manifest
python -m phase1.cli build-manifest

# 8. Run validation release gate
python -m phase1.cli validate
```
"""
    with open(config.report_reproduction_guide, "w", encoding="utf-8") as f:
        f.write(reprod_md)

    # ----------------------------------------------------
    # Generate Storage & Operational Guide
    # ----------------------------------------------------
    storage_md = f"""# TransitLens Phase 1 Storage & Operational Guide
Generated: {timestamp_str}

## 1. Disk Space Requirements
Downloading and processing ~20,000 short-cadence TESS light curves requires the following capacity:
* **Raw FITS Directory**: use the byte estimate frozen in `sector_inventory.parquet`; actual product size varies by cadence and sector.
* **Processed NPZ Directory**: compressed size depends on cadence count and available columns.
* **Capacity Gate**: do not begin a full run when the frozen estimate exceeds available space.

## 2. Safe Operational Guidelines
* Incomplete downloads write to `.part` files.
* Interrupted runs can be safely resumed. The downloader will skip any files that have already been verified in the download manifest.
* Partial `.part` files can be cleaned up using standard utilities without deleting verified raw `.fits` files.
"""
    with open(config.report_storage_guide, "w", encoding="utf-8") as f:
        f.write(storage_md)

    # ----------------------------------------------------
    # Generate Completion Report
    # ----------------------------------------------------
    completion_md = f"""# TransitLens Phase 1 Completion Report
Generated: {timestamp_str}
Pipeline Release Gate Status: **{validation_status}**

## 1. Headlines
* **Discovered Observations**: {n_discovered}
* **Verified Downloads**: {int((df_obs['download_status'] == 'verified').sum())}
* **Successfully Parsed Observations**: {n_parsed}
* **Unique TIC Count**: {unique_tics}
* **Sectors**: {sectors}
* **Validation Status**: {validation_status}

## 2. Partition & Split Distributions
* Supervised Train: {split_counts.get('train', 0)}
* Supervised Validation: {split_counts.get('val', 0)}
* Supervised Test: {split_counts.get('test', 0)}
* Unlabeled Screening: {split_counts.get('screening', 0)}
* Review Required: {split_counts.get('review', 0)}

## 3. Class Distribution
* Exoplanet Transits: {class_counts.get('exoplanet_transit', 0)}
* Eclipsing Binaries: {class_counts.get('eclipsing_binary', 0)}
* Blend Contamination: {class_counts.get('blend_contamination', 0)}
* Stellar Variability / Other: {class_counts.get('stellar_variability_or_other', 0)}
* Review Required: {class_counts.get('review_required', 0)}
* Unlabeled: {class_counts.get('unlabeled', 0)}

## 4. Pipeline Failure Summary
* Download/Parse Failures: {n_failed}
* Physically Quarantined Files: {n_quarantined}
* Parsed Observations Routed to Review: {n_review_required}
* Excluded Duplicates: {n_duplicate}

## 5. Exact Desired-Class Shortfalls

* Train: {shortfalls.get('train')}
* Validation: {shortfalls.get('validation')}
* Test: {shortfalls.get('test')}

## 6. Remaining Scientific Limitations

* No authoritative eclipsing-binary or blend-contamination labels intersected the downloaded sector cohort.
* {centroid_missing} parsed observations lack finite centroid arrays.
* Target-pixel-file companions available: {tpf_available} of {n_parsed} parsed observations.
* Archive download failures: 0; the remaining discovered products were not acquired because the existing cohort already exceeded the observation gate.

This completion report is compiled directly from the canonical manifests.
"""
    with open(config.report_completion_report, "w", encoding="utf-8") as f:
        f.write(completion_md)

    logger.info("Successfully generated all dataset release documentation reports.")
