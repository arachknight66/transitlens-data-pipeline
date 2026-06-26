import os
import yaml
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def load_label_mapping(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def normalize_target_id(raw_id, prefix="KIC"):
    """Normalizes IDs to standard TransitLens formats like KIC-123456 or TIC-123456."""
    if pd.isnull(raw_id):
        return None
    cleaned = str(raw_id).strip().upper().replace("TIC", "").replace("KIC", "").replace("-", "")
    try:
        # standardise integer representation
        cleaned = str(int(float(cleaned)))
    except ValueError:
        pass
    return f"{prefix}-{cleaned}"

def process_archive():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    archive_dir = os.path.join(repo_root, "archive")
    datasets_dir = os.path.join(repo_root, "transitlens-data-pipeline", "datasets")
    splits_dir = os.path.join(datasets_dir, "splits")
    
    os.makedirs(splits_dir, exist_ok=True)
    
    mapping_path = os.path.join(datasets_dir, "label_mapping.yaml")
    mapping = load_label_mapping(mapping_path)
    
    records = []
    rejected_count = 0
    
    # 1. Kepler Cumulative Loading
    kep_path = os.path.join(archive_dir, "cumulative.csv")
    if os.path.exists(kep_path):
        logger.info("Loading Kepler catalogue from %s", kep_path)
        df_kep = pd.read_csv(kep_path)
        for idx, row in df_kep.iterrows():
            kepid = row.get("kepid")
            if pd.isnull(kepid):
                rejected_count += 1
                continue
            tid = normalize_target_id(kepid, "KIC")
            
            disp = str(row.get("koi_disposition", "")).strip().upper()
            label = None
            
            if disp in ("CONFIRMED", "CANDIDATE"):
                label = "exoplanet_transit"
            elif disp == "FALSE POSITIVE":
                ss = row.get("koi_fpflag_ss", 0)
                ec = row.get("koi_fpflag_ec", 0)
                co = row.get("koi_fpflag_co", 0)
                nt = row.get("koi_fpflag_nt", 0)
                
                if ss == 1 or ec == 1:
                    label = "eclipsing_binary"
                elif co == 1:
                    label = "blend_contamination"
                elif nt == 1:
                    label = "stellar_variability_or_other"
                else:
                    label = "stellar_variability_or_other"
            
            if label:
                records.append({
                    "target_id": tid,
                    "label": label,
                    "period_days": float(row.get("koi_period", 0.0)) if pd.notnull(row.get("koi_period")) else 0.0,
                    "depth_frac": (float(row.get("koi_depth", 0.0)) / 1e6) if pd.notnull(row.get("koi_depth")) else 0.0,
                    "duration_days": (float(row.get("koi_duration", 0.0)) / 24.0) if pd.notnull(row.get("koi_duration")) else 0.0,
                    "source": "kepler"
                })
            else:
                rejected_count += 1
                
    # 2. TESS TOI Loading
    toi_path = os.path.join(archive_dir, "TOI_2026.06.25_21.21.19.csv")
    if os.path.exists(toi_path):
        logger.info("Loading TESS TOI catalogue from %s", toi_path)
        df_toi = pd.read_csv(toi_path, comment="#")
        for idx, row in df_toi.iterrows():
            ticid = row.get("tid")
            if pd.isnull(ticid):
                rejected_count += 1
                continue
            tid = normalize_target_id(ticid, "TIC")
            
            disp = str(row.get("tfopwg_disp", "")).strip().upper()
            label = None
            
            if disp in ("CP", "KP", "PC"):
                label = "exoplanet_transit"
            elif disp in ("EB", "OEB", "V"):
                label = "eclipsing_binary"
            elif disp in ("NEB", "BEB", "BC"):
                label = "blend_contamination"
            elif disp == "FA":
                label = "stellar_variability_or_other"
            elif disp == "FP":
                depth_ppm = float(row.get("pl_trandep", 0.0)) if pd.notnull(row.get("pl_trandep")) else 0.0
                if depth_ppm > 30000.0:  # > 3% fractional depth
                    label = "eclipsing_binary"
                else:
                    label = "blend_contamination"
            else:
                label = "stellar_variability_or_other"
                
            records.append({
                "target_id": tid,
                "label": label,
                "period_days": float(row.get("pl_orbper", 0.0)) if pd.notnull(row.get("pl_orbper")) else 0.0,
                "depth_frac": (float(row.get("pl_trandep", 0.0)) / 1e6) if pd.notnull(row.get("pl_trandep")) else 0.0,
                "duration_days": (float(row.get("pl_trandurh", 0.0)) / 24.0) if pd.notnull(row.get("pl_trandurh")) else 0.0,
                "source": "tess"
            })
            
    df_all = pd.DataFrame(records)
    logger.info("Processed %d rows, rejected %d rows due to missing metadata/labels.", len(df_all), rejected_count)
    
    # De-duplicate to prevent target leak
    df_all = df_all.drop_duplicates(subset=["target_id"])
    logger.info("De-duplicated records count: %d targets", len(df_all))
    
    # Split Strictly by target_id
    rng = np.random.default_rng(42)
    shuffled_idx = rng.permutation(df_all.index)
    df_shuffled = df_all.loc[shuffled_idx].reset_index(drop=True)
    
    n_total = len(df_shuffled)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    
    df_train = df_shuffled.iloc[:n_train]
    df_val = df_shuffled.iloc[n_train:n_train+n_val]
    df_test = df_shuffled.iloc[n_train+n_val:]
    
    # Save splits
    df_train[["target_id", "label"]].to_csv(os.path.join(splits_dir, "train_targets.csv"), index=False)
    df_val[["target_id", "label"]].to_csv(os.path.join(splits_dir, "val_targets.csv"), index=False)
    df_test[["target_id", "label"]].to_csv(os.path.join(splits_dir, "test_targets.csv"), index=False)
    
    logger.info("Saved splits: Train=%d, Val=%d, Test=%d", len(df_train), len(df_val), len(df_test))
    
    # Class balance report
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        counts = df["label"].value_counts()
        logger.info("%s split balance:\n%s", name, counts.to_string())
        
    # Generate Gold Set
    gold_records = []
    # confirmed exoplanets
    confirmed_exo = df_shuffled[df_shuffled["label"] == "exoplanet_transit"].head(3)
    # eclipsing binaries
    confirmed_eb = df_shuffled[df_shuffled["label"] == "eclipsing_binary"].head(3)
    # blend contamination
    confirmed_blend = df_shuffled[df_shuffled["label"] == "blend_contamination"].head(3)
    # noise
    confirmed_noise = df_shuffled[df_shuffled["label"] == "stellar_variability_or_other"].head(3)
    
    df_gold = pd.concat([confirmed_exo, confirmed_eb, confirmed_blend, confirmed_noise])
    gold_path = os.path.join(datasets_dir, "gold_set.csv")
    df_gold.to_csv(gold_path, index=False)
    logger.info("Saved Gold Set to %s", gold_path)

if __name__ == "__main__":
    process_archive()
