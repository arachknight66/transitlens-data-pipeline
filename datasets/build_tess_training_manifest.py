import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def normalize_tic_id(raw_id):
    """Normalize TIC ID to integer and string formats."""
    if pd.isnull(raw_id):
        return None, None
    cleaned = str(raw_id).strip().upper().replace("TIC", "").replace("-", "")
    try:
        val = int(float(cleaned))
        return val, f"TIC-{val}"
    except ValueError:
        return None, None

def audit_archive(toi_path, tce_path):
    """Audit archive files and print the audit report to stdout."""
    print("=" * 60)
    print("  NASA Archive Audit Report")
    print("=" * 60)
    
    # 1. Audit TOI
    print(f"Loading TOI file: {toi_path}")
    toi_df = pd.read_csv(toi_path, comment="#")
    row_count_toi = len(toi_df)
    cols_toi = list(toi_df.columns)
    print(f"TOI Row Count: {row_count_toi}")
    print(f"TOI Columns: {cols_toi}")
    
    missing_tic_toi = toi_df["tid"].isnull().sum()
    print(f"TOI Missing TIC IDs: {missing_tic_toi}")
    
    unique_tic_toi = toi_df["tid"].dropna().nunique()
    print(f"TOI Unique TIC Count: {unique_tic_toi}")
    
    disp_counts = toi_df["tfopwg_disp"].value_counts(dropna=False).to_dict()
    print(f"TOI Disposition Counts: {disp_counts}")
    
    # 2. Audit TCE
    print(f"Loading TCE file: {tce_path}")
    tce_df = pd.read_csv(tce_path)
    row_count_tce = len(tce_df)
    cols_tce = list(tce_df.columns)
    print(f"TCE Row Count: {row_count_tce}")
    print(f"TCE Columns: {cols_tce}")
    
    missing_tic_tce = tce_df["ticid"].isnull().sum()
    print(f"TCE Missing TIC IDs: {missing_tic_tce}")
    
    unique_tic_tce = tce_df["ticid"].dropna().nunique()
    print(f"TCE Unique TIC Count: {unique_tic_tce}")
    print("=" * 60)
    
    return toi_df, tce_df

def main():
    parser = argparse.ArgumentParser(description="Build TESS Training Targets Manifest")
    parser.add_argument("--archive", required=True, help="Path to main TOI labels CSV")
    parser.add_argument("--tce", required=True, help="Path to TCE stats CSV")
    parser.add_argument("--output", required=True, help="Path to write the Parquet manifest")
    parser.add_argument("--label-policy", required=True, help="Path to label policy YAML")
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 1. Load label policy
    print(f"Loading label policy from {args.label_policy}")
    with open(args.label_policy, "r") as f:
        policy = yaml.safe_load(f)
    mappings = policy.get("mappings", {})
    
    # 2. Audit and Load CSVs
    toi_df, tce_df = audit_archive(args.archive, args.tce)
    
    # Prepare exclusions log
    exclusions = []
    
    # 3. Parse and join TESS targets
    processed_records = []
    
    # Create TCE lookup dictionary: ticid -> list of TCE rows
    tce_lookup = {}
    for idx, row in tce_df.iterrows():
        raw_tic = row.get("ticid")
        tic_val, _ = normalize_tic_id(raw_tic)
        if tic_val is not None:
            tce_lookup.setdefault(tic_val, []).append(row)
            
    # Process TOI rows
    source_filename = os.path.basename(args.archive)
    for idx, row in toi_df.iterrows():
        raw_tic = row.get("tid")
        tic_val, target_id = normalize_tic_id(raw_tic)
        
        if tic_val is None:
            exclusions.append({
                "tic_id": None,
                "toi": row.get("toi"),
                "reason": "Missing or invalid TIC ID in TOI row",
                "source_row": idx
            })
            continue
            
        disp = str(row.get("tfopwg_disp", "")).strip()
        if pd.isnull(row.get("tfopwg_disp")) or disp == "" or disp == "nan":
            disp_key = "missing"
        else:
            disp_key = disp
            
        # Match disposition mapping
        mapping = mappings.get(disp_key)
        if not mapping:
            exclusions.append({
                "tic_id": tic_val,
                "toi": row.get("toi"),
                "reason": f"Unknown tfopwg_disp '{disp}' in policy",
                "source_row": idx
            })
            continue
            
        action = mapping.get("action", "exclude")
        if action == "exclude":
            exclusions.append({
                "tic_id": tic_val,
                "toi": row.get("toi"),
                "reason": mapping.get("reason", "Excluded by label policy"),
                "source_row": idx
            })
            continue
            
        class_label = mapping.get("label")
        label_strength = mapping.get("strength")
        
        # Get Candidate number
        toi_id = str(row.get("toi", ""))
        try:
            # e.g. "1000.01" -> candidate 1
            candidate_id = int(float(toi_id.split(".")[-1]))
        except (ValueError, IndexError):
            candidate_id = 1
            
        # Match Sector 78 TCE stats
        matched_tce = None
        tce_list = tce_lookup.get(tic_val, [])
        for t in tce_list:
            t_pl = t.get("tce_plnt_num")
            if pd.notnull(t_pl) and int(t_pl) == candidate_id:
                matched_tce = t
                break
        if matched_tce is None and len(tce_list) > 0:
            # If no plant number match, default to first TCE
            matched_tce = tce_list[0]
            
        # Determine parameters
        # Duration: convert hours to days
        duration_hours = float(row.get("pl_trandurh", 0.0)) if pd.notnull(row.get("pl_trandurh")) else 0.0
        duration_days = duration_hours / 24.0
        
        period_days = float(row.get("pl_orbper", 0.0)) if pd.notnull(row.get("pl_orbper")) else 0.0
        depth_ppm = float(row.get("pl_trandep", 0.0)) if pd.notnull(row.get("pl_trandep")) else 0.0
        
        sector = None
        if matched_tce is not None:
            # If we matched Sector 78 event stats
            sector = 78
            
        record = {
            "target_id": target_id,
            "tic_id": int(tic_val),
            "toi": toi_id,
            "candidate_id": candidate_id,
            "ra": float(row.get("ra", 0.0)) if pd.notnull(row.get("ra")) else 0.0,
            "dec": float(row.get("dec", 0.0)) if pd.notnull(row.get("dec")) else 0.0,
            "sector": sector,
            "source_file": source_filename,
            "source_row": idx,
            "source_disposition": disp,
            "class_label": class_label,
            "label_strength": label_strength,
            "period_days": period_days,
            "duration_days": duration_days,
            "depth_ppm": depth_ppm,
            "download_status": "pending",
            "data_product": "SPOC",
            "raw_fits_path": "",
            "processed_path": "",
            "failure_reason": "",
            "attempt_count": 0,
            "download_timestamp": "",
            "checksum": ""
        }
        processed_records.append(record)
        
    df_processed = pd.DataFrame(processed_records)
    print(f"Total processed candidates before deduplication: {len(df_processed)}")
    
    # 4. Deduplicate by TIC and Candidate with Conflict Resolution
    deduped_records = []
    if len(df_processed) > 0:
        # Group by (tic_id, candidate_id)
        grouped = df_processed.groupby(["tic_id", "candidate_id"])
        strength_val = {"strong": 3, "medium": 2, "weak": 1, "none": 0}
        
        for name, group in grouped:
            if len(group) == 1:
                deduped_records.append(group.iloc[0].to_dict())
            else:
                # Sort by strength
                group = group.copy()
                group["strength_score"] = group["label_strength"].map(strength_val)
                group = group.sort_values(by="strength_score", ascending=False)
                
                # Check for contradictions among the highest-strength entries
                best_strength = group.iloc[0]["strength_score"]
                best_entries = group[group["strength_score"] == best_strength]
                
                unique_labels = best_entries["class_label"].unique()
                if len(unique_labels) > 1:
                    # Contradictory classes at same strength level: exclude the target!
                    exclusions.append({
                        "tic_id": int(name[0]),
                        "toi": str(group.iloc[0]["toi"]),
                        "reason": f"Contradictory labels {list(unique_labels)} for same target",
                        "source_row": int(group.iloc[0]["source_row"])
                    })
                else:
                    deduped_records.append(best_entries.iloc[0].drop("strength_score").to_dict())
                    
    df_deduped = pd.DataFrame(deduped_records)
    print(f"Total candidates after deduplication: {len(df_deduped)}")
    
    # 5. Split before downloading, grouping by tic_id
    if len(df_deduped) > 0:
        # Group split by tic_id to prevent leakage
        unique_tics = df_deduped["tic_id"].unique()
        
        # Build representative class for stratification
        tic_repr_class = {}
        for tic in unique_tics:
            tic_rows = df_deduped[df_deduped["tic_id"] == tic]
            # Prioritize exoplanet_transit for stratification
            classes = tic_rows["class_label"].tolist()
            if "exoplanet_transit" in classes:
                tic_repr_class[tic] = "exoplanet_transit"
            else:
                tic_repr_class[tic] = classes[0]
                
        # Split unique tics stratified by representative class
        rng = np.random.default_rng(42)
        train_tics, val_tics, test_tics = [], [], []
        
        # Group tics by class
        class_to_tics = {}
        for tic, cls in tic_repr_class.items():
            class_to_tics.setdefault(cls, []).append(tic)
            
        for cls, tics_list in class_to_tics.items():
            tics_arr = np.array(tics_list)
            rng.shuffle(tics_arr)
            
            n = len(tics_arr)
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)
            
            train_tics.extend(tics_arr[:n_train])
            val_tics.extend(tics_arr[n_train:n_train+n_val])
            test_tics.extend(tics_arr[n_train+n_val:])
            
        train_set = set(train_tics)
        val_set = set(val_tics)
        test_set = set(test_tics)
        
        # Map split back to targets dataframe
        splits = []
        for idx, row in df_deduped.iterrows():
            tic = row["tic_id"]
            if tic in train_set:
                splits.append("train")
            elif tic in val_set:
                splits.append("val")
            elif tic in test_set:
                splits.append("test")
            else:
                splits.append("train") # default fallback
                
        df_deduped["split"] = splits
    else:
        df_deduped["split"] = []
        
    # Print stats
    if len(df_deduped) > 0:
        print("\nSplit Distribution:")
        print(df_deduped["split"].value_counts())
        print("\nClass Distribution:")
        print(df_deduped["class_label"].value_counts())
        print("\nSplit Class Balance:")
        print(pd.crosstab(df_deduped["class_label"], df_deduped["split"]))
        
    # Write exclusions report
    exclusions_df = pd.DataFrame(exclusions)
    exclusions_path = Path(args.output).parent / "exclusions_report.csv"
    exclusions_df.to_csv(exclusions_path, index=False)
    print(f"Exclusions report saved to {exclusions_path}")
    print(f"Total excluded records: {len(exclusions_df)}")
    
    # Save manifest Parquet
    df_deduped.to_parquet(args.output, index=False)
    print(f"Saved manifest to {args.output}")

if __name__ == "__main__":
    main()
