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

def save_manifest_atomically(df, dest_path):
    dest_path = Path(dest_path)
    dest_dir = dest_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    temp_path = dest_dir / f"temp_{dest_path.name}"
    df.to_parquet(temp_path, index=False)
    
    try:
        # Validate readability
        pd.read_parquet(temp_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Manifest Parquet validation failed after write: {e}")
        
    os.replace(str(temp_path), str(dest_path))
    print(f"Saved manifest atomically to {dest_path}")

def split_tics_stratified(class_to_tics, seed=42):
    rng = np.random.default_rng(seed)
    train_tics, val_tics, test_tics = [], [], []
    
    for cls, tics_list in class_to_tics.items():
        tics_arr = np.array(tics_list)
        rng.shuffle(tics_arr)
        n = len(tics_arr)
        
        if n == 0:
            continue
        elif n == 1:
            train_tics.append(tics_arr[0])
        elif n == 2:
            train_tics.append(tics_arr[0])
            val_tics.append(tics_arr[1])
        else:
            n_val = max(1, int(np.round(0.15 * n)))
            n_test = max(1, int(np.round(0.15 * n)))
            n_train = n - n_val - n_test
            
            if n_train <= 0:
                n_train = 1
                n_val = 1
                n_test = n - 2
                
            train_tics.extend(tics_arr[:n_train])
            val_tics.extend(tics_arr[n_train:n_train+n_val])
            test_tics.extend(tics_arr[n_train+n_val:n_train+n_val+n_test])
            
            if n_train + n_val + n_test < n:
                train_tics.extend(tics_arr[n_train+n_val+n_test:])
                
    return set(train_tics), set(val_tics), set(test_tics)

def main():
    parser = argparse.ArgumentParser(description="Build TESS Training Targets Manifest")
    parser.add_argument("--archive", required=True, help="Path to main TOI labels CSV")
    parser.add_argument("--tce", required=True, help="Path to TCE stats CSV")
    parser.add_argument("--output", required=True, help="Path to write the Parquet manifest")
    parser.add_argument("--label-policy", required=True, help="Path to label policy YAML")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split reproducibility")
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
            matched_tce = tce_list[0]
            
        duration_hours = float(row.get("pl_trandurh", 0.0)) if pd.notnull(row.get("pl_trandurh")) else 0.0
        duration_days = duration_hours / 24.0
        
        period_days = float(row.get("pl_orbper", 0.0)) if pd.notnull(row.get("pl_orbper")) else 0.0
        depth_ppm = float(row.get("pl_trandep", 0.0)) if pd.notnull(row.get("pl_trandep")) else 0.0
        
        sector = None
        if matched_tce is not None:
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
            "processing_status": "pending",
            "data_product": "SPOC",
            "raw_fits_path": "",
            "processed_path": "",
            "failure_reason": "",
            "attempt_count": 0,
            "download_timestamp": "",
            "checksum": "",
            "cutout_size": 15,
            "aperture_version": "connected_threshold_v1.0"
        }
        processed_records.append(record)
        
    df_processed = pd.DataFrame(processed_records)
    print(f"Total processed candidates before deduplication: {len(df_processed)}")
    
    # 4. Deduplicate by TIC and resolve label conflicts
    deduped_records = []
    if len(df_processed) > 0:
        strength_val = {"strong": 3, "medium": 2, "weak": 1, "none": 0}
        
        # Group by tic_id
        grouped_tic = df_processed.groupby("tic_id")
        for tic_val, group in grouped_tic:
            unique_classes = group["class_label"].unique()
            if len(unique_classes) == 1:
                # No conflict, append all candidates of this TIC
                for _, r in group.iterrows():
                    deduped_records.append(r.to_dict())
            else:
                # Conflict exists: resolve using strength (strong > medium > weak)
                group = group.copy()
                group["strength_score"] = group["label_strength"].map(strength_val)
                max_strength = group["strength_score"].max()
                
                # Filter to only highest-strength rows
                best_entries = group[group["strength_score"] == max_strength]
                best_classes = best_entries["class_label"].unique()
                
                if len(best_classes) > 1:
                    # Contradictory classes at max strength: exclude the entire TIC
                    for _, r in group.iterrows():
                        exclusions.append({
                            "tic_id": int(tic_val),
                            "toi": str(r.get("toi")),
                            "reason": f"Contradictory classes {list(best_classes)} at strength {max_strength} for TIC {tic_val}",
                            "source_row": int(r.get("source_row"))
                        })
                else:
                    # Successfully resolved to best_classes[0]
                    resolved_class = best_classes[0]
                    # Keep only candidates matching the resolved class, exclude the weaker contradicting ones
                    for _, r in group.iterrows():
                        if r["class_label"] == resolved_class:
                            deduped_records.append(r.drop("strength_score").to_dict())
                        else:
                            exclusions.append({
                                "tic_id": int(tic_val),
                                "toi": str(r.get("toi")),
                                "reason": f"Excluded weaker contradicting class '{r['class_label']}' resolved to '{resolved_class}'",
                                "source_row": int(r.get("source_row"))
                            })
                            
    df_deduped = pd.DataFrame(deduped_records)
    print(f"Total candidates after deduplication & TIC-level resolution: {len(df_deduped)}")
    
    # 5. Split before downloading, grouping by tic_id
    if len(df_deduped) > 0:
        unique_tics = df_deduped["tic_id"].unique()
        
        # Build representative class for stratification (all candidates of a TIC have same class now)
        tic_repr_class = {}
        for tic in unique_tics:
            tic_repr_class[tic] = df_deduped[df_deduped["tic_id"] == tic]["class_label"].iloc[0]
            
        class_to_tics = {}
        for tic, cls in tic_repr_class.items():
            class_to_tics.setdefault(cls, []).append(tic)
            
        train_set, val_set, test_set = split_tics_stratified(class_to_tics, seed=args.seed)
        
        # Verify split disjointness
        assert train_set.isdisjoint(val_set), "Train and Val splits overlap!"
        assert train_set.isdisjoint(test_set), "Train and Test splits overlap!"
        assert val_set.isdisjoint(test_set), "Val and Test splits overlap!"
        
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
                splits.append("train")
                
        df_deduped["split"] = splits
        
        # Final validation assertions
        assert not df_deduped["tic_id"].isnull().any(), "Manifest contains rows with null TIC ID!"
        assert not df_deduped["class_label"].isnull().any(), "Manifest contains rows with null class!"
        
        # Verify no TIC has contradictory included labels
        for tic, group in df_deduped.groupby("tic_id"):
            assert group["class_label"].nunique() == 1, f"TIC {tic} has multiple contradictory class labels inside manifest!"
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
    
    # Save manifest Parquet atomically
    save_manifest_atomically(df_deduped, args.output)

if __name__ == "__main__":
    main()
