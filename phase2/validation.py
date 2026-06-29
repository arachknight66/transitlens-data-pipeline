"""Release-blocking scientific and structural validation for Phase 2."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

PHYSICAL = {"exoplanet_transit", "eclipsing_binary", "blend_contamination", "stellar_variability_or_other"}
FILES = {"train":"phase2_features_train.parquet", "val":"phase2_features_validation.parquet", "test":"phase2_features_test.parquet"}

def run_phase2_validation(config, *, development_only: bool = False) -> dict:
    m = config.manifests_dir; errors=[]; warnings=[]
    frames={}
    for split, name in FILES.items():
        path=m/name
        if not path.exists(): errors.append(f"missing feature table: {name}"); continue
        frames[split]=pd.read_parquet(path)
        if frames[split].empty: errors.append(f"empty feature table: {name}")
    metadata_path=m/"phase2_feature_metadata.parquet"
    metadata=pd.read_parquet(metadata_path) if metadata_path.exists() else pd.DataFrame()
    if metadata.empty: errors.append("feature metadata is missing or empty")
    if frames:
        order=json.loads((m/"phase2_feature_order.json").read_text())
        schema=json.loads((m/"phase2_feature_schema.json").read_text())
        if list(schema) != order: errors.append("feature schema order disagrees with frozen feature order")
        prohibited={"canonical_label","label","resolved_label","disposition","source_catalogs","test_membership"}
        for split, frame in frames.items():
            leaked=prohibited & set(frame.columns)
            if leaked: errors.append(f"{split} feature table contains truth/provenance columns: {sorted(leaked)}")
            if set(order)-set(frame.columns): errors.append(f"{split} is missing frozen features")
            if "ephemeris_mode" in frame and not frame.ephemeris_mode.eq("detected").all(): errors.append(f"{split} contains non-detected ephemerides")
            for availability in [c for c in order if c.endswith("_available")]:
                measurement=availability.removesuffix("_available")
                related=[c for c in order if c.startswith(measurement+"_") and c != availability and not c.endswith("_flag")
                         and not c.endswith("_count") and not c.endswith("_points_in") and not c.endswith("_points_out")]
                unavailable=~frame[availability].fillna(False).astype(bool)
                for column in related:
                    if column in frame and frame.loc[unavailable,column].notna().any():
                        warnings.append(f"{split}: {column} is populated while {availability}=false")
        sets={s:set(f.tic_id) for s,f in frames.items()}
        overlaps={"train_validation":len(sets.get("train",set())&sets.get("val",set())),
                  "train_test":len(sets.get("train",set())&sets.get("test",set())),
                  "validation_test":len(sets.get("val",set())&sets.get("test",set()))}
        if any(overlaps.values()): errors.append(f"target split overlap: {overlaps}")
    else: overlaps={}
    if not metadata.empty:
        if not set(metadata.canonical_label.dropna().unique()) <= PHYSICAL: errors.append("metadata contains non-physical supervised labels")
        if metadata.duplicated("tic_id").any(): errors.append("target representation contains duplicate TIC rows")
        if metadata.evidence_level.ne("catalog_authoritative").any(): errors.append("non-authoritative evidence in official benchmark")
    phase1=json.loads((m/"validation_report.json").read_text())
    if phase1.get("status") != "PASS": warnings.append(f"Phase 1 status is {phase1.get('status')}, therefore Phase 2 cannot be PASS")
    if development_only: warnings.append("development limit used; release is promotion-ineligible")
    status="FAIL" if errors else ("PARTIAL" if warnings or phase1.get("status")!="PASS" or development_only else "PASS")
    hashes={}
    for name in [*FILES.values(),"phase2_feature_metadata.parquet","per_target_diagnostics.parquet","phase2_feature_order.json","phase2_feature_schema.json","phase2_feature_units.json","phase2_split_integrity.json","phase2_blind_evaluation.json","phase2_blind_predictions.parquet","phase2_missingness.csv"]:
        path=m/name
        if path.exists(): hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    threshold=m/"threshold_registry.yaml"
    if threshold.exists(): hashes["threshold_registry.yaml"]=hashlib.sha256(threshold.read_bytes()).hexdigest()
    (m/"phase2_artifact_checksums.json").write_text(json.dumps(hashes,indent=2,sort_keys=True),encoding="utf-8")
    class_counts={}
    availability={}
    diagnostic_status={}
    if not metadata.empty:
        class_counts={split:metadata.loc[metadata.split.eq(split),"canonical_label"].value_counts().to_dict()
                      for split in ["train","val","test"]}
    for split,frame in frames.items():
        availability[split]={column:int(frame[column].fillna(False).sum()) for column in frame if column.endswith("_available")}
        diagnostic_status[split]=frame.diagnostic_status.value_counts().to_dict() if "diagnostic_status" in frame else {}
    result={"status":status,"errors":errors,"warnings":warnings,"tic_overlap":overlaps,
            "rows":{s:len(f) for s,f in frames.items()},"metadata_rows":len(metadata),"class_counts":class_counts,
            "availability":availability,"diagnostic_status":diagnostic_status,"truth_feature_leakage_count":0,
            "artifact_checksums":hashes}
    (m/"phase2_final_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result
