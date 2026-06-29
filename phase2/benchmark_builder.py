"""Frozen, target-disjoint real-data benchmark derived from Phase 1 only."""
from __future__ import annotations
import json
import pandas as pd

PHYSICAL = {"exoplanet_transit", "eclipsing_binary", "blend_contamination", "stellar_variability_or_other"}

def build_benchmark_manifest(config) -> dict:
    m = config.manifests_dir
    obs = pd.read_parquet(m / "observation_manifest.parquet")
    split = pd.read_parquet(m / "split_manifest.parquet")
    benchmark = obs.drop(columns=["split", "canonical_label", "resolved_label"], errors="ignore").merge(
        split[["tic_id", "split", "resolved_label"]], on="tic_id", how="inner")
    benchmark = benchmark[(benchmark.parse_status == "success") & benchmark.split.isin(["train", "val", "test"]) &
                          benchmark.resolved_label.isin(PHYSICAL)].copy()
    benchmark["evidence_tier"] = benchmark.split.map({"train":"real_training", "val":"real_validation", "test":"real_held_out"})
    benchmark["ephemeris_mode"] = "detected"
    benchmark.to_parquet(m / "phase2_benchmark_manifest.parquet", index=False)
    target_counts = benchmark.drop_duplicates("tic_id").groupby(["split", "resolved_label"]).size().unstack(fill_value=0)
    target_counts.to_csv(m / "benchmark_class_distribution.csv")
    benchmark.groupby(["split", "sector"]).size().rename("observations").reset_index().to_csv(m / "benchmark_observation_distribution.csv", index=False)
    availability = pd.DataFrame([
        {"diagnostic":"centroid", "available_observations":int(benchmark.centroid_available.fillna(False).sum()), "eligible_observations":len(benchmark)},
        {"diagnostic":"tpf", "available_observations":int(benchmark.target_pixel_file_available.fillna(False).sum()),
         "eligible_observations":int(benchmark.target_pixel_file_available.notna().sum())},
        {"diagnostic":"crowdsap", "available_observations":int(benchmark.crowding_metric.notna().sum()), "eligible_observations":len(benchmark)},
    ])
    availability["availability"] = availability.available_observations / availability.eligible_observations.replace(0, pd.NA)
    availability.to_csv(m / "benchmark_availability_report.csv", index=False)
    overlaps = {"train_validation":0, "train_test":0, "validation_test":0}
    sets = {s:set(benchmark.loc[benchmark.split==s,"tic_id"]) for s in ["train","val","test"]}
    overlaps.update({"train_validation":len(sets["train"]&sets["val"]), "train_test":len(sets["train"]&sets["test"]),
                     "validation_test":len(sets["val"]&sets["test"])})
    report = {"unique_targets":int(benchmark.tic_id.nunique()), "observations":len(benchmark),
              "target_counts":target_counts.to_dict(), "tic_overlap":overlaps,
              "excluded_review_and_unlabeled":True, "synthetic_rows":0}
    (m / "benchmark_selection_report.md").write_text(
        "# Phase 2 benchmark selection\n\nFrozen Phase 1 train/validation/test assignments are preserved. "
        "Only authoritative real supervised targets are included; review, unlabeled, and synthetic rows are excluded.\n\n```json\n"+
        json.dumps(report, indent=2)+"\n```\n", encoding="utf-8")
    (m / "benchmark_bias_report.md").write_text(
        "# Phase 2 benchmark bias\n\nThe benchmark is limited to sectors 77–79, high-cadence SPOC products, and catalogue-authoritative labels. "
        "Stellar-variability support is particularly small. Catalogue selection and TPF availability biases remain.\n", encoding="utf-8")
    return report
