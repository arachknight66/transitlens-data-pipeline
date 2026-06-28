"""Independent aggregation and cross-report agreement checks for Phase 1."""

import json
from pathlib import Path

import pandas as pd


def _counts(series):
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def run_final_verification(config):
    m = config.manifests_dir
    obs = pd.read_parquet(m / "observation_manifest.parquet")
    parsed = obs[obs["parse_status"] == "success"].copy()
    split = pd.read_parquet(m / "split_manifest.parquet")
    download = pd.read_parquet(m / "download_manifest.parquet")
    failures = pd.read_parquet(m / "failures.parquet")
    exclusions = pd.read_parquet(m / "exclusions.parquet")
    duplicates = pd.read_parquet(m / "duplicate_groups.parquet")
    contradictions = pd.read_parquet(m / "contradictions.parquet")

    supervised = {
        name: set(split.loc[split["split"] == name, "tic_id"].astype(int))
        for name in ("train", "val", "test")
    }
    overlaps = {
        "train_validation": len(supervised["train"] & supervised["val"]),
        "train_test": len(supervised["train"] & supervised["test"]),
        "validation_test": len(supervised["val"] & supervised["test"]),
    }

    checksum_report = {}
    if (m / "checksum_report.json").exists():
        checksum_report = json.loads((m / "checksum_report.json").read_text(encoding="utf-8"))

    counts = {
        "discovered_products": int(len(obs)),
        "downloaded_products": int((download["download_status"] == "verified").sum()),
        "verified_fits_products": int((obs["download_status"] == "verified").sum()),
        "parsed_observations": int(len(parsed)),
        "unique_tics": int(parsed["tic_id"].nunique()),
        "tic_sector_observations": int(parsed[["tic_id", "sector"]].drop_duplicates().shape[0]),
        "by_sector": _counts(parsed["sector"]),
        "by_cadence_seconds": _counts(parsed["cadence_seconds"]),
        "by_author": _counts(parsed["author"]),
        "by_canonical_label": _counts(parsed["canonical_label"]),
        "by_evidence_level": _counts(parsed["evidence_level"]),
        "by_split_observations": _counts(parsed["split"]),
        "by_split_unique_tics": _counts(split["split"]),
        "unlabeled_observations": int((parsed["canonical_label"] == "unlabeled").sum()),
        "review_required_observations": int((parsed["canonical_label"] == "review_required").sum()),
        "contradictions": int(len(contradictions)),
        "download_failures": int(download["download_status"].isin(["network_failed", "archive_missing", "checksum_failed"]).sum()),
        "parsing_failures": int((download["parse_status"] == "failed").sum()),
        "quarantined_files": int((download["download_status"] == "quarantined").sum()),
        "duplicate_alternatives": int(len(duplicates)),
        "exclusions": int(len(exclusions)),
        "failure_records": int(len(failures)),
        "checksum_failures": int(len(checksum_report.get("failed_files", [])) + len(checksum_report.get("missing_files", []))),
        "split_overlaps": overlaps,
    }

    mismatches = []
    summary_path = m / "dataset_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key in ("discovered_products", "verified_downloads", "parsed_observations", "unique_tics"):
            actual_key = "verified_fits_products" if key == "verified_downloads" else key
            if int(summary.get(key, -1)) != counts[actual_key]:
                mismatches.append(f"dataset_summary.{key} disagrees with canonical aggregation")

    target_files = {
        "train": "train_targets.parquet",
        "val": "validation_targets.parquet",
        "test": "test_targets.parquet",
        "screening": "unlabeled_screening_targets.parquet",
        "review": "review_required_targets.parquet",
    }
    for split_name, filename in target_files.items():
        target_count = len(pd.read_parquet(m / filename))
        canonical_count = int((split["split"] == split_name).sum())
        if target_count != canonical_count:
            mismatches.append(f"{filename} count {target_count} != split_manifest {canonical_count}")

    validation_status = "FAIL"
    validation_path = m / "validation_report.json"
    if validation_path.exists():
        validation_status = json.loads(validation_path.read_text(encoding="utf-8")).get("status", "FAIL")
    status = "FAIL" if mismatches else validation_status
    result = {"status": status, "counts": counts, "mismatches": mismatches}
    (m / "final_verification.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Phase 1 independent final verification",
        "",
        f"Overall status: **{status}**",
        "",
        f"Cross-report mismatches: **{len(mismatches)}**",
        "",
        "```json",
        json.dumps(counts, indent=2, sort_keys=True),
        "```",
    ]
    if mismatches:
        lines.extend(["", "## Mismatches", ""] + [f"- {item}" for item in mismatches])
    (m / "final_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
