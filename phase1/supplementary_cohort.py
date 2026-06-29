"""Build a provenance-tagged labelled supplement to the full-sector cohort."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from phase1.atomic_io import atomic_write_parquet
from phase1.supplementary_discovery import MAST_INVOKE, _invoke, _chunks


LABEL_TO_CONFIG = {
    "exoplanet_transit": "planets",
    "eclipsing_binary": "ebs",
    "blend_contamination": "blends",
}


def _stable_rank(tic_id, seed):
    return hashlib.sha256(f"{seed}:{int(tic_id)}".encode()).hexdigest()


def _authoritative_candidates(config):
    evidence = pd.read_parquet(config.manifests_dir / "label_evidence.parquet")
    eligible = evidence[
        evidence["canonical_label_candidate"].isin(LABEL_TO_CONFIG)
        & (evidence["evidence_level"] == "catalog_authoritative")
        & evidence["evidence_strength"].isin(["strong", "medium"])
    ][["tic_id", "canonical_label_candidate"]].drop_duplicates()
    # Conflicting authoritative classes require review; they are never sampled
    # merely because one class has a shortfall.
    counts = eligible.groupby("tic_id")["canonical_label_candidate"].nunique()
    unambiguous = set(counts[counts == 1].index.astype(int))
    return eligible[eligible["tic_id"].isin(unambiguous)].drop_duplicates("tic_id")


def _required_additions(config, safety_margin=0.25):
    split = pd.read_parquet(config.manifests_dir / "split_manifest.parquet")
    supervised = split[split["split"].isin(["train", "val", "test"])]
    current = supervised["resolved_label"].value_counts().to_dict()
    desired = {
        label: sum(int(config.min_class_counts.get(name, {}).get(alias, 0)) for name in ("train", "validation", "test"))
        for label, alias in LABEL_TO_CONFIG.items()
    }
    return {
        label: max(0, int((desired[label] - current.get(label, 0)) * (1.0 + safety_margin) + 0.999))
        for label in LABEL_TO_CONFIG
    }, current, desired


def discover_and_merge_supplement(config, metadata_concurrency=4, safety_margin=0.25):
    """Discover one 120-second official SPOC LC for each selected labelled TIC."""
    config.ensure_dirs()
    candidates = _authoritative_candidates(config)
    download = pd.read_parquet(config.manifests_dir / "download_manifest.parquet")
    verified_tics = set(download.loc[download["download_status"] == "verified", "tic_id"].astype(int))
    candidates = candidates[~candidates["tic_id"].isin(verified_tics)].copy()
    candidates["rank"] = candidates["tic_id"].map(lambda value: _stable_rank(value, config.random_seed))
    candidates = candidates.sort_values(["canonical_label_candidate", "rank"])
    additions, current, desired = _required_additions(config, safety_margin=safety_margin)

    # Query additional candidates beyond the selection target because not every
    # catalogue TIC has an eligible 120-second SPOC product.
    query_frames = []
    for label, required in additions.items():
        pool = candidates[candidates["canonical_label_candidate"] == label]
        query_frames.append(pool.head(max(required * 2, required + 500)))
    query_targets = pd.concat(query_frames, ignore_index=True).drop_duplicates("tic_id")
    label_by_tic = query_targets.set_index("tic_id")["canonical_label_candidate"].to_dict()

    cache_dir = config.REPO_ROOT / "data" / "catalogs" / "raw" / "supplementary_discovery_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(_chunks(query_targets["tic_id"], 100))

    def query(chunk):
        digest = hashlib.sha256(",".join(chunk).encode()).hexdigest()[:16]
        cache = cache_dir / f"spoc_labelled_{digest}.json"
        if cache.exists():
            body = json.loads(cache.read_text(encoding="utf-8"))
        else:
            body = _invoke({
                "service": "Mast.Caom.Filtered", "format": "json",
                "params": {
                    "columns": "obs_id,target_name,sequence_number,t_exptime,s_ra,s_dec,provenance_name,dataproduct_type,dataURL",
                    "filters": [
                        {"paramName": "obs_collection", "values": ["TESS"]},
                        {"paramName": "target_name", "values": list(chunk)},
                        {"paramName": "provenance_name", "values": ["SPOC"]},
                        {"paramName": "dataproduct_type", "values": ["timeseries"]},
                    ],
                    "pagesize": max(10000, len(chunk) * 60),
                },
            })
            cache.write_text(json.dumps(body), encoding="utf-8")
        checksum = hashlib.sha256(cache.read_bytes()).hexdigest()
        return body.get("data", []), cache.name, checksum

    records = []
    with ThreadPoolExecutor(max_workers=int(metadata_concurrency)) as executor:
        for (rows, cache_name, cache_checksum) in executor.map(query, chunks):
            for row in rows:
                uri = str(row.get("dataURL", ""))
                target = "".join(char for char in str(row.get("target_name", "")) if char.isdigit())
                cadence = float(row.get("t_exptime") or 0)
                if not target or not uri.endswith("_lc.fits") or not (config.min_cadence_seconds <= cadence <= config.max_cadence_seconds):
                    continue
                tic_id = int(target)
                if tic_id not in label_by_tic:
                    continue
                records.append({
                    "obs_id": row.get("obs_id"), "tic_id": tic_id,
                    "target_id": f"TIC-{tic_id}", "sector": int(row["sequence_number"]),
                    "ra": row.get("s_ra"), "dec": row.get("s_dec"),
                    "t_exptime": cadence, "cadence_seconds": cadence,
                    "mission": "TESS", "product_author": "SPOC", "product_type": "lightcurve",
                    "product_uri": uri, "product_filename": Path(uri).name,
                    "download_url": f"https://mast.stsci.edu/portal/Download/file?uri={uri}",
                    "status": "discovered", "discovery_timestamp": datetime.now(timezone.utc).isoformat(),
                    "archive_endpoint": MAST_INVOKE,
                    "archive_query_parameters": json.dumps({"cohort": "authoritative_label_supplement", "tic_id": tic_id}),
                    "archive_response_cache": cache_name, "archive_response_sha256": cache_checksum,
                    "discovery_schema_version": "1.2.0", "expected_size": 2_000_000,
                    "supplementary_label": label_by_tic[tic_id],
                })
    products = pd.DataFrame(records)
    if products.empty:
        raise RuntimeError("No eligible supplementary SPOC products were discovered")
    products = products.sort_values(["tic_id", "sector", "product_uri"]).drop_duplicates("product_uri")
    representative = products.drop_duplicates("tic_id", keep="first")

    selected_parts = []
    available = {}
    for label, required in additions.items():
        pool = representative[representative["supplementary_label"] == label].copy()
        pool["rank"] = pool["tic_id"].map(lambda value: _stable_rank(value, config.random_seed))
        pool = pool.sort_values(["rank", "sector", "product_uri"])
        available[label] = len(pool)
        selected_parts.append(pool.head(required))
    selected = pd.concat(selected_parts, ignore_index=True).drop(columns=["rank"], errors="ignore")
    atomic_write_parquet(representative, config.manifests_dir / "supplementary_product_inventory.parquet", index=False)
    atomic_write_parquet(selected, config.manifests_dir / "supplementary_selected.parquet", index=False)

    discovery_path = config.manifests_dir / "discovery_manifest.parquet"
    discovery = pd.read_parquet(discovery_path)
    merged = pd.concat([discovery, selected], ignore_index=True, sort=False)
    merged = merged.sort_values(["product_uri", "obs_id"]).drop_duplicates("product_uri", keep="first")
    atomic_write_parquet(merged, discovery_path, index=False)

    summary = {
        "current_unique_supervised": {k: int(v) for k, v in current.items()},
        "desired_unique_supervised": desired,
        "requested_with_safety_margin": additions,
        "eligible_products_available": available,
        "selected_products": int(len(selected)),
        "estimated_download_bytes": int(selected["expected_size"].sum()),
    }
    (config.manifests_dir / "supplementary_cohort_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return selected, summary


def extend_supplement_class(config, label="exoplanet_transit", count=100):
    """Freeze an additional deterministic top-up from cached product inventory."""
    inventory = pd.read_parquet(config.manifests_dir / "supplementary_product_inventory.parquet")
    discovery_path = config.manifests_dir / "discovery_manifest.parquet"
    discovery = pd.read_parquet(discovery_path)
    existing_uris = set(discovery["product_uri"])
    pool = inventory[
        (inventory["supplementary_label"] == label)
        & ~inventory["product_uri"].isin(existing_uris)
    ].copy()
    pool["rank"] = pool["tic_id"].map(lambda value: _stable_rank(value, config.random_seed + 1))
    topup = pool.sort_values(["rank", "sector", "product_uri"]).head(int(count)).drop(columns="rank")
    if len(topup) < count:
        raise RuntimeError(f"Only {len(topup)} unused {label} products are available")
    merged = pd.concat([discovery, topup], ignore_index=True, sort=False)
    merged = merged.sort_values(["product_uri", "obs_id"]).drop_duplicates("product_uri", keep="first")
    atomic_write_parquet(merged, discovery_path, index=False)
    selected_path = config.manifests_dir / "supplementary_selected.parquet"
    selected = pd.read_parquet(selected_path)
    selected = pd.concat([selected, topup], ignore_index=True).drop_duplicates("product_uri")
    atomic_write_parquet(selected, selected_path, index=False)
    return topup
