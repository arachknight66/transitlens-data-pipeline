"""Discover and retrieve targeted SPOC Data Validation XML evidence."""

import hashlib
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from phase1.atomic_io import atomic_write_parquet
from phase1.supplementary_discovery import _invoke


def _chunks(values, size):
    values = list(values)
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def targeted_centroid_pairs(config):
    """Return TIC-sector pairs supported by strong SPOC centroid evidence."""
    evidence = pd.read_parquet(config.manifests_dir / "label_evidence.parquet")
    mask = (
        evidence["source_catalog"].str.match(r"^TESS_TCE_S00(77|78|79)$", na=False)
        & (evidence["canonical_label_candidate"] == "blend_contamination")
        & (evidence["evidence_strength"] == "strong")
        & evidence["sector"].isin([77, 78, 79])
    )
    return (
        evidence.loc[mask, ["tic_id", "sector"]]
        .astype({"tic_id": "int64", "sector": "int64"})
        .drop_duplicates()
        .sort_values(["sector", "tic_id"])
        .reset_index(drop=True)
    )


def discover_targeted_dvr_xml(config, target_chunk_size=75, metadata_concurrency=4):
    """Query MAST metadata and freeze one DVR XML product per TIC-sector."""
    pairs = targeted_centroid_pairs(config)
    pair_set = set(map(tuple, pairs[["tic_id", "sector"]].to_numpy()))
    obs_cache = config.manifests_dir / "dvr_xml_observations.parquet"
    if obs_cache.exists():
        obs = pd.read_parquet(obs_cache)
    else:
        observations = []
        target_chunks = list(_chunks(pairs["tic_id"].astype(str).unique(), target_chunk_size))

        def query_targets(chunk):
            return _invoke({
                "service": "Mast.Caom.Filtered", "format": "json",
                "params": {
                    "columns": "obsid,obs_id,target_name,sequence_number,provenance_name,dataproduct_type",
                    "filters": [
                        {"paramName": "obs_collection", "values": ["TESS"]},
                        {"paramName": "target_name", "values": list(chunk)},
                        {"paramName": "provenance_name", "values": ["SPOC"]},
                    ],
                    "pagesize": max(10000, len(chunk) * 50),
                },
            }).get("data", [])

        with ThreadPoolExecutor(max_workers=metadata_concurrency) as executor:
            for rows in executor.map(query_targets, target_chunks):
                if rows:
                    frame = pd.DataFrame(rows)
                else:
                    continue
                frame["tic_id"] = pd.to_numeric(frame["target_name"], errors="coerce")
                frame["sector"] = pd.to_numeric(frame["sequence_number"], errors="coerce")
                frame = frame.dropna(subset=["tic_id", "sector"])
                frame[["tic_id", "sector"]] = frame[["tic_id", "sector"]].astype("int64")
                frame = frame[
                    frame.apply(lambda row: (row["tic_id"], row["sector"]) in pair_set, axis=1)
                ]
                observations.append(frame)
        if not observations:
            raise RuntimeError("MAST returned no observations for targeted centroid evidence")
        obs = pd.concat(observations, ignore_index=True).drop_duplicates("obsid")
        obs["obsid"] = pd.to_numeric(obs["obsid"], errors="coerce").astype("Int64")
        atomic_write_parquet(obs, obs_cache, index=False)

    product_cache = config.manifests_dir / "dvr_xml_products.parquet"
    if product_cache.exists():
        product = pd.read_parquet(product_cache)
    else:
        def query_products(chunk_ids):
            return _invoke({
                "service": "Mast.Caom.Products", "format": "json",
                "params": {"obsid": ",".join(chunk_ids)},
            }).get("data", [])

        products = []
        obs_chunks = list(_chunks(obs["obsid"].astype(str).tolist(), 50))
        with ThreadPoolExecutor(max_workers=metadata_concurrency) as executor:
            for rows in executor.map(query_products, obs_chunks):
                if rows:
                    products.extend(rows)
        product = pd.DataFrame(products)
        atomic_write_parquet(product, product_cache, index=False)
    product = product[
        (product["productSubGroupDescription"] == "DVR")
        & product["productFilename"].str.lower().str.endswith("_dvr.xml")
    ].copy()
    obs_map = obs.set_index("obsid")[["tic_id", "sector"]]
    product["obsID"] = pd.to_numeric(product["obsID"], errors="coerce").astype("Int64")
    product = product.join(obs_map, on="obsID", how="inner")
    product = product.sort_values(["tic_id", "sector", "productFilename"]).drop_duplicates(
        ["tic_id", "sector"], keep="last"
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    manifest = pd.DataFrame({
        "tic_id": product["tic_id"].astype("int64"),
        "sector": product["sector"].astype("int64"),
        "obs_id": product["obs_id"],
        "mast_obsid": product["obsID"].astype("int64"),
        "product_uri": product["dataURI"],
        "product_filename": product["productFilename"].map(lambda value: Path(value).name),
        "expected_size": pd.to_numeric(product["size"], errors="coerce").fillna(0).astype("int64"),
        "download_url": product["dataURI"].map(
            lambda uri: f"https://mast.stsci.edu/portal/Download/file?uri={uri}"
        ),
        "discovered_at": timestamp,
        "archive": "MAST/STScI",
        "product_type": "SPOC_DVR_XML",
    })
    manifest["local_path"] = ""
    manifest["sha256"] = ""
    manifest["actual_size"] = 0
    manifest["status"] = "pending"
    manifest["failure_message"] = ""
    output = config.manifests_dir / "dvr_xml_manifest.parquet"
    atomic_write_parquet(manifest, output, index=False)
    missing = len(pair_set - set(map(tuple, manifest[["tic_id", "sector"]].to_numpy())))
    return manifest, missing


def _download_one(row, root, timeout=90):
    filename = Path(str(row.product_filename)).name
    if filename != str(row.product_filename) or not filename.lower().endswith("_dvr.xml"):
        raise ValueError(f"Unsafe DVR XML filename: {row.product_filename!r}")
    destination = Path(root) / f"sector_{int(row.sector):04d}" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            ET.parse(destination)
            return "verified", destination, destination.stat().st_size, _sha256(destination), ""
        except Exception:
            destination.unlink()
    partial = destination.with_suffix(".xml.part")
    headers = {"User-Agent": "TransitLens-Phase1/1.1 (public scientific archive client)"}
    with requests.get(row.download_url, stream=True, timeout=timeout, headers=headers) as response:
        response.raise_for_status()
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(128 * 1024):
                if chunk:
                    handle.write(chunk)
    actual = partial.stat().st_size
    if int(row.expected_size) > 0 and actual != int(row.expected_size):
        partial.unlink(missing_ok=True)
        raise ValueError(f"Size mismatch: expected {row.expected_size}, got {actual}")
    ET.parse(partial)
    checksum = _sha256(partial)
    os.replace(partial, destination)
    return "verified", destination, actual, checksum, ""


def download_targeted_dvr_xml(config, concurrency=4):
    """Download the frozen targeted DVR XML manifest with bounded concurrency."""
    path = config.manifests_dir / "dvr_xml_manifest.parquet"
    if not path.exists():
        raise FileNotFoundError("Run DVR XML discovery before downloading")
    manifest = pd.read_parquet(path)
    pending = manifest[manifest["status"] != "verified"]
    root = config.REPO_ROOT / "data" / "catalogs" / "raw" / "dvr_xml"
    with ThreadPoolExecutor(max_workers=int(concurrency)) as executor:
        futures = {executor.submit(_download_one, row, root): index for index, row in pending.iterrows()}
        for future in as_completed(futures):
            index = futures[future]
            try:
                status, destination, size, checksum, message = future.result()
            except Exception as error:
                status, destination, size, checksum, message = "failed", "", 0, "", str(error)
            manifest.at[index, "status"] = status
            manifest.at[index, "local_path"] = str(destination)
            manifest.at[index, "actual_size"] = int(size)
            manifest.at[index, "sha256"] = checksum
            manifest.at[index, "failure_message"] = message
            # A single writer atomically checkpoints every completed product.
            atomic_write_parquet(manifest, path, index=False)
    return manifest
