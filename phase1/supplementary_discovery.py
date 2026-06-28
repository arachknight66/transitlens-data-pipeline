"""Cached discovery for authoritative labelled supplementary cohorts."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

MAST_INVOKE = "https://mast.stsci.edu/api/v0/invoke"


def _invoke(payload, timeout=120):
    response = requests.post(MAST_INVOKE, data={"request": json.dumps(payload)}, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if body.get("status") not in (None, "COMPLETE"):
        raise RuntimeError(f"MAST request failed: {body.get('msg', body.get('status'))}")
    return body


def _chunks(values, size):
    values = sorted({str(int(value)) for value in values})
    for offset in range(0, len(values), size):
        yield values[offset:offset + size]


def build_kic_tic_aliases(kic_ids, cache_dir, chunk_size=400):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for chunk in _chunks(kic_ids, chunk_size):
        digest = hashlib.sha256(",".join(chunk).encode()).hexdigest()[:16]
        cache = cache_dir / f"tic_alias_{digest}.json"
        if cache.exists():
            body = json.loads(cache.read_text(encoding="utf-8"))
        else:
            payload = {
                "service": "Mast.Catalogs.Filtered.Tic", "format": "json",
                "params": {
                    "columns": "ID,KIC,ra,dec,Tmag",
                    "filters": [{"paramName": "KIC", "values": chunk}],
                    "pagesize": max(1000, len(chunk) * 3),
                },
            }
            body = _invoke(payload)
            cache.write_text(json.dumps(body), encoding="utf-8")
        for row in body.get("data", []):
            if row.get("KIC") is not None and row.get("ID") is not None:
                rows.append({
                    "kic_id": int(row["KIC"]), "tic_id": int(row["ID"]),
                    "ra": row.get("ra"), "dec": row.get("dec"), "tmag": row.get("Tmag"),
                    "source": "MAST_TIC_v8.2", "cache_file": cache.name,
                })
    return pd.DataFrame(rows).drop_duplicates(["kic_id", "tic_id"])


def discover_spoc_products(tic_ids, cache_dir, chunk_size=150):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for chunk in _chunks(tic_ids, chunk_size):
        digest = hashlib.sha256(",".join(chunk).encode()).hexdigest()[:16]
        cache = cache_dir / f"spoc_targets_{digest}.json"
        if cache.exists():
            body = json.loads(cache.read_text(encoding="utf-8"))
        else:
            payload = {
                "service": "Mast.Caom.Filtered", "format": "json",
                "params": {
                    "columns": "obs_id,target_name,sequence_number,t_exptime,s_ra,s_dec,provenance_name,dataproduct_type,dataURL",
                    "filters": [
                        {"paramName": "obs_collection", "values": ["TESS"]},
                        {"paramName": "target_name", "values": chunk},
                        {"paramName": "provenance_name", "values": ["SPOC"]},
                        {"paramName": "dataproduct_type", "values": ["timeseries"]},
                    ],
                    "pagesize": max(10000, len(chunk) * 50),
                },
            }
            body = _invoke(payload)
            cache.write_text(json.dumps(body), encoding="utf-8")
        for row in body.get("data", []):
            uri = str(row.get("dataURL", ""))
            if not uri.endswith("_lc.fits"):
                continue
            target = "".join(char for char in str(row.get("target_name", "")) if char.isdigit())
            if not target:
                continue
            rows.append({
                "obs_id": row.get("obs_id"), "tic_id": int(target),
                "sector": int(row["sequence_number"]), "cadence_seconds": float(row.get("t_exptime") or 0),
                "ra": row.get("s_ra"), "dec": row.get("s_dec"),
                "product_uri": uri, "product_filename": Path(uri).name,
                "download_url": f"https://mast.stsci.edu/portal/Download/file?uri={uri}",
                "product_author": "SPOC", "product_type": "lightcurve",
                "archive_endpoint": MAST_INVOKE, "cache_file": cache.name,
            })
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.drop_duplicates("product_uri")
    return frame


def choose_one_per_tic(products, minimum=110.0, maximum=130.0):
    eligible = products[products["cadence_seconds"].between(minimum, maximum)].copy()
    eligible = eligible.sort_values(["tic_id", "sector", "product_uri"])
    return eligible.drop_duplicates("tic_id", keep="first")
