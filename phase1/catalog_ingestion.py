import os
import time
import hashlib
import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def get_file_checksum(filepath):
    """Computes SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def normalize_tic_id(raw_id):
    """Parses and normalizes a TIC ID, returns int or None."""
    if pd.isnull(raw_id):
        return None
    cleaned = str(raw_id).strip().upper().replace("TIC", "").replace("-", "")
    try:
        return int(float(cleaned))
    except ValueError:
        return None

def normalize_kic_id(raw_id):
    """Parses and normalizes a KIC ID, returns int or None."""
    if pd.isnull(raw_id):
        return None
    cleaned = str(raw_id).strip().upper().replace("KIC", "").replace("-", "")
    try:
        return int(float(cleaned))
    except ValueError:
        return None

def coord_match(ra1, dec1, ra2, dec2, tol_arcsec=15.0):
    """Determines if two coordinates match within a tolerance in arcseconds."""
    if pd.isnull(ra1) or pd.isnull(dec1) or pd.isnull(ra2) or pd.isnull(dec2):
        return False
    # Simple angular distance approximation
    dra = (ra1 - ra2) * np.cos(np.radians(dec1))
    ddec = dec1 - dec2
    dist = np.sqrt(dra**2 + ddec**2) * 3600.0
    return dist <= tol_arcsec

def ingest_all_catalogs(config):
    """
    Ingests all input catalogs from archive/ and outputs a normalized evidence
    table label_evidence.parquet.
    """
    config.ensure_dirs()
    manifests_dir = config.manifests_dir
    
    evidence_rows = []
    
    ingest_time = datetime.now(timezone.utc).isoformat()
    adapter_version = "1.0.0"
    
    # Keep track of unique evidence row index
    evidence_id_counter = 1
    
    # ----------------------------------------------------
    # 1. Ingest TESS TOI Catalog
    # ----------------------------------------------------
    with open(config.label_policy_file, "r", encoding="utf-8") as handle:
        toi_policy = yaml.safe_load(handle)
    if str(toi_policy.get("version")) != str(config.label_policy_version):
        raise ValueError("TOI label policy version does not match dataset configuration")
    toi_mappings = toi_policy.get("mappings", {})
    toi_path = Path(config.toi_catalog)
    if toi_path.exists():
        logger.info(f"Ingesting TESS TOI Catalog from {toi_path}...")
        csum = get_file_checksum(toi_path)
        
        # Read file, skipping comment lines starting with #
        df_toi = pd.read_csv(toi_path, comment="#")
        
        for idx, row in df_toi.iterrows():
            raw_tid = row.get("tid")
            tic_id = normalize_tic_id(raw_tid)
            if tic_id is None:
                continue
                
            disp = str(row.get("tfopwg_disp", "")).strip()
            
            mapping = toi_mappings.get(disp, toi_mappings.get("missing", {}))
            label_candidate = mapping.get("label", "unlabeled")
            strength = mapping.get("strength", "none")
            notes = mapping.get("reason", f"Versioned TOI policy mapping for {disp or 'missing'}")
                
            evidence_rows.append({
                "evidence_id": f"EVI-TOI-{evidence_id_counter:06d}",
                "tic_id": tic_id,
                "canonical_label_candidate": label_candidate,
                "original_label": disp,
                "original_disposition": disp,
                "source_catalog": "TESS_TOI",
                "source_version": "2026.06.25",
                "source_row_identifier": str(idx),
                "evidence_level": "catalog_authoritative",
                "evidence_strength": strength,
                "disposition_date": str(row.get("rowupdate", "")),
                "target_name": f"TIC-{tic_id}",
                "sector": None,
                "ephemeris": str(row.get("pl_tranmid", "")),
                "period": float(row.get("pl_orbper")) if pd.notnull(row.get("pl_orbper")) else None,
                "depth": float(row.get("pl_trandep")) / 1e6 if pd.notnull(row.get("pl_trandep")) else None,
                "duration": float(row.get("pl_trandurh")) / 24.0 if pd.notnull(row.get("pl_trandurh")) else None,
                "centroid_evidence": "",
                "contamination_evidence": f"crowding_metric: {row.get('st_rad', '')}",
                "source_checksum": csum,
                "ingestion_timestamp": ingest_time,
                "adapter_version": adapter_version,
                "notes": notes
            })
            evidence_id_counter += 1
    else:
        logger.warning(f"TOI catalog not found at {toi_path}")
        
    # ----------------------------------------------------
    # 2. Ingest TESS TCE Catalog (Sector 78 stats)
    # ----------------------------------------------------
    tce_path = Path(config.tce_catalog)
    if tce_path.exists():
        logger.info(f"Ingesting TESS Sector 78 TCE Catalog from {tce_path}...")
        csum = get_file_checksum(tce_path)
        df_tce = pd.read_csv(tce_path)
        
        for idx, row in df_tce.iterrows():
            raw_tic = row.get("ticid")
            tic_id = normalize_tic_id(raw_tic)
            if tic_id is None:
                continue
                
            tce_id = str(row.get("tceid", ""))
            fap = float(row.get("boot_fap", 1.0)) if pd.notnull(row.get("boot_fap")) else 1.0
            
            # Map candidate canonical label (TCE alone is weak candidate)
            label_candidate = "exoplanet_transit"
            strength = "weak"
            notes = f"Threshold Crossing Event (TCE) {tce_id} in Sector 78 with FAP {fap:.2e}"
            
            evidence_rows.append({
                "evidence_id": f"EVI-TCE-{evidence_id_counter:06d}",
                "tic_id": tic_id,
                "canonical_label_candidate": label_candidate,
                "original_label": "TCE",
                "original_disposition": "TCE",
                "source_catalog": "TESS_TCE_S0078",
                "source_version": "1.0",
                "source_row_identifier": tce_id,
                "evidence_level": "catalog_weak",
                "evidence_strength": strength,
                "disposition_date": str(row.get("lastUpdate", "")),
                "target_name": f"TIC-{tic_id}",
                "sector": 78,
                "ephemeris": str(row.get("tce_time0bt", "")),
                "period": float(row.get("tce_period")) if pd.notnull(row.get("tce_period")) else None,
                "depth": float(row.get("tce_depth")) / 1e6 if pd.notnull(row.get("tce_depth")) else None,
                "duration": float(row.get("tce_duration")) / 24.0 if pd.notnull(row.get("tce_duration")) else None,
                "centroid_evidence": f"dicco_msky: {row.get('tce_dicco_msky', '')}",
                "contamination_evidence": "",
                "source_checksum": csum,
                "ingestion_timestamp": ingest_time,
                "adapter_version": adapter_version,
                "notes": notes
            })
            evidence_id_counter += 1
    else:
        logger.warning(f"TCE catalog not found at {tce_path}")

    # ----------------------------------------------------
    # 3. Ingest Kepler Cumulative Catalog
    # ----------------------------------------------------
    cum_path = Path(config.cumulative_catalog)
    if cum_path.exists():
        logger.info(f"Ingesting Kepler Cumulative Catalog from {cum_path}...")
        csum = get_file_checksum(cum_path)
        df_cum = pd.read_csv(cum_path)
        
        # We need coordinates from discovery manifest to cross-match Kepler targets to TICs!
        discovery_path = manifests_dir / "discovery_manifest.parquet"
        df_disc_coords = pd.DataFrame(columns=["tic_id", "ra", "dec"])
        if discovery_path.exists():
            df_disc = pd.read_parquet(discovery_path)
            df_disc_coords = df_disc[["tic_id", "ra", "dec"]].drop_duplicates().copy()
            
        tol_deg = 15.0 / 3600.0  # 15 arcseconds in degrees
        
        for idx, row in df_cum.iterrows():
            kepid = row.get("kepid")
            kic_id = normalize_kic_id(kepid)
            if kic_id is None:
                continue
                
            disp = str(row.get("koi_disposition", "")).strip().upper()
            label_candidate = "unlabeled"
            strength = "none"
            
            if disp in ("CONFIRMED", "CANDIDATE"):
                label_candidate = "exoplanet_transit"
                strength = "strong" if disp == "CONFIRMED" else "medium"
            elif disp == "FALSE POSITIVE":
                # Check Kepler false positive flags
                ss = int(row.get("koi_fpflag_ss", 0))
                ec = int(row.get("koi_fpflag_ec", 0))
                co = int(row.get("koi_fpflag_co", 0))
                nt = int(row.get("koi_fpflag_nt", 0))
                
                if ss == 1 or ec == 1:
                    label_candidate = "eclipsing_binary"
                    strength = "strong"
                elif co == 1:
                    label_candidate = "blend_contamination"
                    strength = "strong"
                elif nt == 1:
                    label_candidate = "stellar_variability_or_other"
                    strength = "strong"
                else:
                    label_candidate = "stellar_variability_or_other"
                    strength = "medium"
                    
            # Check if this Kepler target coordinates match any of our TESS targets
            target_tic_id = None
            kra = float(row.get("ra", 0))
            kdec = float(row.get("dec", 0))
            
            # Fast vectorized bounding box filter
            subset = df_disc_coords[
                (df_disc_coords["ra"] >= kra - tol_deg) & (df_disc_coords["ra"] <= kra + tol_deg) &
                (df_disc_coords["dec"] >= kdec - tol_deg) & (df_disc_coords["dec"] <= kdec + tol_deg)
            ]
            
            for _, tc in subset.iterrows():
                dra = (tc["ra"] - kra) * np.cos(np.radians(kdec))
                ddec = tc["dec"] - kdec
                dist = np.sqrt(dra**2 + ddec**2) * 3600.0
                if dist <= 15.0:
                    target_tic_id = int(tc["tic_id"])
                    break
                    
            if target_tic_id is not None:
                evidence_rows.append({
                    "evidence_id": f"EVI-KEP-{evidence_id_counter:06d}",
                    "tic_id": target_tic_id,
                    "canonical_label_candidate": label_candidate,
                    "original_label": disp,
                    "original_disposition": disp,
                    "source_catalog": "Kepler_Cumulative",
                    "source_version": "DR25",
                    "source_row_identifier": str(kic_id),
                    "evidence_level": "catalog_authoritative",
                    "evidence_strength": strength,
                    "disposition_date": "",
                    "target_name": f"KIC-{kic_id}",
                    "sector": None,
                    "ephemeris": str(row.get("koi_time0bk", "")),
                    "period": float(row.get("koi_period")) if pd.notnull(row.get("koi_period")) else None,
                    "depth": float(row.get("koi_depth")) / 1e6 if pd.notnull(row.get("koi_depth")) else None,
                    "duration": float(row.get("koi_duration")) / 24.0 if pd.notnull(row.get("koi_duration")) else None,
                    "centroid_evidence": "",
                    "contamination_evidence": f"fpflag_co: {co}",
                    "source_checksum": csum,
                    "ingestion_timestamp": ingest_time,
                    "adapter_version": adapter_version,
                    "notes": f"Kepler target KIC-{kic_id} coordinate matched to TESS TIC-{target_tic_id}"
                })
                evidence_id_counter += 1
    else:
        logger.warning(f"Kepler Cumulative catalog not found at {cum_path}")

    # ----------------------------------------------------
    # 4. Ingest NASA Planets Catalog
    # ----------------------------------------------------
    planets_path = Path(config.planets_catalog)
    if planets_path.exists():
        logger.info(f"Ingesting NASA Planets Catalog from {planets_path}...")
        csum = get_file_checksum(planets_path)
        df_planets = pd.read_csv(planets_path)
        
        # Load TIC coordinates to match
        discovery_path = manifests_dir / "discovery_manifest.parquet"
        df_disc_coords = pd.DataFrame(columns=["tic_id", "ra", "dec"])
        if discovery_path.exists():
            df_disc = pd.read_parquet(discovery_path)
            df_disc_coords = df_disc[["tic_id", "ra", "dec"]].drop_duplicates().copy()
            
        tol_deg = 15.0 / 3600.0  # 15 arcseconds in degrees
        
        for idx, row in df_planets.iterrows():
            host = str(row.get("pl_hostname", ""))
            letter = str(row.get("pl_letter", ""))
            planet_id = f"{host} {letter}"
            
            pra = float(row.get("ra", 0))
            pdec = float(row.get("dec", 0))
            
            target_tic_id = None
            # Fast vectorized bounding box filter
            subset = df_disc_coords[
                (df_disc_coords["ra"] >= pra - tol_deg) & (df_disc_coords["ra"] <= pra + tol_deg) &
                (df_disc_coords["dec"] >= pdec - tol_deg) & (df_disc_coords["dec"] <= pdec + tol_deg)
            ]
            
            for _, tc in subset.iterrows():
                dra = (tc["ra"] - pra) * np.cos(np.radians(pdec))
                ddec = tc["dec"] - pdec
                dist = np.sqrt(dra**2 + ddec**2) * 3600.0
                if dist <= 15.0:
                    target_tic_id = int(tc["tic_id"])
                    break
                    
            if target_tic_id is not None:
                evidence_rows.append({
                    "evidence_id": f"EVI-PLANET-{evidence_id_counter:06d}",
                    "tic_id": target_tic_id,
                    "canonical_label_candidate": "exoplanet_transit",
                    "original_label": "CONFIRMED",
                    "original_disposition": "CONFIRMED",
                    "source_catalog": "NASA_Planets",
                    "source_version": "2025.02.03",
                    "source_row_identifier": planet_id,
                    "evidence_level": "catalog_authoritative",
                    "evidence_strength": "strong",
                    "disposition_date": str(row.get("rowupdate", "")),
                    "target_name": host,
                    "sector": None,
                    "ephemeris": "",
                    "period": float(row.get("pl_orbper")) if pd.notnull(row.get("pl_orbper")) else None,
                    "depth": float(row.get("pl_trandep")) / 1e6 if pd.notnull(row.get("pl_trandep")) else None,
                    "duration": float(row.get("pl_trandurh")) / 24.0 if pd.notnull(row.get("pl_trandurh")) else None,
                    "centroid_evidence": "",
                    "contamination_evidence": "",
                    "source_checksum": csum,
                    "ingestion_timestamp": ingest_time,
                    "adapter_version": adapter_version,
                    "notes": f"Confirmed planet {planet_id} coordinate matched to TESS TIC-{target_tic_id}"
                })
                evidence_id_counter += 1
    else:
        logger.warning(f"NASA Planets catalog not found at {planets_path}")

    # Build and save label_evidence.parquet
    if len(evidence_rows) > 0:
        df_evidence = pd.DataFrame(evidence_rows)
    else:
        # Create empty dataframe with schema
        df_evidence = pd.DataFrame(columns=[
            "evidence_id", "tic_id", "canonical_label_candidate", "original_label",
            "original_disposition", "source_catalog", "source_version", "source_row_identifier",
            "evidence_level", "evidence_strength", "disposition_date", "target_name", "sector",
            "ephemeris", "period", "depth", "duration", "centroid_evidence", "contamination_evidence",
            "source_checksum", "ingestion_timestamp", "adapter_version", "notes"
        ])

    # Canonical provenance aliases are materialized rather than inferred later.
    df_evidence["source_catalog_version"] = df_evidence["source_version"]
    df_evidence["source_record_identifier"] = df_evidence["source_row_identifier"]
    df_evidence["catalogue_checksum"] = df_evidence["source_checksum"]
    df_evidence["label_policy_version"] = config.label_policy_version
    df_evidence["provenance_reference"] = df_evidence["source_catalog"].map({
        "TESS_TOI": config.toi_catalog.name,
        "TESS_TCE_S0078": config.tce_catalog.name,
        "Kepler_Cumulative": config.cumulative_catalog.name,
        "NASA_Planets": config.planets_catalog.name,
    }).fillna("")
        
    output_path = manifests_dir / "label_evidence.parquet"
    df_evidence.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(df_evidence)} normalized evidence rows to {output_path}")
    return df_evidence
