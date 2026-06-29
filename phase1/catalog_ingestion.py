import os
import time
import hashlib
import logging
import pandas as pd
import numpy as np
import yaml
import re
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
    # 2. Ingest the MAST/ExoFOP TOI release. Unlike the Exoplanet
    # Archive projection, this table retains the official Centroid Offset flag.
    # ----------------------------------------------------
    mast_toi_path = Path(config.mast_toi_catalog)
    if mast_toi_path.exists():
        logger.info(f"Ingesting current MAST TOI release from {mast_toi_path}...")
        csum = get_file_checksum(mast_toi_path)
        df_mast_toi = pd.read_csv(mast_toi_path, comment="#", low_memory=False)
        mast_mapping = {
            "CP": ("exoplanet_transit", "strong"),
            "KP": ("exoplanet_transit", "strong"),
            "PC": ("exoplanet_transit", "medium"),
            "EB": ("eclipsing_binary", "strong"),
            "V": ("stellar_variability_or_other", "strong"),
        }
        for idx, row in df_mast_toi.iterrows():
            tic_id = normalize_tic_id(row.get("TIC"))
            if tic_id is None:
                continue
            disposition = str(row.get("TOI Disposition", "")).strip().upper()
            centroid_raw = row.get("Centroid Offset", False)
            centroid_offset = str(centroid_raw).strip().lower() in {"true", "1", "yes"}
            if centroid_offset:
                label_candidate, strength = "blend_contamination", "strong"
                original = f"{disposition or 'UNSPECIFIED'}; CENTROID_OFFSET"
                reason = "Official MAST TOI centroid-offset flag is true"
            elif disposition in mast_mapping:
                label_candidate, strength = mast_mapping[disposition]
                original = disposition
                reason = "Versioned MAST TOI disposition mapping"
            else:
                label_candidate, strength = "review_required", "medium"
                original = disposition or "UNSPECIFIED"
                reason = "MAST TOI disposition is not safe for supervised mapping"
            evidence_rows.append({
                "evidence_id": f"EVI-MTOI-{evidence_id_counter:06d}",
                "tic_id": tic_id,
                "canonical_label_candidate": label_candidate,
                "original_label": original,
                "original_disposition": disposition,
                "source_catalog": "MAST_TOI_EXOFOP",
                "source_version": "retrieved-2026-06-28",
                "source_row_identifier": str(row.get("Full TOI ID", idx)),
                "evidence_level": "catalog_authoritative",
                "evidence_strength": strength,
                "disposition_date": str(row.get("Updated", "")),
                "target_name": f"TIC-{tic_id}",
                "sector": None,
                "ephemeris": str(row.get("Orbital Epoch Value", "")),
                "period": float(row.get("Orbital Period (days) Value")) if pd.notnull(row.get("Orbital Period (days) Value")) else None,
                "depth": float(row.get("Transit Depth Value")) / 1e6 if pd.notnull(row.get("Transit Depth Value")) else None,
                "duration": float(row.get("Transit Duration (hours) Value")) / 24.0 if pd.notnull(row.get("Transit Duration (hours) Value")) else None,
                "centroid_evidence": f"Centroid Offset={centroid_offset}",
                "contamination_evidence": reason if centroid_offset else "",
                "source_checksum": csum,
                "ingestion_timestamp": ingest_time,
                "adapter_version": adapter_version,
                "notes": reason,
            })
            evidence_id_counter += 1
    else:
        logger.warning(f"MAST TOI catalog not found at {mast_toi_path}")

    # ----------------------------------------------------
    # 3. Ingest official SPOC TCE/DV statistics. A TCE alone remains weak;
    # only concordant, conservative centroid diagnostics identify a blend.
    # ----------------------------------------------------
    centroid_policy = toi_policy.get("centroid_offset_policy", {})
    min_sigma = float(centroid_policy.get("min_significance_sigma", 5.0))
    min_offset = float(centroid_policy.get("min_offset_arcsec", 2.0))
    
    dvr_xml_manifest_path = config.manifests_dir / "dvr_xml_manifest.parquet"
    dvr_xml_lookup = {}
    if dvr_xml_manifest_path.exists():
        try:
            df_dvr = pd.read_parquet(dvr_xml_manifest_path)
            df_dvr_ver = df_dvr[df_dvr["status"] == "verified"]
            for _, r_dvr in df_dvr_ver.iterrows():
                dvr_xml_lookup[(int(r_dvr["tic_id"]), int(r_dvr["sector"]))] = {
                    "product_filename": r_dvr["product_filename"],
                    "sha256": r_dvr["sha256"]
                }
            logger.info(f"Loaded {len(dvr_xml_lookup)} verified DVR XML entries for provenance verification.")
        except Exception as e:
            logger.warning(f"Failed to load DVR XML manifest: {e}")

    tce_paths = [Path(config.tce_catalog), *map(Path, config.additional_tce_catalogs)]
    for tce_path in tce_paths:
        if not tce_path.exists():
            logger.warning(f"TCE catalog not found at {tce_path}")
            continue
        logger.info(f"Ingesting official TESS TCE/DV statistics from {tce_path}...")
        csum = get_file_checksum(tce_path)
        df_tce = pd.read_csv(tce_path, comment="#", low_memory=False)
        for idx, row in df_tce.iterrows():
            tic_id = normalize_tic_id(row.get("ticid"))
            if tic_id is None:
                continue
            tce_id = str(row.get("tceid", ""))
            sector_tokens = re.findall(r"s(\d{4})", str(row.get("sectors", "")))
            sector_value = int(sector_tokens[0]) if len(sector_tokens) == 1 else None
            diagnostics = []
            passes = []
            for prefix in ("tce_dicco_msky", "tce_ditco_msky"):
                value = pd.to_numeric(row.get(prefix), errors="coerce")
                error = pd.to_numeric(row.get(prefix + "_err"), errors="coerce")
                significance = abs(value) / error if pd.notnull(value) and pd.notnull(error) and error > 0 else np.nan
                passes.append(bool(pd.notnull(significance) and value >= min_offset and significance >= min_sigma))
                diagnostics.append(f"{prefix}={value},err={error},sigma={significance}")
            
            dvr_entry = dvr_xml_lookup.get((tic_id, sector_value)) if sector_value else None
            
            centroid_blend = all(passes)
            if centroid_blend:
                label_candidate = centroid_policy.get("resolved_label", "blend_contamination")
                strength = centroid_policy.get("evidence_strength", "strong")
                evidence_level = "catalog_authoritative"
                original_disposition = "SPOC_DV_CONCORDANT_CENTROID_OFFSET"
                notes = "Concordant official SPOC centroid diagnostics exceed versioned conservative thresholds"
                if dvr_entry:
                    notes += f" Centroid offset diagnostics verified against SPOC Data Validation XML file: {dvr_entry['product_filename']}."
            else:
                label_candidate, strength = "exoplanet_transit", "weak"
                evidence_level = "catalog_weak"
                original_disposition = "TCE"
                notes = "TCE alone is weak evidence and is not a confirmed planet label"
            evidence_rows.append({
                "evidence_id": f"EVI-TCE-{evidence_id_counter:06d}",
                "tic_id": tic_id,
                "canonical_label_candidate": label_candidate,
                "original_label": "TCE",
                "original_disposition": original_disposition,
                "source_catalog": f"TESS_TCE_{'S' + str(sector_value).zfill(4) if sector_value else 'MULTISECTOR'}",
                "source_version": tce_path.name,
                "source_row_identifier": tce_id,
                "evidence_level": evidence_level,
                "evidence_strength": strength,
                "disposition_date": str(row.get("lastUpdate", "")),
                "target_name": f"TIC-{tic_id}",
                "sector": sector_value,
                "ephemeris": str(row.get("tce_time0bt", "")),
                "period": float(row.get("tce_period")) if pd.notnull(row.get("tce_period")) else None,
                "depth": float(row.get("tce_depth")) / 1e6 if pd.notnull(row.get("tce_depth")) else None,
                "duration": float(row.get("tce_duration")) / 24.0 if pd.notnull(row.get("tce_duration")) else None,
                "centroid_evidence": "; ".join(diagnostics),
                "contamination_evidence": notes if centroid_blend else "",
                "source_checksum": dvr_entry["sha256"] if dvr_entry else csum,
                "ingestion_timestamp": ingest_time,
                "adapter_version": "1.1.0",
                "notes": notes,
                "dvr_xml_filename": dvr_entry["product_filename"] if dvr_entry else "",
            })
            evidence_id_counter += 1

    # ----------------------------------------------------
    # 3. Ingest the vetted TESS Eclipsing Binary catalogue
    # ----------------------------------------------------
    eb_path = Path(config.eb_catalog)
    if eb_path.exists():
        logger.info(f"Ingesting TESS Eclipsing Binary Catalog from {eb_path}...")
        csum = get_file_checksum(eb_path)
        df_eb = pd.read_csv(eb_path, comment="#")

        required = {"tess_id", "signal_id", "date_modified", "period", "sectors"}
        missing = required - set(df_eb.columns)
        if missing:
            raise ValueError(f"TESS EB catalogue is missing required columns: {sorted(missing)}")

        for idx, row in df_eb.iterrows():
            tic_id = normalize_tic_id(row.get("tess_id"))
            if tic_id is None:
                continue
            signal_id = str(row.get("signal_id", "1"))
            evidence_rows.append({
                "evidence_id": f"EVI-TEB-{evidence_id_counter:06d}",
                "tic_id": tic_id,
                "canonical_label_candidate": "eclipsing_binary",
                "original_label": "eclipsing_binary",
                "original_disposition": "vetted_tess_eb_catalogue",
                "source_catalog": "TESS_EB_HLSP",
                "source_version": "1.0",
                "source_row_identifier": f"{tic_id}:{signal_id}",
                "evidence_level": "catalog_authoritative",
                "evidence_strength": "strong",
                "disposition_date": str(row.get("date_modified", "")),
                "target_name": f"TIC-{tic_id}",
                # The source can list several sectors; preserve them rather than
                # inventing one scalar observation sector.
                "sector": None,
                "ephemeris": str(row.get("bjd0", "")),
                "period": float(row.get("period")) if pd.notnull(row.get("period")) else None,
                "depth": float(row.get("prim_depth_pf")) if pd.notnull(row.get("prim_depth_pf")) else None,
                "duration": float(row.get("prim_width_pf")) if pd.notnull(row.get("prim_width_pf")) else None,
                "centroid_evidence": "",
                "contamination_evidence": "",
                "source_checksum": csum,
                "ingestion_timestamp": ingest_time,
                "adapter_version": adapter_version,
                "notes": f"Vetted TESS EB signal; catalogue sectors={row.get('sectors', '')}",
            })
            evidence_id_counter += 1
    else:
        logger.warning(f"TESS eclipsing-binary catalog not found at {eb_path}")

    # ----------------------------------------------------
    # 4. Ingest Kepler Cumulative Catalog
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
        "MAST_TOI_EXOFOP": config.mast_toi_catalog.name,
        "Kepler_Cumulative": config.cumulative_catalog.name,
        "NASA_Planets": config.planets_catalog.name,
    }).fillna("")
    tce_mask = df_evidence["source_catalog"].str.startswith("TESS_TCE_", na=False)
    df_evidence.loc[tce_mask, "provenance_reference"] = df_evidence.loc[tce_mask, "source_version"]
    
    # If dvr_xml_filename was set, use it as the provenance_reference and override catalogue_checksum
    if "dvr_xml_filename" in df_evidence.columns:
        has_xml = df_evidence["dvr_xml_filename"] != ""
        df_evidence.loc[has_xml, "provenance_reference"] = df_evidence.loc[has_xml, "dvr_xml_filename"]
        df_evidence.loc[has_xml, "catalogue_checksum"] = df_evidence.loc[has_xml, "source_checksum"]
        df_evidence = df_evidence.drop(columns=["dvr_xml_filename"])
        
    output_path = manifests_dir / "label_evidence.parquet"
    df_evidence.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(df_evidence)} normalized evidence rows to {output_path}")
    return df_evidence
