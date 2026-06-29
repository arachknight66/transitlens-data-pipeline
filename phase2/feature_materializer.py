"""Leakage-safe Phase 2 feature materialization using detected ephemerides only."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import logging
import os
import tempfile

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

IDENTITY_COLUMNS = [
    "tic_id", "observation_id", "sector", "split", "source_checksum",
    "diagnostics_version", "feature_schema_version", "ephemeris_mode",
    "candidate_detected", "diagnostic_status", "diagnostic_failure_reason",
]

# All are inference-time measurements or explicit availability states. Truth and
# catalogue provenance stay in the separate metadata table.
FEATURE_COLUMNS = [
    "bls_power", "snr", "period_days", "duration_days", "depth", "transit_count",
    "alias_warning", "odd_even_available", "odd_event_count", "even_event_count",
    "odd_depth", "odd_depth_uncertainty", "even_depth", "even_depth_uncertainty",
    "odd_even_depth_difference", "odd_even_fractional_difference",
    "odd_even_significance", "odd_even_p_value", "odd_even_evidence_flag",
    "secondary_available", "secondary_phase", "secondary_depth",
    "secondary_depth_uncertainty", "secondary_significance",
    "secondary_primary_depth_ratio", "secondary_delta_bic", "secondary_global_p_value",
    "secondary_evidence_flag", "morphology_available", "trapezoid_depth",
    "trapezoid_duration", "ingress_duration", "egress_duration", "ingress_fraction",
    "egress_fraction", "ingress_egress_asymmetry", "flat_bottom_duration",
    "v_shape_score", "grazing_probability_proxy", "morphology_fit_quality",
    "morphology_evidence_flag", "harmonic_available", "orbital_amplitude",
    "orbital_amplitude_uncertainty", "first_harmonic_amplitude",
    "first_harmonic_uncertainty", "ellipsoidal_amplitude", "ellipsoidal_significance",
    "reflection_amplitude", "reflection_significance", "beaming_amplitude",
    "beaming_significance", "harmonic_delta_bic", "harmonic_evidence_flag",
    "centroid_available", "centroid_shift_column_pixels", "centroid_shift_row_pixels",
    "centroid_shift_pixels", "centroid_shift_arcsec", "centroid_shift_uncertainty_pixels",
    "centroid_shift_significance", "centroid_mahalanobis_distance",
    "centroid_permutation_p_value", "centroid_points_in", "centroid_points_out",
    "centroid_evidence_flag", "difference_image_available", "difference_image_snr",
    "source_target_offset_pixels", "source_target_offset_arcsec",
    "source_target_offset_uncertainty_pixels", "source_target_offset_significance",
    "difference_flux", "difference_flux_uncertainty", "difference_image_evidence_flag",
    "gaia_available", "gaia_neighbor_count", "nearest_neighbor_sep_arcsec",
    "nearest_neighbor_delta_gmag", "nearest_neighbor_delta_tmag",
    "summed_neighbor_flux_ratio", "aperture_weighted_neighbor_flux_ratio",
    "gaia_evidence_flag", "crowding_available", "crowdsap", "flfrcsap",
    "contamination_fraction", "observed_depth", "observed_depth_uncertainty",
    "dilution_corrected_depth", "dilution_corrected_depth_uncertainty",
    "correction_factor", "crowding_evidence_flag", "multi_aperture_available",
    "aperture_count", "aperture_depth_slope", "aperture_depth_slope_uncertainty",
    "aperture_depth_consistency_chi2", "aperture_depth_consistency_p_value",
    "multi_aperture_evidence_flag", "eb_risk_score", "blend_risk_score",
    "independent_eb_evidence_count", "independent_blend_evidence_count", "review_required",
]

BOOLEAN_FEATURES = {name for name in FEATURE_COLUMNS if name.endswith("_available") or name.endswith("_flag")} | {
    "alias_warning", "candidate_detected", "review_required"
}

def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def _best_observation_per_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Label-independent selection: usable fraction, points, cadence, observation id."""
    ranked = frame.copy()
    ranked["_usable"] = pd.to_numeric(ranked.get("usable_fraction"), errors="coerce").fillna(-1)
    ranked["_points"] = pd.to_numeric(ranked.get("n_points_usable"), errors="coerce").fillna(-1)
    ranked["_cadence"] = pd.to_numeric(ranked.get("median_cadence_seconds"), errors="coerce").fillna(np.inf)
    ranked = ranked.sort_values(["tic_id", "_usable", "_points", "_cadence", "observation_id"],
                                ascending=[True, False, False, True, True])
    return ranked.drop_duplicates("tic_id", keep="first").drop(columns=["_usable", "_points", "_cadence"])

def _process_observation(row: dict, ml_core: str, tpf_map: dict, diagnostics_config: dict) -> tuple[dict, dict]:
    import sys
    if ml_core not in sys.path: sys.path.insert(0, ml_core)
    from core.bls_detector import detect
    from diagnostics import run_diagnostics

    feature = {name: None for name in IDENTITY_COLUMNS + FEATURE_COLUMNS}
    for name in BOOLEAN_FEATURES: feature[name] = False
    feature.update({
        "tic_id": int(row["tic_id"]), "observation_id": str(row["observation_id"]),
        "sector": int(row["sector"]), "split": str(row["split"]),
        "source_checksum": str(row.get("processed_sha256") or ""),
        "diagnostics_version": str(diagnostics_config.get("General", {}).get("diagnostics_version", "2.0.0")),
        "feature_schema_version": "phase2-features-2.1.1", "ephemeris_mode": "detected",
        "diagnostic_status": "unavailable", "diagnostic_failure_reason": "",
    })
    metadata = {
        **{name: feature[name] for name in IDENTITY_COLUMNS},
        "target_id": str(row.get("target_id", f"TIC-{row['tic_id']}")),
        "canonical_label": str(row["resolved_label"]),
        "label_strength": str(row.get("label_strength", "")),
        "evidence_level": str(row.get("evidence_level", "")),
        "label_policy_version": str(row.get("label_policy_version", "")),
        "source_type": "real_tess_spoc",
        "aggregation_policy": "best_observation_label_independent_v1",
    }
    try:
        path = Path(row["processed_path"])
        if not path.exists(): raise FileNotFoundError(f"processed light curve missing: {path}")
        with np.load(path) as data:
            time = np.asarray(data["time"], dtype=float)
            flux = np.asarray(data["flux"], dtype=float)
            centroid_x = np.asarray(data["centroid_column"], dtype=float) if "centroid_column" in data else None
            centroid_y = np.asarray(data["centroid_row"], dtype=float) if "centroid_row" in data else None
            quality = np.asarray(data["quality"], dtype=int) if "quality" in data else None
        bls = detect(time, flux, diagnostics_config.get("BLS", {}))
        feature.update({"bls_power": float(bls.bls_power_peak), "snr": float(bls.snr),
                        "alias_warning": bool(bls.alias_warning), "candidate_detected": bool(bls.candidate_detected)})
        if not bls.candidate_detected:
            feature["diagnostic_status"] = "low_snr"
            feature["diagnostic_failure_reason"] = bls.detection_reason
            feature["review_required"] = True
            metadata.update({k: feature[k] for k in IDENTITY_COLUMNS})
            return feature, metadata
        duration = float(bls.best_duration); period = float(bls.best_period); epoch = float(bls.best_t0); depth = float(bls.best_depth)
        feature.update({"period_days": period, "duration_days": duration, "depth": depth,
                        "transit_count": int(np.unique(np.rint((time - epoch) / period)).size)})
        key = f"{int(row['tic_id'])}:{int(row['sector'])}"
        diag_metadata = {
            "target_id": metadata["target_id"], "tic_id": int(row["tic_id"]), "sector": int(row["sector"]),
            "observation_id": str(row["observation_id"]), "fits_filename": str(row.get("raw_path", "")),
            "source_checksum": feature["source_checksum"], "ephemeris_mode": "detected",
            "ephemeris_source": "transitlens_bls", "ra": _finite_or_none(row.get("ra")),
            "dec": _finite_or_none(row.get("dec")), "crowding_metric": _finite_or_none(row.get("crowding_metric")),
            "flux_fraction": _finite_or_none(row.get("flux_fraction")), "tpf_path": tpf_map.get(key),
        }
        diag = run_diagnostics(time, flux, period, epoch, duration, depth, centroid_x, centroid_y, quality,
                               diag_metadata, diagnostics_config)
        for name in FEATURE_COLUMNS:
            if name in diag: feature[name] = diag[name]
        feature.update({"bls_power": float(bls.bls_power_peak), "snr": float(bls.snr), "period_days": period,
                        "duration_days": duration, "depth": depth, "candidate_detected": True,
                        "alias_warning": bool(bls.alias_warning), "diagnostic_status": "success"})
    except Exception as exc:
        logger.exception("Phase 2 observation failed: %s", row.get("observation_id"))
        feature["diagnostic_status"] = "numerical_failure"
        feature["diagnostic_failure_reason"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        feature["review_required"] = True
    metadata.update({k: feature[k] for k in IDENTITY_COLUMNS})
    return feature, metadata

def _finite_or_none(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None

def materialize_features(config, limit: int | None = None, *, workers: int = 1,
                         split: str | None = None, resume: bool = False,
                         ephemeris_mode: str = "detected", dry_run: bool = False) -> dict:
    if ephemeris_mode != "detected":
        raise ValueError("official Phase 2 materialization permits only detected ephemerides")
    m = config.manifests_dir
    obs = pd.read_parquet(m / "observation_manifest.parquet")
    assignments = pd.read_parquet(m / "split_manifest.parquet")
    merged = obs.drop(columns=["split", "canonical_label", "resolved_label"], errors="ignore").merge(
        assignments[["tic_id", "split", "resolved_label"]], on="tic_id", how="inner"
    )
    allowed = {"train", "val", "test"}
    merged = merged[(merged["split"].isin(allowed)) & (merged["parse_status"] == "success") &
                    (merged["resolved_label"].isin(["exoplanet_transit", "eclipsing_binary",
                                                    "blend_contamination", "stellar_variability_or_other"]))]
    if split:
        normalized = "val" if split in {"val", "validation"} else split
        if normalized not in allowed: raise ValueError("split must be train, validation, or test")
        merged = merged[merged["split"] == normalized]
    selected = _best_observation_per_target(merged)
    if limit is not None: selected = selected.head(limit)
    if dry_run:
        return {"status": "DEVELOPMENT_ONLY", "selected_targets": len(selected),
                "class_counts": selected["resolved_label"].value_counts().to_dict()}

    cfg_path = config.REPO_ROOT / "transitlens-ml-core" / "config" / "phase2_diagnostics.yaml"
    import yaml
    diagnostics_config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    diagnostics_config.setdefault("Gaia", {})["offline_only"] = True
    diagnostics_config.setdefault("General", {})["development_limit"] = limit is not None
    tpf_map = {}
    tpf_manifest = m / "tpf_discovery_manifest.parquet"
    if tpf_manifest.exists():
        tpf = pd.read_parquet(tpf_manifest)
        for _, item in tpf.iterrows():
            candidate = item.get("local_path") or item.get("tpf_path")
            if candidate and Path(str(candidate)).exists(): tpf_map[f"{int(item.tic_id)}:{int(item.sector)}"] = str(candidate)

    rows = selected.to_dict("records")
    ml_core = str(config.REPO_ROOT / "transitlens-ml-core")
    process = lambda row: _process_observation(row, ml_core, tpf_map, diagnostics_config)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool: results = list(pool.map(process, rows))
    else:
        results = [process(row) for row in rows]
    feature_frame = pd.DataFrame([item[0] for item in results], columns=IDENTITY_COLUMNS + FEATURE_COLUMNS)
    metadata_frame = pd.DataFrame([item[1] for item in results])

    output_dir = m if limit is None else m / "phase2_development"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = {"train": "phase2_features_train.parquet", "val": "phase2_features_validation.parquet", "test": "phase2_features_test.parquet"}
    for split_name, output_name in output_names.items():
        part = feature_frame[feature_frame["split"] == split_name].copy()
        _atomic_parquet(part, output_dir / output_name)
    _atomic_parquet(metadata_frame, output_dir / "phase2_feature_metadata.parquet")
    diagnostics_frame = feature_frame.merge(metadata_frame[["observation_id", "canonical_label"]], on="observation_id", how="left", validate="one_to_one")
    _atomic_parquet(diagnostics_frame, output_dir / "per_target_diagnostics.parquet")

    order = FEATURE_COLUMNS
    schema = {name: ("boolean" if name in BOOLEAN_FEATURES else "nullable_float64") for name in order}
    units = {name: _unit_for(name) for name in order}
    _atomic_text(json.dumps(order, indent=2), output_dir / "phase2_feature_order.json")
    _atomic_text(json.dumps(schema, indent=2), output_dir / "phase2_feature_schema.json")
    _atomic_text(json.dumps(units, indent=2), output_dir / "phase2_feature_units.json")
    integrity = {
        "representation": "one_best_observation_per_tic_label_independent_v1",
        "ephemeris_mode": "detected", "development_only": limit is not None,
        "train_count": int((feature_frame.split == "train").sum()),
        "val_count": int((feature_frame.split == "val").sum()),
        "test_count": int((feature_frame.split == "test").sum()),
        "unique_tics": int(feature_frame.tic_id.nunique()),
        "overlaps": _overlaps(feature_frame),
    }
    _atomic_text(json.dumps(integrity, indent=2), output_dir / "phase2_split_integrity.json")
    config_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    card = f"""# Phase 2 feature card

- Version: phase2-features-2.1.1
- Rows: one label-independent best observation per TIC
- Ephemeris: TransitLens BLS detected only
- Features: {len(FEATURE_COLUMNS)} ordered measurements and availability flags
- Missingness: null/NaN plus explicit availability flags; no imputation
- Labels: stored only in `phase2_feature_metadata.parquet`
- Configuration SHA-256: `{config_hash}`
- Development-only: `{limit is not None}`
"""
    _atomic_text(card, output_dir / "phase2_feature_card.md")
    return {"status": "DEVELOPMENT_ONLY" if limit is not None else "COMPLETE",
            "features_count": len(feature_frame), "output_dir": str(output_dir), "split_integrity": integrity,
            "diagnostic_status": feature_frame.diagnostic_status.value_counts().to_dict()}

def _overlaps(frame: pd.DataFrame) -> dict:
    sets = {name: set(frame.loc[frame.split == name, "tic_id"]) for name in ("train", "val", "test")}
    return {"train_validation": len(sets["train"] & sets["val"]), "train_test": len(sets["train"] & sets["test"]),
            "validation_test": len(sets["val"] & sets["test"])}

def _unit_for(name: str) -> str:
    if name.endswith("_days") or name in {"period_days", "duration_days"}: return "day"
    if name.endswith("_arcsec"): return "arcsec"
    if "pixels" in name: return "pixel"
    if name.endswith("_p_value") or name.endswith("_ratio") or name.endswith("_score"): return "dimensionless"
    if name.endswith("_available") or name.endswith("_flag") or name in {"alias_warning", "review_required"}: return "boolean"
    return "dimensionless"
