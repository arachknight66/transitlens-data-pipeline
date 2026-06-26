"""
interface.py
─────────────
Single entry-point module that `transitlens-ml-core` imports from.

This is the ONLY file in `transitlens-data-pipeline` that ml-core is
allowed to depend on. Everything else (synthetic/, real_tess/,
datasets/) is an internal implementation detail that can change
without breaking the contract, as long as `load_light_curve()` keeps
returning the shape documented below.

Output contract
----------------
{
    "time":      List[float],        # BTJD timestamps, length N
    "flux":      List[float],        # normalised flux, median ~1.0, length N
    "target_id": str,
    "source":    str,                # "synthetic" | "tess" | "csv"
    "n_points":  int,                # len(time) == len(flux)
    "metadata": {
        "cadence_min":    float,
        "time_span_days": float,
        "sector":         int | None,
        "label":          str | None,
        "true_period":    float | None,
        "true_depth":     float | None,
        "true_duration":  float | None,
    }
}

`time` and `flux` are always plain Python lists of floats (not numpy
arrays) so the result is JSON-serialisable.
"""

import json
import os

import numpy as np
import pandas as pd
import yaml


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

_VALID_SOURCES = ("synthetic", "tess", "csv", "fits")
_VALID_LABELS = {
    "exoplanet_transit",
    "eclipsing_binary",
    "blend_contamination",
    "stellar_variability_or_other",
    "exoplanet_like",
    "eclipsing_binary_like",
    "noise_or_other",
}

_ALIAS_MAP = {
    "exoplanet_like": "exoplanet_transit",
    "eclipsing_binary_like": "eclipsing_binary",
    "noise_or_other": "stellar_variability_or_other",
}


# ─────────────────────────────────────────────
# Exception Hierarchy
# ─────────────────────────────────────────────

class DataPipelineError(Exception):
    """Base exception for all transitlens-data-pipeline errors."""
    pass

class DataShapeError(DataPipelineError):
    pass

class DataQualityError(DataPipelineError):
    pass

class DataNormalisationError(DataPipelineError):
    pass

class InvalidSourceError(DataPipelineError):
    pass

class InvalidLabelError(DataPipelineError):
    pass


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def load_light_curve(source, target_id, config=None):
    """
    Single entry point consumed by transitlens-ml-core.

    Parameters
    ----------
    source : str
        One of "synthetic", "tess", or "csv".
        "synthetic" — loads from synthetic/cases/{target_id}.csv
        "tess"      — loads from real_tess/cache/ via mast_loader
        "csv"       — loads from an arbitrary file path passed in config["path"]

    target_id : str
        For synthetic: one of "candidate_a", "candidate_b", "candidate_c"
        For tess: TIC ID string, e.g. "TIC-25155310"
        For csv: descriptive name chosen by caller

    config : dict, optional
        Override any default parameters.
        Valid keys:
            "path"         — used when source="csv"
            "sector"       — override sector selection for TESS
            "cadence_min"  — override cadence assumption
            "generate"     — if True and synthetic case CSV missing, generate it

    Returns
    -------
    dict matching the output contract documented at the top of this file.

    Raises
    ------
    FileNotFoundError  — source file or cache entry not found
    ValueError         — unknown source type or malformed CSV
    ImportError        — real_tess source requested but lightkurve not installed
    """
    config = config or {}

    if source == "synthetic":
        return _load_synthetic(target_id, config)
    elif source == "tess":
        return _load_tess(target_id, config)
    elif source == "csv":
        return _load_csv(target_id, config)
    elif source == "fits":
        return _load_fits(target_id, config)
    else:
        raise InvalidSourceError(
            f"Unknown source: {source!r}. Must be one of {_VALID_SOURCES}."
        )


# ─────────────────────────────────────────────
# source = "synthetic"
# ─────────────────────────────────────────────

def _load_synthetic(target_id, config):
    cases_dir = config.get(
        "cases_dir", os.path.join(_REPO_ROOT, "synthetic", "cases")
    )
    config_path = config.get(
        "config_path", os.path.join(_REPO_ROOT, "synthetic", "config.yaml")
    )
    metadata_path = config.get(
        "metadata_path", os.path.join(_REPO_ROOT, "datasets", "metadata.json")
    )

    csv_path = os.path.join(cases_dir, f"{target_id}.csv")

    if not os.path.exists(csv_path):
        if config.get("generate", False):
            _generate_synthetic_case(target_id, config_path, cases_dir)
        else:
            raise FileNotFoundError(
                f"Synthetic case '{target_id}' not found at {csv_path}. "
                f"Pass config={{'generate': True}} to auto-generate it, or "
                f"run synthetic.generator.generate_all_cases() first."
            )

    df = _read_light_curve_csv(csv_path)

    time = df["time"].astype(float).tolist()
    flux = df["flux"].astype(float).tolist()

    metadata = _resolve_synthetic_metadata(target_id, metadata_path, config_path)

    return _build_result(time, flux, target_id, "synthetic", metadata)


def _generate_synthetic_case(target_id, config_path, cases_dir):
    """
    Self-healing fallback: if a synthetic case CSV is missing and the
    caller passed config={'generate': True}, generate it on the fly
    using the same generator as Phase 1.
    """
    # Imported lazily so importing interface.py never requires numpy's
    # generation dependencies unless a case is actually missing.
    from synthetic.generator import generate_from_config

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    if target_id not in full_config["cases"]:
        raise FileNotFoundError(
            f"Cannot auto-generate '{target_id}': no such case in {config_path}."
        )

    time, flux, _ = generate_from_config(full_config, target_id)

    os.makedirs(cases_dir, exist_ok=True)
    csv_path = os.path.join(cases_dir, f"{target_id}.csv")
    pd.DataFrame({"time": time, "flux": flux}).to_csv(csv_path, index=False)


def _resolve_synthetic_metadata(target_id, metadata_path, config_path):
    """
    Loads metadata for a synthetic target_id.

    Prefers datasets/metadata.json (written by Phase 2's
    build_from_synthetic) since it is the canonical source of truth.
    Falls back to reading synthetic/config.yaml directly — this keeps
    load_light_curve() working even for a freshly auto-generated case
    that hasn't been folded into metadata.json yet.
    """
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            all_metadata = json.load(f)
        if target_id in all_metadata:
            entry = all_metadata[target_id]
            return {
                "cadence_min": entry["cadence_min"],
                "time_span_days": entry["time_span_days"],
                "sector": entry["sector"],
                "label": entry["label"],
                "true_period": entry["true_period"],
                "true_depth": entry["true_depth"],
                "true_duration": entry["true_duration"],
            }

    # Fall back to config.yaml directly
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    if target_id not in full_config["cases"]:
        raise FileNotFoundError(
            f"No metadata available for '{target_id}': not present in "
            f"{metadata_path} or in {config_path}."
        )

    gen = full_config["generation"]
    case = full_config["cases"][target_id]

    return {
        "cadence_min": gen["cadence_minutes"],
        "time_span_days": gen["time_span_days"],
        "sector": None,
        "label": case.get("label"),
        "true_period": case.get("period_days"),
        "true_depth": case.get("depth"),
        "true_duration": case.get("duration_days"),
    }


# ─────────────────────────────────────────────
# source = "tess"
# ─────────────────────────────────────────────

def _load_tess(target_id, config):
    """
    Loads a real TESS light curve via Lightkurve/MAST.

    This is the Phase 5 stretch-goal path. It is wired up here so
    ml-core can call load_light_curve("tess", ...) without caring
    whether Phase 5 has been implemented yet, but the heavy lifting
    lives in real_tess/mast_loader.py.
    """
    try:
        import lightkurve  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "source='tess' requires the 'lightkurve' package, which is not "
            "installed. Uncomment lightkurve/astroquery in requirements.txt "
            "and `pip install -r requirements.txt` to enable real TESS data."
        ) from exc

    try:
        from real_tess.mast_loader import fetch_light_curve
    except ImportError as exc:
        raise ImportError(
            "real_tess.mast_loader could not be imported."
        ) from exc

    sector = config.get("sector")
    cache_dir = config.get(
        "cache_dir", os.path.join(_REPO_ROOT, "real_tess", "cache")
    )

    time, flux, resolved_sector = fetch_light_curve(
        target_id, sector=sector, cache_dir=cache_dir
    )

    metadata = {
        "cadence_min": config.get("cadence_min", 2.0),
        "time_span_days": float(time[-1] - time[0]) if len(time) else 0.0,
        "sector": resolved_sector,
        "label": config.get("label"),
        "true_period": config.get("true_period"),
        "true_depth": config.get("true_depth"),
        "true_duration": config.get("true_duration"),
    }

    return _build_result(
        list(map(float, time)), list(map(float, flux)),
        target_id, "tess", metadata,
    )


# ─────────────────────────────────────────────
# source = "csv"
# ─────────────────────────────────────────────

def _load_csv(target_id, config):
    """
    Loads a light curve from an arbitrary CSV path passed via
    config["path"]. Useful for ad-hoc files (e.g. a light curve
    exported from another tool) without going through synthetic
    generation or the MAST pipeline.
    """
    path = config.get("path")
    if not path:
        raise ValueError(
            "source='csv' requires config={'path': '/path/to/file.csv'}."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}.")

    df = _read_light_curve_csv(path)

    time = df["time"].astype(float).tolist()
    flux = df["flux"].astype(float).tolist()

    cadence_min = config.get("cadence_min")
    if cadence_min is None and len(time) > 1:
        # Infer cadence from the median spacing between points (in minutes).
        cadence_min = float(np.median(np.diff(time)) * 1440.0)

    time_span_days = float(time[-1] - time[0]) if len(time) > 1 else 0.0

    metadata = {
        "cadence_min": cadence_min,
        "time_span_days": time_span_days,
        "sector": config.get("sector"),
        "label": config.get("label"),
        "true_period": config.get("true_period"),
        "true_depth": config.get("true_depth"),
        "true_duration": config.get("true_duration"),
    }

    return _build_result(time, flux, target_id, "csv", metadata)


# ─────────────────────────────────────────────
# source = "fits"
# ─────────────────────────────────────────────

def _load_fits(target_id, config):
    """
    Loads a light curve directly from a local FITS file.
    """
    path = config.get("path")
    if not path:
        raise ValueError("source='fits' requires config={'path': '/path/to/file.fits'}.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"FITS file not found at {path}.")
        
    from real_tess.fits_parser import load_fits_and_normalize
    
    try:
        parsed = load_fits_and_normalize(path, config)
    except Exception as e:
        raise DataPipelineError(f"Failed to parse FITS: {e}")
        
    resolved_target_id = target_id if target_id and target_id != "unknown" else (parsed.get("target_id") or "unknown")
    
    return _build_result(
        parsed["time"],
        parsed["flux"],
        resolved_target_id,
        "fits",
        parsed["metadata"]
    )



# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _read_light_curve_csv(csv_path):
    df = pd.read_csv(csv_path)
    missing = {"time", "flux"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required column(s) {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def _build_result(time, flux, target_id, source, metadata):
    if len(time) != len(flux):
        raise DataShapeError(
            f"time and flux length mismatch for '{target_id}': "
            f"{len(time)} vs {len(flux)}."
        )

    if any(time[i] >= time[i+1] for i in range(len(time)-1)):
        raise DataQualityError("time array must be monotonically increasing.")

    if abs(np.nanmedian(flux) - 1.0) >= 0.01:
        raise DataNormalisationError(
            f"Flux median must be ~1.0, got {np.nanmedian(flux)}"
        )

    if source not in _VALID_SOURCES:
        raise InvalidSourceError(f"Unknown source: {source!r}. Must be one of {_VALID_SOURCES}.")

    label = metadata.get("label")
    if label is not None:
        label = _ALIAS_MAP.get(label, label)
        metadata["label"] = label
        if label not in _VALID_LABELS:
            raise InvalidLabelError(
                f"Invalid label '{label}' for '{target_id}'. "
                f"Must be one of {sorted(_VALID_LABELS)} or None."
            )

    return {
        "time": time,
        "flux": flux,
        "target_id": target_id,
        "source": source,
        "n_points": len(time),
        "metadata": metadata,
    }