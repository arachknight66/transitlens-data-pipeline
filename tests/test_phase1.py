import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits

from phase1.config import Config
import phase1.fits_parser as parser
import phase1.label_resolver as resolver
import phase1.deduplication as dedup
import phase1.split_builder as splits
import phase1.validation as validator
import phase1.checksums as csums

def make_dummy_fits(path, tic_id, sector, exptime=120.0, cadence_diff_sec=120.0, n_points=200, corrupt=False):
    """Helper to write a valid or corrupt TESS-like dummy FITS file."""
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["OBJECT"] = f"TIC {tic_id}"
    primary_hdu.header["TICID"] = tic_id
    primary_hdu.header["SECTOR"] = sector
    primary_hdu.header["CAMERA"] = 1
    primary_hdu.header["CCD"] = 2
    primary_hdu.header["RA_OBJ"] = 100.0
    primary_hdu.header["DEC_OBJ"] = -45.0
    primary_hdu.header["TESSMAG"] = 10.5
    primary_hdu.header["EXPTIME"] = exptime
    primary_hdu.header["DATE"] = "2026-06-28"
    primary_hdu.header["PROCVER"] = "1.0"
    primary_hdu.header["DATA_REL"] = 1
    primary_hdu.header["OBSID"] = f"tess_obs_{tic_id}_{sector}"
    primary_hdu.header["TELESCOP"] = "TESS"
    
    if corrupt:
        hdul = fits.HDUList([primary_hdu])
        hdul.writeto(path, overwrite=True)
        return

    # Create table columns
    time = 2460000.0 + np.arange(n_points) * (cadence_diff_sec / 86400.0)
    flux_raw = 1000.0 + np.random.normal(0, 1, n_points)
    flux_err = np.ones(n_points)
    quality = np.zeros(n_points, dtype=np.int64)
    # add some bad quality flags
    quality[5] = 128
    
    col_time = fits.Column(name="TIME", format="D", array=time)
    col_sap = fits.Column(name="SAP_FLUX", format="E", array=flux_raw)
    col_sap_err = fits.Column(name="SAP_FLUX_ERR", format="E", array=flux_err)
    col_pdc = fits.Column(name="PDCSAP_FLUX", format="E", array=flux_raw)
    col_pdc_err = fits.Column(name="PDCSAP_FLUX_ERR", format="E", array=flux_err)
    col_qual = fits.Column(name="QUALITY", format="J", array=quality)
    col_cad = fits.Column(name="CADENCENO", format="J", array=np.arange(n_points, dtype=np.int64))
    
    tb_hdu = fits.BinTableHDU.from_columns([col_time, col_sap, col_sap_err, col_pdc, col_pdc_err, col_qual, col_cad], name="LIGHTCURVE")
    tb_hdu.header["CROWDSAP"] = 0.99
    tb_hdu.header["FLFRCSAP"] = 0.98
    
    hdul = fits.HDUList([primary_hdu, tb_hdu])
    hdul.writeto(path, overwrite=True)

@pytest.fixture
def mock_config(tmp_path):
    """Fixture to generate a temporary Config object with customized folders."""
    config_yaml = f"""
dataset_name: "test-dataset"
dataset_version: "1.0.0"
random_seed: 42
label_policy_version: "1.0.0"
cadence_limits:
  min_cadence_seconds: 110.0
  max_cadence_seconds: 130.0
selected_sectors: [78]
automatic_sector_selection: false
minimum_successful_observations: 5
download_concurrency: 2
download_retry_policy:
  retries: 1
  backoff_factor: 1.0
  timeout_seconds: 10
output_directories:
  raw_dir: "{tmp_path.as_posix()}/raw"
  processed_dir: "{tmp_path.as_posix()}/processed"
  manifests_dir: "{tmp_path.as_posix()}/manifests"
checksum_algorithm: "SHA-256"
allowed_fits_authors: ["SPOC"]
quality_mask_policy:
  clean_quality: true
  quality_bitmask: 128
preprocessing_limits:
  minimum_points: 10
  minimum_usable_fraction: 0.1
  minimum_time_span_days: 0.1
  normalization_method: "median_division"
duplicate_policy:
  prefer_spoc: true
  prefer_latest_release: true
  prefer_highest_usable_fraction: true
split_policy:
  ratios:
    train: 0.60
    val: 0.20
    test: 0.20
  grouping_field: "tic_id"
  stratification: true
minimum_desired_class_counts:
  train: {{planets: 1, ebs: 1, blends: 1}}
  validation: {{planets: 1, ebs: 1, blends: 1}}
  test: {{planets: 1, ebs: 1, blends: 1}}
catalogue_locations:
  toi_catalog: "{tmp_path.as_posix()}/archive/toi.csv"
  tce_catalog: "{tmp_path.as_posix()}/archive/tce.csv"
  cumulative_catalog: "{tmp_path.as_posix()}/archive/cumulative.csv"
  planets_catalog: "{tmp_path.as_posix()}/archive/planets.csv"
quarantine_rules:
  max_failure_rate_threshold: 0.2
  reasons: ["Corrupted FITS structure"]
reporting_paths:
  dataset_card: "{tmp_path.as_posix()}/docs/card.md"
  provenance_doc: "{tmp_path.as_posix()}/docs/provenance.md"
  label_policy_doc: "{tmp_path.as_posix()}/docs/label_policy.md"
  split_methodology: "{tmp_path.as_posix()}/docs/splits.md"
  reproduction_guide: "{tmp_path.as_posix()}/docs/reproduce.md"
  storage_guide: "{tmp_path.as_posix()}/docs/storage.md"
  completion_report: "{tmp_path.as_posix()}/docs/completion.md"
"""
    yaml_path = tmp_path / "test_config.yaml"
    with open(yaml_path, "w") as f:
        f.write(config_yaml)
        
    config = Config(yaml_path)
    # Mock REPO_ROOT as tmp_path to isolate filesystem side-effects
    config.REPO_ROOT = tmp_path
    return config

def test_fits_parser_success(mock_config, tmp_path):
    """Test strict FITS parser succeeds on a well-formed TESS dummy file."""
    fpath = tmp_path / "valid.fits"
    make_dummy_fits(fpath, tic_id=12345, sector=78, exptime=120.0, n_points=100)
    
    arrays, metadata = parser.parse_single_fits(fpath, "mock_checksum", mock_config)
    
    assert metadata["tic_id"] == 12345
    assert metadata["sector"] == 78
    assert metadata["camera"] == 1
    assert metadata["ccd"] == 2
    assert metadata["cadence_seconds"] == 120.0
    assert metadata["source_checksum"] == "mock_checksum"
    assert metadata["n_points_raw"] == 100
    assert metadata["n_points_usable"] < 100  # bad quality points should be cleaned out
    
    # check normalized arrays
    flux = np.array(arrays["flux"])
    assert abs(np.median(flux) - 1.0) < 0.01
    assert {
        "time_btjd", "sap_flux", "pdcsap_flux", "quality_raw",
        "finite_mask", "archive_quality_mask", "usable_mask",
        "normalization_mask", "raw_cadence_number",
    }.issubset(arrays)

def test_fits_parser_out_of_bounds_cadence(mock_config, tmp_path):
    """Test FITS parser fails when the cadence is outside configured limits."""
    fpath = tmp_path / "bad_cadence.fits"
    # Create file with 20s cadence and enough points to pass duration check (1000 * 20s = 0.23 days >= 0.1 days)
    make_dummy_fits(fpath, tic_id=12345, sector=78, exptime=20.0, cadence_diff_sec=20.0, n_points=1000)
    
    with pytest.raises(parser.ParsingError, match="cadence"):
        parser.parse_single_fits(fpath, "mock_checksum", mock_config)

def test_label_resolution(mock_config, tmp_path):
    """Test deterministic label priority resolution and contradiction routing."""
    mock_config.ensure_dirs()
    
    # Write mock discovery manifest first
    df_disc = pd.DataFrame([{
        "obs_id": "obs_1", "tic_id": 111, "target_id": "TIC-111", "sector": 78,
        "ra": 100.0, "dec": -45.0, "t_exptime": 120.0, "product_uri": "uri_1",
        "product_filename": "fn_1", "download_url": "url_1", "status": "discovered"
    }, {
        "obs_id": "obs_2", "tic_id": 222, "target_id": "TIC-222", "sector": 78,
        "ra": 100.0, "dec": -45.0, "t_exptime": 120.0, "product_uri": "uri_2",
        "product_filename": "fn_2", "download_url": "url_2", "status": "discovered"
    }])
    df_disc.to_parquet(mock_config.manifests_dir / "discovery_manifest.parquet")
    
    # 1. Non-conflicting evidence
    df_evidence = pd.DataFrame([{
        "evidence_id": "EVI-1", "tic_id": 111, "canonical_label_candidate": "exoplanet_transit",
        "original_label": "CP", "original_disposition": "CP", "source_catalog": "TESS_TOI",
        "source_version": "1", "source_row_identifier": "1", "evidence_level": "catalog_authoritative",
        "evidence_strength": "strong", "disposition_date": "2026-01-01", "target_name": "TIC-111",
        "sector": 78, "ephemeris": "", "period": 10.0, "depth": 0.01, "duration": 0.1,
        "centroid_evidence": "", "contamination_evidence": "", "source_checksum": "sum",
        "ingestion_timestamp": "", "adapter_version": "1", "notes": ""
    }, {
        # 2. Conflicting evidence (equal strength: strong exoplanet vs strong EB)
        "evidence_id": "EVI-2", "tic_id": 222, "canonical_label_candidate": "exoplanet_transit",
        "original_label": "CP", "original_disposition": "CP", "source_catalog": "TESS_TOI",
        "source_version": "1", "source_row_identifier": "2", "evidence_level": "catalog_authoritative",
        "evidence_strength": "strong", "disposition_date": "2026-01-01", "target_name": "TIC-222",
        "sector": 78, "ephemeris": "", "period": 5.0, "depth": 0.01, "duration": 0.1,
        "centroid_evidence": "", "contamination_evidence": "", "source_checksum": "sum",
        "ingestion_timestamp": "", "adapter_version": "1", "notes": ""
    }, {
        "evidence_id": "EVI-3", "tic_id": 222, "canonical_label_candidate": "eclipsing_binary",
        "original_label": "EB", "original_disposition": "EB", "source_catalog": "TESS_TOI",
        "source_version": "1", "source_row_identifier": "3", "evidence_level": "catalog_authoritative",
        "evidence_strength": "strong", "disposition_date": "2026-01-01", "target_name": "TIC-222",
        "sector": 78, "ephemeris": "", "period": 5.0, "depth": 0.05, "duration": 0.2,
        "centroid_evidence": "", "contamination_evidence": "", "source_checksum": "sum",
        "ingestion_timestamp": "", "adapter_version": "1", "notes": ""
    }])
    df_evidence.to_parquet(mock_config.manifests_dir / "label_evidence.parquet")
    
    # Run resolver
    df_resolved = resolver.resolve_labels(mock_config)
    
    # Assertions
    res_111 = df_resolved[df_resolved["tic_id"] == 111].iloc[0]
    assert res_111["resolved_label"] == "exoplanet_transit"
    assert res_111["label_subtype"] == "confirmed"
    assert "EVI-1" in res_111["winning_evidence_ids"]
    assert len(res_111["rejected_evidence_ids"]) == 0
    assert not res_111["requires_review"]
    
    res_222 = df_resolved[df_resolved["tic_id"] == 222].iloc[0]
    assert res_222["resolved_label"] == "review_required"
    assert res_222["requires_review"]

def test_deduplication(mock_config):
    """Test duplicate TESS observation resolution prioritizes SPOC and larger file sizes."""
    # Build a mock download manifest
    df_dl = pd.DataFrame([{
        "obs_id": "obs_spoc", "tic_id": 999, "sector": 78, "product_uri": "uri_spoc",
        "product_filename": "fn_spoc", "final_status": "verified", "actual_size": 2000, "sha256": "sha1",
        "download_url": "", "status": "", "local_path": ""
    }, {
        "obs_id": "obs_tess_spoc", "tic_id": 999, "sector": 78, "product_uri": "uri_tess_spoc",
        "product_filename": "fn_tess_spoc", "final_status": "verified", "actual_size": 1500, "sha256": "sha2",
        "download_url": "", "status": "", "local_path": ""
    }])
    mock_config.ensure_dirs()
    df_dl.to_parquet(mock_config.manifests_dir / "download_manifest.parquet")
    
    selected = dedup.resolve_duplicates(mock_config)
    
    assert "obs_spoc" in selected
    assert "obs_tess_spoc" not in selected

def test_splits_builder_disjoint_and_grouping(mock_config):
    """Verify target-disjoint, group-aware partitioning in split generation."""
    mock_config.ensure_dirs()
    
    # 1. Labels
    df_labels = pd.DataFrame([
        {"tic_id": 111, "resolved_label": "exoplanet_transit", "label_subtype": "confirmed", "requires_review": False},
        {"tic_id": 222, "resolved_label": "eclipsing_binary", "label_subtype": "eb", "requires_review": False},
        {"tic_id": 333, "resolved_label": "blend_contamination", "label_subtype": "blend", "requires_review": False},
        {"tic_id": 444, "resolved_label": "unlabeled", "label_subtype": "unlabeled", "requires_review": False},
        {"tic_id": 555, "resolved_label": "review_required", "label_subtype": "review", "requires_review": True}
    ])
    df_labels.to_parquet(mock_config.manifests_dir / "resolved_labels.parquet")
    
    # 2. Download status (all verified)
    df_dl = pd.DataFrame([
        {"obs_id": "obs_1", "tic_id": 111, "sector": 78, "final_status": "verified"},
        {"obs_id": "obs_2", "tic_id": 222, "sector": 78, "final_status": "verified"},
        {"obs_id": "obs_3", "tic_id": 333, "sector": 78, "final_status": "verified"},
        {"obs_id": "obs_4", "tic_id": 444, "sector": 78, "final_status": "verified"},
        {"obs_id": "obs_5", "tic_id": 555, "sector": 78, "final_status": "verified"}
    ])
    df_dl.to_parquet(mock_config.manifests_dir / "download_manifest.parquet")
    
    # Run split builder
    splits.build_splits(mock_config)
    
    df_split = pd.read_parquet(mock_config.manifests_dir / "split_manifest.parquet")
    
    train_tics = df_split[df_split["split"] == "train"]["tic_id"].tolist()
    val_tics = df_split[df_split["split"] == "val"]["tic_id"].tolist()
    test_tics = df_split[df_split["split"] == "test"]["tic_id"].tolist()
    screening_tics = df_split[df_split["split"] == "screening"]["tic_id"].tolist()
    review_tics = df_split[df_split["split"] == "review"]["tic_id"].tolist()
    
    # Disjoint check
    assert len(set(train_tics).intersection(set(val_tics))) == 0
    assert len(set(train_tics).intersection(set(test_tics))) == 0
    assert len(set(val_tics).intersection(set(test_tics))) == 0
    assert len(set(train_tics + val_tics + test_tics).intersection(set(screening_tics))) == 0
    
    # Unlabeled / review isolation check
    assert 444 in screening_tics
    assert 555 in review_tics
    assert 444 not in train_tics
    assert 555 not in train_tics

def test_validation_gate_fails_low_obs(mock_config):
    """Verify that validation release gate fails when successful observation count is below threshold."""
    mock_config.ensure_dirs()
    
    # Create empty/skeleton manifests
    for m in [
        "discovery_manifest.parquet", "download_manifest.parquet", "label_evidence.parquet",
        "resolved_labels.parquet", "split_manifest.parquet", "train_targets.parquet",
        "validation_targets.parquet", "test_targets.parquet", "unlabeled_screening_targets.parquet",
        "failures.parquet", "exclusions.parquet", "duplicate_groups.parquet", "contradictions.parquet"
    ]:
        pd.DataFrame().to_parquet(mock_config.manifests_dir / m)
        
    # Write a dummy empty checksums file
    with open(mock_config.manifests_dir / "checksums.sha256", "w") as f:
        f.write("")
        
    # Build core manifest with only 1 parsed observation (threshold is 5 in mock config)
    df_obs = pd.DataFrame([{
        "observation_id": "obs_1", "tic_id": 111, "target_id": "TIC-111", "sector": 78,
        "camera": 1, "ccd": 2, "cadence_seconds": 120.0, "mission": "TESS", "archive": "MAST",
        "author": "SPOC", "pipeline_name": "SPOC", "pipeline_version": "1.0", "data_release": 1,
        "product_type": "lightcurve", "product_uri": "uri_1", "raw_path": "path_1", "processed_path": "path_2",
        "raw_sha256": "sha_raw", "processed_sha256": "sha_proc", "n_points_raw": 100, "n_points_finite": 100,
        "n_points_usable": 90, "usable_fraction": 0.9, "time_span_days": 10.0, "median_cadence_seconds": 120.0,
        "gap_count": 0, "selected_flux_column": "PDCSAP_FLUX", "normalization_method": "median_division",
        "ra": 100.0, "dec": -45.0, "tess_magnitude": 10.5, "crowding_metric": 0.99, "flux_fraction": 0.98,
        "centroid_available": True, "target_pixel_file_available": False, "canonical_label": "exoplanet_transit",
        "label_subtype": "confirmed", "evidence_level": "catalog_authoritative", "label_strength": "strong",
        "label_policy_version": "1.0.0", "requires_review": False, "is_supervised_eligible": True,
        "split": "train", "split_group_id": "111", "split_seed": 42, "split_version": "1.0.0",
        "discovery_status": "discovered", "download_status": "verified", "parse_status": "success",
        "validation_status": "passed", "exclusion_reason": "", "quarantine_reason": "",
        "discovery_run_id": "run_1", "download_run_id": "run_1", "processing_run_id": "run_1",
        "code_version": "1.0.0", "created_at": "", "updated_at": ""
    }])
    df_obs.to_parquet(mock_config.manifests_dir / "observation_manifest.parquet")
    
    # Regenerate checksums file to include observation manifest
    csums.generate_checksums_file(mock_config)
    
    # Run validation
    res = validator.run_release_validation(mock_config)
    
    # Must fail because count (1) is < 5 (minimum in mock config)
    assert res["status"] == "FAIL"
    assert any("gate unmet" in e for e in res["errors"])


def test_release_floor_cannot_be_lowered_by_development_config(mock_config):
    assert mock_config.minimum_successful_observations == 5
    source = Path(validator.__file__).read_text(encoding="utf-8")
    assert "max(20_000, config.minimum_successful_observations)" in source


def test_generic_toi_false_positive_requires_review(mock_config, tmp_path):
    import phase1.catalog_ingestion as ingestion

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True)
    pd.DataFrame([{
        "tid": 12345, "tfopwg_disp": "FP", "pl_trandep": 90000,
        "rowupdate": "2026-01-01", "pl_orbper": 2.0,
        "pl_tranmid": 1.0, "pl_trandurh": 2.0,
    }]).to_csv(mock_config.toi_catalog, index=False)

    evidence = ingestion.ingest_all_catalogs(mock_config)
    row = evidence[evidence["tic_id"] == 12345].iloc[0]
    assert row["canonical_label_candidate"] == "review_required"
    assert "depth" not in row["notes"].lower()
