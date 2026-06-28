import os
import yaml
from pathlib import Path

# Compute Repo Root path (assumes this file is in <repo_root>/transitlens-data-pipeline/phase1/config.py)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "phase1_dataset.yaml"

class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        self.config_path = config_path
        self.REPO_ROOT = REPO_ROOT

        # Populate attributes
        self.dataset_name = self._cfg.get("dataset_name", "transitlens-phase1-highcadence")
        self.dataset_version = self._cfg.get("dataset_version", "1.0.0")
        self.random_seed = int(self._cfg.get("random_seed", 42))
        self.label_policy_version = self._cfg.get("label_policy_version", "1.0.0")
        self.label_policy_file = REPO_ROOT / self._cfg.get("label_policy_file", "config/toi_label_policy.yaml")
        
        self.archive_provider = self._cfg.get("archive_provider", "MAST")
        self.product_types = self._cfg.get("product_types", ["timeseries"])
        
        cadence_cfg = self._cfg.get("cadence_limits", {})
        self.min_cadence_seconds = float(cadence_cfg.get("min_cadence_seconds", 110.0))
        self.max_cadence_seconds = float(cadence_cfg.get("max_cadence_seconds", 130.0))

        self.selected_sectors = [int(s) for s in self._cfg.get("selected_sectors", [78])]
        self.automatic_sector_selection = bool(self._cfg.get("automatic_sector_selection", False))
        self.minimum_successful_observations = int(self._cfg.get("minimum_successful_observations", 20000))
        
        self.download_concurrency = int(self._cfg.get("download_concurrency", 8))
        
        dl_retry = self._cfg.get("download_retry_policy", {})
        self.download_retries = int(dl_retry.get("retries", 3))
        self.download_backoff_factor = float(dl_retry.get("backoff_factor", 1.5))
        self.download_timeout = float(dl_retry.get("timeout_seconds", 45))

        # Output Directories: resolve relative to REPO_ROOT
        out_dirs = self._cfg.get("output_directories", {})
        self.raw_dir = REPO_ROOT / out_dirs.get("raw_dir", "data/raw/spoc")
        self.processed_dir = REPO_ROOT / out_dirs.get("processed_dir", "data/processed/phase1")
        self.manifests_dir = REPO_ROOT / out_dirs.get("manifests_dir", "data/manifests/phase1")

        self.checksum_algorithm = self._cfg.get("checksum_algorithm", "SHA-256")
        self.allowed_fits_authors = self._cfg.get("allowed_fits_authors", ["SPOC", "TESS-SPOC"])

        qm_policy = self._cfg.get("quality_mask_policy", {})
        self.clean_quality = bool(qm_policy.get("clean_quality", True))
        self.quality_bitmask = int(qm_policy.get("quality_bitmask", 0))

        limits = self._cfg.get("preprocessing_limits", {})
        self.minimum_points = int(limits.get("minimum_points", 100))
        self.minimum_usable_fraction = float(limits.get("minimum_usable_fraction", 0.1))
        self.minimum_time_span_days = float(limits.get("minimum_time_span_days", 5.0))
        self.normalization_method = limits.get("normalization_method", "median_division")

        dup_policy = self._cfg.get("duplicate_policy", {})
        self.prefer_spoc = bool(dup_policy.get("prefer_spoc", True))
        self.prefer_latest_release = bool(dup_policy.get("prefer_latest_release", True))
        self.prefer_highest_usable_fraction = bool(dup_policy.get("prefer_highest_usable_fraction", True))

        split_policy = self._cfg.get("split_policy", {})
        self.split_ratios = split_policy.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
        self.split_grouping_field = split_policy.get("grouping_field", "tic_id")
        self.split_stratification = bool(split_policy.get("stratification", True))

        self.min_class_counts = self._cfg.get("minimum_desired_class_counts", {})

        # Catalog locations: resolve relative to REPO_ROOT
        cat_locs = self._cfg.get("catalogue_locations", {})
        self.toi_catalog = REPO_ROOT / cat_locs.get("toi_catalog", "archive/TOI_2026.06.25_21.21.19.csv")
        self.mast_toi_catalog = REPO_ROOT / cat_locs.get(
            "mast_toi_catalog", "archive/phase1_catalogs/2026-06-28/mast_toi_current.csv"
        )
        self.tce_catalog = REPO_ROOT / cat_locs.get("tce_catalog", "archive/tess s0078-s0078_tcestats.csv")
        self.additional_tce_catalogs = [
            REPO_ROOT / value for value in cat_locs.get("additional_tce_catalogs", [])
        ]
        self.eb_catalog = REPO_ROOT / cat_locs.get(
            "eb_catalog", "archive/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv"
        )
        self.cumulative_catalog = REPO_ROOT / cat_locs.get("cumulative_catalog", "archive/cumulative.csv")
        self.planets_catalog = REPO_ROOT / cat_locs.get("planets_catalog", "archive/planets.csv")

        q_rules = self._cfg.get("quarantine_rules", {})
        self.max_failure_rate_threshold = float(q_rules.get("max_failure_rate_threshold", 0.05))
        self.quarantine_reasons = q_rules.get("reasons", [])

        # Reporting paths: resolve relative to REPO_ROOT
        rep_paths = self._cfg.get("reporting_paths", {})
        self.report_dataset_card = REPO_ROOT / rep_paths.get("dataset_card", "docs/phase1_dataset_card.md")
        self.report_provenance_doc = REPO_ROOT / rep_paths.get("provenance_doc", "docs/phase1_provenance.md")
        self.report_label_policy_doc = REPO_ROOT / rep_paths.get("label_policy_doc", "docs/phase1_label_policy.md")
        self.report_split_methodology = REPO_ROOT / rep_paths.get("split_methodology", "docs/phase1_splits.md")
        self.report_reproduction_guide = REPO_ROOT / rep_paths.get("reproduction_guide", "docs/phase1_reproduction.md")
        self.report_storage_guide = REPO_ROOT / rep_paths.get("storage_guide", "docs/phase1_storage.md")
        self.report_completion_report = REPO_ROOT / rep_paths.get("completion_report", "docs/phase1_completion_report.md")

    def ensure_dirs(self):
        """Creates output, processed, and raw directories."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "lightcurves").mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "quarantine").mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "validation").mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
