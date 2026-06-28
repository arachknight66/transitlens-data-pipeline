"""Versioned Phase 1 table contracts.

The constants are intentionally plain Python so schema validation stays usable
without a dataframe-validation framework or live network access.
"""

SCHEMA_VERSION = "1.1.0"

OBSERVATION_REQUIRED_COLUMNS = frozenset({
    "observation_id", "tic_id", "target_id", "sector", "camera", "ccd",
    "cadence_seconds", "mission", "archive", "author", "pipeline_name",
    "pipeline_version", "data_release", "product_type", "product_uri",
    "raw_path", "processed_path", "raw_sha256", "processed_sha256",
    "n_points_raw", "n_points_finite", "n_points_usable", "usable_fraction",
    "time_span_days", "median_cadence_seconds", "gap_count",
    "selected_flux_column", "normalization_method", "ra", "dec",
    "canonical_label", "evidence_level", "label_strength",
    "label_policy_version", "requires_review", "is_supervised_eligible",
    "split", "split_group_id", "split_seed", "split_version",
    "discovery_status", "download_status", "parse_status",
    "validation_status", "discovery_run_id", "download_run_id",
    "processing_run_id", "code_version", "created_at", "updated_at",
})

LABEL_EVIDENCE_REQUIRED_COLUMNS = frozenset({
    "evidence_id", "tic_id", "canonical_label_candidate", "original_label",
    "original_disposition", "source_catalog", "source_version",
    "source_row_identifier", "evidence_level", "evidence_strength",
    "disposition_date", "source_checksum", "ingestion_timestamp",
    "adapter_version", "notes",
})

RESOLVED_LABEL_REQUIRED_COLUMNS = frozenset({
    "tic_id", "resolved_label", "label_subtype", "winning_evidence_ids",
    "rejected_evidence_ids", "conflict_count", "resolution_reason",
    "evidence_level", "label_strength", "requires_review", "policy_version",
})

SPLIT_REQUIRED_COLUMNS = frozenset({
    "tic_id", "split", "resolved_label", "label_subtype",
})

CANONICAL_LABELS = frozenset({
    "exoplanet_transit", "eclipsing_binary", "blend_contamination",
    "stellar_variability_or_other", "review_required", "unlabeled",
})


def missing_columns(frame, required):
    """Return sorted required columns absent from a pandas-like frame."""
    return sorted(set(required) - set(frame.columns))
