"""
tests/test_schema.py
──────────────────────
Validates that the assembled labeled_dataset.csv conforms to
datasets/schema.md.

Builds a fresh, isolated copy of the dataset in a tmp directory
(rather than depending on a pre-existing datasets/labeled_dataset.csv)
so these tests are reproducible on a clean checkout and don't get
stale if the committed CSV falls out of sync with the generators.
"""

import os

import pytest

from datasets.build_dataset import build_from_synthetic
from synthetic.generator import generate_all_cases

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL_CONFIG_PATH = os.path.join(REPO_ROOT, "synthetic", "config.yaml")

REQUIRED_COLUMNS = [
    "target_id", "time", "flux", "source", "label",
    "true_period", "true_depth", "true_duration",
    "cadence_min", "sector",
]

REQUIRED_NON_NULL_COLUMNS = ["target_id", "time", "flux", "source", "cadence_min"]

VALID_LABELS = {
    "exoplanet_transit",
    "eclipsing_binary",
    "blend_contamination",
    "stellar_variability_or_other",
    "exoplanet_like",
    "eclipsing_binary_like",
    "noise_or_other",
}


@pytest.fixture(scope="module")
def labeled_dataset(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("schema_test")
    cases_dir = tmp_path / "cases"
    generate_all_cases(_REAL_CONFIG_PATH, str(cases_dir))

    output_path = tmp_path / "labeled_dataset.csv"
    dataset = build_from_synthetic(str(cases_dir), _REAL_CONFIG_PATH, str(output_path))
    return dataset


def test_labeled_dataset_columns(labeled_dataset):
    # labeled_dataset.csv must have all required columns from schema.md
    assert list(labeled_dataset.columns) == REQUIRED_COLUMNS


def test_no_nulls_in_required_columns(labeled_dataset):
    # time, flux, target_id, source, cadence_min must never be null
    for col in REQUIRED_NON_NULL_COLUMNS:
        assert labeled_dataset[col].isnull().sum() == 0, f"unexpected nulls in {col}"


def test_flux_normalised(labeled_dataset):
    # median of flux column must be between 0.99 and 1.01 per target
    for target_id, group in labeled_dataset.groupby("target_id"):
        median_flux = group["flux"].median()
        assert 0.99 <= median_flux <= 1.01, (target_id, median_flux)


def test_label_values_valid(labeled_dataset):
    # all non-null labels must be in the valid label set
    labels = set(labeled_dataset["label"].dropna().unique())
    assert labels <= VALID_LABELS


def test_time_strictly_increasing_per_target(labeled_dataset):
    # validation rule 1 in schema.md
    for target_id, group in labeled_dataset.groupby("target_id"):
        assert group["time"].is_monotonic_increasing, target_id


def test_sector_none_for_synthetic(labeled_dataset):
    # sector must be None for all synthetic cases
    assert labeled_dataset["sector"].isnull().all()