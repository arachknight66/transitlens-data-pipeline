"""
tests/conftest.py
──────────────────
Shared fixtures used across the test suite.

Also includes a session-scoped autouse fixture that makes the test
suite self-contained: if `synthetic/cases/*.csv` or
`datasets/metadata.json` don't exist yet (e.g. on a fresh checkout
before anyone has run the generation scripts), they're generated
once before any test runs. This mirrors the handoff checklist in the
plan ("pytest tests/ -v passes with zero failures") without forcing
contributors to manually run Phase 1/2 scripts first.
"""

import os
import shutil

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL_CONFIG_PATH = os.path.join(REPO_ROOT, "synthetic", "config.yaml")
_REAL_CASES_DIR = os.path.join(REPO_ROOT, "synthetic", "cases")
_REAL_DATASET_PATH = os.path.join(REPO_ROOT, "datasets", "labeled_dataset.csv")
_REAL_METADATA_PATH = os.path.join(REPO_ROOT, "datasets", "metadata.json")


@pytest.fixture(scope="session", autouse=True)
def ensure_pipeline_artifacts_exist():
    """
    Guarantees synthetic/cases/*.csv and datasets/metadata.json exist
    in the real repo locations before any test runs, since
    tests/test_interface.py exercises load_light_curve() against the
    real default paths (not a tmp_path copy).
    """
    expected_cases = ["candidate_a.csv", "candidate_b.csv", "candidate_c.csv"]
    cases_missing = not all(
        os.path.exists(os.path.join(_REAL_CASES_DIR, f)) for f in expected_cases
    )

    if cases_missing:
        from synthetic.generator import generate_all_cases
        generate_all_cases(_REAL_CONFIG_PATH, _REAL_CASES_DIR)

    if not os.path.exists(_REAL_METADATA_PATH):
        from datasets.build_dataset import build_from_synthetic
        build_from_synthetic(_REAL_CASES_DIR, _REAL_CONFIG_PATH, _REAL_DATASET_PATH)

    yield


@pytest.fixture
def synthetic_time():
    """Returns a small 500-point time array for fast tests."""
    return list(np.linspace(0, 5.0, 500))


@pytest.fixture
def synthetic_flux():
    """Returns a flat normalised flux array with light Gaussian noise."""
    rng = np.random.default_rng(seed=0)
    return list(1.0 + rng.normal(0, 0.002, 500))


@pytest.fixture
def config_path(tmp_path):
    """Copies the real config.yaml to a tmp location for test isolation."""
    dst = tmp_path / "config.yaml"
    shutil.copy(_REAL_CONFIG_PATH, dst)
    return str(dst)