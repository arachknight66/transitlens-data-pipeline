"""
tests/test_loader.py
──────────────────────
Tests the raw CSV-loading mechanics that sit underneath interface.py
and the synthetic generation pipeline.

This is deliberately narrower than test_interface.py (which tests the
full load_light_curve() contract) and test_schema.py (which tests the
assembled, multi-target labeled_dataset.csv). Here we're checking the
loader behaviour at the single-CSV level: does a generated case file
have the right columns, no NaNs, plausible value ranges, and does the
shared CSV-reading helper fail loudly on malformed input.
"""

import os

import numpy as np
import pandas as pd
import pytest

from interface import _read_light_curve_csv
from synthetic.generator import generate_all_cases

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REAL_CONFIG_PATH = os.path.join(REPO_ROOT, "synthetic", "config.yaml")

EXPECTED_CASES = ["candidate_a", "candidate_b", "candidate_c"]


@pytest.fixture(scope="module")
def cases_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("loader_test")
    cases_dir = tmp_path / "cases"
    generate_all_cases(_REAL_CONFIG_PATH, str(cases_dir))
    return str(cases_dir)


@pytest.mark.parametrize("case_name", EXPECTED_CASES)
def test_case_csv_exists(cases_dir, case_name):
    csv_path = os.path.join(cases_dir, f"{case_name}.csv")
    assert os.path.exists(csv_path)


@pytest.mark.parametrize("case_name", EXPECTED_CASES)
def test_case_csv_has_exactly_time_and_flux_columns(cases_dir, case_name):
    df = pd.read_csv(os.path.join(cases_dir, f"{case_name}.csv"))
    assert list(df.columns) == ["time", "flux"]


@pytest.mark.parametrize("case_name", EXPECTED_CASES)
def test_case_csv_has_no_missing_values(cases_dir, case_name):
    df = pd.read_csv(os.path.join(cases_dir, f"{case_name}.csv"))
    assert df["time"].isnull().sum() == 0
    assert df["flux"].isnull().sum() == 0


@pytest.mark.parametrize("case_name", EXPECTED_CASES)
def test_case_csv_flux_in_plausible_range(cases_dir, case_name):
    # Even candidate_b's 18% eclipse depth and noise shouldn't push
    # flux outside a physically sane [0.5, 1.5] envelope.
    df = pd.read_csv(os.path.join(cases_dir, f"{case_name}.csv"))
    assert df["flux"].min() > 0.5
    assert df["flux"].max() < 1.5


@pytest.mark.parametrize("case_name", EXPECTED_CASES)
def test_case_csv_time_is_strictly_increasing(cases_dir, case_name):
    df = pd.read_csv(os.path.join(cases_dir, f"{case_name}.csv"))
    assert df["time"].is_monotonic_increasing


def test_case_csv_row_count_matches_27_day_27min_cadence_expectation(cases_dir):
    # ~27 days at 2-min cadence with ~2% simulated gaps -> a few hundred
    # short of 18000, comfortably within [17000, 18000].
    df = pd.read_csv(os.path.join(cases_dir, "candidate_a.csv"))
    assert 17000 <= len(df) <= 18000


# ─────────────────────────────────────────────
# interface._read_light_curve_csv — the shared helper used by every
# source type ("synthetic", "csv") to parse a raw CSV.
# ─────────────────────────────────────────────

def test_read_light_curve_csv_valid_file(tmp_path):
    csv_path = tmp_path / "valid.csv"
    pd.DataFrame({"time": [0.0, 1.0, 2.0], "flux": [1.0, 0.99, 1.01]}).to_csv(
        csv_path, index=False
    )
    df = _read_light_curve_csv(str(csv_path))
    assert list(df.columns) == ["time", "flux"]
    assert len(df) == 3


def test_read_light_curve_csv_missing_flux_column_raises_value_error(tmp_path):
    csv_path = tmp_path / "missing_flux.csv"
    pd.DataFrame({"time": [0.0, 1.0, 2.0]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        _read_light_curve_csv(str(csv_path))


def test_read_light_curve_csv_missing_time_column_raises_value_error(tmp_path):
    csv_path = tmp_path / "missing_time.csv"
    pd.DataFrame({"flux": [1.0, 0.99, 1.01]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        _read_light_curve_csv(str(csv_path))


def test_read_light_curve_csv_extra_columns_are_tolerated(tmp_path):
    # A CSV with extra columns (e.g. flux_err) should still load fine --
    # interface.py only requires time/flux to be present, it doesn't
    # reject unexpected extras.
    csv_path = tmp_path / "extra_columns.csv"
    pd.DataFrame(
        {"time": [0.0, 1.0], "flux": [1.0, 0.99], "flux_err": [0.001, 0.001]}
    ).to_csv(csv_path, index=False)
    df = _read_light_curve_csv(str(csv_path))
    assert {"time", "flux"} <= set(df.columns)
    assert len(df) == 2


def test_read_light_curve_csv_nonexistent_file_raises():
    with pytest.raises((FileNotFoundError, OSError)):
        _read_light_curve_csv("/nonexistent/path/to/file.csv")