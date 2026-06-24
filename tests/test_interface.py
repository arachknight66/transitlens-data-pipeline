"""
tests/test_interface.py
──────────────────────────
Tests for interface.py — the single entry point transitlens-ml-core
depends on (Phase 3).
"""

import statistics

import pytest

from interface import load_light_curve

REQUIRED_TOP_LEVEL_KEYS = {"time", "flux", "target_id", "source", "n_points", "metadata"}
REQUIRED_METADATA_KEYS = {
    "cadence_min", "time_span_days", "sector", "label",
    "true_period", "true_depth", "true_duration",
}


def test_load_synthetic_candidate_a():
    # returns dict with correct shape
    result = load_light_curve("synthetic", "candidate_a")

    # time and flux are lists, not numpy arrays
    assert isinstance(result["time"], list)
    assert isinstance(result["flux"], list)
    assert isinstance(result["time"][0], float)
    assert isinstance(result["flux"][0], float)

    # n_points == len(time) == len(flux)
    assert result["n_points"] == len(result["time"]) == len(result["flux"])


@pytest.mark.parametrize("target_id", ["candidate_a", "candidate_b", "candidate_c"])
def test_output_contract_all_keys_present(target_id):
    # result must contain: time, flux, target_id, source, n_points, metadata
    result = load_light_curve("synthetic", target_id)
    assert REQUIRED_TOP_LEVEL_KEYS <= set(result.keys())
    assert result["target_id"] == target_id
    assert result["source"] == "synthetic"


@pytest.mark.parametrize("target_id", ["candidate_a", "candidate_b", "candidate_c"])
def test_metadata_keys_present(target_id):
    # metadata must contain all required keys
    result = load_light_curve("synthetic", target_id)
    assert REQUIRED_METADATA_KEYS <= set(result["metadata"].keys())


@pytest.mark.parametrize("target_id", ["candidate_a", "candidate_b", "candidate_c"])
def test_flux_values_normalised(target_id):
    # abs(median(flux) - 1.0) < 0.01
    result = load_light_curve("synthetic", target_id)
    assert abs(statistics.median(result["flux"]) - 1.0) < 0.01


def test_unknown_source_raises_invalid_source_error():
    # load_light_curve("unknown_source", "x") must raise InvalidSourceError
    from interface import InvalidSourceError
    with pytest.raises(InvalidSourceError):
        load_light_curve("unknown_source", "x")


def test_missing_case_raises_file_not_found():
    # load_light_curve("synthetic", "nonexistent") must raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_light_curve("synthetic", "nonexistent_case_xyz")


def test_csv_source_without_path_raises_value_error():
    with pytest.raises(ValueError):
        load_light_curve("csv", "some_target")


def test_tess_source_without_lightkurve_raises_import_error():
    # requirements.txt keeps lightkurve commented out for the offline
    # hackathon demo, so this should normally raise. If a future
    # environment happens to have it installed, skip rather than fail
    # since this test specifically targets the "not installed" path.
    try:
        import lightkurve  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            load_light_curve("tess", "TIC-25155310")
    else:
        pytest.skip("lightkurve is installed in this environment; "
                    "the ImportError path cannot be exercised here.")


def test_candidate_labels_match_expected():
    # ground-truth labels should match synthetic/config.yaml
    expected = {
        "candidate_a": "exoplanet_like",
        "candidate_b": "eclipsing_binary_like",
        "candidate_c": "noise_or_other",
    }
    for target_id, label in expected.items():
        result = load_light_curve("synthetic", target_id)
        assert result["metadata"]["label"] == label