"""
tests/test_real_tess.py
─────────────────────────
Unit tests for the Phase 5 stretch-goal real_tess/ package.

flux_normaliser.py and sector_selector.py are pure, offline
functions and are tested directly with synthetic arrays / fake
search-result objects. mast_loader.py's network-touching code path
is only exercised for the "lightkurve not installed -> ImportError"
contract (the one path interface.py actually depends on for the
offline hackathon demo); its pure helper functions (TIC ID
normalisation, cache filename convention, cache lookup) are tested
directly without needing lightkurve installed.
"""

import os

import numpy as np
import pytest

from real_tess.flux_normaliser import normalise_pdcsap
from real_tess.mast_loader import (
    _cache_filename,
    _find_cached_file,
    _normalise_tic_id,
    fetch_light_curve,
)
from real_tess.sector_selector import select_best_sector


# ─────────────────────────────────────────────
# flux_normaliser.py
# ─────────────────────────────────────────────

def test_normalise_pdcsap_median_is_one():
    rng = np.random.default_rng(0)
    raw = 50000 + rng.normal(0, 200, 5000)
    norm = normalise_pdcsap(raw)
    assert abs(np.nanmedian(norm) - 1.0) < 0.01


def test_normalise_pdcsap_flags_stay_nan():
    raw = np.full(100, 50000.0)
    quality = np.zeros(100, dtype=int)
    quality[10:15] = 1
    norm = normalise_pdcsap(raw, quality_flags=quality)
    assert np.all(np.isnan(norm[10:15]))
    assert np.all(np.isfinite(norm[:10]))


def test_normalise_pdcsap_clips_outlier_without_being_skewed_by_it():
    # A single extreme outlier must not inflate its own clip threshold
    # (i.e. sigma must be estimated robustly, not via plain std).
    rng = np.random.default_rng(1)
    raw = 50000 + rng.normal(0, 100, 2000)
    raw[50] = 1e9  # momentum-dump-style spike

    norm = normalise_pdcsap(raw)
    assert np.isfinite(norm[50])
    assert norm[50] < 10.0  # clipped down close to the baseline, not left huge


def test_normalise_pdcsap_all_flagged_raises_value_error():
    with pytest.raises(ValueError):
        normalise_pdcsap(np.ones(10), quality_flags=np.ones(10))


def test_normalise_pdcsap_zero_median_raises_value_error():
    with pytest.raises(ValueError):
        normalise_pdcsap(np.zeros(10))


def test_normalise_pdcsap_quality_length_mismatch_raises_value_error():
    with pytest.raises(ValueError):
        normalise_pdcsap(np.ones(10), quality_flags=np.ones(5))


# ─────────────────────────────────────────────
# sector_selector.py
# ─────────────────────────────────────────────

class _FakeTable(dict):
    @property
    def colnames(self):
        return list(self.keys())


class _FakeSearchResult:
    def __init__(self, exptimes, sectors):
        self.table = _FakeTable(
            exptime=np.array(exptimes), sequence_number=np.array(sectors)
        )

    def __len__(self):
        return len(self.table["exptime"])


def test_select_best_sector_prefers_2min_and_recency():
    # sector 10 is 30-min FFI; sectors 15 and 20 are 2-min; 20 is most recent
    sr = _FakeSearchResult(exptimes=[1800, 120, 120], sectors=[10, 15, 20])
    idx = select_best_sector(sr)
    assert idx == 2


def test_select_best_sector_ffi_only_prefers_most_recent():
    sr = _FakeSearchResult(exptimes=[1800, 1800], sectors=[5, 8])
    idx = select_best_sector(sr)
    assert idx == 1


def test_select_best_sector_empty_raises_value_error():
    with pytest.raises(ValueError):
        select_best_sector(_FakeSearchResult(exptimes=[], sectors=[]))


class _FakeFluxObj:
    def __init__(self, values):
        self.value = np.array(values)


class _FakeLightCurve:
    def __init__(self, flux, exptime, sector):
        self.flux = _FakeFluxObj(flux)
        self.exptime = exptime
        self.sector = sector


def test_select_best_sector_by_actual_point_count():
    # candidate 0 has 50 valid points, candidate 1 has 90 valid points
    c0 = _FakeLightCurve(flux=[1.0] * 50 + [np.nan] * 50, exptime=120, sector=10)
    c1 = _FakeLightCurve(flux=[1.0] * 90 + [np.nan] * 10, exptime=120, sector=11)
    idx = select_best_sector([None, None], candidates=[c0, c1])
    assert idx == 1


# ─────────────────────────────────────────────
# mast_loader.py — pure helpers + the ImportError contract
# ─────────────────────────────────────────────

@pytest.mark.parametrize("raw_id", ["TIC-25155310", "TIC 25155310", "25155310", 25155310])
def test_normalise_tic_id(raw_id):
    assert _normalise_tic_id(raw_id) == "25155310"


@pytest.mark.parametrize("raw_id", ["", "TIC", "not-a-tic", "123ABC"])
def test_normalise_tic_id_rejects_invalid_values(raw_id):
    with pytest.raises(ValueError, match="Invalid TIC ID"):
        _normalise_tic_id(raw_id)


def test_cache_filename_convention():
    assert _cache_filename("25155310", 15) == "TIC25155310_sector015.fits"


def test_find_cached_file_specific_sector(tmp_path):
    cache_dir = str(tmp_path)
    open(os.path.join(cache_dir, "TIC25155310_sector015.fits"), "w").close()
    open(os.path.join(cache_dir, "TIC25155310_sector020.fits"), "w").close()

    result = _find_cached_file("25155310", 20, cache_dir)
    assert result == (os.path.join(cache_dir, "TIC25155310_sector020.fits"), 20)


def test_find_cached_file_no_match_returns_none(tmp_path):
    cache_dir = str(tmp_path)
    assert _find_cached_file("25155310", None, cache_dir) is None


def test_fetch_light_curve_without_lightkurve_raises_import_error(tmp_path):
    # requirements.txt keeps lightkurve commented out for the offline
    # hackathon demo, so this should normally raise. If a future
    # environment happens to have it installed, skip rather than fail.
    try:
        import lightkurve  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            fetch_light_curve("TIC-25155310", cache_dir=str(tmp_path))
    else:
        pytest.skip("lightkurve is installed in this environment; "
                    "the ImportError path cannot be exercised here.")
