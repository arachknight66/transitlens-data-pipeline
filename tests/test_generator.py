"""
tests/test_generator.py
─────────────────────────
Unit tests for synthetic/generator.py (Phase 1B).
"""

import os

import numpy as np
import pandas as pd

from synthetic.generator import (
    generate_all_cases,
    generate_from_config,
    make_base_flux,
    make_time_array,
)


def test_time_array_length():
    # n_points=1000 -> array of length ~980-1000 (gaps reduce it slightly)
    n_points = 1000
    cadence_minutes = 2.0
    # Pick a time span that yields exactly n_points before gap removal.
    time_span_days = n_points * (cadence_minutes / 1440.0)

    time = make_time_array(
        n_points=n_points,
        time_span_days=time_span_days,
        cadence_minutes=cadence_minutes,
        seed=1,
    )

    assert 950 <= len(time) <= n_points


def test_time_array_monotonic():
    time = make_time_array(
        n_points=1000, time_span_days=1.39, cadence_minutes=2.0, seed=1
    )
    # time must always be strictly increasing
    assert np.all(np.diff(time) > 0)


def test_base_flux_all_ones():
    # base flux before noise/transit must be exactly 1.0
    flux = make_base_flux(250)
    assert flux.shape == (250,)
    assert np.all(flux == 1.0)


def test_generate_candidate_a(tmp_path, config_path):
    # runs full generation for candidate_a
    time, flux, metadata = generate_from_config(config_path, "candidate_a")

    assert len(time) == len(flux)
    assert len(time) > 0
    assert metadata["label"] == "exoplanet_like"
    assert metadata["true_period"] == 3.42

    # checks output CSV exists and has correct columns
    output_dir = tmp_path / "cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "candidate_a.csv"
    pd.DataFrame({"time": time, "flux": flux}).to_csv(csv_path, index=False)

    assert csv_path.exists()
    loaded = pd.read_csv(csv_path)
    assert list(loaded.columns) == ["time", "flux"]
    assert len(loaded) == len(time)


def test_all_cases_generate(tmp_path, config_path):
    # loops over all three cases, verifies each CSV is written
    output_dir = tmp_path / "cases"
    generate_all_cases(config_path, str(output_dir))

    for case_name in ["candidate_a", "candidate_b", "candidate_c"]:
        csv_path = output_dir / f"{case_name}.csv"
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["time", "flux"]
        assert len(df) > 0
        # not completely flat -- noise and/or transit must be present
        assert df["flux"].std() > 0.0