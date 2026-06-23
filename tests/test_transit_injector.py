"""
tests/test_transit_injector.py
─────────────────────────────────
Unit tests for synthetic/transit_injector.py (Phase 1D).
"""

import numpy as np

from synthetic.transit_injector import (
    compute_transit_count,
    inject_secondary_eclipse,
    inject_transit,
)


def test_transit_creates_dip(synthetic_time, synthetic_flux):
    # after injection, min(flux) must be significantly below 1.0
    time = np.array(synthetic_time)
    flux = np.array(synthetic_flux)

    inject_transit(flux, time, period_days=1.0, depth=0.05, duration_days=0.1)

    assert flux.min() < 1.0 - 0.03


def test_box_transit_depth():
    # flux inside transit window should be approximately 1.0 - depth
    period_days = 1.0
    depth = 0.05
    duration_days = 0.1
    t0 = period_days / 4.0

    time = np.linspace(0, 3, 3000)
    flux = np.ones(len(time))

    inject_transit(
        flux, time, period_days=period_days, depth=depth,
        duration_days=duration_days, v_shape=False, t0=t0,
    )

    # Recompute in-transit mask the same way the injector does
    phase = ((time - t0) % period_days) / period_days
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    half_dur = (duration_days / period_days) / 2.0
    in_transit = np.abs(phase) < half_dur

    assert np.allclose(flux[in_transit], 1.0 - depth, atol=1e-9)
    assert np.allclose(flux[~in_transit], 1.0, atol=1e-9)


def test_v_shape_deeper_at_center():
    # for v_shape=True, the minimum should be at the midpoint of the transit
    period_days = 1.0
    depth = 0.1
    duration_days = 0.2
    t0 = period_days / 4.0

    # Restrict to a single transit window so there's an unambiguous minimum
    time = np.linspace(t0 - duration_days, t0 + duration_days, 2001)
    flux = np.ones(len(time))

    inject_transit(
        flux, time, period_days=period_days, depth=depth,
        duration_days=duration_days, v_shape=True, t0=t0,
    )

    idx_min = np.argmin(flux)
    assert abs(time[idx_min] - t0) < 1e-3
    # depth at the very centre should be (close to) the full depth
    assert flux.min() < 1.0 - depth * 0.9


def test_transit_count():
    # with period=3.42 days and span=27 days -> expect 7-8 transits
    time = np.linspace(0, 27, 17600)
    count = compute_transit_count(time, period_days=3.42, duration_days=0.16)
    assert 7 <= count <= 8


def test_secondary_eclipse_at_half_phase():
    # secondary eclipse minimum should be at phase 0.5
    period_days = 1.0
    secondary_depth = 0.05
    duration_days = 0.1
    t0 = period_days / 4.0
    t0_secondary = t0 + period_days / 2.0

    time = np.linspace(0, 3, 3000)
    flux = np.ones(len(time))

    inject_secondary_eclipse(
        flux, time, period_days=period_days, secondary_depth=secondary_depth,
        duration_days=duration_days, t0=t0,
    )

    # nearest sample to the first secondary-eclipse midpoint should be dimmed
    idx_nearest = np.argmin(np.abs(time - t0_secondary))
    assert flux[idx_nearest] < 1.0 - secondary_depth * 0.9

    # primary transit phase (t0) should be untouched by the secondary injector
    idx_primary = np.argmin(np.abs(time - t0))
    assert flux[idx_primary] == 1.0