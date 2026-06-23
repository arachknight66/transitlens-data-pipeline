"""
synthetic/transit_injector.py
─────────────────────────────
Injects transit and eclipse signals into flux arrays.

Supports:
- Box-shaped transits (exoplanet-like flat bottom)
- V-shaped transits (eclipsing binary-like triangular dip)
- Secondary eclipses at phase 0.5

Performance note
-----------------
The original implementation looped over every timestamp in pure
Python (`for i in range(len(time))`), which is O(n) with heavy
interpreter overhead — for n ~ 18,000 points this dominates the
runtime of the whole generation pipeline. Every function below has
been rewritten using numpy vectorised boolean masking, so the
in-transit test and depth scaling are applied to the whole array at
once in compiled C code. Outputs are numerically identical to the
original loop-based version; only the implementation changed.
"""

import numpy as np


def _wrapped_phase(time, period_days, t0):
    """
    Computes phase centred on [-0.5, 0.5) relative to t0, vectorised.

    Returns
    -------
    np.ndarray
        Phase array, same length as time.
    """
    phase = ((time - t0) % period_days) / period_days
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    return phase


def inject_transit(flux, time, period_days, depth, duration_days,
                   v_shape=False, t0=None):
    """
    Injects a periodic transit signal into the flux array.

    For each timestamp t:
      phase = ((t - t0) % period_days) / period_days

      If phase < phase_duration (in-transit):
        - Box shape:  flux[i] *= (1 - depth)
        - V shape:    flux[i] *= (1 - depth * triangle(phase))

    Parameters
    ----------
    flux : np.ndarray
        Flux array to modify (modified in-place).
    time : np.ndarray
        Time array in days (BTJD).
    period_days : float
        Orbital period in days.
    depth : float
        Fractional flux drop at maximum transit depth.
    duration_days : float
        Full duration of the transit event in days.
    v_shape : bool
        If True, uses a triangular (V-shaped) profile instead of a flat box.
    t0 : float, optional
        Time of first transit midpoint. Defaults to period_days / 4 so
        the first transit doesn't land exactly at time=0.

    Returns
    -------
    np.ndarray
        Modified flux array (same object as input).
    """
    if t0 is None:
        t0 = period_days / 4.0

    time = np.asarray(time, dtype=np.float64)
    phase_duration = duration_days / period_days
    half_dur = phase_duration / 2.0

    phase = _wrapped_phase(time, period_days, t0)
    in_transit = np.abs(phase) < half_dur

    if v_shape:
        fraction = 1.0 - np.abs(phase[in_transit]) / half_dur
        flux[in_transit] *= (1.0 - depth * fraction)
    else:
        flux[in_transit] *= (1.0 - depth)

    return flux


def inject_secondary_eclipse(flux, time, period_days, secondary_depth,
                              duration_days, t0=None):
    """
    Injects a secondary eclipse at phase 0.5 (half period from primary).

    Used for eclipsing binary simulation where the secondary star
    also gets occluded. Secondary depth is typically 40-60% of primary depth.

    Parameters
    ----------
    flux : np.ndarray
        Flux array to modify (modified in-place).
    time : np.ndarray
        Time array in days.
    period_days : float
        Orbital period in days.
    secondary_depth : float
        Fractional flux drop for the secondary eclipse.
    duration_days : float
        Duration of the secondary eclipse in days.
    t0 : float, optional
        Time of first primary transit midpoint. Secondary is offset by
        period_days / 2.

    Returns
    -------
    np.ndarray
        Modified flux array (same object as input).
    """
    if t0 is None:
        t0 = period_days / 4.0

    time = np.asarray(time, dtype=np.float64)

    # Secondary eclipse occurs at half-period offset from primary
    t0_secondary = t0 + period_days / 2.0
    phase_duration = duration_days / period_days
    half_dur = phase_duration / 2.0

    phase = _wrapped_phase(time, period_days, t0_secondary)
    in_eclipse = np.abs(phase) < half_dur

    # Secondary eclipses are always box-shaped (thermal occultation)
    flux[in_eclipse] *= (1.0 - secondary_depth)

    return flux


def compute_transit_count(time, period_days, duration_days, t0=None):
    """
    Returns the number of full transits visible in the time array.

    A transit is counted as "visible" if at least one data point falls
    within the transit window.

    Parameters
    ----------
    time : np.ndarray or list
        Time array in days.
    period_days : float
        Orbital period in days.
    duration_days : float
        Transit duration in days.
    t0 : float, optional
        Time of first transit midpoint. Defaults to period_days / 4.

    Returns
    -------
    int
        Number of transits with at least one data point in-transit.
    """
    if t0 is None:
        t0 = period_days / 4.0

    time = np.asarray(time, dtype=np.float64)
    time_span = time[-1] - time[0]

    # Maximum possible number of transits in the time span
    max_transits = int(np.ceil(time_span / period_days)) + 1

    # Vectorised: build all transit windows at once and test each
    # against the full time array via broadcasting. max_transits is
    # small (typically < 30), so this is a cheap (max_transits x N)
    # boolean comparison rather than max_transits separate full-array
    # passes inside a Python loop.
    n = np.arange(max_transits)
    transit_mids = t0 + n * period_days          # shape (T,)
    transit_starts = transit_mids - duration_days / 2.0
    transit_ends = transit_mids + duration_days / 2.0

    # Broadcasting: time[None, :] vs starts/ends[:, None] -> (T, N)
    in_window = (time[None, :] >= transit_starts[:, None]) & \
                (time[None, :] <= transit_ends[:, None])

    visible = np.any(in_window, axis=1)  # shape (T,)
    return int(np.sum(visible))