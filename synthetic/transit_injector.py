"""
synthetic/transit_injector.py
─────────────────────────────
Injects transit and eclipse signals into flux arrays.

Supports:
- Box-shaped transits (exoplanet-like flat bottom)
- V-shaped transits (eclipsing binary-like triangular dip)
- Secondary eclipses at phase 0.5

All functions modify flux in-place for efficiency.
"""

import numpy as np


def inject_transit(flux, time, period_days, depth, duration_days,
                   v_shape=False, t0=None):
    """
    Injects a periodic transit signal into the flux array.

    For each timestamp t:
      phase = ((t - t0) % period_days) / period_days
      phase_duration = duration_days / period_days

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

    phase_duration = duration_days / period_days

    for i in range(len(time)):
        # Compute phase within [0, 1)
        phase = ((time[i] - t0) % period_days) / period_days

        # Centre the transit on phase=0 by shifting to [-0.5, 0.5)
        if phase > 0.5:
            phase -= 1.0

        half_dur = phase_duration / 2.0

        if abs(phase) < half_dur:
            if v_shape:
                # Triangle: max depth at centre (phase=0), zero at edges
                # fraction goes from 1.0 at centre to 0.0 at edge
                fraction = 1.0 - abs(phase) / half_dur
                flux[i] *= (1.0 - depth * fraction)
            else:
                # Box: uniform depth across the entire transit window
                flux[i] *= (1.0 - depth)

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

    # Secondary eclipse occurs at half-period offset from primary
    t0_secondary = t0 + period_days / 2.0
    phase_duration = duration_days / period_days

    for i in range(len(time)):
        phase = ((time[i] - t0_secondary) % period_days) / period_days

        # Centre on phase=0
        if phase > 0.5:
            phase -= 1.0

        half_dur = phase_duration / 2.0

        if abs(phase) < half_dur:
            # Secondary eclipses are always box-shaped (thermal occultation)
            flux[i] *= (1.0 - secondary_depth)

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

    time = np.asarray(time)
    time_span = time[-1] - time[0]

    # Maximum possible number of transits in the time span
    max_transits = int(np.ceil(time_span / period_days)) + 1

    count = 0
    for n in range(max_transits):
        transit_mid = t0 + n * period_days
        transit_start = transit_mid - duration_days / 2.0
        transit_end = transit_mid + duration_days / 2.0

        # Check if any data points fall within this transit window
        in_transit = np.any((time >= transit_start) & (time <= transit_end))
        if in_transit:
            count += 1

    return count
