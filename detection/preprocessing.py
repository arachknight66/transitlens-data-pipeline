"""
detection/preprocessing.py
────────────────────────────
Cleans a (time, flux) light curve before it goes into the BLS search.

Design note
-----------
Synthetic light curves from transitlens-data-pipeline are already flat
outside injected transits (the only "noise" is exactly the Gaussian/red
noise the generator added on purpose) — they don't need detrending, and
detrending them risks shaving real depth off the very signal we're
trying to detect. Real TESS data is messier and often needs flattening
to remove long-term instrumental trends.

Because of this, `clean_light_curve()` (the function `detection/bls.py`
actually calls) only does the always-safe steps — NaN removal and
sigma-clipping — by default. `flatten()` is provided as an explicit,
opt-in step for real data; it is never applied automatically.
"""

import numpy as np
from scipy.signal import savgol_filter


def remove_nans(time, flux):
    """
    Drops any cadence where either time or flux is NaN/non-finite.

    Parameters
    ----------
    time : np.ndarray
    flux : np.ndarray

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Filtered (time, flux), same relative order, NaNs removed.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if len(time) != len(flux):
        raise ValueError(
            f"time and flux must be the same length, got {len(time)} vs {len(flux)}."
        )

    mask = np.isfinite(time) & np.isfinite(flux)
    return time[mask], flux[mask]


def sigma_clip(time, flux, sigma=5.0, max_iters=3):
    """
    Removes outliers more than `sigma` robust-standard-deviations from
    the median, iterating up to `max_iters` times since removing one
    outlier can reveal previously-hidden ones at the same threshold.

    Uses the median absolute deviation (MAD), scaled to approximate a
    Gaussian sigma, rather than `np.std` — the same robustness fix
    used in transitlens-data-pipeline's `real_tess/flux_normaliser.py`,
    since a single huge outlier otherwise inflates its own threshold.

    Parameters
    ----------
    time : np.ndarray
    flux : np.ndarray
    sigma : float
        Clipping threshold in robust-sigma units.
    max_iters : int
        Maximum number of clipping passes.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Filtered (time, flux) with outliers removed.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    for _ in range(max_iters):
        if len(flux) == 0:
            break

        median = np.median(flux)
        mad = np.median(np.abs(flux - median))
        robust_sigma = 1.4826 * mad

        if robust_sigma == 0:
            break

        keep = np.abs(flux - median) < sigma * robust_sigma

        if np.all(keep):
            break

        time, flux = time[keep], flux[keep]

    return time, flux


def flatten(time, flux, window_length=401, polyorder=2):
    """
    Removes long-term trends with a Savitzky-Golay filter, dividing
    the raw flux by its smoothed trend. Opt-in only — see module
    docstring for why this isn't applied automatically.

    `window_length` must be substantially larger than the transit
    duration (in cadences) or the filter will partially flatten out
    the transit itself, biasing the depth low. As a rule of thumb,
    window_length should span at least ~5-10x the longest expected
    transit duration.

    Parameters
    ----------
    time : np.ndarray
    flux : np.ndarray
    window_length : int
        Savitzky-Golay window size in cadences. Must be odd; will be
        rounded up to the nearest odd number if necessary, and capped
        to the input length if larger.
    polyorder : int
        Polynomial order for the filter (must be < window_length).

    Returns
    -------
    np.ndarray
        Flattened flux, same length as input, median-normalised to 1.0.

    Raises
    ------
    ValueError
        If there are too few points to apply the filter at all.
    """
    flux = np.asarray(flux, dtype=np.float64)
    n = len(flux)

    if n < 5:
        raise ValueError(
            f"flatten() needs at least 5 points to fit a trend, got {n}."
        )

    w = min(window_length, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    w = max(w, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)

    trend = savgol_filter(flux, window_length=w, polyorder=polyorder)
    flattened = flux / trend
    return flattened * np.median(flux) / np.median(flattened)


def clean_light_curve(time, flux, sigma=5.0):
    """
    The default preprocessing pipeline `detection/bls.py` calls:
    NaN removal followed by sigma-clipping. Does NOT flatten/detrend
    — see module docstring.

    Parameters
    ----------
    time : np.ndarray or list
    flux : np.ndarray or list
    sigma : float
        Passed through to `sigma_clip`.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Cleaned (time, flux), sorted by time.
    """
    time, flux = remove_nans(time, flux)

    order = np.argsort(time)
    time, flux = time[order], flux[order]

    time, flux = sigma_clip(time, flux, sigma=sigma)

    return time, flux