"""
synthetic/noise_models.py
─────────────────────────
Adds realistic photometric noise to base flux arrays.

Provides three noise models:
- Gaussian (white) noise: i.i.d. normal, typical for bright quiet stars
- Red (correlated) noise: AR(1) process, mimics instrumental systematics
- Stellar variability: sinusoidal rotation signal for variable stars

Performance note
-----------------
The AR(1) red-noise recursion is inherently sequential (each sample
depends on the previous one), so it cannot be fully vectorised with
plain numpy broadcasting. Instead of a Python `for` loop (slow,
O(n) with per-iteration interpreter overhead), this version uses
`scipy.signal.lfilter`, which runs the same IIR recursion as
compiled C code. For n ~ 18,000 points this is roughly 50-100x
faster than the pure-Python loop while producing numerically
identical results.
"""

import numpy as np
from scipy.signal import lfilter


def add_gaussian_noise(flux, sigma, seed=None):
    """
    Adds i.i.d. Gaussian noise with standard deviation sigma.

    Parameters
    ----------
    flux : np.ndarray
        Base flux array (will not be modified in-place).
    sigma : float
        Standard deviation of the noise. ~0.002 means 2000 ppm noise floor,
        realistic for a V=11 star observed by TESS.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy flux array (same length as input).
    """
    flux = np.asarray(flux, dtype=np.float64)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, len(flux))
    return flux + noise


def add_red_noise(flux, sigma, correlation=0.3, seed=None):
    """
    Adds correlated (red) noise via an AR(1) process.

    AR(1): noise[i] = correlation * noise[i-1] + white_noise[i]
    Mimics instrumental systematics and stellar variability trends.

    Implementation detail
    ----------------------
    The recursion noise[i] = correlation*noise[i-1] + white[i] is an
    IIR filter with transfer function 1 / (1 - correlation * z^-1).
    scipy.signal.lfilter applies exactly this recursion using a
    compiled loop, giving identical output to the naive Python loop
    but without per-sample Python interpreter overhead.

    Parameters
    ----------
    flux : np.ndarray
        Base flux array (will not be modified in-place).
    sigma : float
        Controls the overall amplitude of the noise.
    correlation : float
        AR(1) autoregressive coefficient. 0.0 = white noise, 1.0 = random walk.
        Default 0.3 produces mild correlated drift.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy flux array with correlated noise superimposed.
    """
    flux = np.asarray(flux, dtype=np.float64)
    n = len(flux)
    rng = np.random.default_rng(seed)
    white = rng.normal(0, sigma, n)

    # AR(1): red[i] = correlation*red[i-1] + white[i]
    # Equivalent IIR filter: b = [1], a = [1, -correlation]
    b = [1.0]
    a = [1.0, -correlation]
    red = lfilter(b, a, white)

    # Scale so that the overall std is approximately sigma
    # The theoretical std of AR(1) is sigma / sqrt(1 - correlation^2)
    # We rescale to match the target sigma
    if correlation < 1.0:
        scale_factor = np.sqrt(1 - correlation ** 2)
        red = red * scale_factor

    return flux + red


def add_stellar_variability(flux, time, amplitude=0.005, period_days=12.0,
                            seed=None):
    """
    Adds a sinusoidal stellar rotation signal.

    Mimics the photometric modulation caused by starspots rotating
    in and out of view. Amplitude ~0.5% is typical for a moderately
    active star.

    Parameters
    ----------
    flux : np.ndarray
        Base flux array.
    time : np.ndarray
        Time array in days.
    amplitude : float
        Peak-to-peak amplitude of the sinusoidal variation (fractional).
    period_days : float
        Stellar rotation period in days.
    seed : int, optional
        Random seed for reproducibility (used to randomise phase offset).

    Returns
    -------
    np.ndarray
        Flux array with stellar variability superimposed.
    """
    flux = np.asarray(flux, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)

    rng = np.random.default_rng(seed)
    # Random phase offset so it doesn't always start at the same phase
    phase_offset = rng.uniform(0, 2 * np.pi)
    variability = amplitude * np.sin(2 * np.pi * time / period_days + phase_offset)
    return flux + variability