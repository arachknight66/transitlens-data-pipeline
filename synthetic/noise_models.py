"""
synthetic/noise_models.py
─────────────────────────
Adds realistic photometric noise to base flux arrays.

Provides three noise models:
- Gaussian (white) noise: i.i.d. normal, typical for bright quiet stars
- Red (correlated) noise: AR(1) process, mimics instrumental systematics
- Stellar variability: sinusoidal rotation signal for variable stars
"""

import numpy as np


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
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, len(flux))
    return flux + noise


def add_red_noise(flux, sigma, correlation=0.3, seed=None):
    """
    Adds correlated (red) noise via an AR(1) process.

    AR(1): noise[i] = correlation * noise[i-1] + white_noise[i]
    Mimics instrumental systematics and stellar variability trends.

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
    rng = np.random.default_rng(seed)
    n = len(flux)
    white = rng.normal(0, sigma, n)

    # Build AR(1) noise series
    red = np.zeros(n)
    red[0] = white[0]
    for i in range(1, n):
        red[i] = correlation * red[i - 1] + white[i]

    # Scale so that the overall std is approximately sigma
    # The theoretical std of AR(1) is sigma / sqrt(1 - correlation^2)
    # We rescale to match the target sigma
    if correlation < 1.0:
        scale_factor = np.sqrt(1 - correlation**2)
        red *= scale_factor

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
    rng = np.random.default_rng(seed)
    # Random phase offset so it doesn't always start at the same phase
    phase_offset = rng.uniform(0, 2 * np.pi)
    variability = amplitude * np.sin(2 * np.pi * time / period_days + phase_offset)
    return flux + variability
