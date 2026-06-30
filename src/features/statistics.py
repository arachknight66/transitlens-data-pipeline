"""Deterministic statistical feature generation."""

import numpy as np

from features.exceptions import FeatureError
from features.models import StatisticalFeatures
from preprocessing.models import PreprocessedLightCurve


def generate_statistics(light_curve: PreprocessedLightCurve) -> StatisticalFeatures:
    """Compute scalar features from wavelet-denoised normalized flux.

    Standard deviation and variance use population definitions (``ddof=0``).
    RMS is the root-mean-square deviation from the median flux baseline. Signal
    to noise is the absolute mean divided by population standard deviation and
    is undefined (``None``) for a constant signal. Cadence is the median finite
    timestamp difference and is undefined for a single sample.

    Args:
        light_curve: Fully preprocessed and cadence-aligned light curve.

    Returns:
        Validated deterministic statistical features.

    Raises:
        FeatureError: If arrays are non-finite, misaligned, or temporally invalid.
    """
    time = np.asarray(light_curve.time, dtype=np.float64)
    flux = np.asarray(light_curve.wavelet_flux, dtype=np.float64)
    if time.ndim != 1 or flux.ndim != 1 or time.size == 0:
        raise FeatureError("feature generation requires non-empty 1D time and flux")
    if time.size != flux.size:
        raise FeatureError("feature generation requires aligned time and flux")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(flux)):
        raise FeatureError("feature generation requires finite time and flux")

    differences = np.diff(time)
    if differences.size and np.any(differences <= 0.0):
        raise FeatureError("feature generation requires strictly increasing time")
    mean = float(np.mean(flux))
    standard_deviation = float(np.std(flux, ddof=0))
    median = float(np.median(flux))
    rms = float(np.sqrt(np.mean(np.square(flux - median))))
    variance = float(np.var(flux, ddof=0))
    signal_to_noise = (
        None if standard_deviation == 0.0 else abs(mean) / standard_deviation
    )
    duration = float(time[-1] - time[0])
    cadence = None if not differences.size else float(np.median(differences))
    return StatisticalFeatures(
        sample_count=int(time.size),
        mean=mean,
        standard_deviation=standard_deviation,
        rms=rms,
        signal_to_noise_ratio=signal_to_noise,
        flux_variance=variance,
        observation_duration=duration,
        cadence=cadence,
    )
