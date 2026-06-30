"""Robust deterministic flux normalization."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from preprocessing.exceptions import NormalizationError


def normalize_flux(flux: ArrayLike) -> NDArray[np.float64]:
    """Normalize finite flux by its robust median baseline.

    Args:
        flux: One-dimensional finite flux measurements.

    Returns:
        Independent read-only flux values centered around unity.

    Raises:
        NormalizationError: If the signal is empty, non-finite, or has an
            unusable median baseline.
    """
    values = np.asarray(flux, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise NormalizationError("flux must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise NormalizationError("flux must contain only finite values")
    baseline = float(np.median(values))
    tolerance = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(values))))
    if not np.isfinite(baseline) or baseline <= tolerance:
        message = "flux median must be positive and safely separated from zero"
        raise NormalizationError(message)
    normalized = np.array(values / baseline, dtype=np.float64, copy=True)
    normalized.setflags(write=False)
    return normalized
