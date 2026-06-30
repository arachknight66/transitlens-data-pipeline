"""Conservative median filtering for normalized light curves."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import median_filter

from preprocessing.exceptions import FilteringError


def median_filter_flux(
    flux: ArrayLike,
    window: int = 5,
) -> NDArray[np.float64]:
    """Apply a small centered median filter without zero-padded edges.

    Args:
        flux: Finite normalized flux measurements.
        window: Odd cadence window of at least three samples.

    Returns:
        Independent read-only median-filtered flux.

    Raises:
        FilteringError: If the signal or window is invalid.
    """
    values = np.asarray(flux, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise FilteringError("flux must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise FilteringError("median filtering requires finite flux values")
    if window < 3 or window % 2 == 0:
        raise FilteringError(
            "median filter window must be an odd integer of at least 3"
        )
    if values.size < 3:
        result = np.array(values, dtype=np.float64, copy=True)
    else:
        effective_window = min(window, _largest_odd(values.size))
        result = np.asarray(
            median_filter(values, size=effective_window, mode="nearest"),
            dtype=np.float64,
        ).copy()
    result.setflags(write=False)
    return result


def _largest_odd(value: int) -> int:
    """Return the largest odd integer not exceeding a positive value."""
    return value if value % 2 else value - 1
