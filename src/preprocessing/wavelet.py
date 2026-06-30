"""Transit-preserving adaptive wavelet denoising."""

from typing import Literal

import numpy as np
import pywt
from numpy.typing import ArrayLike, NDArray

from preprocessing.exceptions import FilteringError

ThresholdMode = Literal["soft"]


def wavelet_denoise(
    flux: ArrayLike,
    *,
    wavelet: str = "db4",
    threshold_mode: ThresholdMode = "soft",
    threshold_scale: float = 0.5,
    max_level: int = 2,
) -> NDArray[np.float64]:
    """Denoise only the highest-frequency wavelet detail coefficients.

    Limiting shrinkage to the finest detail band avoids suppressing the
    lower-frequency transit depth, ingress, and egress morphology.

    Args:
        flux: Finite median-filtered normalized flux.
        wavelet: PyWavelets wavelet name; ``db4`` is the project default.
        threshold_mode: PyWavelets threshold mode. Only conservative soft
            shrinkage is supported.
        threshold_scale: Fraction of the adaptive universal threshold.
        max_level: Maximum wavelet decomposition depth.

    Returns:
        Independent read-only denoised flux with unchanged length.

    Raises:
        FilteringError: If parameters or signal values are invalid.
    """
    values = np.array(flux, dtype=np.float64, copy=True)
    if values.ndim != 1 or values.size == 0:
        raise FilteringError("flux must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise FilteringError("wavelet denoising requires finite flux values")
    if threshold_mode != "soft":
        raise FilteringError("wavelet threshold mode must be 'soft'")
    if not 0.0 <= threshold_scale <= 1.0:
        raise FilteringError("wavelet threshold scale must be between 0 and 1")
    if max_level < 1:
        raise FilteringError("wavelet maximum level must be at least 1")
    try:
        wavelet_object = pywt.Wavelet(wavelet)
    except ValueError as error:
        raise FilteringError(f"unknown wavelet '{wavelet}'") from error

    possible_level = pywt.dwt_max_level(values.size, wavelet_object.dec_len)
    level = min(max_level, possible_level)
    if level == 0 or threshold_scale == 0.0:
        return _immutable(values)

    coefficients = pywt.wavedec(
        values,
        wavelet_object,
        mode="symmetric",
        level=level,
    )
    finest_detail = coefficients[-1]
    sigma = _median_absolute_deviation(finest_detail) / 0.6744897501960817
    if sigma > 0.0 and finest_detail.size > 1:
        threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(values.size))
        coefficients[-1] = pywt.threshold(
            finest_detail,
            threshold,
            mode=threshold_mode,
        )
    reconstructed = pywt.waverec(coefficients, wavelet_object, mode="symmetric")
    return _immutable(reconstructed[: values.size])


def _median_absolute_deviation(values: NDArray[np.float64]) -> float:
    """Return a deterministic robust scale estimate."""
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _immutable(values: ArrayLike) -> NDArray[np.float64]:
    """Return an independent read-only float64 array."""
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result
