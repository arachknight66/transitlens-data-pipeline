"""Tests for median and wavelet filters."""

import numpy as np
import pytest

from preprocessing.exceptions import FilteringError
from preprocessing.median_filter import median_filter_flux
from preprocessing.wavelet import wavelet_denoise


def test_median_filter_removes_impulse_without_edge_zero_padding() -> None:
    """A small median filter removes isolated spikes and preserves edges."""
    flux = np.array([1.0, 1.0, 5.0, 1.0, 1.0])

    filtered = median_filter_flux(flux, 3)

    np.testing.assert_array_equal(filtered, np.ones(5))
    assert not filtered.flags.writeable


def test_median_filter_handles_short_signal_and_caps_window() -> None:
    """Short observations remain valid without oversized filter behavior."""
    np.testing.assert_array_equal(median_filter_flux([1.0, 2.0], 5), [1.0, 2.0])
    np.testing.assert_array_equal(
        median_filter_flux([1.0, 9.0, 1.0, 1.0], 9),
        [1.0, 1.0, 1.0, 1.0],
    )


@pytest.mark.parametrize("window", [0, 2, 4])
def test_median_filter_rejects_invalid_windows(window: int) -> None:
    """Median windows must be centered odd values."""
    with pytest.raises(FilteringError, match="odd integer"):
        median_filter_flux([1.0, 2.0, 3.0], window)


@pytest.mark.parametrize("flux", [[], [[1.0]], [1.0, np.nan]])
def test_median_filter_rejects_invalid_signals(flux: list[object]) -> None:
    """Median filtering requires non-empty finite one-dimensional flux."""
    with pytest.raises(FilteringError):
        median_filter_flux(flux)


def test_wavelet_denoising_is_deterministic_and_reduces_high_frequency() -> None:
    """Adaptive db4 shrinkage reduces deterministic detector-like noise."""
    samples = np.arange(256, dtype=np.float64)
    clean = 1.0 + 0.001 * np.sin(2.0 * np.pi * samples / 80.0)
    noisy = clean + 0.002 * np.sin(2.0 * np.pi * samples * 0.43)

    first = wavelet_denoise(noisy)
    second = wavelet_denoise(noisy)

    np.testing.assert_array_equal(first, second)
    assert np.std(first - clean) < np.std(noisy - clean)
    assert len(first) == len(noisy)
    assert not first.flags.writeable


def test_wavelet_short_and_zero_threshold_are_exact_copies() -> None:
    """Signals without a usable detail level are returned unchanged."""
    np.testing.assert_array_equal(wavelet_denoise([1.0, 2.0]), [1.0, 2.0])
    np.testing.assert_array_equal(
        wavelet_denoise(np.arange(16.0), threshold_scale=0.0),
        np.arange(16.0),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold_mode": "hard"}, "must be 'soft'"),
        ({"threshold_scale": -0.1}, "between 0 and 1"),
        ({"threshold_scale": 1.1}, "between 0 and 1"),
        ({"max_level": 0}, "at least 1"),
        ({"wavelet": "not-a-wavelet"}, "unknown wavelet"),
    ],
)
def test_wavelet_rejects_invalid_parameters(
    kwargs: dict[str, object], message: str
) -> None:
    """Unsupported wavelet settings fail explicitly."""
    with pytest.raises(FilteringError, match=message):
        wavelet_denoise(np.arange(32.0), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("flux", [[], [[1.0]], [1.0, np.inf]])
def test_wavelet_rejects_invalid_signals(flux: list[object]) -> None:
    """Wavelet filtering requires non-empty finite one-dimensional flux."""
    with pytest.raises(FilteringError):
        wavelet_denoise(flux)
