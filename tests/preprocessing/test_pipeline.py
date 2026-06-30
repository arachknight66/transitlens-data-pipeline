"""End-to-end scientific tests for Phase 4 preprocessing."""

import numpy as np
import pytest
from pydantic import ValidationError

from preprocessing.models import PreprocessingConfig
from preprocessing.pipeline import preprocess_light_curve


def _trapezoid_transit(samples: int = 512) -> np.ndarray:
    """Create a noiseless two-percent transit with finite ingress and egress."""
    center = samples // 2
    distance = np.abs(np.arange(samples) - center)
    shape = np.zeros(samples, dtype=np.float64)
    shape[distance <= 20] = 1.0
    transition = (distance > 20) & (distance < 30)
    shape[transition] = (30.0 - distance[transition]) / 10.0
    return 1.0 - 0.02 * shape


def _transit_depth(signal: np.ndarray) -> float:
    """Measure robust out-of-transit minus flat-bottom flux."""
    center = len(signal) // 2
    indices = np.arange(len(signal))
    outside = np.abs(indices - center) > 45
    bottom = np.abs(indices - center) <= 15
    return float(np.median(signal[outside]) - np.median(signal[bottom]))


def test_pipeline_cleans_filters_and_records_sample_counts(
    light_curve_factory,
) -> None:
    """Operations run in order and retain auditable sample accounting."""
    light_curve = light_curve_factory(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([100.0, np.nan, 98.0, 99.0, 101.0]),
        np.array([0, 0, 1, 0, 0]),
    )

    processed = preprocess_light_curve(light_curve)

    np.testing.assert_array_equal(processed.time, [1.0, 4.0, 5.0])
    assert processed.metadata.input_samples == 5
    assert processed.metadata.non_finite_removed == 1
    assert processed.metadata.quality_removed == 1
    assert processed.metadata.output_samples == 3
    assert len(processed.normalized_flux) == len(processed.wavelet_flux) == 3
    assert not processed.wavelet_flux.flags.writeable


def test_pipeline_preserves_transit_depth_ingress_and_egress(
    light_curve_factory,
) -> None:
    """Conservative filters reduce noise without erasing transit morphology."""
    clean = _trapezoid_transit()
    samples = np.arange(clean.size, dtype=np.float64)
    detector_noise = 0.0015 * np.sin(2.0 * np.pi * samples * 0.41)
    detector_noise += 0.0007 * np.sin(2.0 * np.pi * samples * 0.23)
    noisy = clean + detector_noise
    light_curve = light_curve_factory(samples, noisy * 10_000.0)

    processed = preprocess_light_curve(light_curve)

    clean_normalized = clean / np.median(clean)
    expected_depth = _transit_depth(clean_normalized)
    actual_depth = _transit_depth(processed.wavelet_flux)
    assert actual_depth == pytest.approx(expected_depth, rel=0.05)

    center = clean.size // 2
    ingress = slice(center - 30, center - 19)
    egress = slice(center + 20, center + 31)
    ingress_error = np.sqrt(
        np.mean((processed.wavelet_flux[ingress] - clean_normalized[ingress]) ** 2)
    )
    egress_error = np.sqrt(
        np.mean((processed.wavelet_flux[egress] - clean_normalized[egress]) ** 2)
    )
    assert ingress_error < 0.0015
    assert egress_error < 0.0015

    outside = np.abs(samples - center) > 45
    input_noise = np.std(processed.normalized_flux[outside] - clean_normalized[outside])
    output_noise = np.std(processed.wavelet_flux[outside] - clean_normalized[outside])
    assert output_noise < input_noise


def test_pipeline_is_bitwise_deterministic(light_curve_factory) -> None:
    """Identical raw arrays and settings always produce identical output."""
    samples = np.arange(128, dtype=np.float64)
    flux = 1000.0 + np.sin(samples)
    light_curve = light_curve_factory(samples, flux)

    first = preprocess_light_curve(light_curve)
    second = preprocess_light_curve(light_curve)

    np.testing.assert_array_equal(first.time, second.time)
    np.testing.assert_array_equal(first.normalized_flux, second.normalized_flux)
    np.testing.assert_array_equal(first.wavelet_flux, second.wavelet_flux)
    assert first.metadata == second.metadata


def test_preprocessing_config_requires_odd_window() -> None:
    """Configuration rejects a median window without a center cadence."""
    with pytest.raises(ValidationError, match="must be odd"):
        PreprocessingConfig(median_window=4)
