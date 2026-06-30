"""Composition of the deterministic Phase 4 preprocessing sequence."""

from fits.models import LightCurve
from preprocessing.median_filter import median_filter_flux
from preprocessing.models import (
    PreprocessedLightCurve,
    PreprocessingConfig,
    PreprocessingMetadata,
)
from preprocessing.normalize import normalize_flux
from preprocessing.quality import filter_quality, remove_non_finite
from preprocessing.wavelet import wavelet_denoise


def preprocess_light_curve(
    light_curve: LightCurve,
    config: PreprocessingConfig | None = None,
) -> PreprocessedLightCurve:
    """Run the required preprocessing operations in their documented order.

    Args:
        light_curve: Raw structured light curve from Phase 3.
        config: Optional immutable processing parameters.

    Returns:
        Cleaned, normalized, median-filtered, and wavelet-denoised light curve.
    """
    resolved_config = config or PreprocessingConfig()
    time, flux, quality = remove_non_finite(light_curve)
    finite_samples = len(time)
    time, flux, quality = filter_quality(
        time,
        flux,
        quality,
        light_curve.metadata.mission,
        resolved_config.quality_bitmask,
    )
    normalized = normalize_flux(flux)
    median_filtered = median_filter_flux(normalized, resolved_config.median_window)
    denoised = wavelet_denoise(
        median_filtered,
        wavelet=resolved_config.wavelet,
        threshold_mode=resolved_config.wavelet_threshold_mode,
        threshold_scale=resolved_config.wavelet_threshold_scale,
        max_level=resolved_config.wavelet_max_level,
    )
    metadata = PreprocessingMetadata(
        source=light_curve.metadata,
        config=resolved_config,
        input_samples=len(light_curve.time),
        non_finite_removed=len(light_curve.time) - finite_samples,
        quality_removed=finite_samples - len(time),
        output_samples=len(time),
    )
    return PreprocessedLightCurve(
        time=time,
        flux=flux,
        quality=quality,
        normalized_flux=normalized,
        median_filtered_flux=median_filtered,
        wavelet_flux=denoised,
        metadata=metadata,
    )
