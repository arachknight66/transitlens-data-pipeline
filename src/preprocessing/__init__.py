"""Deterministic light-curve preprocessing package."""

from preprocessing.exceptions import (
    FilteringError,
    InvalidMeasurementsError,
    NormalizationError,
    PreprocessingError,
)
from preprocessing.median_filter import median_filter_flux
from preprocessing.models import (
    PreprocessedLightCurve,
    PreprocessingConfig,
    PreprocessingMetadata,
)
from preprocessing.normalize import normalize_flux
from preprocessing.pipeline import preprocess_light_curve
from preprocessing.quality import filter_quality, remove_non_finite
from preprocessing.wavelet import wavelet_denoise

__all__ = [
    "FilteringError",
    "InvalidMeasurementsError",
    "NormalizationError",
    "PreprocessedLightCurve",
    "PreprocessingConfig",
    "PreprocessingError",
    "PreprocessingMetadata",
    "filter_quality",
    "median_filter_flux",
    "normalize_flux",
    "preprocess_light_curve",
    "remove_non_finite",
    "wavelet_denoise",
]
