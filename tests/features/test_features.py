"""Tests for deterministic statistical features and metadata."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from features.exceptions import FeatureError
from features.metadata import generate_feature_record, generate_metadata
from features.statistics import generate_statistics
from fits.models import LightCurveMetadata
from mast.models import Mission
from preprocessing.models import (
    PreprocessedLightCurve,
    PreprocessingConfig,
    PreprocessingMetadata,
)


def _processed(
    time: np.ndarray,
    flux: np.ndarray,
    quality: np.ndarray | None = None,
) -> PreprocessedLightCurve:
    """Create a compact processed light curve with stable provenance."""
    arrays = [np.array(time, dtype=np.float64), np.array(flux, dtype=np.float64)]
    for array in arrays:
        array.setflags(write=False)
    quality_array = (
        None if quality is None else np.array(quality, dtype=np.int64, copy=True)
    )
    if quality_array is not None:
        quality_array.setflags(write=False)
    config = PreprocessingConfig()
    metadata = PreprocessingMetadata(
        source=LightCurveMetadata(
            mission=Mission.TESS,
            source_path=Path("source.fits"),
            hdu_index=1,
            hdu_name="LIGHTCURVE",
            flux_column="PDCSAP_FLUX",
            quality_column=None if quality is None else "QUALITY",
            target_name="TIC 1",
            observation_id="obs-1",
        ),
        config=config,
        input_samples=len(time) + 2,
        non_finite_removed=1,
        quality_removed=1,
        output_samples=len(time),
    )
    return PreprocessedLightCurve(
        time=arrays[0],
        flux=arrays[1],
        quality=quality_array,
        normalized_flux=arrays[1],
        median_filtered_flux=arrays[1],
        wavelet_flux=arrays[1],
        metadata=metadata,
    )


def test_statistics_use_documented_population_definitions() -> None:
    """Known samples produce exact duration, cadence, RMS, and SNR semantics."""
    processed = _processed(
        np.array([1.0, 3.0, 5.0]),
        np.array([0.9, 1.0, 1.1]),
    )

    features = generate_statistics(processed)

    assert features.sample_count == 3
    assert features.mean == pytest.approx(1.0)
    assert features.standard_deviation == pytest.approx(np.sqrt(0.02 / 3.0))
    assert features.rms == pytest.approx(np.sqrt(0.02 / 3.0))
    assert features.signal_to_noise_ratio == pytest.approx(1.0 / np.sqrt(0.02 / 3.0))
    assert features.flux_variance == pytest.approx(0.02 / 3.0)
    assert features.observation_duration == pytest.approx(4.0)
    assert features.cadence == pytest.approx(2.0)


def test_constant_single_sample_has_undefined_snr_and_cadence() -> None:
    """Undefined scalar concepts remain null instead of becoming infinity."""
    features = generate_statistics(_processed(np.array([2.0]), np.array([1.0])))

    assert features.standard_deviation == 0.0
    assert features.rms == 0.0
    assert features.signal_to_noise_ratio is None
    assert features.observation_duration == 0.0
    assert features.cadence is None


@pytest.mark.parametrize(
    ("time", "flux", "message"),
    [
        (np.array([]), np.array([]), "non-empty"),
        (np.array([1.0, 2.0]), np.array([1.0]), "aligned"),
        (np.array([1.0, np.nan]), np.array([1.0, 1.0]), "finite"),
        (np.array([2.0, 1.0]), np.array([1.0, 1.0]), "strictly increasing"),
        (np.array([[1.0]]), np.array([[1.0]]), "non-empty 1D"),
    ],
)
def test_statistics_reject_invalid_series(
    time: np.ndarray, flux: np.ndarray, message: str
) -> None:
    """Malformed processed-like inputs cannot generate feature records."""
    value = SimpleNamespace(time=time, wavelet_flux=flux)
    with pytest.raises(FeatureError, match=message):
        generate_statistics(value)  # type: ignore[arg-type]


def test_metadata_and_feature_record_are_stable() -> None:
    """Metadata captures source, configuration, and sample accounting."""
    processed = _processed(
        np.array([1.0, 2.0]),
        np.array([0.99, 1.01]),
        np.array([0, 0]),
    )

    metadata = generate_metadata(processed)
    record = generate_feature_record(processed)

    assert metadata.schema_version == "1.0"
    assert metadata.pipeline_version == "0.1.0"
    assert metadata.mission is Mission.TESS
    assert metadata.target_name == "TIC 1"
    assert metadata.input_samples == 4
    assert metadata.non_finite_removed == 1
    assert metadata.quality_removed == 1
    assert metadata.output_samples == 2
    assert record.metadata == metadata
    assert record.statistics.sample_count == 2
    assert generate_feature_record(processed) == record
