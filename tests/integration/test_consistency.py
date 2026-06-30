"""Whole-pipeline repeatability and artifact consistency tests."""

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from exporters import export_numpy, export_parquet
from features import generate_feature_record
from fits import read_fits
from mast.models import Mission
from preprocessing import preprocess_light_curve

MissionFitsFactory = Callable[[Path, Mission, int, bool], Path]


def _digest(path: Path) -> str:
    """Return a complete artifact SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.integration
def test_identical_fits_produces_identical_arrays_features_and_artifacts(
    tmp_path: Path,
    mission_fits_factory: MissionFitsFactory,
) -> None:
    """Repeat the entire local pipeline and compare every stable output."""
    source = mission_fits_factory(
        tmp_path / "source.fits",
        Mission.TESS,
        512,
        True,
    )

    first = preprocess_light_curve(read_fits(source))
    second = preprocess_light_curve(read_fits(source))
    first_record = generate_feature_record(first)
    second_record = generate_feature_record(second)

    for first_array, second_array in (
        (first.time, second.time),
        (first.flux, second.flux),
        (first.normalized_flux, second.normalized_flux),
        (first.median_filtered_flux, second.median_filtered_flux),
        (first.wavelet_flux, second.wavelet_flux),
        (first.quality, second.quality),
    ):
        np.testing.assert_array_equal(first_array, second_array)
    assert first_record == second_record

    first_numpy = export_numpy(first, first_record, tmp_path / "first.npz")
    second_numpy = export_numpy(second, second_record, tmp_path / "second.npz")
    first_parquet = export_parquet(first, first_record, tmp_path / "first.parquet")
    second_parquet = export_parquet(
        second,
        second_record,
        tmp_path / "second.parquet",
    )
    assert _digest(first_numpy) == _digest(second_numpy)
    assert _digest(first_parquet) == _digest(second_parquet)
