"""Round-trip and byte-consistency tests for dataset exporters."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from exporters import ExportError, export_numpy, export_parquet
from features.metadata import generate_feature_record
from fits.models import LightCurveMetadata
from mast.models import Mission
from preprocessing.models import (
    PreprocessedLightCurve,
    PreprocessingConfig,
    PreprocessingMetadata,
)


def _processed(with_quality: bool = True) -> PreprocessedLightCurve:
    """Create a representative immutable dataset for export."""
    time = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    flux = np.array([100.0, 99.0, 101.0], dtype=np.float64)
    normalized = np.array([1.0, 0.99, 1.01], dtype=np.float64)
    median = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    wavelet = np.array([1.0, 0.995, 1.005], dtype=np.float64)
    quality = np.array([0, 0, 2], dtype=np.int64) if with_quality else None
    for array in (time, flux, normalized, median, wavelet, quality):
        if array is not None:
            array.setflags(write=False)
    metadata = PreprocessingMetadata(
        source=LightCurveMetadata(
            mission=Mission.KEPLER,
            source_path=Path("fixture.fits"),
            hdu_index=1,
            hdu_name="LIGHTCURVE",
            flux_column="PDCSAP_FLUX",
            quality_column="SAP_QUALITY" if with_quality else None,
            target_name="KIC 1",
            observation_id="obs-1",
        ),
        config=PreprocessingConfig(),
        input_samples=4,
        non_finite_removed=1,
        quality_removed=0,
        output_samples=3,
    )
    return PreprocessedLightCurve(
        time=time,
        flux=flux,
        quality=quality,
        normalized_flux=normalized,
        median_filtered_flux=median,
        wavelet_flux=wavelet,
        metadata=metadata,
    )


def _sha256(path: Path) -> str:
    """Return the complete artifact digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_numpy_export_round_trips_without_pickle(tmp_path: Path) -> None:
    """NPZ output contains canonical arrays and JSON-only feature metadata."""
    processed = _processed()
    features = generate_feature_record(processed)

    path = export_numpy(processed, features, tmp_path / "nested" / "dataset.npz")

    with np.load(path, allow_pickle=False) as dataset:
        assert sorted(dataset.files) == [
            "features_json",
            "flux",
            "median_filtered_flux",
            "normalized_flux",
            "quality",
            "quality_present",
            "time",
            "wavelet_flux",
        ]
        np.testing.assert_array_equal(dataset["time"], processed.time)
        np.testing.assert_array_equal(dataset["wavelet_flux"], processed.wavelet_flux)
        np.testing.assert_array_equal(dataset["quality"], processed.quality)
        assert bool(dataset["quality_present"]) is True
        decoded = json.loads(dataset["features_json"].tobytes().decode("utf-8"))
    assert decoded["metadata"]["schema_version"] == "1.0"
    assert decoded["statistics"]["sample_count"] == 3


def test_numpy_export_without_quality_uses_explicit_presence_flag(
    tmp_path: Path,
) -> None:
    """Missing quality remains distinguishable from a zero-valued flag array."""
    processed = _processed(with_quality=False)
    path = export_numpy(
        processed,
        generate_feature_record(processed),
        tmp_path / "dataset.npz",
    )

    with np.load(path, allow_pickle=False) as dataset:
        assert dataset["quality"].size == 0
        assert bool(dataset["quality_present"]) is False


def test_numpy_export_is_byte_deterministic(tmp_path: Path) -> None:
    """Fixed ZIP metadata makes identical NumPy exports byte-for-byte equal."""
    processed = _processed()
    features = generate_feature_record(processed)

    first = export_numpy(processed, features, tmp_path / "first.npz")
    second = export_numpy(processed, features, tmp_path / "second.npz")

    assert _sha256(first) == _sha256(second)


def test_parquet_export_round_trips_arrays_and_schema_metadata(
    tmp_path: Path,
) -> None:
    """Parquet preserves ordered cadence columns and canonical feature JSON."""
    processed = _processed()
    features = generate_feature_record(processed)

    path = export_parquet(processed, features, tmp_path / "dataset.parquet")
    frame = pd.read_parquet(path)
    table = pq.read_table(path)

    assert list(frame.columns) == [
        "time",
        "flux",
        "normalized_flux",
        "median_filtered_flux",
        "wavelet_flux",
        "quality",
    ]
    np.testing.assert_array_equal(frame["time"].to_numpy(), processed.time)
    np.testing.assert_array_equal(
        frame["wavelet_flux"].to_numpy(), processed.wavelet_flux
    )
    metadata = table.schema.metadata or {}
    decoded = json.loads(metadata[b"transitlens.feature_record"].decode("utf-8"))
    assert metadata[b"transitlens.schema_version"] == b"1.0"
    assert decoded["statistics"]["sample_count"] == 3


def test_parquet_export_preserves_missing_quality(tmp_path: Path) -> None:
    """Absent source quality flags become a nullable all-null Parquet column."""
    processed = _processed(with_quality=False)
    path = export_parquet(
        processed,
        generate_feature_record(processed),
        tmp_path / "dataset.parquet",
    )

    assert pd.read_parquet(path)["quality"].isna().all()


def test_parquet_export_is_byte_deterministic(tmp_path: Path) -> None:
    """Identical tables and metadata produce byte-identical Parquet artifacts."""
    processed = _processed()
    features = generate_feature_record(processed)

    first = export_parquet(processed, features, tmp_path / "first.parquet")
    second = export_parquet(processed, features, tmp_path / "second.parquet")

    assert _sha256(first) == _sha256(second)


@pytest.mark.parametrize(
    ("exporter", "filename", "extension"),
    [
        (export_numpy, "dataset.parquet", ".npz"),
        (export_parquet, "dataset.npz", ".parquet"),
    ],
)
def test_exporters_reject_incorrect_extensions(
    tmp_path: Path, exporter, filename: str, extension: str
) -> None:
    """Format mismatches are rejected before writing an artifact."""
    processed = _processed()
    with pytest.raises(ExportError, match=extension):
        exporter(processed, generate_feature_record(processed), tmp_path / filename)


def test_exporter_reports_unusable_output_directory(tmp_path: Path) -> None:
    """Filesystem preparation failures become descriptive export errors."""
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("occupied", encoding="utf-8")
    processed = _processed()

    with pytest.raises(ExportError, match="prepare export directory"):
        export_numpy(
            processed,
            generate_feature_record(processed),
            blocking_file / "dataset.npz",
        )
