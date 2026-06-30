"""Deterministic Parquet processed-dataset export."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from exporters.common import canonical_json, validate_output_path
from exporters.exceptions import ExportError
from features.models import FeatureRecord
from preprocessing.models import PreprocessedLightCurve


def export_parquet(
    light_curve: PreprocessedLightCurve,
    features: FeatureRecord,
    path: Path,
) -> Path:
    """Atomically export samples and canonical features as Parquet.

    Scalar features and provenance are embedded as canonical JSON in Arrow
    schema metadata, leaving one table row per cadence.

    Args:
        light_curve: Fully processed cadence arrays.
        features: Canonical scalar features and metadata.
        path: Caller-selected ``.parquet`` destination.

    Returns:
        Resolved completed artifact path.

    Raises:
        ExportError: If serialization or atomic placement fails.
    """
    destination = validate_output_path(path, ".parquet")
    temporary_path = _temporary_path(destination)
    try:
        frame = _dataframe(light_curve)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        schema_metadata = {
            b"transitlens.feature_record": canonical_json(features),
            b"transitlens.schema_version": features.metadata.schema_version.encode(
                "utf-8"
            ),
        }
        table = table.replace_schema_metadata(schema_metadata)
        pq.write_table(
            table,
            temporary_path,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
        )
        os.replace(temporary_path, destination)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise ExportError(f"failed to export Parquet dataset: {destination}") from error
    logger.bind(path=str(destination), samples=len(light_curve.time)).info(
        "Exported Parquet processed dataset"
    )
    return destination


def _dataframe(light_curve: PreprocessedLightCurve) -> pd.DataFrame:
    """Create a stable sample table using explicit dtypes and column order."""
    quality = (
        pd.array([pd.NA] * len(light_curve.time), dtype="Int64")
        if light_curve.quality is None
        else pd.array(light_curve.quality, dtype="Int64")
    )
    return pd.DataFrame(
        {
            "time": pd.Series(light_curve.time, dtype="float64"),
            "flux": pd.Series(light_curve.flux, dtype="float64"),
            "normalized_flux": pd.Series(light_curve.normalized_flux, dtype="float64"),
            "median_filtered_flux": pd.Series(
                light_curve.median_filtered_flux, dtype="float64"
            ),
            "wavelet_flux": pd.Series(light_curve.wavelet_flux, dtype="float64"),
            "quality": quality,
        }
    )


def _temporary_path(destination: Path) -> Path:
    """Allocate a closed temporary file beside the final artifact."""
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}-",
        suffix=".part",
        delete=False,
    ) as temporary_file:
        return Path(temporary_file.name)
