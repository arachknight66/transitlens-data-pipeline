"""Deterministic compressed NumPy dataset export."""

import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from exporters.common import canonical_json, validate_output_path
from exporters.exceptions import ExportError
from features.models import FeatureRecord
from preprocessing.models import PreprocessedLightCurve

_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def export_numpy(
    light_curve: PreprocessedLightCurve,
    features: FeatureRecord,
    path: Path,
) -> Path:
    """Atomically export a processed dataset as deterministic compressed NPZ.

    Args:
        light_curve: Fully processed cadence arrays.
        features: Canonical scalar features and metadata.
        path: Caller-selected ``.npz`` destination.

    Returns:
        Resolved completed artifact path.

    Raises:
        ExportError: If serialization or atomic placement fails.
    """
    destination = validate_output_path(path, ".npz")
    arrays = _dataset_arrays(light_curve, features)
    temporary_path = _temporary_path(destination)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(arrays):
                _write_array(archive, name, arrays[name])
        os.replace(temporary_path, destination)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise ExportError(f"failed to export NumPy dataset: {destination}") from error
    logger.bind(path=str(destination), samples=len(light_curve.time)).info(
        "Exported NumPy processed dataset"
    )
    return destination


def _dataset_arrays(
    light_curve: PreprocessedLightCurve,
    features: FeatureRecord,
) -> dict[str, NDArray[np.generic]]:
    """Build a stable collection of NumPy-safe arrays."""
    quality_present = light_curve.quality is not None
    quality = (
        np.array([], dtype=np.int64)
        if light_curve.quality is None
        else np.asarray(light_curve.quality, dtype=np.int64)
    )
    return {
        "features_json": np.frombuffer(canonical_json(features), dtype=np.uint8).copy(),
        "flux": np.asarray(light_curve.flux, dtype=np.float64),
        "median_filtered_flux": np.asarray(
            light_curve.median_filtered_flux, dtype=np.float64
        ),
        "normalized_flux": np.asarray(light_curve.normalized_flux, dtype=np.float64),
        "quality": quality,
        "quality_present": np.array(quality_present, dtype=np.bool_),
        "time": np.asarray(light_curve.time, dtype=np.float64),
        "wavelet_flux": np.asarray(light_curve.wavelet_flux, dtype=np.float64),
    }


def _write_array(
    archive: zipfile.ZipFile,
    name: str,
    array: NDArray[np.generic],
) -> None:
    """Write one NPY member using fixed ZIP metadata."""
    content = io.BytesIO()
    np.lib.format.write_array(content, array, allow_pickle=False)
    member = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIMESTAMP)
    member.compress_type = zipfile.ZIP_DEFLATED
    member.external_attr = 0o600 << 16
    member.create_system = 3
    archive.writestr(member, content.getvalue(), compress_type=zipfile.ZIP_DEFLATED)


def _temporary_path(destination: Path) -> Path:
    """Allocate a closed temporary file beside the final artifact."""
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}-",
        suffix=".part",
        delete=False,
    ) as temporary_file:
        return Path(temporary_file.name)
