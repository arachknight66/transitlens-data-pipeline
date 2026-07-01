"""CSV light-curve validation and conversion."""

from pathlib import Path

import numpy as np
import pandas as pd

from fits.exceptions import FitsColumnError, FitsReadError
from fits.models import LightCurve, LightCurveMetadata
from fits.validator import as_float_array, as_quality_array
from mast.models import Mission

_FLUX_COLUMNS = ("PDCSAP_FLUX", "SAP_FLUX", "FLUX")
_QUALITY_COLUMNS = ("QUALITY", "SAP_QUALITY")


def validate_csv(path: Path) -> None:
    """Validate that a CSV contains a usable light-curve schema.

    Args:
        path: Local CSV file to inspect.

    Raises:
        FitsReadError: If pandas cannot parse a non-empty CSV.
        FitsColumnError: If required TIME or flux columns are unavailable.
    """
    has_samples = False
    try:
        for frame in pd.read_csv(path, chunksize=10_000):
            if frame.empty:
                continue
            _extract_columns(frame)
            has_samples = True
    except (
        OSError,
        UnicodeError,
        ValueError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        raise FitsReadError(f"could not read CSV light curve: {path}") from error
    if not has_samples:
        raise FitsReadError("CSV light curve contains no samples")


def read_csv_light_curve(path: Path, mission: Mission) -> LightCurve:
    """Read a CSV light curve into the canonical immutable domain model.

    Args:
        path: Local validated CSV file.
        mission: Mission defining quality-flag semantics.

    Returns:
        Structured light curve compatible with existing preprocessing.

    Raises:
        FitsReadError: If the CSV cannot be read.
        FitsColumnError: If required columns are unavailable.
        FitsValidationError: If extracted arrays are invalid.
    """
    source_path = path.expanduser().resolve()
    frame = _read_frame(source_path)
    time_column, flux_column, quality_column = _extract_columns(frame)
    quality = (
        None
        if quality_column is None
        else as_quality_array(frame[quality_column].to_numpy(), quality_column)
    )
    return LightCurve(
        time=as_float_array(frame[time_column].to_numpy(), time_column),
        flux=as_float_array(frame[flux_column].to_numpy(), flux_column),
        quality=quality,
        metadata=LightCurveMetadata(
            mission=mission,
            source_path=source_path,
            hdu_index=0,
            hdu_name="CSV",
            flux_column=flux_column,
            quality_column=quality_column,
        ),
    )


def _read_frame(path: Path) -> pd.DataFrame:
    """Read CSV data with stable parsing behavior and descriptive failures."""
    try:
        frame = pd.read_csv(path)
    except (
        OSError,
        UnicodeError,
        ValueError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        raise FitsReadError(f"could not read CSV light curve: {path}") from error
    if frame.empty:
        raise FitsReadError("CSV light curve contains no samples")
    return frame


def _extract_columns(frame: pd.DataFrame) -> tuple[str, str, str | None]:
    """Select exact CSV column spellings using FITS-compatible preferences."""
    lookup = {str(column).strip().upper(): str(column) for column in frame.columns}
    time_column = lookup.get("TIME")
    if time_column is None:
        raise FitsColumnError("CSV light curve is missing required TIME column")
    flux_column = next(
        (lookup[column] for column in _FLUX_COLUMNS if column in lookup),
        None,
    )
    if flux_column is None:
        expected = ", ".join(_FLUX_COLUMNS)
        raise FitsColumnError(
            f"CSV is missing a flux column; expected one of {expected}"
        )
    quality_column = next(
        (lookup[column] for column in _QUALITY_COLUMNS if column in lookup),
        None,
    )
    _validate_numeric(frame, time_column, flux_column, quality_column)
    return time_column, flux_column, quality_column


def _validate_numeric(
    frame: pd.DataFrame,
    time_column: str,
    flux_column: str,
    quality_column: str | None,
) -> None:
    """Reject columns that cannot be represented by canonical numeric arrays."""
    try:
        np.asarray(frame[time_column], dtype=np.float64)
        np.asarray(frame[flux_column], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise FitsReadError("CSV TIME and flux columns must be numeric") from error
    if quality_column is not None and not pd.api.types.is_integer_dtype(
        frame[quality_column].dtype
    ):
        raise FitsReadError("CSV quality column must contain integers")
