"""Structural validation for extracted FITS light-curve arrays."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fits.exceptions import FitsValidationError


def as_float_array(values: ArrayLike, column_name: str) -> NDArray[np.float64]:
    """Create a validated, immutable one-dimensional float array.

    Args:
        values: Values extracted from a FITS table column.
        column_name: Source column name used in descriptive errors.

    Returns:
        An independent, read-only float64 array.

    Raises:
        FitsValidationError: If the column is not numeric and one-dimensional.
    """
    try:
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        message = f"FITS column '{column_name}' must contain numeric values"
        raise FitsValidationError(message) from error
    _validate_one_dimensional(array, column_name)
    array.setflags(write=False)
    return array


def as_quality_array(values: ArrayLike, column_name: str) -> NDArray[np.int64]:
    """Create a validated, immutable one-dimensional quality array.

    Args:
        values: Quality flags extracted from a FITS table column.
        column_name: Source column name used in descriptive errors.

    Returns:
        An independent, read-only int64 array.

    Raises:
        FitsValidationError: If flags are non-integral or not one-dimensional.
    """
    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.integer):
        message = f"FITS quality column '{column_name}' must contain integers"
        raise FitsValidationError(message)
    array = np.array(raw, dtype=np.int64, copy=True)
    _validate_one_dimensional(array, column_name)
    array.setflags(write=False)
    return array


def validate_series(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    quality: NDArray[np.int64] | None,
) -> None:
    """Validate alignment and minimal scientific usability of a light curve.

    Non-finite samples are retained for the explicitly separate Phase 4 cleaning
    step. Finite timestamps must nevertheless remain strictly increasing.

    Args:
        time: Extracted cadence timestamps.
        flux: Extracted raw flux measurements.
        quality: Optional aligned quality flags.

    Raises:
        FitsValidationError: If arrays are empty, misaligned, unusable, or
            ordered inconsistently.
    """
    lengths = [len(time), len(flux)]
    if quality is not None:
        lengths.append(len(quality))
    if not lengths[0]:
        raise FitsValidationError("FITS light curve contains no samples")
    if len(set(lengths)) != 1:
        message = "TIME, FLUX, and quality columns must have equal lengths"
        raise FitsValidationError(message)
    finite_time = time[np.isfinite(time)]
    if finite_time.size == 0:
        raise FitsValidationError("TIME column contains no finite samples")
    if not np.any(np.isfinite(flux)):
        raise FitsValidationError("FLUX column contains no finite samples")
    if finite_time.size > 1 and np.any(np.diff(finite_time) <= 0):
        message = "finite TIME samples must be strictly increasing"
        raise FitsValidationError(message)


def _validate_one_dimensional(array: np.ndarray, column_name: str) -> None:
    """Require a scalar value per cadence."""
    if array.ndim != 1:
        message = f"FITS column '{column_name}' must be one-dimensional"
        raise FitsValidationError(message)
