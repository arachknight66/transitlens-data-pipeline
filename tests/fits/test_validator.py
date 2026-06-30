"""Unit tests for extracted light-curve validation."""

import numpy as np
import pytest

from fits.exceptions import FitsValidationError
from fits.validator import as_float_array, as_quality_array, validate_series


def test_float_array_rejects_non_numeric_and_multidimensional_values() -> None:
    """Flux and time columns must be scalar numeric samples."""
    with pytest.raises(FitsValidationError, match="numeric"):
        as_float_array(["bad"], "FLUX")
    with pytest.raises(FitsValidationError, match="one-dimensional"):
        as_float_array([[1.0, 2.0]], "FLUX")


def test_quality_array_requires_integral_flags() -> None:
    """Floating-point quality flags are rejected rather than truncated."""
    with pytest.raises(FitsValidationError, match="must contain integers"):
        as_quality_array([0.0, 1.0], "QUALITY")
    with pytest.raises(FitsValidationError, match="one-dimensional"):
        as_quality_array(np.array([[0, 1]]), "QUALITY")


def test_series_requires_aligned_arrays() -> None:
    """All present cadence columns must have identical lengths."""
    with pytest.raises(FitsValidationError, match="equal lengths"):
        validate_series(
            np.array([1.0, 2.0]),
            np.array([1.0]),
            np.array([0, 0]),
        )


def test_series_requires_finite_time_and_flux() -> None:
    """A light curve needs at least one usable time and flux sample."""
    with pytest.raises(FitsValidationError, match="TIME column"):
        validate_series(np.array([np.nan]), np.array([1.0]), None)
    with pytest.raises(FitsValidationError, match="FLUX column"):
        validate_series(np.array([1.0]), np.array([np.nan]), None)
