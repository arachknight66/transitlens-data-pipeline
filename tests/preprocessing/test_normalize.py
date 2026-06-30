"""Tests for robust flux normalization."""

import numpy as np
import pytest

from preprocessing.exceptions import NormalizationError
from preprocessing.normalize import normalize_flux


def test_normalize_flux_uses_median_without_mutating_input() -> None:
    """Median normalization centers flux at unity and preserves input."""
    original = np.array([8.0, 10.0, 12.0])
    snapshot = original.copy()

    normalized = normalize_flux(original)

    np.testing.assert_array_equal(original, snapshot)
    np.testing.assert_allclose(normalized, [0.8, 1.0, 1.2])
    assert float(np.median(normalized)) == pytest.approx(1.0)
    assert not normalized.flags.writeable


@pytest.mark.parametrize(
    ("flux", "message"),
    [
        (np.array([]), "non-empty"),
        (np.array([[1.0]]), "one-dimensional"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([-1.0, 0.0, 1.0]), "must be positive"),
        (np.array([-3.0, -2.0, -1.0]), "must be positive"),
    ],
)
def test_normalize_flux_rejects_unusable_input(flux: np.ndarray, message: str) -> None:
    """Unsafe normalization inputs produce descriptive errors."""
    with pytest.raises(NormalizationError, match=message):
        normalize_flux(flux)
