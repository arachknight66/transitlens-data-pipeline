"""Tests for aligned non-finite and mission quality filtering."""

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from fits.models import LightCurve
from mast.models import Mission
from preprocessing.exceptions import InvalidMeasurementsError
from preprocessing.quality import filter_quality, remove_non_finite

LightCurveFactory = Callable[
    [np.ndarray, np.ndarray, np.ndarray | None, Mission], LightCurve
]


def test_remove_non_finite_preserves_alignment(light_curve_factory) -> None:
    """Time, flux, and quality are filtered using one shared finite mask."""
    light_curve = light_curve_factory(
        np.array([1.0, 2.0, np.nan, 4.0]),
        np.array([10.0, np.inf, 12.0, 13.0]),
        np.array([0, 1, 2, 3]),
    )

    time, flux, quality = remove_non_finite(light_curve)

    np.testing.assert_array_equal(time, [1.0, 4.0])
    np.testing.assert_array_equal(flux, [10.0, 13.0])
    np.testing.assert_array_equal(quality, [0, 3])
    assert not time.flags.writeable
    assert not flux.flags.writeable
    assert quality is not None and not quality.flags.writeable


def test_remove_non_finite_rejects_empty_result() -> None:
    """Cleaning cannot silently produce an empty observation."""
    light_curve = SimpleNamespace(
        time=np.array([np.nan]),
        flux=np.array([np.inf]),
        quality=None,
    )

    with pytest.raises(InvalidMeasurementsError, match="no finite"):
        remove_non_finite(light_curve)  # type: ignore[arg-type]


@pytest.mark.parametrize("mission", [Mission.KEPLER, Mission.K2, Mission.TESS])
def test_default_quality_mask_removes_compromised_cadence(
    mission: Mission,
) -> None:
    """Lightkurve default masks are applied using mission semantics."""
    time = np.array([1.0, 2.0, 3.0])
    flux = np.array([10.0, 11.0, 12.0])
    quality = np.array([0, 1, 0], dtype=np.int64)

    filtered_time, filtered_flux, filtered_quality = filter_quality(
        time, flux, quality, mission
    )

    np.testing.assert_array_equal(filtered_time, [1.0, 3.0])
    np.testing.assert_array_equal(filtered_flux, [10.0, 12.0])
    np.testing.assert_array_equal(filtered_quality, [0, 0])


def test_missing_quality_and_none_bitmask_preserve_cadences() -> None:
    """Absent flags or an explicit none mask do not discard measurements."""
    time = np.array([1.0, 2.0])
    flux = np.array([10.0, 11.0])

    no_quality = filter_quality(time, flux, None, Mission.TESS)
    no_mask = filter_quality(
        time,
        flux,
        np.array([1, 1], dtype=np.int64),
        Mission.TESS,
        "none",
    )

    np.testing.assert_array_equal(no_quality[0], time)
    assert no_quality[2] is None
    np.testing.assert_array_equal(no_mask[2], [1, 1])


def test_quality_filter_rejects_misalignment_and_empty_result() -> None:
    """Invalid alignment and masks rejecting every cadence are explicit."""
    with pytest.raises(InvalidMeasurementsError, match="aligned"):
        filter_quality(
            np.array([1.0, 2.0]),
            np.array([1.0]),
            None,
            Mission.TESS,
        )

    with pytest.raises(InvalidMeasurementsError, match="every TESS cadence"):
        filter_quality(
            np.array([1.0]),
            np.array([1.0]),
            np.array([1], dtype=np.int64),
            Mission.TESS,
        )
