"""Aligned invalid-measurement and mission quality filtering."""

import warnings
from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import NDArray

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Warning: the tpfmodel submodule is not available.*",
        category=UserWarning,
    )
    from lightkurve.utils import KeplerQualityFlags, TessQualityFlags

from fits.models import LightCurve
from mast.models import Mission
from preprocessing.exceptions import InvalidMeasurementsError

QualityBitmask = Literal["none", "default", "hard", "hardest"]


def remove_non_finite(
    light_curve: LightCurve,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64] | None]:
    """Remove non-finite time/flux samples while preserving row alignment.

    Args:
        light_curve: Validated raw light curve from FITS parsing.

    Returns:
        Immutable aligned time, flux, and optional quality arrays.

    Raises:
        InvalidMeasurementsError: If no finite time/flux pairs remain.
    """
    mask = np.isfinite(light_curve.time) & np.isfinite(light_curve.flux)
    if not np.any(mask):
        raise InvalidMeasurementsError("no finite time and flux pairs remain")
    removed = int(mask.size - np.count_nonzero(mask))
    logger.bind(removed=removed, retained=int(np.count_nonzero(mask))).info(
        "Removed non-finite light-curve measurements"
    )
    quality = None if light_curve.quality is None else light_curve.quality[mask]
    return (
        _immutable(light_curve.time[mask], np.float64),
        _immutable(light_curve.flux[mask], np.float64),
        None if quality is None else _immutable(quality, np.int64),
    )


def filter_quality(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    quality: NDArray[np.int64] | None,
    mission: Mission,
    bitmask: QualityBitmask = "default",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64] | None]:
    """Apply Lightkurve's mission-aware quality mask conservatively.

    Args:
        time: Finite cadence timestamps.
        flux: Finite aligned flux measurements.
        quality: Optional aligned mission quality flags.
        mission: Mission controlling quality-flag semantics.
        bitmask: Named Lightkurve bitmask. ``default`` is scientifically
            conservative and retains flags not designated for rejection.

    Returns:
        Immutable aligned arrays containing accepted cadences.

    Raises:
        InvalidMeasurementsError: If arrays are misaligned or all rejected.
    """
    if len(time) != len(flux) or (quality is not None and len(quality) != len(time)):
        raise InvalidMeasurementsError("quality filtering requires aligned arrays")
    if quality is None or bitmask == "none":
        return (
            _immutable(time, np.float64),
            _immutable(flux, np.float64),
            None if quality is None else _immutable(quality, np.int64),
        )

    quality_flags = TessQualityFlags if mission is Mission.TESS else KeplerQualityFlags
    mask = np.asarray(quality_flags.create_quality_mask(quality, bitmask), dtype=bool)
    if not np.any(mask):
        raise InvalidMeasurementsError(
            f"quality filtering rejected every {mission.value} cadence"
        )
    logger.bind(
        mission=mission.value,
        bitmask=bitmask,
        removed=int(mask.size - np.count_nonzero(mask)),
        retained=int(np.count_nonzero(mask)),
    ).info("Applied mission quality filter")
    return (
        _immutable(time[mask], np.float64),
        _immutable(flux[mask], np.float64),
        _immutable(quality[mask], np.int64),
    )


def _immutable(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Return an independent, read-only array with a stable dtype."""
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result
