"""Shared fixtures for deterministic preprocessing tests."""

from pathlib import Path

import numpy as np
import pytest

from fits.models import LightCurve, LightCurveMetadata
from mast.models import Mission


@pytest.fixture
def light_curve_factory():
    """Return a factory for compact validated raw light curves."""

    def factory(
        time: np.ndarray,
        flux: np.ndarray,
        quality: np.ndarray | None = None,
        mission: Mission = Mission.TESS,
    ) -> LightCurve:
        copied_time = np.array(time, dtype=np.float64, copy=True)
        copied_flux = np.array(flux, dtype=np.float64, copy=True)
        copied_quality = (
            None if quality is None else np.array(quality, dtype=np.int64, copy=True)
        )
        for array in (copied_time, copied_flux, copied_quality):
            if array is not None:
                array.setflags(write=False)
        return LightCurve(
            time=copied_time,
            flux=copied_flux,
            quality=copied_quality,
            metadata=LightCurveMetadata(
                mission=mission,
                source_path=Path("fixture.fits"),
                hdu_index=1,
                hdu_name="LIGHTCURVE",
                flux_column="PDCSAP_FLUX",
                quality_column=None if quality is None else "QUALITY",
            ),
        )

    return factory
