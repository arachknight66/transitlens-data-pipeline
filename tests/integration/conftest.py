"""Mission FITS factories for cross-module integration tests."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits as astropy_fits

from mast.models import Mission

MissionFitsFactory = Callable[[Path, Mission, int, bool], Path]


@pytest.fixture
def mission_fits_factory() -> MissionFitsFactory:
    """Return a deterministic Kepler, K2, or TESS FITS writer."""

    def write(
        path: Path,
        mission: Mission,
        samples: int = 256,
        include_invalid: bool = True,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        cadence = 0.02043365
        time = np.arange(samples, dtype=np.float64) * cadence
        center = samples // 2
        distance = np.abs(np.arange(samples) - center)
        transit = np.where(distance <= max(3, samples // 30), 0.99, 1.0)
        noise = 0.0005 * np.sin(2.0 * np.pi * np.arange(samples) * 0.37)
        flux = 10_000.0 * (transit + noise)
        quality = np.zeros(samples, dtype=np.int32)
        if include_invalid and samples > 20:
            flux[10] = np.nan
            quality[11] = 1

        primary = astropy_fits.PrimaryHDU()
        primary.header["MISSION"] = mission.value
        primary.header["TELESCOP"] = "TESS" if mission is Mission.TESS else "Kepler"
        primary.header["OBJECT"] = f"{mission.value} Integration Target"
        primary.header["OBS_ID"] = f"{mission.value.lower()}-integration"
        flux_column = "SAP_FLUX" if mission is Mission.K2 else "PDCSAP_FLUX"
        quality_column = "QUALITY" if mission is Mission.TESS else "SAP_QUALITY"
        table = astropy_fits.BinTableHDU.from_columns(
            [
                astropy_fits.Column(name="TIME", format="D", array=time),
                astropy_fits.Column(name=flux_column, format="E", array=flux),
                astropy_fits.Column(name=quality_column, format="J", array=quality),
            ],
            name="LIGHTCURVE",
        )
        astropy_fits.HDUList([primary, table]).writeto(path, overwrite=True)
        return path

    return write
