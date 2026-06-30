"""Mission and failure-path tests for FITS light-curve reading."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits as astropy_fits

from fits import (
    FitsColumnError,
    FitsHduError,
    FitsReadError,
    FitsValidationError,
    UnsupportedMissionError,
    read_fits,
)
from mast.models import Mission


def _write_light_curve(
    path: Path,
    *,
    mission: str,
    flux_columns: tuple[str, ...] = ("PDCSAP_FLUX", "SAP_FLUX"),
    quality_column: str | None = "SAP_QUALITY",
    time: np.ndarray | None = None,
    flux: np.ndarray | None = None,
    hdu_name: str = "LIGHTCURVE",
    campaign: int | None = None,
) -> Path:
    """Write a compact mission-like FITS light curve fixture."""
    times = np.array([1.0, 2.0, 3.0]) if time is None else time
    fluxes = np.array([100.0, 99.0, 101.0]) if flux is None else flux
    primary = astropy_fits.PrimaryHDU()
    primary.header["MISSION"] = mission
    primary.header["TELESCOP"] = "TESS" if mission == "TESS" else "Kepler"
    primary.header["OBJECT"] = "Test Target"
    primary.header["OBS_ID"] = "observation-1"
    if campaign is not None:
        primary.header["CAMPAIGN"] = campaign
    columns = [astropy_fits.Column(name="TIME", format="D", array=times)]
    for index, name in enumerate(flux_columns):
        columns.append(
            astropy_fits.Column(
                name=name,
                format="E",
                array=fluxes + float(index),
            )
        )
    if quality_column is not None:
        columns.append(
            astropy_fits.Column(
                name=quality_column,
                format="J",
                array=np.array([0, 1, 0], dtype=np.int32),
            )
        )
    table = astropy_fits.BinTableHDU.from_columns(columns, name=hdu_name)
    astropy_fits.HDUList([primary, table]).writeto(path)
    return path


def test_kepler_prefers_pdcsap_flux_and_extracts_quality(tmp_path: Path) -> None:
    """Kepler corrected flux and SAP quality flags are selected."""
    path = _write_light_curve(tmp_path / "kepler.fits", mission="Kepler")

    light_curve = read_fits(path)

    assert light_curve.metadata.mission is Mission.KEPLER
    assert light_curve.metadata.flux_column == "PDCSAP_FLUX"
    assert light_curve.metadata.quality_column == "SAP_QUALITY"
    assert light_curve.metadata.target_name == "Test Target"
    assert light_curve.metadata.observation_id == "observation-1"
    np.testing.assert_array_equal(light_curve.time, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(light_curve.flux, [100.0, 99.0, 101.0])
    np.testing.assert_array_equal(light_curve.quality, [0, 1, 0])
    assert light_curve.time.flags.writeable is False
    assert light_curve.flux.flags.writeable is False
    assert light_curve.quality is not None
    assert light_curve.quality.flags.writeable is False


def test_k2_detection_uses_campaign_and_falls_back_to_sap_flux(
    tmp_path: Path,
) -> None:
    """K2 campaign headers override the shared Kepler telescope name."""
    path = _write_light_curve(
        tmp_path / "k2.fits",
        mission="Kepler",
        campaign=5,
        flux_columns=("SAP_FLUX",),
    )

    light_curve = read_fits(path)

    assert light_curve.metadata.mission is Mission.K2
    assert light_curve.metadata.flux_column == "SAP_FLUX"


def test_tess_extracts_generic_flux_and_quality(tmp_path: Path) -> None:
    """TESS QUALITY and generic FLUX columns are supported."""
    path = _write_light_curve(
        tmp_path / "tess.fits",
        mission="TESS",
        flux_columns=("FLUX",),
        quality_column="QUALITY",
    )

    light_curve = read_fits(path, Mission.TESS)

    assert light_curve.metadata.mission is Mission.TESS
    assert light_curve.metadata.flux_column == "FLUX"
    assert light_curve.metadata.quality_column == "QUALITY"


def test_missing_quality_is_retained_as_none(tmp_path: Path) -> None:
    """Quality flags are optional when a mission product omits them."""
    path = _write_light_curve(
        tmp_path / "no-quality.fits",
        mission="TESS",
        quality_column=None,
    )

    assert read_fits(path).quality is None


def test_lightcurve_hdu_is_preferred_over_other_time_tables(tmp_path: Path) -> None:
    """A named LIGHTCURVE extension wins over unrelated time tables."""
    path = _write_light_curve(tmp_path / "multiple.fits", mission="TESS")
    with astropy_fits.open(path, mode="update") as hdus:
        unrelated = astropy_fits.BinTableHDU.from_columns(
            [
                astropy_fits.Column(name="TIME", format="D", array=[9.0]),
                astropy_fits.Column(name="FLUX", format="E", array=[9.0]),
            ],
            name="EVENTS",
        )
        hdus.insert(1, unrelated)

    light_curve = read_fits(path)

    assert light_curve.metadata.hdu_name == "LIGHTCURVE"
    assert light_curve.metadata.hdu_index == 2
    assert len(light_curve.time) == 3


def test_expected_mission_mismatch_is_rejected(tmp_path: Path) -> None:
    """A caller cannot accidentally parse one mission as another."""
    path = _write_light_curve(tmp_path / "kepler.fits", mission="Kepler")

    with pytest.raises(UnsupportedMissionError, match="does not match"):
        read_fits(path, Mission.TESS)


def test_unsupported_mission_is_rejected(tmp_path: Path) -> None:
    """Unknown mission headers produce a descriptive exception."""
    path = _write_light_curve(tmp_path / "unknown.fits", mission="Roman")
    with astropy_fits.open(path, mode="update") as hdus:
        del hdus[0].header["TELESCOP"]

    with pytest.raises(UnsupportedMissionError, match="missing or unsupported"):
        read_fits(path)


def test_missing_time_table_is_rejected(tmp_path: Path) -> None:
    """FITS files without a TIME table cannot become light curves."""
    path = tmp_path / "no-time.fits"
    primary = astropy_fits.PrimaryHDU()
    primary.header["MISSION"] = "TESS"
    astropy_fits.HDUList([primary]).writeto(path)

    with pytest.raises(FitsHduError, match="TIME column"):
        read_fits(path)


def test_missing_flux_column_is_rejected(tmp_path: Path) -> None:
    """A TIME table must also contain a supported flux column."""
    path = _write_light_curve(
        tmp_path / "no-flux.fits",
        mission="TESS",
        flux_columns=(),
    )

    with pytest.raises(FitsColumnError, match="missing a flux column"):
        read_fits(path)


def test_empty_light_curve_is_rejected(tmp_path: Path) -> None:
    """A table with no cadences is structurally invalid."""
    path = _write_light_curve(
        tmp_path / "empty.fits",
        mission="TESS",
        time=np.array([], dtype=np.float64),
        flux=np.array([], dtype=np.float32),
        quality_column=None,
    )

    with pytest.raises(FitsValidationError, match="no samples"):
        read_fits(path)


def test_non_monotonic_finite_time_is_rejected(tmp_path: Path) -> None:
    """Finite cadence timestamps must preserve strict temporal order."""
    path = _write_light_curve(
        tmp_path / "unordered.fits",
        mission="TESS",
        time=np.array([1.0, 3.0, 2.0]),
    )

    with pytest.raises(FitsValidationError, match="strictly increasing"):
        read_fits(path)


def test_non_finite_samples_are_retained_for_phase_four(tmp_path: Path) -> None:
    """Individual NaNs remain aligned for the later cleaning phase."""
    path = _write_light_curve(
        tmp_path / "nan.fits",
        mission="TESS",
        time=np.array([1.0, np.nan, 3.0]),
        flux=np.array([100.0, np.nan, 99.0]),
    )

    light_curve = read_fits(path)

    assert np.isnan(light_curve.time[1])
    assert np.isnan(light_curve.flux[1])
    assert len(light_curve.time) == len(light_curve.flux) == 3


def test_missing_and_invalid_files_raise_read_errors(tmp_path: Path) -> None:
    """Filesystem and Astropy failures share a descriptive read exception."""
    with pytest.raises(FitsReadError, match="does not exist"):
        read_fits(tmp_path / "missing.fits")

    invalid = tmp_path / "invalid.fits"
    invalid.write_text("not a FITS file", encoding="utf-8")
    with pytest.raises(FitsReadError, match="could not read"):
        read_fits(invalid)
