"""Mission-aware parsing of open FITS HDU lists."""

from pathlib import Path
from typing import Any

from astropy.io import fits

from fits.exceptions import FitsColumnError, FitsHduError, UnsupportedMissionError
from fits.models import LightCurve, LightCurveMetadata
from fits.validator import as_float_array, as_quality_array
from mast.models import Mission

_FLUX_COLUMNS = ("PDCSAP_FLUX", "SAP_FLUX", "FLUX")
_QUALITY_COLUMNS: dict[Mission, tuple[str, ...]] = {
    Mission.KEPLER: ("SAP_QUALITY", "QUALITY"),
    Mission.K2: ("SAP_QUALITY", "QUALITY"),
    Mission.TESS: ("QUALITY", "SAP_QUALITY"),
}


def parse_light_curve(
    hdus: fits.HDUList,
    source_path: Path,
    mission: Mission | None = None,
) -> LightCurve:
    """Extract a validated light curve from an open FITS file.

    Args:
        hdus: Open Astropy FITS HDU list.
        source_path: Source path retained as provenance.
        mission: Optional expected mission. Auto-detected when omitted.

    Returns:
        Structured raw light curve with immutable arrays and provenance.

    Raises:
        UnsupportedMissionError: If mission detection fails or conflicts.
        FitsHduError: If no binary table contains a TIME column.
        FitsColumnError: If a suitable table has no supported flux column.
    """
    detected_mission = _detect_mission(hdus)
    if mission is not None and detected_mission != mission:
        message = (
            f"FITS mission '{detected_mission.value}' does not match expected "
            f"mission '{mission.value}'"
        )
        raise UnsupportedMissionError(message)
    resolved_mission = mission or detected_mission
    hdu_index, table_hdu = _find_light_curve_hdu(hdus)
    columns = _column_lookup(table_hdu)
    flux_column = _select_column(columns, _FLUX_COLUMNS)
    if flux_column is None:
        expected = ", ".join(_FLUX_COLUMNS)
        message = (
            f"light-curve HDU is missing a flux column; expected one of {expected}"
        )
        raise FitsColumnError(message)
    quality_column = _select_column(columns, _QUALITY_COLUMNS[resolved_mission])
    data = table_hdu.data
    if data is None:
        raise FitsHduError("light-curve table HDU contains no data")

    time = as_float_array(data[columns["TIME"]], columns["TIME"])
    flux = as_float_array(data[flux_column], flux_column)
    quality = (
        None
        if quality_column is None
        else as_quality_array(data[quality_column], quality_column)
    )
    metadata = _metadata(
        hdus,
        source_path,
        resolved_mission,
        hdu_index,
        table_hdu,
        flux_column,
        quality_column,
    )
    return LightCurve(time=time, flux=flux, quality=quality, metadata=metadata)


def _detect_mission(hdus: fits.HDUList) -> Mission:
    """Identify Kepler, K2, or TESS from mission headers."""
    for hdu in hdus:
        header = hdu.header
        mission_value = str(header.get("MISSION", "")).strip().upper()
        telescope = str(header.get("TELESCOP", "")).strip().upper()
        if mission_value == "K2" or "CAMPAIGN" in header:
            return Mission.K2
        if mission_value == "TESS" or telescope == "TESS":
            return Mission.TESS
        if mission_value == "KEPLER" or telescope == "KEPLER":
            return Mission.KEPLER
    message = "FITS mission is missing or unsupported; expected Kepler, K2, or TESS"
    raise UnsupportedMissionError(message)


def _find_light_curve_hdu(hdus: fits.HDUList) -> tuple[int, fits.BinTableHDU]:
    """Find the preferred binary table containing a TIME column."""
    candidates: list[tuple[int, fits.BinTableHDU]] = []
    for index, hdu in enumerate(hdus):
        if isinstance(hdu, fits.BinTableHDU) and "TIME" in _column_lookup(hdu):
            candidates.append((index, hdu))
    if not candidates:
        raise FitsHduError("FITS file contains no binary table HDU with a TIME column")
    return min(
        candidates,
        key=lambda item: (
            0 if item[1].name.strip().upper() == "LIGHTCURVE" else 1,
            item[0],
        ),
    )


def _column_lookup(hdu: fits.BinTableHDU) -> dict[str, str]:
    """Map case-insensitive FITS column names to their exact spellings."""
    names = hdu.columns.names or []
    return {name.strip().upper(): name for name in names}


def _select_column(columns: dict[str, str], preferences: tuple[str, ...]) -> str | None:
    """Select the first available column from a stable preference list."""
    for preferred in preferences:
        if preferred in columns:
            return columns[preferred]
    return None


def _metadata(
    hdus: fits.HDUList,
    source_path: Path,
    mission: Mission,
    hdu_index: int,
    table_hdu: fits.BinTableHDU,
    flux_column: str,
    quality_column: str | None,
) -> LightCurveMetadata:
    """Collect stable extraction provenance from primary and table headers."""
    primary = hdus[0].header
    table = table_hdu.header
    return LightCurveMetadata(
        mission=mission,
        source_path=source_path,
        hdu_index=hdu_index,
        hdu_name=table_hdu.name,
        flux_column=flux_column,
        quality_column=quality_column,
        target_name=_first_header_value(table, primary, keys=("OBJECT",)),
        observation_id=_first_header_value(
            table,
            primary,
            keys=("OBS_ID", "KEPLERID", "TICID"),
        ),
    )


def _first_header_value(
    first: fits.Header,
    second: fits.Header,
    *,
    keys: tuple[str, ...],
) -> str | None:
    """Return the first non-empty value from ordered headers and keys."""
    for header in (first, second):
        for key in keys:
            value: Any = header.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return None
