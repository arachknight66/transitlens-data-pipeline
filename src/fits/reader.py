"""Safe file-level entry point for Astropy FITS parsing."""

from pathlib import Path

from astropy.io import fits
from loguru import logger

from fits.exceptions import FitsError, FitsReadError
from fits.models import LightCurve
from fits.parser import parse_light_curve
from mast.models import Mission


def read_fits(path: Path, mission: Mission | None = None) -> LightCurve:
    """Read and parse one local FITS light curve with Astropy.

    Args:
        path: Local FITS file path.
        mission: Optional expected mission used for consistency validation.

    Returns:
        Structured, validated light curve detached from the closed FITS file.

    Raises:
        FitsReadError: If the path is unavailable or Astropy cannot open it.
        FitsError: If mission detection, parsing, or validation fails.
    """
    source_path = path.expanduser().resolve()
    if not source_path.is_file():
        message = f"FITS file does not exist: {source_path}"
        raise FitsReadError(message)
    logger.bind(path=str(source_path)).info("Reading FITS light curve")
    try:
        with fits.open(
            source_path,
            mode="readonly",
            memmap=False,
            lazy_load_hdus=False,
        ) as hdus:
            light_curve = parse_light_curve(hdus, source_path, mission)
    except FitsError:
        raise
    except Exception as error:
        logger.bind(path=str(source_path), error_type=type(error).__name__).warning(
            "FITS file could not be opened"
        )
        message = f"could not read FITS file: {source_path}"
        raise FitsReadError(message) from error
    logger.bind(
        path=str(source_path),
        mission=light_curve.metadata.mission.value,
        samples=len(light_curve.time),
    ).info("FITS light curve parsed")
    return light_curve
