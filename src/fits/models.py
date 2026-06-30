"""Structured light-curve contracts produced by FITS parsing."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from fits.validator import validate_series
from mast.models import Mission


class LightCurveMetadata(BaseModel):
    """Provenance describing how a FITS light curve was extracted."""

    model_config = ConfigDict(frozen=True)

    mission: Mission
    source_path: Path
    hdu_index: int
    hdu_name: str
    flux_column: str
    quality_column: str | None
    target_name: str | None = None
    observation_id: str | None = None


class LightCurve(BaseModel):
    """Validated raw light curve extracted from one FITS table."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    time: NDArray[np.float64]
    flux: NDArray[np.float64]
    quality: NDArray[np.int64] | None
    metadata: LightCurveMetadata

    def model_post_init(self, __context: object) -> None:
        """Verify alignment and scientific usability after model validation."""
        validate_series(self.time, self.flux, self.quality)
