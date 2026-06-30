"""Configuration and output contracts for light-curve preprocessing."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fits.models import LightCurveMetadata


class PreprocessingConfig(BaseModel):
    """Validated deterministic preprocessing parameters."""

    model_config = ConfigDict(frozen=True)

    quality_bitmask: Literal["none", "default", "hard", "hardest"] = "default"
    median_window: int = Field(default=5, ge=3)
    wavelet: str = "db4"
    wavelet_threshold_mode: Literal["soft"] = "soft"
    wavelet_threshold_scale: float = Field(default=0.5, ge=0.0, le=1.0)
    wavelet_max_level: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_odd_window(self) -> "PreprocessingConfig":
        """Require a centered odd median-filter window."""
        if self.median_window % 2 == 0:
            raise ValueError("median_window must be odd")
        return self


class PreprocessingMetadata(BaseModel):
    """Provenance and sample accounting for a preprocessing run."""

    model_config = ConfigDict(frozen=True)

    source: LightCurveMetadata
    config: PreprocessingConfig
    input_samples: int
    non_finite_removed: int
    quality_removed: int
    output_samples: int


class PreprocessedLightCurve(BaseModel):
    """Cleaned and denoised light curve produced by Phase 4."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    time: NDArray[np.float64]
    flux: NDArray[np.float64]
    quality: NDArray[np.int64] | None
    normalized_flux: NDArray[np.float64]
    median_filtered_flux: NDArray[np.float64]
    wavelet_flux: NDArray[np.float64]
    metadata: PreprocessingMetadata

    @model_validator(mode="after")
    def validate_alignment(self) -> "PreprocessedLightCurve":
        """Require every processed representation to remain cadence-aligned."""
        lengths = {
            len(self.time),
            len(self.flux),
            len(self.normalized_flux),
            len(self.median_filtered_flux),
            len(self.wavelet_flux),
        }
        if self.quality is not None:
            lengths.add(len(self.quality))
        if len(lengths) != 1:
            raise ValueError("processed light-curve arrays must have equal lengths")
        return self
