"""Validated REST request and response contracts."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from features.models import FeatureRecord
from mast.models import Mission
from preprocessing.models import PreprocessingConfig, PreprocessingMetadata


class StatusResponse(BaseModel):
    """Service readiness response."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    version: str
    supported_missions: tuple[Mission, ...]


class DownloadRequest(BaseModel):
    """Request to cache the preferred FITS product for one observation."""

    model_config = ConfigDict(frozen=True)

    mast_id: str

    @field_validator("mast_id")
    @classmethod
    def validate_mast_id(cls, value: str) -> str:
        """Reject an empty MAST observation identifier."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("mast_id must not be empty")
        return normalized


class DownloadResponse(BaseModel):
    """Cached FITS download response."""

    model_config = ConfigDict(frozen=True)

    mast_id: str
    product_filename: str
    data_uri: str
    path: Path
    from_cache: bool


class ProcessRequest(BaseModel):
    """Request to process a FITS file already held in the service cache."""

    model_config = ConfigDict(frozen=True)

    fits_path: Path
    mission: Mission | None = None
    preprocessing: PreprocessingConfig | None = None


class ProcessResponse(BaseModel):
    """Canonical JSON representation of a processed light curve."""

    model_config = ConfigDict(frozen=True)

    time: list[float]
    flux: list[float]
    normalized_flux: list[float]
    median_filtered_flux: list[float]
    wavelet_flux: list[float]
    quality: list[int] | None
    metadata: PreprocessingMetadata
    features: FeatureRecord


class ErrorResponse(BaseModel):
    """Stable error payload for domain failures."""

    model_config = ConfigDict(frozen=True)

    code: str
    detail: str
