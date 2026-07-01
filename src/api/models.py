"""Validated REST request and response contracts."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

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


class UploadResponse(BaseModel):
    """Opaque reference returned for a validated temporary upload."""

    model_config = ConfigDict(frozen=True)

    file_id: str
    media_type: Literal["fits", "fit", "csv"]
    size_bytes: int


class ProcessRequest(BaseModel):
    """Request to process an opaque upload or legacy cached FITS path."""

    model_config = ConfigDict(frozen=True)

    file_id: str | None = None
    fits_path: Path | None = None
    mission: Mission | None = None
    preprocessing: PreprocessingConfig | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "ProcessRequest":
        """Require exactly one opaque identifier or legacy cached path."""
        if (self.file_id is None) == (self.fits_path is None):
            raise ValueError("provide exactly one of file_id or fits_path")
        return self


class ProcessResponse(BaseModel):
    """Canonical JSON representation of a processed light curve."""

    model_config = ConfigDict(frozen=True)

    file_id: str | None = None
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
