"""Validated data contracts for MAST operations."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Mission(StrEnum):
    """MAST observation collections supported by TransitLens."""

    KEPLER = "Kepler"
    K2 = "K2"
    TESS = "TESS"


class ObservationSearch(BaseModel):
    """Validated criteria for a target-centered MAST observation search."""

    model_config = ConfigDict(frozen=True)

    target: str
    missions: tuple[Mission, ...] = tuple(Mission)
    radius_deg: float = Field(default=0.001, gt=0.0, le=5.0)
    limit: int = Field(default=100, ge=1, le=1000)

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        """Reject an empty target name and normalize surrounding whitespace."""
        target = value.strip()
        if not target:
            message = "target must not be empty"
            raise ValueError(message)
        return target

    @field_validator("missions")
    @classmethod
    def validate_missions(cls, value: tuple[Mission, ...]) -> tuple[Mission, ...]:
        """Require at least one mission and remove duplicates predictably."""
        if not value:
            message = "at least one mission is required"
            raise ValueError(message)
        return tuple(dict.fromkeys(value))


class Observation(BaseModel):
    """Stable representation of a searchable MAST observation."""

    model_config = ConfigDict(frozen=True)

    mast_id: str
    observation_id: str
    target_name: str
    mission: Mission
    product_type: str
    start_time: float | None = None
    end_time: float | None = None


class DownloadedFits(BaseModel):
    """Metadata describing a locally cached MAST FITS product."""

    model_config = ConfigDict(frozen=True)

    mast_id: str
    product_filename: str
    data_uri: str
    path: Path
    from_cache: bool
