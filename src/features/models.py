"""Stable feature and dataset metadata contracts."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mast.models import Mission
from preprocessing.models import PreprocessingConfig


class StatisticalFeatures(BaseModel):
    """Deterministic scalar features computed from denoised normalized flux."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    sample_count: int = Field(ge=1)
    mean: float
    standard_deviation: float = Field(ge=0.0)
    rms: float = Field(ge=0.0)
    signal_to_noise_ratio: float | None
    flux_variance: float = Field(ge=0.0)
    observation_duration: float = Field(ge=0.0)
    cadence: float | None = Field(default=None, gt=0.0)


class DatasetMetadata(BaseModel):
    """Stable provenance accompanying exported processed datasets."""

    model_config = ConfigDict(frozen=True)

    schema_version: str
    pipeline_version: str
    mission: Mission
    source_path: Path
    target_name: str | None
    observation_id: str | None
    flux_column: str
    quality_column: str | None
    preprocessing: PreprocessingConfig
    input_samples: int = Field(ge=1)
    non_finite_removed: int = Field(ge=0)
    quality_removed: int = Field(ge=0)
    output_samples: int = Field(ge=1)


class FeatureRecord(BaseModel):
    """Canonical deterministic features and their dataset provenance."""

    model_config = ConfigDict(frozen=True)

    statistics: StatisticalFeatures
    metadata: DatasetMetadata
