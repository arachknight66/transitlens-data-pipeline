"""Stable metadata generation for processed datasets."""

from importlib.metadata import version

from features.models import DatasetMetadata, FeatureRecord
from features.statistics import generate_statistics
from preprocessing.models import PreprocessedLightCurve

DATASET_SCHEMA_VERSION = "1.0"


def generate_metadata(light_curve: PreprocessedLightCurve) -> DatasetMetadata:
    """Generate deterministic export provenance without runtime timestamps.

    Args:
        light_curve: Fully preprocessed light curve and its source provenance.

    Returns:
        Stable validated dataset metadata.
    """
    preprocessing = light_curve.metadata
    source = preprocessing.source
    return DatasetMetadata(
        schema_version=DATASET_SCHEMA_VERSION,
        pipeline_version=version("transitlens-data-pipeline"),
        mission=source.mission,
        source_path=source.source_path,
        target_name=source.target_name,
        observation_id=source.observation_id,
        flux_column=source.flux_column,
        quality_column=source.quality_column,
        preprocessing=preprocessing.config,
        input_samples=preprocessing.input_samples,
        non_finite_removed=preprocessing.non_finite_removed,
        quality_removed=preprocessing.quality_removed,
        output_samples=preprocessing.output_samples,
    )


def generate_feature_record(light_curve: PreprocessedLightCurve) -> FeatureRecord:
    """Generate the canonical feature record for an exported light curve.

    Args:
        light_curve: Fully preprocessed light curve.

    Returns:
        Deterministic statistics paired with stable dataset metadata.
    """
    return FeatureRecord(
        statistics=generate_statistics(light_curve),
        metadata=generate_metadata(light_curve),
    )
