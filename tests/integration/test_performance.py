"""Local pipeline performance acceptance test."""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import pytest

from exporters import export_numpy, export_parquet
from features import generate_feature_record
from fits import read_fits
from mast.models import Mission
from preprocessing import preprocess_light_curve

MissionFitsFactory = Callable[[Path, Mission, int, bool], Path]


@pytest.mark.integration
@pytest.mark.performance
def test_local_pipeline_completes_under_three_seconds(
    tmp_path: Path,
    mission_fits_factory: MissionFitsFactory,
) -> None:
    """Parse, process, feature, and export a 50k-cadence local observation."""
    source = mission_fits_factory(
        tmp_path / "large-tess.fits",
        Mission.TESS,
        50_000,
        False,
    )

    started = perf_counter()
    processed = preprocess_light_curve(read_fits(source))
    record = generate_feature_record(processed)
    export_numpy(processed, record, tmp_path / "large.npz")
    export_parquet(processed, record, tmp_path / "large.parquet")
    elapsed = perf_counter() - started

    assert record.statistics.sample_count == 50_000
    assert elapsed < 3.0, f"local pipeline took {elapsed:.3f} seconds"
