"""Cross-module search-to-export and REST workflow integration tests."""

from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from api.app import create_app
from api.services import ApiServices
from config import Settings
from exporters import export_numpy, export_parquet
from features import generate_feature_record
from fits import read_fits
from mast import (
    FitsCache,
    Mission,
    ObservationSearch,
    download_fits,
    search_observations,
)
from preprocessing import preprocess_light_curve

MissionFitsFactory = Callable[[Path, Mission, int, bool], Path]


class MissionArchive:
    """In-process MAST substitute that serves a real mission FITS fixture."""

    def __init__(self, mission: Mission, writer: MissionFitsFactory) -> None:
        """Configure the represented mission and FITS writer."""
        self.mission = mission
        self.writer = writer
        self.download_calls = 0

    def query_criteria(
        self, *args: object, **kwargs: object
    ) -> list[dict[str, object]]:
        """Return one supported observation."""
        return [
            {
                "obsid": f"{self.mission.value}-42",
                "obs_id": f"{self.mission.value.lower()}-observation",
                "target_name": "Integration Target",
                "obs_collection": self.mission.value,
                "dataproduct_type": "timeseries",
                "t_min": 0.0,
                "t_max": 10.0,
            }
        ]

    def get_product_list(self, observations: object) -> list[dict[str, str]]:
        """Return one mission-compatible light-curve product."""
        suffix = "_lc.fits" if self.mission is Mission.TESS else "_llc.fits"
        return [
            {
                "productFilename": f"integration{suffix}",
                "dataURI": f"mast:integration/{self.mission.value}",
                "productType": "SCIENCE",
            }
        ]

    def download_file(
        self,
        uri: str,
        *,
        local_path: str,
        cache: bool,
        verbose: bool,
    ) -> tuple[str, str | None, str | None]:
        """Write a valid mission FITS file to the requested cache path."""
        self.download_calls += 1
        self.writer(Path(local_path), self.mission, 256, True)
        return "COMPLETE", None, None


@pytest.mark.integration
@pytest.mark.parametrize("mission", list(Mission))
def test_search_to_export_for_every_supported_mission(
    tmp_path: Path,
    mission: Mission,
    mission_fits_factory: MissionFitsFactory,
) -> None:
    """Every mission completes search, download, parse, process, and export."""
    archive = MissionArchive(mission, mission_fits_factory)
    observations = search_observations(
        ObservationSearch(target="Integration Target", missions=(mission,)),
        archive,
    )
    downloaded = download_fits(
        observations[0].mast_id,
        archive,
        FitsCache(tmp_path / "cache"),
    )
    raw = read_fits(downloaded.path, mission)
    processed = preprocess_light_curve(raw)
    record = generate_feature_record(processed)
    numpy_path = export_numpy(processed, record, tmp_path / f"{mission}.npz")
    parquet_path = export_parquet(
        processed,
        record,
        tmp_path / f"{mission}.parquet",
    )

    assert raw.metadata.mission is mission
    assert record.metadata.mission is mission
    assert record.statistics.sample_count == 254
    assert processed.metadata.non_finite_removed == 1
    assert processed.metadata.quality_removed == 1
    with np.load(numpy_path, allow_pickle=False) as dataset:
        np.testing.assert_array_equal(dataset["time"], processed.time)
    parquet = pd.read_parquet(parquet_path)
    np.testing.assert_array_equal(parquet["wavelet_flux"], processed.wavelet_flux)
    assert archive.download_calls == 1


@pytest.fixture
def anyio_backend() -> str:
    """Run the REST workflow using asyncio."""
    return "asyncio"


@pytest.mark.integration
@pytest.mark.anyio
async def test_complete_rest_search_download_process_workflow(
    tmp_path: Path,
    mission_fits_factory: MissionFitsFactory,
) -> None:
    """The platform-facing route sequence returns one canonical result."""
    archive = MissionArchive(Mission.TESS, mission_fits_factory)
    settings = Settings(cache_dir=tmp_path / "cache", log_level="WARNING")
    application = create_app(
        settings,
        ApiServices(mast_client_provider=lambda: archive),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        search = await client.get(
            "/search",
            params={"target": "Integration Target", "missions": "TESS"},
        )
        download = await client.post(
            "/download",
            json={"mast_id": search.json()[0]["mast_id"]},
        )
        process = await client.post(
            "/process",
            json={"fits_path": download.json()["path"], "mission": "TESS"},
        )

    assert search.status_code == 200
    assert download.status_code == 200
    assert process.status_code == 200
    assert process.json()["features"]["statistics"]["sample_count"] == 254
    assert len(process.json()["wavelet_flux"]) == 254
