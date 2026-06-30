"""Integration tests for the four documented REST endpoints."""

from pathlib import Path

import httpx
import pytest
from astropy.io import fits as astropy_fits

from api.app import create_app
from api.services import ApiServices
from config import Settings
from mast.exceptions import MastAuthenticationError


class FakeMastClient:
    """Configurable Astroquery-compatible client for API tests."""

    def __init__(self) -> None:
        """Initialize deterministic observation and product responses."""
        self.search_error: Exception | None = None
        self.products: list[dict[str, str]] = [
            {
                "productFilename": "target_lc.fits",
                "dataURI": "mast:product/lightcurve",
                "productType": "SCIENCE",
            }
        ]
        self.query: dict[str, object] = {}
        self.download_calls = 0

    def query_criteria(
        self, *args: object, **kwargs: object
    ) -> list[dict[str, object]]:
        """Return one observation or raise a configured network error."""
        self.query = kwargs
        if self.search_error is not None:
            raise self.search_error
        return [
            {
                "obsid": "42",
                "obs_id": "tess-42",
                "target_name": "TIC 42",
                "obs_collection": "TESS",
                "dataproduct_type": "timeseries",
                "t_min": 1.0,
                "t_max": 2.0,
            }
        ]

    def get_product_list(self, observations: object) -> list[dict[str, str]]:
        """Return configured MAST products."""
        return self.products

    def download_file(
        self,
        uri: str,
        *,
        local_path: str,
        cache: bool,
        verbose: bool,
    ) -> tuple[str, str | None, str | None]:
        """Write a deterministic placeholder FITS download."""
        self.download_calls += 1
        Path(local_path).write_bytes(b"SIMPLE  = T")
        return "COMPLETE", None, None


@pytest.fixture
def anyio_backend() -> str:
    """Run ASGI tests on the installed asyncio backend only."""
    return "asyncio"


def _app(tmp_path: Path, client: FakeMastClient | None = None):
    """Create an API application with an isolated cache and fake MAST."""
    mast_client = client or FakeMastClient()
    settings = Settings(cache_dir=tmp_path / "cache", log_level="WARNING")
    return create_app(settings, ApiServices(mast_client_provider=lambda: mast_client))


def _write_tess_fits(path: Path) -> Path:
    """Write a small valid cached TESS light curve."""
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = astropy_fits.PrimaryHDU()
    primary.header["MISSION"] = "TESS"
    primary.header["OBJECT"] = "TIC 42"
    table = astropy_fits.BinTableHDU.from_columns(
        [
            astropy_fits.Column(name="TIME", format="D", array=[1.0, 2.0, 3.0]),
            astropy_fits.Column(
                name="PDCSAP_FLUX", format="E", array=[100.0, 99.0, 101.0]
            ),
            astropy_fits.Column(name="QUALITY", format="J", array=[0, 0, 0]),
        ],
        name="LIGHTCURVE",
    )
    astropy_fits.HDUList([primary, table]).writeto(path)
    return path


@pytest.mark.anyio
async def test_status_reports_readiness_without_calling_mast(tmp_path: Path) -> None:
    """Status is local, stable, and advertises supported missions."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path)),
        base_url="http://test",
    ) as client:
        response = await client.get("/status")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "0.1.0",
        "supported_missions": ["Kepler", "K2", "TESS"],
    }


@pytest.mark.anyio
async def test_search_validates_and_returns_observations(tmp_path: Path) -> None:
    """GET search translates query values to the MAST search contract."""
    mast = FakeMastClient()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path, mast)),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/search",
            params={"target": "TIC 42", "missions": "TESS", "limit": 5},
        )

    assert response.status_code == 200
    assert response.json()[0]["mast_id"] == "42"
    assert response.json()[0]["mission"] == "TESS"
    assert mast.query["obs_collection"] == ["TESS"]


@pytest.mark.anyio
async def test_search_network_failure_maps_to_stable_error(tmp_path: Path) -> None:
    """Upstream MAST failures become a 502 domain response."""
    mast = FakeMastClient()
    mast.search_error = ConnectionError("offline")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path, mast)),
        base_url="http://test",
    ) as client:
        response = await client.get("/search", params={"target": "TIC 42"})

    assert response.status_code == 502
    assert response.json()["code"] == "mast_search_failed"
    assert "TIC 42" in response.json()["detail"]


@pytest.mark.anyio
async def test_search_rejects_whitespace_target(tmp_path: Path) -> None:
    """Blank target names fail request validation before reaching MAST."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path)),
        base_url="http://test",
    ) as client:
        response = await client.get("/search", params={"target": "   "})

    assert response.status_code == 422


@pytest.mark.anyio
async def test_download_caches_and_reuses_fits_product(tmp_path: Path) -> None:
    """POST download returns a reusable cache path without redownloading."""
    mast = FakeMastClient()
    app = _app(tmp_path, mast)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        first = await client.post("/download", json={"mast_id": "42"})
        second = await client.post("/download", json={"mast_id": "42"})

    assert first.status_code == 200
    assert first.json()["from_cache"] is False
    assert second.json()["from_cache"] is True
    assert first.json()["path"] == second.json()["path"]
    assert Path(first.json()["path"]).read_bytes() == b"SIMPLE  = T"
    assert mast.download_calls == 1


@pytest.mark.anyio
async def test_download_missing_product_maps_to_404(tmp_path: Path) -> None:
    """An observation without a light-curve FITS product returns 404."""
    mast = FakeMastClient()
    mast.products = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path, mast)),
        base_url="http://test",
    ) as client:
        response = await client.post("/download", json={"mast_id": "42"})

    assert response.status_code == 404
    assert response.json()["code"] == "mast_product_not_found"


@pytest.mark.anyio
async def test_process_runs_canonical_pipeline_and_features(tmp_path: Path) -> None:
    """POST process returns aligned processed arrays, metadata, and features."""
    fits_path = _write_tess_fits(tmp_path / "cache" / "target_lc.fits")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path)),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/process",
            json={"fits_path": str(fits_path), "mission": "TESS"},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["time"] == [1.0, 2.0, 3.0]
    assert len(payload["normalized_flux"]) == len(payload["wavelet_flux"]) == 3
    assert payload["quality"] == [0, 0, 0]
    assert payload["metadata"]["source"]["mission"] == "TESS"
    assert payload["features"]["statistics"]["sample_count"] == 3
    assert payload["features"]["metadata"]["schema_version"] == "1.0"


@pytest.mark.anyio
async def test_process_restricts_path_to_configured_cache(tmp_path: Path) -> None:
    """POST process cannot read arbitrary server filesystem paths."""
    outside = _write_tess_fits(tmp_path / "outside.fits")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path)),
        base_url="http://test",
    ) as client:
        response = await client.post("/process", json={"fits_path": str(outside)})

    assert response.status_code == 403
    assert response.json()["code"] == "cached_path_required"


@pytest.mark.anyio
async def test_process_invalid_fits_maps_to_422(tmp_path: Path) -> None:
    """Malformed cached input returns a descriptive FITS error response."""
    invalid = tmp_path / "cache" / "invalid.fits"
    invalid.parent.mkdir(parents=True)
    invalid.write_text("invalid", encoding="utf-8")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path)),
        base_url="http://test",
    ) as client:
        response = await client.post("/process", json={"fits_path": str(invalid)})

    assert response.status_code == 422
    assert response.json()["code"] == "fits_read_failed"


@pytest.mark.anyio
async def test_authentication_failure_maps_to_401(tmp_path: Path) -> None:
    """Configured MAST authentication failures return a stable 401 response."""
    services = ApiServices(
        mast_client_provider=lambda: (_ for _ in ()).throw(
            MastAuthenticationError("invalid token")
        )
    )
    app = create_app(
        Settings(cache_dir=tmp_path / "cache", log_level="WARNING"), services
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/search", params={"target": "TIC 42"})

    assert response.status_code == 401
    assert response.json() == {
        "code": "mast_authentication_failed",
        "detail": "invalid token",
    }
