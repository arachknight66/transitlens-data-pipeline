"""Tests for MAST product selection and cached FITS downloads."""

from pathlib import Path

import pytest

from mast.cache import FitsCache
from mast.download import download_fits
from mast.exceptions import MastDownloadError, MastProductNotFoundError


class DownloadClient:
    """Configurable fake client for download behavior."""

    def __init__(
        self,
        products: object,
        status: tuple[str, str | None, str | None] = ("COMPLETE", None, None),
    ) -> None:
        """Configure product discovery and download status."""
        self.products = products
        self.status = status
        self.download_calls = 0

    def query_criteria(self, *args: object, **kwargs: object) -> list[object]:
        """Return no observations."""
        return []

    def get_product_list(self, observations: object) -> object:
        """Return configured products or raise the configured error."""
        if isinstance(self.products, Exception):
            raise self.products
        return self.products

    def download_file(
        self,
        uri: str,
        *,
        local_path: str,
        cache: bool,
        verbose: bool,
    ) -> tuple[str, str | None, str | None]:
        """Materialize a fake FITS file for successful downloads."""
        self.download_calls += 1
        if self.status[0] == "COMPLETE":
            Path(local_path).write_bytes(b"SIMPLE  = T")
        return self.status


def _products() -> list[dict[str, str]]:
    """Return mixed products with multiple valid light curves."""
    return [
        {
            "productFilename": "target_tpf.fits",
            "dataURI": "mast:product/tpf",
            "productType": "SCIENCE",
        },
        {
            "productFilename": "target_llc.fits",
            "dataURI": "mast:product/long",
            "productType": "SCIENCE",
        },
        {
            "productFilename": "target_lc.fits",
            "dataURI": "mast:product/lightcurve",
            "productType": "SCIENCE",
        },
    ]


def test_download_selects_preferred_product_and_caches_it(tmp_path: Path) -> None:
    """The preferred light curve is downloaded once and reused thereafter."""
    client = DownloadClient(_products())
    cache = FitsCache(tmp_path / "cache")

    downloaded = download_fits("123", client, cache)
    cached = download_fits("123", client, cache)

    assert downloaded.product_filename == "target_lc.fits"
    assert downloaded.data_uri == "mast:product/lightcurve"
    assert downloaded.path.read_bytes() == b"SIMPLE  = T"
    assert downloaded.from_cache is False
    assert cached.from_cache is True
    assert cached.path == downloaded.path
    assert client.download_calls == 1


@pytest.mark.parametrize(
    "filename",
    ["target.txt", "target_tpf.fits", "target_targetpixel.fits", "generic.fits"],
)
def test_unsupported_products_are_rejected(tmp_path: Path, filename: str) -> None:
    """Only recognized mission light-curve FITS products are downloadable."""
    client = DownloadClient(
        [
            {
                "productFilename": filename,
                "dataURI": "mast:product/unsupported",
                "productType": "SCIENCE",
            }
        ]
    )

    with pytest.raises(MastProductNotFoundError, match="no supported"):
        download_fits("123", client, FitsCache(tmp_path))


def test_non_science_product_is_rejected(tmp_path: Path) -> None:
    """Auxiliary products are not mistaken for science light curves."""
    client = DownloadClient(
        [
            {
                "productFilename": "target_lc.fits",
                "dataURI": "mast:product/aux",
                "productType": "AUXILIARY",
            }
        ]
    )

    with pytest.raises(MastProductNotFoundError):
        download_fits("123", client, FitsCache(tmp_path))


def test_product_lookup_failure_is_descriptive(tmp_path: Path) -> None:
    """Product-list network errors retain observation context."""
    client = DownloadClient(ConnectionError("offline"))

    with pytest.raises(MastDownloadError, match="observation '123'"):
        download_fits("123", client, FitsCache(tmp_path))


def test_unsuccessful_download_status_cleans_temporary_file(tmp_path: Path) -> None:
    """Failed downloads never leave partial cache entries."""
    cache_root = tmp_path / "cache"
    client = DownloadClient(_products(), ("ERROR", "service unavailable", None))

    with pytest.raises(MastDownloadError, match="service unavailable"):
        download_fits("123", client, FitsCache(cache_root))

    assert list(cache_root.iterdir()) == []


def test_empty_successful_download_is_rejected(tmp_path: Path) -> None:
    """A nominal success without file content cannot enter the cache."""
    client = DownloadClient(_products())

    def empty_download(
        uri: str,
        *,
        local_path: str,
        cache: bool,
        verbose: bool,
    ) -> tuple[str, str | None, str | None]:
        return "COMPLETE", None, None

    client.download_file = empty_download  # type: ignore[method-assign]

    with pytest.raises(MastDownloadError, match="failed to download"):
        download_fits("123", client, FitsCache(tmp_path / "cache"))
