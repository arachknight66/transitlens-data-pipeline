"""Cached download of supported MAST light-curve FITS products."""

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from loguru import logger

from mast.auth import MastClient
from mast.cache import FitsCache
from mast.exceptions import MastDownloadError, MastProductNotFoundError
from mast.models import DownloadedFits


class _Product(NamedTuple):
    """Selected fields from a MAST product record."""

    filename: str
    data_uri: str
    priority: int


def download_fits(
    mast_id: str,
    client: MastClient,
    cache: FitsCache,
) -> DownloadedFits:
    """Download and cache the preferred light-curve FITS product.

    Args:
        mast_id: MAST observation identifier returned by observation search.
        client: Configured MAST client.
        cache: Caller-configured FITS cache.

    Returns:
        Metadata and path for the cached FITS product.

    Raises:
        MastProductNotFoundError: If no supported light-curve FITS exists.
        MastDownloadError: If product discovery or download fails.
    """
    try:
        products = client.get_product_list(mast_id)
        product = _select_product(products)
    except MastProductNotFoundError:
        raise
    except Exception as error:
        message = f"could not retrieve products for MAST observation '{mast_id}'"
        raise MastDownloadError(message) from error

    cached_path = cache.find(product.data_uri, product.filename)
    if cached_path is not None:
        logger.bind(mast_id=mast_id, path=str(cached_path)).info(
            "Using cached MAST FITS product"
        )
        return _download_result(mast_id, product, cached_path, from_cache=True)

    cache.root.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_download_path(cache.root)
    try:
        status, detail, _ = client.download_file(
            product.data_uri,
            local_path=str(temporary_path),
            cache=False,
            verbose=False,
        )
        if status.upper() != "COMPLETE":
            message = detail or "MAST returned an unsuccessful download status"
            raise MastDownloadError(message)
        final_path = cache.store(
            product.data_uri,
            product.filename,
            temporary_path,
        )
    except MastDownloadError:
        temporary_path.unlink(missing_ok=True)
        raise
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        message = f"failed to download MAST FITS product '{product.filename}'"
        raise MastDownloadError(message) from error

    logger.bind(mast_id=mast_id, path=str(final_path)).info(
        "Downloaded and cached MAST FITS product"
    )
    return _download_result(mast_id, product, final_path, from_cache=False)


def _select_product(rows: object) -> _Product:
    """Choose the preferred science light-curve FITS deterministically."""
    candidates: list[_Product] = []
    for row in rows:  # type: ignore[union-attr]
        filename = str(_row_value(row, "productFilename") or "")
        data_uri = str(_row_value(row, "dataURI") or "")
        product_type = str(_row_value(row, "productType") or "SCIENCE")
        priority = _light_curve_priority(filename)
        if data_uri and product_type.upper() == "SCIENCE" and priority is not None:
            candidates.append(_Product(filename, data_uri, priority))

    if not candidates:
        message = "observation contains no supported light-curve FITS product"
        raise MastProductNotFoundError(message)
    return min(
        candidates, key=lambda item: (item.priority, item.filename, item.data_uri)
    )


def _light_curve_priority(filename: str) -> int | None:
    """Rank known Kepler, K2, and TESS light-curve filenames."""
    name = filename.casefold()
    if not name.endswith(".fits") or "tpf" in name or "targetpixel" in name:
        return None
    suffixes = ("_lc.fits", "_llc.fits", "_slc.fits")
    for priority, suffix in enumerate(suffixes):
        if name.endswith(suffix):
            return priority
    return None


def _row_value(row: object, name: str) -> Any | None:
    """Read a value from an Astropy row or mapping."""
    column_names = getattr(row, "colnames", ())
    if name in column_names:
        return row[name]  # type: ignore[index]
    if isinstance(row, Mapping):
        return row.get(name)
    return None


def _temporary_download_path(cache_root: Path) -> Path:
    """Allocate a closed temporary file within the cache filesystem."""
    with tempfile.NamedTemporaryFile(
        dir=cache_root,
        prefix="mast-",
        suffix=".part",
        delete=False,
    ) as temporary_file:
        return Path(temporary_file.name)


def _download_result(
    mast_id: str,
    product: _Product,
    path: Path,
    *,
    from_cache: bool,
) -> DownloadedFits:
    """Build the stable public download result."""
    return DownloadedFits(
        mast_id=mast_id,
        product_filename=product.filename,
        data_uri=product.data_uri,
        path=path,
        from_cache=from_cache,
    )
