"""Tests for the deterministic FITS filesystem cache."""

from pathlib import Path

import pytest

from mast.cache import FitsCache


def test_destination_is_stable_and_confined(tmp_path: Path) -> None:
    """Cache paths use stable URI hashes and discard path traversal."""
    cache = FitsCache(tmp_path)

    first = cache.destination("mast:product/1", "../unsafe/lightcurve_lc.fits")
    second = cache.destination("mast:product/1", "../unsafe/lightcurve_lc.fits")

    assert first == second
    assert first.parent == tmp_path
    assert first.name.endswith("-lightcurve_lc.fits")


def test_empty_filename_is_rejected(tmp_path: Path) -> None:
    """A cache entry cannot be created without a product filename."""
    with pytest.raises(ValueError, match="must not be empty"):
        FitsCache(tmp_path).destination("mast:product/1", "")


def test_find_ignores_missing_and_empty_entries(tmp_path: Path) -> None:
    """Incomplete files are never treated as valid cached downloads."""
    cache = FitsCache(tmp_path)
    assert cache.find("mast:product/1", "target_lc.fits") is None

    destination = cache.destination("mast:product/1", "target_lc.fits")
    destination.write_bytes(b"")

    assert cache.find("mast:product/1", "target_lc.fits") is None


def test_store_atomically_places_nonempty_file(tmp_path: Path) -> None:
    """Completed downloads are moved to their deterministic destination."""
    source = tmp_path / "temporary.part"
    source.write_bytes(b"FITS")
    cache = FitsCache(tmp_path / "cache")

    destination = cache.store("mast:product/1", "target_lc.fits", source)

    assert destination.read_bytes() == b"FITS"
    assert not source.exists()
    assert cache.find("mast:product/1", "target_lc.fits") == destination


def test_store_rejects_missing_or_empty_file(tmp_path: Path) -> None:
    """Only complete non-empty downloads may enter the cache."""
    cache = FitsCache(tmp_path / "cache")
    with pytest.raises(ValueError, match="missing or empty"):
        cache.store("mast:product/1", "target_lc.fits", tmp_path / "missing")
