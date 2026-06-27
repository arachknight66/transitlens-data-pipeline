"""
real_tess/mast_loader.py
─────────────────────────
Downloads and caches real TESS light curves from MAST via Lightkurve.

This is the Phase 5 stretch-goal data source, and the *only* file in
real_tess/ that talks to the network. sector_selector.py and
flux_normaliser.py are pure, offline post-processing so they can be
unit-tested without ever hitting MAST. interface.py's `_load_tess()`
checks for `lightkurve` and calls `fetch_light_curve()` below; it
never imports lightkurve itself.

Caveat: the hackathon offline demo must work with zero network
access. requirements.txt keeps lightkurve/astroquery commented out
by default — this module is only reachable if a contributor
deliberately installs them.
"""

import os

import numpy as np

from real_tess.flux_normaliser import normalise_pdcsap
from real_tess.sector_selector import select_best_sector


class TessDataUnavailableError(Exception):
    """Raised when no TESS light curve can be found or downloaded for a TIC ID."""


def fetch_light_curve(tic_id, sector=None, cache_dir="real_tess/cache", use_cache=True):
    """
    Fetches a real TESS light curve for a given TIC ID, using the
    on-disk cache when available and falling back to a MAST download
    via Lightkurve otherwise.

    Parameters
    ----------
    tic_id : str or int
        TIC identifier, with or without a "TIC" / "TIC-" prefix
        (e.g. "TIC-25155310", "TIC 25155310", "25155310", 25155310).
    sector : int, optional
        Restrict to this specific sector. If None, the best available
        sector is chosen automatically via sector_selector.
    cache_dir : str
        Directory to look in / save cached .fits files to. Created
        if it doesn't exist.
    use_cache : bool
        If False, skip the cache check and always hit MAST (still
        writes the freshly downloaded result back to cache_dir).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, int)
        (time, flux, sector) — time in BTJD, flux normalised to
        median = 1.0 via flux_normaliser.normalise_pdcsap.

    Raises
    ------
    ImportError
        If the 'lightkurve' package is not installed.
    TessDataUnavailableError
        If no observations exist for this TIC ID/sector, or the
        network is unreachable and nothing is cached.
    """
    clean_id = _normalise_tic_id(tic_id)
    os.makedirs(cache_dir, exist_ok=True)

    if use_cache:
        cached = _find_cached_file(clean_id, sector, cache_dir)
        if cached is not None:
            cache_path, cached_sector = cached
            time, flux, quality = _read_fits_cache(cache_path)
            flux = normalise_pdcsap(flux, quality_flags=quality)
            return time, flux, cached_sector

    try:
        import lightkurve as lk
    except ImportError as exc:
        raise ImportError(
            "fetch_light_curve() requires the 'lightkurve' package, which is "
            "not installed. Uncomment lightkurve/astroquery in "
            "requirements.txt and `pip install -r requirements.txt` to "
            "enable real TESS data, or rely on the synthetic source for the "
            "offline demo."
        ) from exc

    search_kwargs = {"mission": "TESS"}
    if sector is not None:
        search_kwargs["sector"] = sector

    try:
        search_result = lk.search_lightcurve(f"TIC {clean_id}", **search_kwargs)
    except Exception as exc:
        # Covers network-unreachable / MAST-down conditions from
        # astroquery, which don't share one consistent exception type.
        raise TessDataUnavailableError(
            f"Could not reach MAST to search for TIC {clean_id}: {exc}"
        ) from exc

    if len(search_result) == 0:
        raise TessDataUnavailableError(
            f"No TESS light curve observations found on MAST for TIC {clean_id}"
            + (f", sector {sector}" if sector is not None else "") + "."
        )

    best_index = 0 if sector is not None else select_best_sector(search_result)

    lc = _download_with_retry(search_result, best_index)

    resolved_sector = (
        int(lc.sector) if getattr(lc, "sector", None) is not None
        else (sector if sector is not None else -1)
    )

    cache_path = os.path.join(cache_dir, _cache_filename(clean_id, resolved_sector))
    lc.to_fits(cache_path, overwrite=True)

    time = np.asarray(lc.time.value, dtype=np.float64)
    flux_raw = np.asarray(lc.flux.value, dtype=np.float64)
    quality = np.asarray(lc.quality, dtype=np.int64) if hasattr(lc, "quality") else None

    flux = normalise_pdcsap(flux_raw, quality_flags=quality)

    return time, flux, resolved_sector


def _download_with_retry(search_result, index, retries=1):
    """Downloads search_result[index], retrying once on timeout/failure."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return search_result[index].download()
        except Exception as exc:  # noqa: BLE001 - network errors vary by backend
            last_exc = exc
    raise TessDataUnavailableError(
        f"Download failed after {retries + 1} attempt(s): {last_exc}"
    ) from last_exc


def _normalise_tic_id(tic_id):
    """Strips 'TIC'/'TIC-' prefixes and whitespace so cache filenames
    and lookups are consistent regardless of how the caller wrote it."""
    clean_id = str(tic_id).upper().replace("TIC", "").replace("-", "").strip()
    clean_id = "".join(clean_id.split())
    if not clean_id.isdigit():
        raise ValueError(
            f"Invalid TIC ID '{tic_id}'. Enter the numeric TESS Input Catalog identifier."
        )
    return clean_id


def _cache_filename(clean_id, sector):
    return f"TIC{clean_id}_sector{sector:03d}.fits"


def _find_cached_file(clean_id, sector, cache_dir):
    """Looks for a previously-downloaded .fits file in cache_dir
    matching this TIC ID (and sector, if specified)."""
    if not os.path.isdir(cache_dir):
        return None

    prefix = f"TIC{clean_id}_sector"
    for fname in sorted(os.listdir(cache_dir)):
        if not (fname.startswith(prefix) and fname.endswith(".fits")):
            continue
        try:
            file_sector = int(fname[len(prefix):].split(".")[0])
        except ValueError:
            continue
        if sector is None or file_sector == sector:
            return os.path.join(cache_dir, fname), file_sector

    return None


def _read_fits_cache(cache_path):
    """Read cached SPOC/QLP FITS without requiring Lightkurve.

    The shared Astropy parser supports standard TESS light-curve files and
    Lightkurve exports. This keeps cached TIC retrieval functional in the
    normal ml-core environment, where Astropy is installed but Lightkurve may
    intentionally be absent.
    """
    from real_tess.fits_parser import read_fits_lightcurve

    parsed = read_fits_lightcurve(cache_path)
    return parsed["time"], parsed["flux_raw"], parsed["quality"]
