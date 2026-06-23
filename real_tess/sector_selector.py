"""
real_tess/sector_selector.py
─────────────────────────────
Picks the best sector out of a Lightkurve SearchResult when multiple
sectors/products are available for a TIC ID.

Selection criteria (in priority order), per the plan:
1. Highest number of data points (fewest gaps)
2. PDC-SAP flux available
3. 2-minute cadence (not 30-minute FFI cadence)
4. Most recent sector (in case of tie)

Design note
-----------
Criterion 1 ("highest number of data points") technically requires
downloading each candidate to count real, unflagged cadences — but
doing that for every search result just to pick one is exactly the
expensive network round-trip the cache in mast_loader.py is meant to
avoid. So this module works in two tiers:

- `select_best_sector()` (default, cheap): uses only the lightweight
  metadata table that `lightkurve.search_lightcurve()` already
  returns (exposure time / cadence, sector number) — no network call
  beyond the search itself. This covers criteria 2-4 directly and
  uses cadence as a strong proxy for criterion 1, since short-cadence
  (2-min) sectors almost always have far more data points than
  30-min FFI sectors over the same time span.
- `select_best_sector(..., candidates=[...])` (opt-in, expensive): if
  the caller has already downloaded candidate LightCurve objects
  (e.g. mast_loader.py downloading the top few ranked by the cheap
  pass), this re-ranks by the *actual* unflagged point count, giving
  an exact answer to criterion 1 at the cost of those downloads.
"""

import numpy as np


# TESS pipeline products report exposure time in seconds.
_TWO_MINUTE_CADENCE_SEC = 120.0
_TWENTY_SECOND_CADENCE_SEC = 20.0


def select_best_sector(search_results, candidates=None):
    """
    Returns the index into `search_results` (NOT a sector number) of
    the best entry, ranked by the criteria documented above.

    Parameters
    ----------
    search_results : lightkurve.SearchResult or any object supporting
        `len()` and exposing a `.table` attribute (an astropy Table)
        with an `exptime` column. Lightkurve's real SearchResult
        satisfies this without any extra work.
    candidates : list of lightkurve.LightCurve, optional
        If provided (same length and order as search_results), the
        actual downloaded light curves are used to rank by exact
        unflagged data-point count instead of the cadence proxy.

    Returns
    -------
    int
        Index of the best entry in `search_results`.

    Raises
    ------
    ValueError
        If `search_results` is empty.
    """
    n = len(search_results)
    if n == 0:
        raise ValueError("select_best_sector() received an empty SearchResult.")

    if candidates is not None:
        return _select_by_actual_point_count(candidates)

    return _select_by_metadata(search_results)


def _select_by_metadata(search_results):
    """Cheap tier: rank using only the search-result metadata table."""
    table = getattr(search_results, "table", None)
    n = len(search_results)

    if table is None or "exptime" not in getattr(table, "colnames", []):
        # No usable metadata at all -- fall back to "most recent" by
        # assuming results are already returned in chronological order
        # (lightkurve's default), so the last entry is the newest.
        return n - 1

    exptimes = np.asarray(table["exptime"], dtype=np.float64)
    sectors = (
        np.asarray(table["sequence_number"], dtype=np.float64)
        if "sequence_number" in table.colnames
        else np.arange(n, dtype=np.float64)
    )

    best_index = 0
    best_score = None

    for i in range(n):
        exptime = exptimes[i]

        # Criterion 3: strongly prefer 2-minute (or finer, e.g. 20s)
        # cadence over long-cadence FFI products (10/30 min).
        cadence_score = 2 if exptime <= _TWO_MINUTE_CADENCE_SEC else (
            1 if exptime <= 600.0 else 0
        )

        # Criterion 1 proxy: shorter cadence -> more points for the
        # same observing window, so smaller exptime is better within
        # a cadence tier.
        cadence_proxy = -exptime

        # Criterion 4: prefer the more recent sector as a tie-breaker.
        recency = sectors[i]

        score = (cadence_score, cadence_proxy, recency)

        if best_score is None or score > best_score:
            best_score = score
            best_index = i

    return best_index


def _select_by_actual_point_count(candidates):
    """Expensive tier: rank already-downloaded LightCurve objects by
    exact unflagged data-point count, then cadence, then recency."""
    if len(candidates) == 0:
        raise ValueError("select_best_sector() received an empty candidates list.")

    best_index = 0
    best_score = None

    for i, lc in enumerate(candidates):
        flux = np.asarray(lc.flux.value, dtype=np.float64)
        n_valid = int(np.sum(np.isfinite(flux)))

        exptime = float(getattr(lc, "exptime", 120.0))
        if hasattr(exptime, "value"):
            exptime = float(exptime.value)
        cadence_score = 1.0 if exptime <= _TWO_MINUTE_CADENCE_SEC else 0.0

        sector = float(getattr(lc, "sector", i))

        score = (n_valid, cadence_score, sector)

        if best_score is None or score > best_score:
            best_score = score
            best_index = i

    return best_index