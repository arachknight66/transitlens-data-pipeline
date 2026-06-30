"""
real_tess/flux_normaliser.py
──────────────────────────────
Normalises raw PDC-SAP flux from real TESS light curves to the same
median=1.0 convention used everywhere else in this pipeline (see the
output contract documented in interface.py).

This module has no dependency on lightkurve/astroquery and does no
network I/O, so it can be unit-tested with plain numpy arrays even
when Phase 5's MAST download path isn't installed/available.
"""

import numpy as np


def normalise_pdcsap(flux_raw, quality_flags=None, clip_sigma=50.0):
    """
    Normalises PDC-SAP flux to median = 1.0.

    Steps:
    1. Remove flagged cadences (quality_flags != 0) by setting to NaN
    2. Compute median of unflagged values
    3. Divide entire array by median
    4. Clip extreme outliers beyond `clip_sigma` (default 5-sigma)
    5. Return normalised array

    Parameters
    ----------
    flux_raw : np.ndarray
        Raw PDC-SAP flux values (typically electrons/second, as
        reported by the TESS pipeline). Not modified in-place.
    quality_flags : np.ndarray, optional
        TESS quality bitmask per cadence, same length as flux_raw.
        Any non-zero value marks the cadence as flagged/unreliable
        and it is set to NaN before computing the median.
    clip_sigma : float
        Number of standard deviations (computed on the *normalised*
        data) beyond which values are clipped. Matches the plan's
        "clip extreme outliers beyond 5-sigma" spec by default.

    Returns
    -------
    np.ndarray
        Normalised flux array, same length and dtype as flux_raw.
        Flagged cadences remain NaN; extreme outliers are clipped,
        not removed, so the array length never changes (downstream
        code can always assume len(time) == len(flux)).

    Raises
    ------
    ValueError
        If every cadence is flagged/non-finite, or the computed
        median is zero/non-finite (normalisation would be undefined).
    """
    flux = np.array(flux_raw, dtype=np.float64, copy=True)

    # Step 1: mask out flagged cadences
    if quality_flags is not None:
        quality_flags = np.asarray(quality_flags)
        if len(quality_flags) != len(flux):
            raise ValueError(
                f"quality_flags length ({len(quality_flags)}) must match "
                f"flux_raw length ({len(flux)})."
            )
        flux[quality_flags != 0] = np.nan

    # Step 2: median of unflagged, finite values
    finite_mask = np.isfinite(flux)
    if not np.any(finite_mask):
        raise ValueError(
            "normalise_pdcsap() received no finite, unflagged flux values "
            "to compute a median from."
        )
    median = float(np.nanmedian(flux[finite_mask]))

    if median == 0.0 or not np.isfinite(median):
        raise ValueError(
            f"normalise_pdcsap() computed a non-finite or zero median ({median}); "
            "cannot normalise."
        )

    # Step 3: normalise so the unflagged median becomes 1.0
    flux = flux / median

    # Step 4: clip extreme outliers beyond clip_sigma. Using plain
    # np.std here would be circular: a single huge outlier (TESS
    # momentum-dump spikes can be many orders of magnitude off) can
    # inflate the std enough that the clip threshold no longer
    # excludes it. The median absolute deviation (MAD), scaled by
    # 1.4826 to approximate a Gaussian sigma, is robust to exactly
    # this failure mode since the median barely moves when a handful
    # of points are extreme. np.clip leaves NaNs untouched, so
    # flagged cadences stay NaN rather than being clipped to a bound.
    normalised_finite = flux[np.isfinite(flux)]
    mad = float(np.nanmedian(np.abs(normalised_finite - 1.0)))
    sigma = 1.4826 * mad
    if sigma > 0:
        lower = 1.0 - clip_sigma * sigma
        upper = 1.0 + clip_sigma * sigma
        flux = np.clip(flux, lower, upper)

    # Step 5
    return flux