# aperture_photometry.py
# --------------------
# Aperture photometry on TPF files: nested aperture definitions and flux extraction.

from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)

def perform_aperture_photometry(
    flux_cube: np.ndarray,
    flux_err_cube: np.ndarray,
    aperture_mask: np.ndarray,
    aperture_type: str = "nominal",
) -> dict:
    """
    Computes flux sum and error propagation on a TPF flux cube using the specified aperture.
    Types: "small", "nominal", "expanded"
    """
    nominal_mask = (aperture_mask & 2) > 0
    if nominal_mask.sum() == 0:
        nominal_mask = aperture_mask > 0
        
    if aperture_type == "small":
        # Find center of nominal aperture
        y_indices, x_indices = np.where(nominal_mask)
        if len(x_indices) == 0:
            active_mask = nominal_mask
        else:
            center_x = int(np.median(x_indices))
            center_y = int(np.median(y_indices))
            rows, cols = nominal_mask.shape
            y_grid, x_grid = np.mgrid[0:rows, 0:cols]
            active_mask = nominal_mask & (np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2) <= 1.0)
            if active_mask.sum() == 0:
                active_mask = nominal_mask
    elif aperture_type == "expanded":
        active_mask = nominal_mask.copy()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                active_mask = active_mask | np.roll(np.roll(nominal_mask, dy, axis=0), dx, axis=1)
        active_mask = active_mask & (aperture_mask >= 0)
    else: # nominal
        active_mask = nominal_mask
        
    n_pixels = int(active_mask.sum())
    
    # Calculate sum of flux and propagate error
    # sum_flux = sum(pixels)
    # sum_err = sqrt(sum(err^2))
    sum_flux = np.sum(flux_cube[:, active_mask], axis=1)
    sum_err = np.sqrt(np.sum(flux_err_cube[:, active_mask]**2, axis=1))
    
    # Check for saturation (flux > 1.5e5 or flat peaks)
    saturated = bool(np.any(flux_cube[:, active_mask] > 1.5e5))
    
    return {
        "aperture_type": aperture_type,
        "n_pixels": n_pixels,
        "flux": sum_flux,
        "flux_err": sum_err,
        "saturated": saturated,
        "mask": active_mask,
    }
