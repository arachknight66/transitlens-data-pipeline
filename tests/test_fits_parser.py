import os
import pytest
import numpy as np
from astropy.io import fits
from real_tess.fits_parser import read_fits_lightcurve, load_fits_and_normalize, InvalidFITSStructureError
from interface import load_light_curve

# Paths for testing
CACHE_DIR = r"C:\Users\arach\Documents\Projects\Transitlens\transitlens-data-pipeline\real_tess\cache"
VALID_FITS = os.path.join(CACHE_DIR, "TIC261136679_sector095.fits")

def test_read_valid_fits():
    """Verify that read_fits_lightcurve correctly reads a valid FITS file."""
    assert os.path.exists(VALID_FITS), f"Test requires cached TESS FITS: {VALID_FITS}"
    
    parsed = read_fits_lightcurve(VALID_FITS)
    assert "time" in parsed
    assert "flux_raw" in parsed
    assert "quality" in parsed
    assert "metadata" in parsed
    
    assert len(parsed["time"]) > 0
    assert len(parsed["time"]) == len(parsed["flux_raw"])
    assert parsed["metadata"]["target_id"] is not None
    assert parsed["metadata"]["sector"] == 95

def test_load_fits_and_normalize_valid():
    """Verify that load_fits_and_normalize correctly normalizes and cleans the data."""
    parsed = load_fits_and_normalize(VALID_FITS)
    
    time = parsed["time"]
    flux = parsed["flux"]
    metadata = parsed["metadata"]
    
    assert isinstance(time, list)
    assert isinstance(flux, list)
    assert len(time) == len(flux)
    assert len(time) >= 100
    
    # Enforce normalization (median ~ 1.0)
    assert abs(np.median(flux) - 1.0) < 0.01
    
    # Enforce strictly monotonic time
    assert all(time[i] < time[i+1] for i in range(len(time)-1))
    
    # Check metadata keys
    assert "cadence_min" in metadata
    assert "time_span_days" in metadata
    assert "sector" in metadata
    assert "flux_type_used" in metadata

def test_load_light_curve_fits():
    """Verify interface entry point with source='fits'."""
    result = load_light_curve(
        source="fits",
        target_id="TIC 261136679",
        config={"path": VALID_FITS}
    )
    
    assert result["source"] == "fits"
    assert result["target_id"] == "TIC 261136679"
    assert len(result["time"]) == result["n_points"]
    assert abs(np.median(result["flux"]) - 1.0) < 0.001

def test_fits_missing_file():
    """Verify appropriate error when FITS file is missing."""
    with pytest.raises(FileNotFoundError):
        read_fits_lightcurve("nonexistent_file.fits")

def test_fits_invalid_structure(tmp_path):
    """Verify error raised on invalid FITS structure (e.g. empty primary HDU only)."""
    invalid_path = os.path.join(tmp_path, "invalid.fits")
    
    # Create empty FITS file
    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu])
    hdul.writeto(invalid_path)
    
    with pytest.raises(InvalidFITSStructureError):
        read_fits_lightcurve(invalid_path)

def test_fits_missing_columns(tmp_path):
    """Verify error raised on FITS missing expected columns."""
    invalid_cols_path = os.path.join(tmp_path, "invalid_cols.fits")
    
    primary_hdu = fits.PrimaryHDU()
    # Create binary table without TIME
    col = fits.Column(name="FLUX", format="E", array=np.ones(10))
    tb_hdu = fits.BinTableHDU.from_columns([col], name="LIGHTCURVE")
    hdul = fits.HDUList([primary_hdu, tb_hdu])
    hdul.writeto(invalid_cols_path)
    
    with pytest.raises(InvalidFITSStructureError):
        read_fits_lightcurve(invalid_cols_path)
        
    # Create binary table with TIME but without FLUX
    invalid_flux_path = os.path.join(tmp_path, "invalid_flux.fits")
    col_time = fits.Column(name="TIME", format="D", array=np.arange(10))
    tb_hdu_flux = fits.BinTableHDU.from_columns([col_time], name="LIGHTCURVE")
    hdul_flux = fits.HDUList([primary_hdu, tb_hdu_flux])
    hdul_flux.writeto(invalid_flux_path)
    
    with pytest.raises(InvalidFITSStructureError):
        read_fits_lightcurve(invalid_flux_path)

def test_fits_invalid_flux_dimensions(tmp_path):
    """Verify error raised when FITS contains multidimensional flux column."""
    invalid_dim_path = os.path.join(tmp_path, "invalid_dim.fits")
    
    primary_hdu = fits.PrimaryHDU()
    col_time = fits.Column(name="TIME", format="D", array=np.arange(10))
    # Create 3D flux data shape (10, 20, 20)
    flux_3d = np.ones((10, 20, 20))
    col_flux = fits.Column(name="FLUX", format="400E", array=flux_3d, dim="(20,20)")
    tb_hdu = fits.BinTableHDU.from_columns([col_time, col_flux], name="LIGHTCURVE")
    hdul = fits.HDUList([primary_hdu, tb_hdu])
    hdul.writeto(invalid_dim_path)
    
    with pytest.raises(InvalidFITSStructureError) as excinfo:
        read_fits_lightcurve(invalid_dim_path)
    assert "must be 1-dimensional" in str(excinfo.value)
