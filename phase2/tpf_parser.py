# tpf_parser.py
# -------------
# Parser for TESS Target Pixel Files (TPFs).

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

class TpfParser:
    """Parses TESS Target Pixel Files (TPF) and loads coordinate/flux arrays."""
    def __init__(self, tpf_path: str | Path):
        self.tpf_path = Path(tpf_path)
        if not self.tpf_path.exists():
            raise FileNotFoundError(f"TPF file not found: {self.tpf_path}")
            
    def parse(self) -> dict:
        """Parses the TPF fits file and returns a dictionary of data products."""
        with fits.open(self.tpf_path, memmap=False) as hdul:
            if len(hdul) < 3:
                raise ValueError("TPF file has insufficient extensions (< 3).")
                
            primary_hdr = hdul[0].header
            tpf_data = hdul[1].data
            aperture_mask = hdul[2].data
            
            # WCS column coordinates
            target_col = float(primary_hdr.get("1CRPX4", 0.0))
            target_row = float(primary_hdr.get("2CRPX4", 0.0))
            
            # Time and cadence numbers
            time = np.array(tpf_data["TIME"], dtype=np.float64)
            cadenceno = np.array(tpf_data["CADENCENO"], dtype=np.int64)
            quality = np.array(tpf_data["QUALITY"], dtype=np.int64)
            
            # Pixel cubes
            flux_cube = np.array(tpf_data["FLUX"], dtype=np.float64)
            flux_err_cube = np.array(tpf_data["FLUX_ERR"], dtype=np.float64)
            bkg_cube = np.array(tpf_data["FLUX_BKG"], dtype=np.float64) if "FLUX_BKG" in tpf_data.columns.names else None
            
            # Metadata
            meta = {
                "tic_id": int(primary_hdr.get("TICID", 0)),
                "sector": int(primary_hdr.get("SECTOR", 0)),
                "camera": int(primary_hdr.get("CAMERA", 0)),
                "ccd": int(primary_hdr.get("CCD", 0)),
                "ra_obj": float(primary_hdr.get("RA_OBJ", 0.0)),
                "dec_obj": float(primary_hdr.get("DEC_OBJ", 0.0)),
            }
            
            return {
                "time": time,
                "cadenceno": cadenceno,
                "flux_cube": flux_cube,
                "flux_err_cube": flux_err_cube,
                "background_cube": bkg_cube,
                "quality": quality,
                "aperture_mask": aperture_mask,
                "target_column": target_col,
                "target_row": target_row,
                "metadata": meta,
            }
