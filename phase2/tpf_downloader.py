# tpf_downloader.py
# ----------------
# Resumable downloader for TESS Target Pixel Files (TPFs).

from __future__ import annotations
import logging
import os
import hashlib
from pathlib import Path
import urllib.request
from astropy.io import fits

logger = logging.getLogger(__name__)

def file_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

class TpfDownloader:
    """Handles resumable, audited downloads of TESS Target Pixel Files."""
    def __init__(self, dest_dir: Path):
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        
    def download_tpf(
        self,
        tic_id: int,
        sector: int,
        data_uri: str,
        expected_checksum: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Downloads a single TPF from MAST.
        """
        filename = f"tess2024{tic_id:012d}_sector-{sector:04d}_tpf.fits"
        dest_path = self.dest_dir / filename
        
        result = {
            "tic_id": tic_id,
            "sector": sector,
            "filename": filename,
            "local_path": str(dest_path),
            "status": "pending",
            "error": "",
        }
        
        # Check if already present and valid
        if dest_path.exists():
            try:
                with fits.open(dest_path, memmap=False) as hdul:
                    pass # valid FITS
                result["status"] = "verified"
                return result
            except Exception:
                # corrupt file, delete and redownload
                logger.warning(f"Local file {dest_path} is corrupt. Redownloading.")
                dest_path.unlink()
                
        if dry_run:
            result["status"] = "dry_run"
            return result
            
        # Standardize MAST data URI to HTTP URL
        # e.g., mast:TESS/product/tess2019112060037-s0011-0000000261136679-0143-s_tp.fits
        # url: https://mast.stsci.edu/api/v0.1/retrieve?uri=mast:TESS/product/...
        if data_uri.startswith("mast:"):
            url = f"https://mast.stsci.edu/api/v0.1/retrieve?uri={data_uri}"
        else:
            url = data_uri
            
        try:
            logger.info(f"Downloading TPF for TIC {tic_id} from {url}...")
            
            # Simple chunked download with urllib
            temp_path = dest_path.with_suffix(".tmp")
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
                chunk_size = 1024 * 1024 # 1MB
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    
            os.replace(temp_path, dest_path)
            
            # Verify FITS structure
            with fits.open(dest_path, memmap=False) as hdul:
                pass
                
            # Verify checksum if provided
            if expected_checksum:
                actual = file_sha256(dest_path)
                if actual != expected_checksum:
                    dest_path.unlink()
                    result["status"] = "checksum_failed"
                    result["error"] = f"Checksum mismatch: expected {expected_checksum}, got {actual}"
                    return result
                    
            result["status"] = "verified"
            
        except Exception as e:
            logger.error(f"Download failed for TIC {tic_id} sector {sector}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            result["status"] = "failed"
            result["error"] = str(e)
            
        return result
