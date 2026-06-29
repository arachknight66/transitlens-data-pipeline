# tpf_downloader.py
# ----------------
# Resumable downloader for TESS Target Pixel Files (TPFs).

from __future__ import annotations
import logging
import os
import hashlib
from pathlib import Path
import urllib.request
import urllib.parse
import shutil
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
        filename = Path(urllib.parse.urlparse(data_uri).path).name if "product/" in data_uri else ""
        if not filename or not filename.endswith("_tp.fits"):
            filename = f"TIC{tic_id}_sector{sector:04d}_tp.fits"
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
            url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={data_uri}"
        else:
            url = data_uri
            
        try:
            logger.info(f"Downloading TPF for TIC {tic_id} from {url}...")
            
            # Simple chunked download with urllib
            temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
            offset = temp_path.stat().st_size if temp_path.exists() else 0
            headers = {'User-Agent': 'TransitLens/2.2'}
            if offset: headers['Range'] = f'bytes={offset}-'
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                append = offset > 0 and response.status == 206
                mode = "ab" if append else "wb"
                if not append: offset = 0
                with open(temp_path, mode) as out_file:
                    while True:
                        chunk = response.read(1024 * 1024)
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
            # Preserve partial data for a later ranged resume.
            result["status"] = "failed"
            result["error"] = str(e)
            
        return result
