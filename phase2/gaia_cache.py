# gaia_cache.py
# -------------
# Offline cache manager for Gaia DR3 cone queries.

from __future__ import annotations
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class GaiaCacheManager:
    """Manages reading and writing local cached Gaia query results."""
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_key(self, ra: float, dec: float, radius_arcsec: float) -> str:
        return f"gaia_ra{ra:.5f}_dec{dec:.5f}_r{radius_arcsec:.1f}"
        
    def get_cache_file(self, ra: float, dec: float, radius_arcsec: float) -> Path:
        key = self.get_cache_key(ra, dec, radius_arcsec)
        return self.cache_dir / f"{key}.json"
        
    def lookup(self, ra: float, dec: float, radius_arcsec: float) -> list[dict] | None:
        cfile = self.get_cache_file(ra, dec, radius_arcsec)
        if cfile.exists():
            try:
                with open(cfile, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached Gaia file {cfile}: {e}")
        return None
        
    def save(self, ra: float, dec: float, radius_arcsec: float, data: list[dict]) -> None:
        cfile = self.get_cache_file(ra, dec, radius_arcsec)
        try:
            with open(cfile, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved Gaia cache to {cfile}")
        except Exception as e:
            logger.warning(f"Failed to save Gaia cache to {cfile}: {e}")
