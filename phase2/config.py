# config.py
# ---------
# Pipeline configuration settings and path directory overrides for TransitLens Phase 2.

from __future__ import annotations
from pathlib import Path
import yaml
from phase1.config import Config as Phase1Config

class Phase2Config(Phase1Config):
    """
    Extends Phase 1 Config to include Target Pixel File paths,
    Gaia cache paths, and Phase 2 run directories.
    """
    def __init__(self, config_path: str | None = None):
        super().__init__(config_path)
        
        # Override or set Phase 2 paths
        self.tpf_dir = self.raw_dir / "tpf"
        self.gaia_cache_dir = self.REPO_ROOT / "data" / "cache" / "gaia"
        
    def ensure_dirs(self):
        super().ensure_dirs()
        self.tpf_dir.mkdir(parents=True, exist_ok=True)
        self.gaia_cache_dir.mkdir(parents=True, exist_ok=True)
