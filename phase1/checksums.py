import os
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def file_sha256(filepath):
    """Computes SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def generate_checksums_file(config):
    """
    Computes checksums for all parquet files and JSON summaries in the manifests directory
    and writes them to data/manifests/phase1/checksums.sha256.
    """
    config.ensure_dirs()
    manifests_dir = config.manifests_dir
    
    lines = []
    
    # Freeze canonical inputs/manifests only. Derived validation reports are
    # regenerated after this file and must not checksum themselves.
    for f in sorted(manifests_dir.iterdir()):
        if f.is_file() and f.name != "checksums.sha256" and (
            f.suffix == ".parquet" or f.name.startswith("mast_discovery_")
        ):
            h = file_sha256(f)
            lines.append(f"{h}  {f.name}\n")
            
    checksums_path = manifests_dir / "checksums.sha256"
    with open(checksums_path, "w", encoding="utf-8") as f_out:
        f_out.writelines(lines)
        
    logger.info(f"Generated checksums file at {checksums_path}")
    return checksums_path

def verify_checksums_file(config):
    """
    Verifies all checksums in data/manifests/phase1/checksums.sha256.
    Returns (success, list_of_failed_files, list_of_missing_files).
    """
    manifests_dir = config.manifests_dir
    checksums_path = manifests_dir / "checksums.sha256"
    
    if not checksums_path.exists():
        return False, [], [checksums_path.name]
        
    failed = []
    missing = []
    
    with open(checksums_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                # try single space or tabs
                parts = line.split(None, 1)
            if len(parts) != 2:
                continue
                
            expected_hash, fname = parts
            fpath = manifests_dir / fname
            
            if not fpath.exists():
                missing.append(fname)
                continue
                
            actual_hash = file_sha256(fpath)
            if actual_hash != expected_hash:
                failed.append(fname)
                
    success = (len(failed) == 0) and (len(missing) == 0)
    return success, failed, missing
