"""Small atomic-write helpers used by resumable Phase 1 stages."""

import os
import uuid
from pathlib import Path


def atomic_write_parquet(frame, destination, *, index=False):
    """Write a dataframe beside its destination and atomically promote it."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        frame.to_parquet(temporary, index=index)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

