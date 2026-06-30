"""Filesystem cache for downloaded MAST FITS products."""

import hashlib
import os
from pathlib import Path


class FitsCache:
    """Store downloaded FITS products under stable, collision-safe paths."""

    def __init__(self, root: Path) -> None:
        """Initialize a cache rooted at a caller-configured directory.

        Args:
            root: Directory where cached FITS files will be stored.
        """
        self._root = root

    @property
    def root(self) -> Path:
        """Return the configured cache root."""
        return self._root

    def destination(self, data_uri: str, product_filename: str) -> Path:
        """Return the deterministic path for a MAST product.

        Args:
            data_uri: Globally identifying MAST product URI.
            product_filename: Server-provided product filename.

        Returns:
            A path contained directly inside the configured cache root.
        """
        digest = hashlib.sha256(data_uri.encode("utf-8")).hexdigest()[:16]
        safe_filename = Path(product_filename).name
        if not safe_filename:
            message = "product filename must not be empty"
            raise ValueError(message)
        return self._root / f"{digest}-{safe_filename}"

    def find(self, data_uri: str, product_filename: str) -> Path | None:
        """Return a valid cached product, if present and non-empty."""
        path = self.destination(data_uri, product_filename)
        return path if path.is_file() and path.stat().st_size > 0 else None

    def store(
        self,
        data_uri: str,
        product_filename: str,
        temporary_path: Path,
    ) -> Path:
        """Atomically place a completed temporary download in the cache.

        Args:
            data_uri: Globally identifying MAST product URI.
            product_filename: Server-provided product filename.
            temporary_path: Completed download awaiting cache placement.

        Returns:
            Final cached file path.

        Raises:
            ValueError: If the temporary download is missing or empty.
        """
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            message = "downloaded FITS file is missing or empty"
            raise ValueError(message)
        self._root.mkdir(parents=True, exist_ok=True)
        destination = self.destination(data_uri, product_filename)
        os.replace(temporary_path, destination)
        return destination
