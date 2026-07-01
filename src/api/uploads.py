"""Bounded streaming storage and validation for uploaded light curves."""

from __future__ import annotations

import os
import re
import secrets
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile
from loguru import logger

from fits.csv_reader import validate_csv
from fits.exceptions import FitsError
from fits.reader import read_fits

_ALLOWED_SUFFIXES = frozenset({".fits", ".fit", ".csv"})
_IDENTIFIER_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class UploadError(RuntimeError):
    """Base exception for upload storage and validation failures."""


class UnsupportedUploadError(UploadError):
    """Raised when an upload filename or format is unsupported."""


class InvalidUploadError(UploadError):
    """Raised when uploaded content is empty, malformed, or unusable."""


class UploadTooLargeError(UploadError):
    """Raised when streamed content exceeds the configured size limit."""


class UploadNotFoundError(UploadError):
    """Raised when an opaque upload identifier is invalid or unavailable."""


@dataclass(frozen=True)
class StoredUpload:
    """Internal metadata for one validated temporary upload."""

    file_id: str
    path: Path
    media_type: str
    size_bytes: int


class UploadStore:
    """Filesystem-backed temporary upload cache addressed by opaque IDs."""

    def __init__(
        self,
        root: Path,
        *,
        max_size_bytes: int,
        chunk_size_bytes: int,
        retention_seconds: int,
    ) -> None:
        """Configure bounded streaming storage.

        Args:
            root: Dedicated upload cache directory.
            max_size_bytes: Maximum accepted file content size.
            chunk_size_bytes: Maximum bytes read from the request at once.
            retention_seconds: Age after which stored uploads are deleted.
        """
        if min(max_size_bytes, chunk_size_bytes, retention_seconds) <= 0:
            raise ValueError("upload size, chunk size, and retention must be positive")
        self._root = root.expanduser().resolve()
        self._max_size_bytes = max_size_bytes
        self._chunk_size_bytes = chunk_size_bytes
        self._retention_seconds = retention_seconds

    async def save(self, upload: UploadFile) -> StoredUpload:
        """Stream, validate, and atomically store one multipart upload.

        Args:
            upload: FastAPI multipart file stream.

        Returns:
            Internal upload metadata including an opaque random identifier.

        Raises:
            UnsupportedUploadError: If the filename extension is unsupported.
            UploadTooLargeError: If content exceeds the configured limit.
            InvalidUploadError: If content is empty or scientifically invalid.
        """
        temporary_path: Path | None = None
        size_bytes = 0
        try:
            suffix = _validated_suffix(upload.filename)
            self._root.mkdir(parents=True, exist_ok=True)
            self.cleanup_expired()
            temporary_path = self._temporary_path()
            read_size = min(self._chunk_size_bytes, self._max_size_bytes + 1)
            with temporary_path.open("wb") as destination:
                while chunk := await upload.read(read_size):
                    size_bytes += len(chunk)
                    if size_bytes > self._max_size_bytes:
                        message = (
                            "upload exceeds maximum size of "
                            f"{self._max_size_bytes} bytes"
                        )
                        raise UploadTooLargeError(message)
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if size_bytes == 0:
                raise InvalidUploadError("uploaded file is empty")
            _validate_content(temporary_path, suffix)
            file_id, final_path = self._new_destination(suffix)
            os.replace(temporary_path, final_path)
        except UploadError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        except FitsError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise InvalidUploadError(str(error)) from error
        except OSError as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise UploadError("uploaded file could not be stored") from error
        finally:
            try:
                await upload.close()
            except Exception as error:
                logger.bind(error_type=type(error).__name__).warning(
                    "Could not close completed upload stream"
                )

        logger.bind(
            file_id=file_id,
            media_type=suffix.removeprefix("."),
            size_bytes=size_bytes,
        ).info("Stored validated upload")
        return StoredUpload(
            file_id=file_id,
            path=final_path,
            media_type=suffix.removeprefix("."),
            size_bytes=size_bytes,
        )

    def resolve(self, file_id: str) -> StoredUpload:
        """Resolve an opaque identifier to an unexpired internal upload.

        Args:
            file_id: Identifier previously returned by ``save``.

        Returns:
            Internal upload metadata.

        Raises:
            UploadNotFoundError: If the ID is malformed, expired, or unavailable.
        """
        self.cleanup_expired()
        if not _IDENTIFIER_PATTERN.fullmatch(file_id):
            raise UploadNotFoundError("uploaded file identifier is invalid or expired")
        for suffix in sorted(_ALLOWED_SUFFIXES):
            path = self._root / f"{file_id}{suffix}"
            if path.is_file():
                return StoredUpload(
                    file_id=file_id,
                    path=path,
                    media_type=suffix.removeprefix("."),
                    size_bytes=path.stat().st_size,
                )
        raise UploadNotFoundError("uploaded file identifier is invalid or expired")

    def cleanup_expired(self) -> int:
        """Delete expired completed uploads and abandoned partial files.

        Returns:
            Number of cache files successfully removed.
        """
        if not self._root.is_dir():
            return 0
        cutoff = time.time() - self._retention_seconds
        removed = 0
        for path in self._root.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                logger.bind(path_name=path.name).warning(
                    "Could not remove expired upload cache file"
                )
        if removed:
            logger.bind(removed=removed).info("Cleaned expired upload cache files")
        return removed

    def _temporary_path(self) -> Path:
        """Allocate a unique closed partial file inside the upload cache."""
        with tempfile.NamedTemporaryFile(
            dir=self._root,
            prefix=".upload-",
            suffix=".part",
            delete=False,
        ) as temporary_file:
            return Path(temporary_file.name)

    def _new_destination(self, suffix: str) -> tuple[str, Path]:
        """Allocate a collision-free opaque identifier and final path."""
        while True:
            file_id = secrets.token_hex(16)
            destination = self._root / f"{file_id}{suffix}"
            if not destination.exists():
                return file_id, destination


def _validated_suffix(filename: str | None) -> str:
    """Extract a safe supported extension without trusting directory segments."""
    if filename is None or "\x00" in filename:
        raise UnsupportedUploadError("upload filename is missing or invalid")
    basename = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1].strip()
    if not basename:
        raise UnsupportedUploadError("upload filename is missing or invalid")
    suffix = Path(basename).suffix.casefold()
    if suffix not in _ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(_ALLOWED_SUFFIXES))
        raise UnsupportedUploadError(f"unsupported upload type; expected {allowed}")
    return suffix


def _validate_content(path: Path, suffix: str) -> None:
    """Validate uploaded content using the relevant production parser."""
    if suffix == ".csv":
        validate_csv(path)
    else:
        read_fits(path)
