"""Security, streaming, cleanup, and workflow tests for uploads."""

import io
import os
import re
import time
from pathlib import Path

import httpx
import pytest
from astropy.io import fits as astropy_fits

from api.app import create_app
from api.services import ApiServices
from api.uploads import UploadStore
from config import Settings


@pytest.fixture
def anyio_backend() -> str:
    """Run upload ASGI tests using asyncio."""
    return "asyncio"


def _app(tmp_path: Path, **overrides: object):
    """Create an isolated upload-capable application."""
    settings = Settings(
        cache_dir=tmp_path / "cache",
        log_level="WARNING",
        **overrides,
    )
    services = ApiServices(mast_client_provider=lambda: None)  # type: ignore[arg-type]
    return create_app(settings, services)


def _fits_bytes() -> bytes:
    """Return a compact valid TESS light-curve FITS payload."""
    primary = astropy_fits.PrimaryHDU()
    primary.header["MISSION"] = "TESS"
    primary.header["OBJECT"] = "Uploaded Target"
    table = astropy_fits.BinTableHDU.from_columns(
        [
            astropy_fits.Column(name="TIME", format="D", array=[1.0, 2.0, 3.0]),
            astropy_fits.Column(
                name="PDCSAP_FLUX", format="E", array=[100.0, 99.0, 101.0]
            ),
            astropy_fits.Column(name="QUALITY", format="J", array=[0, 0, 0]),
        ],
        name="LIGHTCURVE",
    )
    output = io.BytesIO()
    astropy_fits.HDUList([primary, table]).writeto(output)
    return output.getvalue()


def _csv_bytes() -> bytes:
    """Return a valid generic CSV light curve."""
    return b"TIME,FLUX,QUALITY\n1.0,100.0,0\n2.0,99.0,0\n3.0,101.0,0\n"


@pytest.mark.anyio
@pytest.mark.parametrize("extension", ["fits", "fit", "FITS"])
async def test_upload_accepts_fits_and_returns_only_opaque_metadata(
    tmp_path: Path,
    extension: str,
) -> None:
    """FITS variants are validated and never expose their internal path."""
    content = _fits_bytes()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path)),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/upload",
            files={"file": (f"target.{extension}", content, "application/fits")},
        )

    assert response.status_code == 201
    payload = response.json()
    assert re.fullmatch(r"[0-9a-f]{32}", payload["file_id"])
    assert payload["media_type"] == extension.casefold()
    assert payload["size_bytes"] == len(content)
    assert set(payload) == {"file_id", "media_type", "size_bytes"}
    stored = list((tmp_path / "cache" / "uploads").iterdir())
    assert [path.name for path in stored] == [
        f"{payload['file_id']}.{extension.casefold()}"
    ]


@pytest.mark.anyio
async def test_uploaded_fits_processes_by_id_without_path_disclosure(
    tmp_path: Path,
) -> None:
    """The opaque identifier drives the existing canonical processing pipeline."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path)),
        base_url="http://test",
    ) as client:
        uploaded = await client.post(
            "/upload",
            files={"file": ("target.fits", _fits_bytes(), "application/fits")},
        )
        file_id = uploaded.json()["file_id"]
        processed = await client.post(
            "/process",
            json={"file_id": file_id, "mission": "TESS"},
        )

    assert processed.status_code == 200
    payload = processed.json()
    assert payload["file_id"] == file_id
    assert payload["time"] == [1.0, 2.0, 3.0]
    assert payload["metadata"]["source"]["source_path"] == file_id
    assert payload["features"]["metadata"]["source_path"] == file_id
    assert str(tmp_path) not in processed.text
    assert "uploads" not in processed.text


@pytest.mark.anyio
async def test_csv_upload_requires_mission_then_processes(tmp_path: Path) -> None:
    """CSV uses the same LightCurve contract once mission semantics are supplied."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path)),
        base_url="http://test",
    ) as client:
        uploaded = await client.post(
            "/upload",
            files={"file": ("curve.csv", _csv_bytes(), "text/csv")},
        )
        file_id = uploaded.json()["file_id"]
        missing_mission = await client.post("/process", json={"file_id": file_id})
        processed = await client.post(
            "/process",
            json={"file_id": file_id, "mission": "TESS"},
        )

    assert uploaded.status_code == 201
    assert uploaded.json()["media_type"] == "csv"
    assert missing_mission.status_code == 422
    assert missing_mission.json()["code"] == "invalid_upload"
    assert processed.status_code == 200
    assert processed.json()["time"] == [1.0, 2.0, 3.0]
    assert processed.json()["metadata"]["source"]["hdu_name"] == "CSV"


@pytest.mark.anyio
async def test_upload_rejects_unsupported_and_malformed_files(tmp_path: Path) -> None:
    """Extensions and content are independently validated before storage."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path)),
        base_url="http://test",
    ) as client:
        unsupported = await client.post(
            "/upload",
            files={"file": ("curve.txt", _csv_bytes(), "text/plain")},
        )
        bad_fits = await client.post(
            "/upload",
            files={"file": ("curve.fits", b"not fits", "application/fits")},
        )
        bad_csv = await client.post(
            "/upload",
            files={"file": ("curve.csv", b"WRONG,DATA\na,b\n", "text/csv")},
        )
        empty = await client.post(
            "/upload",
            files={"file": ("empty.csv", b"", "text/csv")},
        )

    assert unsupported.status_code == 415
    assert unsupported.json()["code"] == "unsupported_upload_type"
    assert bad_fits.status_code == 422
    assert bad_fits.json()["code"] == "invalid_upload"
    assert bad_csv.status_code == 422
    assert bad_csv.json()["code"] == "invalid_upload"
    assert empty.status_code == 422
    assert empty.json()["code"] == "invalid_upload"
    assert list((tmp_path / "cache" / "uploads").iterdir()) == []


@pytest.mark.anyio
async def test_upload_enforces_streamed_size_limit_and_removes_partial(
    tmp_path: Path,
) -> None:
    """Oversized request content is stopped and its partial file removed."""
    app = _app(
        tmp_path,
        max_upload_size_bytes=16,
        upload_chunk_size_bytes=4,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/upload",
            files={"file": ("large.csv", _csv_bytes(), "text/csv")},
        )

    assert response.status_code == 413
    assert response.json()["code"] == "upload_too_large"
    assert list((tmp_path / "cache" / "uploads").iterdir()) == []


@pytest.mark.anyio
async def test_secure_filename_cannot_control_storage_path(tmp_path: Path) -> None:
    """Traversal segments are discarded and only the validated suffix survives."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path)),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/upload",
            files={"file": ("../../escape.fits", _fits_bytes(), "application/fits")},
        )

    assert response.status_code == 201
    file_id = response.json()["file_id"]
    assert (tmp_path / "cache" / "uploads" / f"{file_id}.fits").is_file()
    assert not (tmp_path / "escape.fits").exists()


@pytest.mark.anyio
async def test_unknown_or_malformed_identifier_returns_404(tmp_path: Path) -> None:
    """Opaque lookup never interprets identifier text as a path."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(tmp_path)),
        base_url="http://test",
    ) as client:
        malformed = await client.post(
            "/process",
            json={"file_id": "../../etc/passwd", "mission": "TESS"},
        )
        missing = await client.post(
            "/process",
            json={"file_id": "0" * 32, "mission": "TESS"},
        )

    assert malformed.status_code == 404
    assert missing.status_code == 404
    assert malformed.json()["code"] == "upload_not_found"


class ChunkedUpload:
    """UploadFile-compatible test stream that records requested read sizes."""

    filename = "stream.csv"

    def __init__(self, content: bytes) -> None:
        """Initialize the in-memory test stream."""
        self._content = content
        self._offset = 0
        self.read_sizes: list[int] = []
        self.closed = False

    async def read(self, size: int) -> bytes:
        """Return at most the requested chunk size."""
        self.read_sizes.append(size)
        chunk = self._content[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    async def close(self) -> None:
        """Record closure by the upload store."""
        self.closed = True


@pytest.mark.anyio
async def test_store_reads_in_bounded_chunks_and_closes_stream(tmp_path: Path) -> None:
    """Storage never requests the complete upload in one read."""
    upload = ChunkedUpload(_csv_bytes())
    store = UploadStore(
        tmp_path / "uploads",
        max_size_bytes=1_000,
        chunk_size_bytes=8,
        retention_seconds=60,
    )

    stored = await store.save(upload)  # type: ignore[arg-type]

    assert stored.size_bytes == len(_csv_bytes())
    assert len(upload.read_sizes) > 2
    assert set(upload.read_sizes) == {8}
    assert upload.closed is True


def test_store_cleans_expired_uploads(tmp_path: Path) -> None:
    """TTL cleanup removes completed and abandoned temporary cache files."""
    root = tmp_path / "uploads"
    root.mkdir()
    completed = root / f"{'a' * 32}.fits"
    partial = root / ".upload-abandoned.part"
    completed.write_bytes(_fits_bytes())
    partial.write_bytes(b"partial")
    expired = time.time() - 120
    os.utime(completed, (expired, expired))
    os.utime(partial, (expired, expired))
    store = UploadStore(
        root,
        max_size_bytes=1_000_000,
        chunk_size_bytes=1024,
        retention_seconds=60,
    )

    assert store.cleanup_expired() == 2
    assert list(root.iterdir()) == []
