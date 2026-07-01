"""TransitLens REST routes."""

from importlib.metadata import version
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from api.exceptions import CachedPathError
from api.models import (
    DownloadRequest,
    DownloadResponse,
    ErrorResponse,
    ProcessRequest,
    ProcessResponse,
    StatusResponse,
    UploadResponse,
)
from api.services import ApiServices
from api.uploads import InvalidUploadError, StoredUpload, UploadStore
from config import Settings
from features.metadata import generate_feature_record
from fits.csv_reader import read_csv_light_curve
from fits.models import LightCurve
from fits.reader import read_fits
from mast.cache import FitsCache
from mast.download import download_fits
from mast.models import Mission, Observation, ObservationSearch
from mast.search import search_observations
from preprocessing.models import PreprocessingConfig
from preprocessing.pipeline import preprocess_light_curve

router = APIRouter()


def get_settings(request: Request) -> Settings:
    """Return settings attached by the application factory."""
    return request.app.state.settings


def get_services(request: Request) -> ApiServices:
    """Return service providers attached by the application factory."""
    return request.app.state.services


@router.get(
    "/status",
    response_model=StatusResponse,
    responses={500: {"model": ErrorResponse}},
)
def status() -> StatusResponse:
    """Return local service readiness without making a network request."""
    return StatusResponse(
        version=version("transitlens-data-pipeline"),
        supported_missions=tuple(Mission),
    )


@router.get(
    "/search",
    response_model=list[Observation],
    responses={401: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
)
def search(
    target: Annotated[str, Query(min_length=1, pattern=r".*\S.*")],
    services: Annotated[ApiServices, Depends(get_services)],
    missions: Annotated[list[Mission] | None, Query()] = None,
    radius_deg: Annotated[float, Query(gt=0.0, le=5.0)] = 0.001,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[Observation]:
    """Search supported MAST collections for target light curves."""
    criteria = ObservationSearch(
        target=target,
        missions=tuple(missions) if missions is not None else tuple(Mission),
        radius_deg=radius_deg,
        limit=limit,
    )
    return search_observations(criteria, services.mast_client_provider())


@router.post(
    "/download",
    response_model=DownloadResponse,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
    },
)
def download(
    payload: DownloadRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    services: Annotated[ApiServices, Depends(get_services)],
) -> DownloadResponse:
    """Download and cache the preferred FITS product for an observation."""
    result = download_fits(
        payload.mast_id,
        services.mast_client_provider(),
        FitsCache(settings.cache_dir),
    )
    return DownloadResponse.model_validate(result, from_attributes=True)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=201,
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def upload(
    file: Annotated[UploadFile, File(...)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UploadResponse:
    """Stream, validate, and temporarily cache a light-curve upload."""
    stored = await _upload_store(settings).save(file)
    return UploadResponse(
        file_id=stored.file_id,
        media_type=stored.media_type,
        size_bytes=stored.size_bytes,
    )


@router.post(
    "/process",
    response_model=ProcessResponse,
    responses={403: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def process(
    payload: ProcessRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProcessResponse:
    """Run parsing, preprocessing, and features for cached input."""
    raw, file_id = _read_process_source(payload, settings)
    preprocessing_config = payload.preprocessing or _default_preprocessing(settings)
    processed = preprocess_light_curve(raw, preprocessing_config)
    features = generate_feature_record(processed)
    return ProcessResponse(
        file_id=file_id,
        time=processed.time.tolist(),
        flux=processed.flux.tolist(),
        normalized_flux=processed.normalized_flux.tolist(),
        median_filtered_flux=processed.median_filtered_flux.tolist(),
        wavelet_flux=processed.wavelet_flux.tolist(),
        quality=None if processed.quality is None else processed.quality.tolist(),
        metadata=processed.metadata,
        features=features,
    )


def _read_process_source(
    payload: ProcessRequest,
    settings: Settings,
) -> tuple[LightCurve, str | None]:
    """Resolve and read an opaque upload or legacy cached FITS path."""
    if payload.file_id is not None:
        stored = _upload_store(settings).resolve(payload.file_id)
        raw = _read_upload(stored, payload.mission)
        opaque_metadata = raw.metadata.model_copy(
            update={"source_path": Path(stored.file_id)}
        )
        return raw.model_copy(update={"metadata": opaque_metadata}), stored.file_id
    if payload.fits_path is None:
        raise ValueError("process request source was not validated")
    fits_path = _validated_cached_path(payload.fits_path, settings.cache_dir)
    return read_fits(fits_path, payload.mission), None


def _read_upload(stored: StoredUpload, mission: Mission | None) -> LightCurve:
    """Read a validated upload through the canonical domain contract."""
    if stored.media_type == "csv":
        if mission is None:
            raise InvalidUploadError("mission is required to process a CSV upload")
        return read_csv_light_curve(stored.path, mission)
    return read_fits(stored.path, mission)


def _upload_store(settings: Settings) -> UploadStore:
    """Create a request-scoped upload store from application settings."""
    root = settings.upload_cache_dir or settings.cache_dir / "uploads"
    return UploadStore(
        root,
        max_size_bytes=settings.max_upload_size_bytes,
        chunk_size_bytes=settings.upload_chunk_size_bytes,
        retention_seconds=settings.upload_retention_seconds,
    )


def _validated_cached_path(path: Path, cache_dir: Path) -> Path:
    """Resolve a requested FITS path and require cache containment."""
    resolved_path = path.expanduser().resolve()
    resolved_cache = cache_dir.expanduser().resolve()
    if not resolved_path.is_relative_to(resolved_cache):
        raise CachedPathError("FITS path must be inside the configured cache directory")
    return resolved_path


def _default_preprocessing(settings: Settings) -> PreprocessingConfig:
    """Translate application settings into validated preprocessing settings."""
    return PreprocessingConfig(
        quality_bitmask=settings.quality_bitmask,
        median_window=settings.median_filter_window,
        wavelet=settings.wavelet,
        wavelet_threshold_mode=settings.wavelet_mode,
        wavelet_threshold_scale=settings.wavelet_threshold_scale,
        wavelet_max_level=settings.wavelet_max_level,
    )
