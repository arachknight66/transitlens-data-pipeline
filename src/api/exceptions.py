"""REST-layer exceptions and domain error mappings."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from api.models import ErrorResponse
from exporters.exceptions import ExportError
from features.exceptions import FeatureError
from fits.exceptions import FitsError, FitsReadError
from mast.exceptions import (
    MastAuthenticationError,
    MastDownloadError,
    MastProductNotFoundError,
    MastSearchError,
)
from preprocessing.exceptions import PreprocessingError


class CachedPathError(ValueError):
    """Raised when processing is requested outside the configured cache."""


def register_exception_handlers(application: FastAPI) -> None:
    """Register stable HTTP mappings for pipeline domain exceptions.

    Args:
        application: FastAPI application receiving exception handlers.
    """
    application.add_exception_handler(
        MastAuthenticationError,
        _handler(status_code=401, code="mast_authentication_failed"),
    )
    application.add_exception_handler(
        MastProductNotFoundError,
        _handler(status_code=404, code="mast_product_not_found"),
    )
    application.add_exception_handler(
        MastSearchError,
        _handler(status_code=502, code="mast_search_failed"),
    )
    application.add_exception_handler(
        MastDownloadError,
        _handler(status_code=502, code="mast_download_failed"),
    )
    application.add_exception_handler(
        CachedPathError,
        _handler(status_code=403, code="cached_path_required"),
    )
    application.add_exception_handler(
        FitsReadError,
        _handler(status_code=422, code="fits_read_failed"),
    )
    application.add_exception_handler(
        FitsError,
        _handler(status_code=422, code="fits_processing_failed"),
    )
    application.add_exception_handler(
        PreprocessingError,
        _handler(status_code=422, code="preprocessing_failed"),
    )
    application.add_exception_handler(
        FeatureError,
        _handler(status_code=422, code="feature_generation_failed"),
    )
    application.add_exception_handler(
        ExportError,
        _handler(status_code=500, code="export_failed"),
    )


def _handler(status_code: int, code: str):
    """Build an asynchronous FastAPI domain exception handler."""

    async def handle(request: Request, error: Exception) -> JSONResponse:
        logger.bind(
            path=request.url.path,
            status_code=status_code,
            error_code=code,
            error_type=type(error).__name__,
        ).warning("API request failed")
        payload = ErrorResponse(code=code, detail=str(error))
        return JSONResponse(status_code=status_code, content=payload.model_dump())

    return handle
