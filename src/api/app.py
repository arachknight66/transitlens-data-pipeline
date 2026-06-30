"""FastAPI application factory."""

from fastapi import FastAPI

from config import Settings, load_settings
from logging_config import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an independently configured FastAPI application.

    Args:
        settings: Optional validated settings. Defaults to environment settings.

    Returns:
        Configured FastAPI application without business routes.
    """
    resolved_settings = settings or load_settings()
    configure_logging(resolved_settings.log_level)

    application = FastAPI(
        title="TransitLens Data Pipeline",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.state.settings = resolved_settings
    return application
