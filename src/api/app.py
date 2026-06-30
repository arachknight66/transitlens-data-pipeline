"""FastAPI application factory."""

from fastapi import FastAPI

from api.exceptions import register_exception_handlers
from api.routes import router
from api.services import ApiServices
from config import Settings, load_settings
from logging_config import configure_logging


def create_app(
    settings: Settings | None = None,
    services: ApiServices | None = None,
) -> FastAPI:
    """Create an independently configured FastAPI application.

    Args:
        settings: Optional validated settings. Defaults to environment settings.
        services: Optional injected external service providers.

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
    application.state.services = services or ApiServices.from_settings(
        resolved_settings
    )
    register_exception_handlers(application)
    application.include_router(router)
    return application
