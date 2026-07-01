"""Smoke tests for the FastAPI application factory."""

from api.app import create_app
from api.routes import router
from config import Settings


def test_create_app_uses_supplied_settings() -> None:
    """The factory attaches supplied immutable settings to the app."""
    settings = Settings(log_level="WARNING")

    application = create_app(settings)

    assert application.state.settings is settings
    assert application.title == "TransitLens Data Pipeline"


def test_application_exposes_only_authorized_business_routes() -> None:
    """The application exposes documented routes plus authorized upload."""
    create_app(Settings(log_level="WARNING"))
    route_paths = {route.path for route in router.routes}

    assert route_paths == {"/search", "/download", "/upload", "/process", "/status"}
    assert "/docs" not in route_paths
