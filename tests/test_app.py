"""Smoke tests for the FastAPI application factory."""

from api.app import create_app
from config import Settings


def test_create_app_uses_supplied_settings() -> None:
    """The factory attaches supplied immutable settings to the app."""
    settings = Settings(log_level="WARNING")

    application = create_app(settings)

    assert application.state.settings is settings
    assert application.title == "TransitLens Data Pipeline"


def test_phase_one_does_not_expose_future_routes() -> None:
    """Business endpoints remain absent until their documented phase."""
    application = create_app(Settings(log_level="WARNING"))
    route_paths = {route.path for route in application.routes}

    assert "/status" not in route_paths
    assert "/docs" not in route_paths
