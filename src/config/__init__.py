"""Typed application configuration."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from configuration and environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="TRANSITLENS_",
        extra="forbid",
        frozen=True,
    )

    cache_dir: Path = Path("data/cache")
    log_level: str = "INFO"
    mast_api_token: str | None = None
    mast_session_token: str | None = None
    median_filter_window: int = Field(default=5, ge=3)
    wavelet: str = "db4"
    wavelet_mode: str = "soft"

    @field_validator("median_filter_window")
    @classmethod
    def validate_odd_window(cls, value: int) -> int:
        """Ensure the median-filter window has a centered sample."""
        if value % 2 == 0:
            message = "median_filter_window must be odd"
            raise ValueError(message)
        return value

    @field_validator("mast_api_token", "mast_session_token", mode="before")
    @classmethod
    def normalize_optional_secret(cls, value: object) -> object:
        """Treat an empty optional credential as absent."""
        return None if value == "" else value


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from an optional TOML file and environment variables.

    Environment variables prefixed with ``TRANSITLENS_`` take precedence over
    values from the configuration file.

    Args:
        config_path: Optional path to a TOML configuration file.

    Returns:
        Validated, immutable runtime settings.

    Raises:
        FileNotFoundError: If the requested configuration file does not exist.
        tomllib.TOMLDecodeError: If the configuration file is invalid TOML.
    """
    file_values: dict[str, Any] = {}
    if config_path is not None:
        with config_path.open("rb") as config_file:
            file_values = tomllib.load(config_file)

    environment_values = _environment_overrides(Settings.model_fields)
    return Settings(**(file_values | environment_values))


def _environment_overrides(fields: Mapping[str, object]) -> dict[str, str]:
    """Return explicitly configured TransitLens environment values."""
    return {
        field: value
        for field in fields
        if (value := os.getenv(f"TRANSITLENS_{field.upper()}")) is not None
    }
