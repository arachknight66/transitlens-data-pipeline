"""Tests for application configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from config import Settings, load_settings


def test_load_settings_from_toml(tmp_path: Path) -> None:
    """Configuration files populate typed runtime settings."""
    config_path = tmp_path / "settings.toml"
    config_path.write_text(
        'cache_dir = "custom-cache"\nmedian_filter_window = 7\n',
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.cache_dir == Path("custom-cache")
    assert settings.median_filter_window == 7


def test_environment_overrides_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment variables take precedence over configuration files."""
    config_path = tmp_path / "settings.toml"
    config_path.write_text('log_level = "INFO"\n', encoding="utf-8")
    monkeypatch.setenv("TRANSITLENS_LOG_LEVEL", "DEBUG")

    assert load_settings(config_path).log_level == "DEBUG"


def test_empty_credentials_are_absent() -> None:
    """Empty credentials select anonymous MAST access."""
    settings = Settings(mast_api_token="")

    assert settings.mast_api_token is None


def test_median_window_must_be_odd() -> None:
    """Even median-filter windows are rejected."""
    with pytest.raises(ValidationError, match="must be odd"):
        Settings(median_filter_window=4)


def test_scientific_modes_are_restricted() -> None:
    """Unsupported filtering modes are rejected during configuration loading."""
    with pytest.raises(ValidationError):
        Settings(wavelet_mode="hard")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Settings(quality_bitmask="unknown")  # type: ignore[arg-type]


def test_missing_requested_configuration_raises(tmp_path: Path) -> None:
    """A missing explicit configuration path is not silently ignored."""
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.toml")
