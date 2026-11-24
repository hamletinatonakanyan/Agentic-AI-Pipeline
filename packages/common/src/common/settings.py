"""Settings base class.

Every package subclasses `BaseAppSettings` and sets its own `env_prefix`, for example:

    class IngestSettings(BaseAppSettings):
        model_config = SettingsConfigDict(env_prefix="INGEST_")
        scanned_text_threshold: int = 50     # reads INGEST_SCANNED_TEXT_THRESHOLD

Values come from the environment, then from `.env`. Real secrets live only in `.env`.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseAppSettings(BaseSettings):
    """Common behaviour: read `.env`, ignore variables that belong to other packages."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `FOO=` in .env (an empty value, as in .env.example) means "use the default",
        # not "an empty string", which would break typed fields such as ints.
        env_ignore_empty=True,
    )


class AppSettings(BaseAppSettings):
    """Settings shared by every app. No prefix: these use the standard names."""

    # Read from APP_API_KEY. Empty means "not configured" and protected routes answer 503.
    app_api_key: str = Field(default="", repr=False)


@lru_cache
def get_app_settings() -> AppSettings:
    """Cached so the `.env` file is read once. Tests override it with `dependency_overrides`."""
    return AppSettings()


def mask_secret(value: str | None) -> str:
    """Safe text to show in a UI or log. Never returns the full secret."""
    if not value:
        return "not set"
    if len(value) < 12:
        return "set (hidden)"
    return f"set (...{value[-4:]})"
