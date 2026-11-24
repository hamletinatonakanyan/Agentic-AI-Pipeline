from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict

from common import BaseAppSettings

# Applied to string attribute values before export.
DEFAULT_REDACT_PATTERNS = [
    r"sk-[A-Za-z0-9_\-]{16,}",  # OpenAI-style keys
    r"llx-[A-Za-z0-9_\-]{16,}",  # LlamaCloud-style keys
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",  # email addresses
]

KNOWN_EXPORTERS = {"local", "console"}


class ObsSettings(BaseAppSettings):
    """Reads OBS_* variables."""

    model_config = SettingsConfigDict(env_prefix="OBS_", env_file=".env", extra="ignore")

    service_name: str = "agentic-blocks"
    enabled: bool = True
    # Comma-separated. `local` = SQLite trace store, `console` = JSON log line per span.
    exporters: str = "local"
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    # In .env this is a JSON list, for example OBS_REDACT_PATTERNS='["secret-\\d+"]'.
    redact_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_REDACT_PATTERNS))
    local_db_path: str = "data/observability.sqlite"

    @field_validator("exporters")
    @classmethod
    def _check_exporters(cls, value: str) -> str:
        names = [n.strip() for n in value.split(",") if n.strip()]
        unknown = set(names) - KNOWN_EXPORTERS
        if unknown:
            raise ValueError(
                f"unknown exporters {sorted(unknown)}; known: {sorted(KNOWN_EXPORTERS)}"
            )
        return ",".join(names)

    @property
    def exporter_names(self) -> list[str]:
        return [n for n in self.exporters.split(",") if n]
