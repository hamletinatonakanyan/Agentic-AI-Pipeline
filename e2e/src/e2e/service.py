"""Business logic for the status report. Pure Python, no HTTP."""

from dataclasses import dataclass

import httpx

from common import AppSettings, mask_secret
from e2e.models import ComponentStatus, StatusReport
from e2e.protocols import DependencyProbe
from e2e.settings import E2ESettings

# The components shown on the status page, and the setting that turns each one off.
# A flag of None means the component has no on/off switch (it is a shared library).
COMPONENTS: list[tuple[str, str | None]] = [
    ("ingest", "enable_ingest"),
    ("observability", "enable_obs"),
    ("llm", None),
    ("retrieval", "enable_retrieval"),
    ("evals", "enable_evals"),
    ("mcp", "enable_mcp"),
    ("agent", "enable_agent"),
]


class QdrantProbe:
    """Real probe: asks Qdrant's liveness endpoint. Any failure means unreachable."""

    def __init__(self, url: str, timeout_s: float = 1.0) -> None:
        self._url = url.rstrip("/")
        self._timeout = timeout_s

    def is_reachable(self) -> bool:
        try:
            return httpx.get(f"{self._url}/healthz", timeout=self._timeout).status_code == 200
        except httpx.HTTPError:
            return False


@dataclass(frozen=True)
class BuiltComponent:
    """A package that is mounted in this app. `missing` names settings it still needs."""

    detail: str = ""
    ui_path: str | None = None
    missing: tuple[str, ...] = ()


class StatusService:
    def __init__(
        self,
        app_settings: AppSettings,
        settings: E2ESettings,
        qdrant: DependencyProbe,
        built: dict[str, BuiltComponent] | None = None,
    ) -> None:
        self._app = app_settings
        self._settings = settings
        self._qdrant = qdrant
        # Components whose package is built and mounted, by name.
        self._built = built or {}

    def report(self) -> StatusReport:
        items = [self._access_key(), self._qdrant_status()]
        items += [self._component(*c) for c in COMPONENTS]
        return StatusReport(components=items)

    def _access_key(self) -> ComponentStatus:
        key = self._app.app_api_key
        return ComponentStatus(
            name="access_key",
            enabled=True,
            configured=bool(key),
            missing=[] if key else ["APP_API_KEY"],
            detail=mask_secret(key),
        )

    def _qdrant_status(self) -> ComponentStatus:
        reachable = self._qdrant.is_reachable()
        return ComponentStatus(
            name="qdrant",
            enabled=True,
            configured=True,
            reachable=reachable,
            detail=self._settings.qdrant_url
            + ("" if reachable else " is not answering. Run `make up` to start it."),
        )

    def _component(self, name: str, flag: str | None) -> ComponentStatus:
        enabled = True if flag is None else bool(getattr(self._settings, flag))
        if not enabled:
            return ComponentStatus(
                name=name,
                enabled=False,
                configured=name in self._built,
                detail=f"turned off by E2E_{(flag or '').upper()}",
            )
        built = self._built.get(name)
        if built is None:
            return ComponentStatus(
                name=name,
                enabled=True,
                configured=False,
                missing=["not mounted in this app"],
            )
        return ComponentStatus(
            name=name,
            enabled=True,
            configured=not built.missing,
            missing=list(built.missing),
            detail=built.detail,
            ui_path=built.ui_path,
        )
