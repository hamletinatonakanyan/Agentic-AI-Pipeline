"""Standalone observability app (port 8005).

    uv run uvicorn observability.app:create_app --factory --port 8005

A factory (not a module-level `app`) so importing this module has no side effects,
such as creating the SQLite file.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from common import install_error_handlers, mount_ui_kit, setup_logging
from observability.api import health_router, router, status_router
from observability.protocols import SpanExporter
from observability.service import ObservabilityService
from observability.settings import ObsSettings
from observability.tracer import init_observability, shutdown_observability

UI_DIR = Path(__file__).parent / "ui"


def create_app(
    settings: ObsSettings | None = None,
    exporters: list[SpanExporter] | None = None,
) -> FastAPI:
    setup_logging()
    handle = init_observability(settings or ObsSettings(), exporters=exporters, force=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        shutdown_observability()

    app = FastAPI(title="Observability", lifespan=lifespan)
    install_error_handlers(app)
    app.state.obs_service = ObservabilityService(handle)

    app.include_router(health_router)  # public
    app.include_router(status_router)  # /status, needs the key
    app.include_router(status_router, prefix="/api/obs")  # the same, where the UI looks for it
    app.include_router(router, prefix="/api/obs")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    mount_ui_kit(app)  # before the broader /ui mount
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app
