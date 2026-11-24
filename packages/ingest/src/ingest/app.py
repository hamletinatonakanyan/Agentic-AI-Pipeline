"""Standalone ingest app (port 8001).

    uv run uvicorn ingest.app:create_app --factory --port 8001      (make run-ingest)

Needs Qdrant for real ingests (`make up`). Previews and the UI work without it.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from common import install_error_handlers, mount_ui_kit, setup_logging
from ingest import api
from ingest.factory import Ingest, build_ingest
from ingest.settings import IngestSettings
from observability import init_observability, shutdown_observability
from observability.settings import ObsSettings

UI_DIR = Path(__file__).parent / "ui"


def create_app(
    settings: IngestSettings | None = None,
    ingest: Ingest | None = None,
    obs_settings: ObsSettings | None = None,
) -> FastAPI:
    """Tests pass a prebuilt `ingest` (fakes) and in-memory `obs_settings`."""
    setup_logging()
    init_observability(obs_settings or ObsSettings(), force=True)
    ingest = ingest or build_ingest(settings or IngestSettings())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        ingest.close()
        shutdown_observability()

    app = FastAPI(title="Ingest", lifespan=lifespan)
    install_error_handlers(app)
    app.state.ingest = ingest

    app.include_router(api.health_router)  # public
    # Standard standalone paths, plus the same routes under /api/ingest so the UI works unchanged
    # here and inside e2e.
    app.include_router(api.status_router)
    app.include_router(api.jobs_router)
    app.include_router(api.status_router, prefix="/api/ingest")
    app.include_router(api.jobs_router, prefix="/api/ingest")
    app.include_router(api.router, prefix="/api/ingest")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    mount_ui_kit(app)
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app
