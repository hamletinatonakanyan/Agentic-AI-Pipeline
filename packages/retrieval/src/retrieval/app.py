"""Standalone retrieval app (port 8002).

uv run uvicorn retrieval.app:create_app --factory --port 8002      (make run-retrieval)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from common import install_error_handlers, mount_ui_kit, setup_logging
from observability import init_observability, shutdown_observability
from observability.settings import ObsSettings
from retrieval import api
from retrieval.factory import Retrieval, build_retrieval
from retrieval.settings import RetrievalSettings

UI_DIR = Path(__file__).parent / "ui"


def create_app(
    settings: RetrievalSettings | None = None,
    retrieval: Retrieval | None = None,
    obs_settings: ObsSettings | None = None,
) -> FastAPI:
    """Tests pass a prebuilt `retrieval` (fakes) and in-memory `obs_settings`."""
    setup_logging()
    init_observability(obs_settings or ObsSettings(), force=True)
    retrieval = retrieval or build_retrieval(settings or RetrievalSettings())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        retrieval.close()
        shutdown_observability()

    app = FastAPI(title="Retrieval", lifespan=lifespan)
    install_error_handlers(app)
    app.state.retrieval = retrieval

    app.include_router(api.health_router)  # public
    app.include_router(api.status_router)  # standard standalone path
    app.include_router(api.status_router, prefix="/api/retrieval")  # where the UI looks
    app.include_router(api.router, prefix="/api/retrieval")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    mount_ui_kit(app)
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app
