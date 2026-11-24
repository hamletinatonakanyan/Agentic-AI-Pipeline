"""Standalone evals app (port 8003).

    uv run uvicorn evals.app:create_app --factory --port 8003      (make run-evals)

Standalone, it can evaluate HTTP targets and the `retrieval_http` preset (a running retrieval
service). Inside e2e there is also an in-process `retrieval` preset.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from common import install_error_handlers, mount_ui_kit, setup_logging
from evals import api
from evals.factory import Evals, build_evals
from evals.settings import EvalSettings
from observability import init_observability, shutdown_observability
from observability.settings import ObsSettings

UI_DIR = Path(__file__).parent / "ui"


def create_app(
    settings: EvalSettings | None = None,
    evals: Evals | None = None,
    obs_settings: ObsSettings | None = None,
) -> FastAPI:
    """Tests pass a prebuilt `evals` (fakes) and in-memory `obs_settings`."""
    setup_logging()
    init_observability(obs_settings or ObsSettings(), force=True)
    evals = evals or build_evals(settings or EvalSettings())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        evals.close()
        shutdown_observability()

    app = FastAPI(title="Evals", lifespan=lifespan)
    install_error_handlers(app)
    app.state.evals = evals

    app.include_router(api.health_router)  # public
    app.include_router(api.status_router)  # standard standalone paths
    app.include_router(api.jobs_router)
    app.include_router(api.status_router, prefix="/api/evals")  # where the UI looks
    app.include_router(api.jobs_router, prefix="/api/evals")
    app.include_router(api.router, prefix="/api/evals")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    mount_ui_kit(app)
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app
