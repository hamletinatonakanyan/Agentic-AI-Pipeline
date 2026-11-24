"""Standalone agent app (port 8006).

    uv run uvicorn agent_graph.app:create_app --factory --port 8006      (make run-agent)

It reaches retrieval (8002) and the MCP tools (8004) over HTTP, so start those too.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from agent_graph import api
from agent_graph.factory import build_standalone
from agent_graph.runtime import Agent
from agent_graph.settings import AgentSettings
from common import install_error_handlers, mount_ui_kit, setup_logging
from observability import init_observability, shutdown_observability
from observability.settings import ObsSettings

UI_DIR = Path(__file__).parent / "ui"


def create_app(
    settings: AgentSettings | None = None,
    agent: Agent | None = None,
    obs_settings: ObsSettings | None = None,
) -> FastAPI:
    """Tests pass a prebuilt `agent` (fakes) and in-memory `obs_settings`."""
    setup_logging()
    init_observability(obs_settings or ObsSettings(), force=True)
    agent = agent or build_standalone(settings or AgentSettings())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with agent.lifespan():  # opens the checkpointer, closes it at shutdown
            yield
        shutdown_observability()

    app = FastAPI(title="Agent", lifespan=lifespan)
    install_error_handlers(app)
    app.state.agent = agent

    app.include_router(api.health_router)
    app.include_router(api.status_router)
    app.include_router(api.status_router, prefix="/api/agent")
    app.include_router(api.router, prefix="/api/agent")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    mount_ui_kit(app)
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app
