"""Standalone MCP tools app (port 8004).

    uv run uvicorn mcp_tools.app:create_app --factory --port 8004      (make run-mcp)

Serves the management API and UI, and our own MCP server over HTTP at /mcp (needs the key).
Over stdio, run `python -m mcp_tools.server` instead (that is what a client launches).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from common import install_error_handlers, mount_ui_kit, setup_logging
from mcp_tools import api
from mcp_tools.factory import Mcp, build_mcp
from mcp_tools.settings import McpSettings
from observability import init_observability, shutdown_observability
from observability.settings import ObsSettings

UI_DIR = Path(__file__).parent / "ui"


def create_app(
    settings: McpSettings | None = None,
    mcp: Mcp | None = None,
    obs_settings: ObsSettings | None = None,
) -> FastAPI:
    """Tests pass a prebuilt `mcp` (fakes) and in-memory `obs_settings`."""
    setup_logging()
    init_observability(obs_settings or ObsSettings(), force=True)
    mcp = mcp or build_mcp(settings or McpSettings())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with mcp.endpoint.lifespan():  # without this, /mcp fails on every request
            yield
        mcp.close()
        shutdown_observability()

    app = FastAPI(title="MCP tools", lifespan=lifespan)
    install_error_handlers(app)
    app.state.mcp = mcp

    app.include_router(api.health_router)  # public
    app.include_router(api.status_router)  # standard standalone path
    app.include_router(api.status_router, prefix="/api/mcp")  # where the UI looks
    app.include_router(api.router, prefix="/api/mcp")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    mcp.endpoint.install(app)  # the MCP protocol itself at /mcp, behind the key
    mount_ui_kit(app)
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app
