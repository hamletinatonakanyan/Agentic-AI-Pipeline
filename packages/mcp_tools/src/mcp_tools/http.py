"""The MCP server over HTTP, behind the same access key as every other route.

Mounted in a FastAPI app at `/mcp`. Two things to know:
- The host app's lifespan must run the SDK's session manager, or every request fails with
  "task group is not initialized". `McpHttpEndpoint.lifespan()` is what to enter.
- The endpoint answers only to the Host names in MCP_HTTP_ALLOWED_HOSTS (protection against
  DNS rebinding), and only with a valid X-API-Key.
"""

import hmac
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from common import ErrorBody, ErrorEnvelope
from mcp_tools.settings import McpSettings


def _envelope(code: str, message: str, status: int) -> JSONResponse:
    body = ErrorEnvelope(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status, content=body.model_dump())


class ApiKeyGuard:
    """ASGI middleware: no valid X-API-Key, no access. It is raw ASGI because FastAPI
    dependencies do not apply to a mounted sub-application."""

    def __init__(self, app: ASGIApp, key_provider: Callable[[], str]) -> None:
        self._app = app
        self._key = key_provider

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        configured = self._key()
        if not configured:
            response = _envelope(
                "not_configured", "APP_API_KEY is not set. Add it to .env and restart.", 503
            )
            await response(scope, receive, send)
            return
        provided = dict(scope["headers"]).get(b"x-api-key")
        if provided is None or not hmac.compare_digest(provided, configured.encode()):
            await _envelope("unauthorized", "Missing or invalid X-API-Key.", 401)(
                scope, receive, send
            )
            return
        await self._app(scope, receive, send)


class _AtRoot:
    """Serves the inner app at its root, for a request to exactly `/mcp` (no trailing slash)."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        inner = dict(scope)
        inner["path"] = "/"
        inner["root_path"] = scope.get("root_path", "") + scope["path"]
        await self._app(inner, receive, send)


class McpHttpEndpoint:
    def __init__(
        self, server: MCPServer, settings: McpSettings, key_provider: Callable[[], str]
    ) -> None:
        self._server = server
        mcp_app = server.streamable_http_app(
            streamable_http_path="/",  # the host app mounts it at /mcp
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=settings.http_allowed_hosts,
                allowed_origins=[],
            ),
        )
        self.app: ASGIApp = ApiKeyGuard(mcp_app, key_provider)

    def routes(self, path: str = "/mcp") -> list[BaseRoute]:
        """Both `/mcp` and `/mcp/...`. A plain Mount serves only the second, but `/mcp` is the URL
        people give to MCP clients, and a catch-all mount later in the app would swallow it."""
        return [Route(path, endpoint=_AtRoot(self.app)), Mount(path, app=self.app)]

    def install(self, app: Any, path: str = "/mcp") -> None:
        """Add the endpoint to a FastAPI app. Call it BEFORE any catch-all mount such as `/`."""
        app.router.routes.extend(self.routes(path))

    def lifespan(self) -> AbstractAsyncContextManager[Any]:
        """Enter this in the host app's lifespan."""
        return self._server.session_manager.run()
