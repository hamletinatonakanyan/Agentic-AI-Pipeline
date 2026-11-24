"""Routes. Three routers, so each app decides where each one is mounted:

- `health_router`: public liveness.
- `status_router`: protected `/status`.
- `router`:        protected, mounted at `/api/mcp`.

The MCP endpoint itself is separate (`/mcp`, see http.py) and checks the same key.
"""

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from common import AppError, require_api_key
from mcp_tools.factory import Mcp
from mcp_tools.models import (
    AllowlistUpdate,
    CallRequest,
    EnabledUpdate,
    ExposedInfo,
    McpStatus,
    ServerSpec,
    ServerStatus,
    ToolResult,
    ToolsListing,
)

health_router = APIRouter()
status_router = APIRouter(dependencies=[Depends(require_api_key)])
router = APIRouter(dependencies=[Depends(require_api_key)])

# A call that is refused because of what the caller asked is an HTTP error. A tool that ran and
# failed is a normal result (`is_error`), because for an agent that is information, not a fault.
_REFUSED = {"not_found": 404, "not_allowed": 403, "disabled": 409, "invalid_arguments": 422}


def get_mcp(request: Request) -> Mcp:
    return request.app.state.mcp


@health_router.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@status_router.get("/status", response_model=McpStatus)
def status(mcp: Mcp = Depends(get_mcp)) -> McpStatus:
    s = mcp.settings
    return McpStatus(
        db_path=s.db_path,
        servers=len(mcp.registry.list()),
        allowed_commands=s.allowed_commands,
        allowed_hosts=s.allowed_hosts,
        allowed_env_refs=s.allowed_env_refs,
        allowed_header_refs=s.allowed_header_refs,
    )


@router.get("/servers", response_model=list[ServerStatus])
def list_servers(mcp: Mcp = Depends(get_mcp)) -> list[ServerStatus]:
    return mcp.manager.list_servers()


@router.post("/servers", response_model=ServerStatus, status_code=201)
def register_server(spec: ServerSpec, mcp: Mcp = Depends(get_mcp)) -> ServerStatus:
    if mcp.registry.get(spec.name):
        raise AppError("server_exists", f"A server named '{spec.name}' already exists.", 409)
    mcp.manager.register(spec)
    return ServerStatus(spec=spec)


@router.delete("/servers/{name}")
def unregister_server(name: str, mcp: Mcp = Depends(get_mcp)) -> dict[str, str]:
    mcp.manager.unregister(name)
    return {"deleted": name}


@router.patch("/servers/{name}/allowlist", response_model=ServerSpec)
def set_allowlist(name: str, body: AllowlistUpdate, mcp: Mcp = Depends(get_mcp)) -> ServerSpec:
    return mcp.manager.update(name, tool_allowlist=sorted(set(body.tool_allowlist)))


@router.patch("/servers/{name}/enabled", response_model=ServerSpec)
def set_enabled(name: str, body: EnabledUpdate, mcp: Mcp = Depends(get_mcp)) -> ServerSpec:
    return mcp.manager.update(name, enabled=body.enabled)


@router.post("/servers/{name}/check", response_model=ServerStatus)
async def check_server(name: str, mcp: Mcp = Depends(get_mcp)) -> ServerStatus:
    return await mcp.manager.check(name)


@router.get("/servers/{name}/tools", response_model=ToolsListing)
async def server_tools(
    name: str, refresh: bool = False, mcp: Mcp = Depends(get_mcp)
) -> ToolsListing:
    return await mcp.manager.list_tools(server=name, refresh=refresh)


@router.post("/servers/{name}/tools/{tool}/call", response_model=ToolResult)
async def call_tool(
    name: str, tool: str, body: CallRequest, mcp: Mcp = Depends(get_mcp)
) -> ToolResult:
    result = await mcp.manager.call_tool(name, tool, body.arguments)
    if result.error_kind in _REFUSED:
        raise AppError(f"tool_{result.error_kind}", result.content, _REFUSED[result.error_kind])
    return result


@router.get("/llm-tools")
async def llm_tools(refresh: bool = False, mcp: Mcp = Depends(get_mcp)) -> JSONResponse:
    """The tools a model would be offered, in the OpenAI function format."""
    tools = await mcp.manager.llm_tools(refresh=refresh)
    payload: dict[str, Any] = {
        "tools": tools.tools,
        # Not for the model: where each tool lives and whether it only reads, so a caller can
        # call it and decide whether to ask a human first.
        "meta": {
            name: {"server": server, "tool": tool, "read_only": tools.read_only.get(name)}
            for name, (server, tool) in tools.index.items()
        },
    }
    return JSONResponse(payload)


@router.get("/exposed", response_model=ExposedInfo)
async def exposed(mcp: Mcp = Depends(get_mcp)) -> ExposedInfo:
    return await mcp.exposed()
