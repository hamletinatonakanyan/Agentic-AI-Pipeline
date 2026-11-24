"""MCPClientManager: discover and call tools on any registered MCP server, safely.

A connection is opened for each operation and closed afterwards. That is simple and cannot leak
processes or sessions. The cost is a process start for each stdio call, which a small cache of
tool lists softens.

Rules every `call_tool` follows, in order, before anything is sent to a server:
  1. the server exists and is enabled
  2. the tool is on the server's allow-list
  3. the arguments match the tool's JSON Schema
Then the call has a timeout, long output is cut and flagged, and every failure comes back as a
`ToolResult(is_error=True)` with a reason, never as an exception, so an agent can recover.
"""

import asyncio
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

from jsonschema import Draft202012Validator
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from common import AppError, Environment
from mcp_tools.llm_tools import LlmToolSet, build_llm_tools
from mcp_tools.models import (
    ErrorKind,
    ServerSpec,
    ServerStatus,
    ToolDescriptor,
    ToolResult,
    ToolsListing,
)
from mcp_tools.policy import check_spec
from mcp_tools.registry import ServerRegistry
from mcp_tools.settings import McpSettings
from observability import span

ClientFactory = Callable[[ServerSpec], Client]


class _Failure(Exception):
    def __init__(self, kind: ErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


def _flatten(exc: BaseException) -> list[BaseException]:
    """Connections run in task groups, so errors often arrive wrapped in an ExceptionGroup."""
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _flatten(sub)]
    return [exc]


def _describe(exc: BaseException) -> str:
    leaves = _flatten(exc)
    names = sorted({type(e).__name__ for e in leaves})
    return ", ".join(names)


class MCPClientManager:
    def __init__(
        self,
        settings: McpSettings,
        registry: ServerRegistry,
        client_factory: ClientFactory | None = None,
        env: Mapping[str, str] | Environment | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._s = settings
        self._registry = registry
        self._factory = client_factory
        self._env = Environment() if env is None else env
        self._clock = clock
        self._cache: dict[str, tuple[float, list[ToolDescriptor]]] = {}
        self._status: dict[str, ServerStatus] = {}

    # ----- registrations -----------------------------------------------------------------

    def register(self, spec: ServerSpec) -> ServerSpec:
        check_spec(spec, self._s)
        self._registry.put(spec)
        self._forget(spec.name)
        return spec

    def unregister(self, name: str) -> None:
        if not self._registry.delete(name):
            raise AppError("server_not_found", f"No server '{name}'.", 404)
        self._forget(name)

    def get_spec(self, name: str) -> ServerSpec:
        spec = self._registry.get(name)
        if spec is None:
            raise AppError("server_not_found", f"No server '{name}'.", 404)
        return spec

    def update(self, name: str, **changes: Any) -> ServerSpec:
        spec = self.get_spec(name).model_copy(update=changes)
        return self.register(ServerSpec.model_validate(spec.model_dump()))

    def list_servers(self) -> list[ServerStatus]:
        out = []
        for spec in self._registry.list():
            known = self._status.get(spec.name)
            status = known.model_copy(update={"spec": spec}) if known else ServerStatus(spec=spec)
            if not spec.enabled:
                status.state = "disabled"
            out.append(status)
        return out

    def _forget(self, name: str) -> None:
        self._cache.pop(name, None)
        self._status.pop(name, None)

    # ----- connecting --------------------------------------------------------------------

    def _command(self, command: str) -> str:
        return sys.executable if command in ("python", "python3") else command

    def _client(self, spec: ServerSpec) -> Client:
        check_spec(spec, self._s)  # policy is checked again at use, not only at registration
        if self._factory is not None:
            return self._factory(spec)
        if spec.transport == "stdio":
            env = {n: v for n in spec.env_refs if (v := self._env.get(n))}
            params = StdioServerParameters(
                command=self._command(spec.command or ""), args=spec.args, env=env
            )
            return Client(params, read_timeout_seconds=spec.timeout_s)

        import httpx2
        from mcp.client.streamable_http import streamable_http_client

        headers = {}
        for header, ref in spec.headers_refs.items():
            value = self._env.get(ref)
            if not value:
                raise _Failure("connection", f"the variable {ref} is not set")
            headers[header] = value
        http = httpx2.AsyncClient(headers=headers, timeout=spec.timeout_s)
        return Client(streamable_http_client(spec.url or "", http_client=http))

    async def _tools_from(self, client: Client, server: str) -> list[ToolDescriptor]:
        listing = await client.list_tools()
        return [
            ToolDescriptor(
                server=server,
                name=t.name,
                description=t.description or "",
                input_schema=t.input_schema or {"type": "object", "properties": {}},
                output_schema=t.output_schema,
                read_only=t.annotations.read_only_hint if t.annotations else None,
            )
            for t in listing.tools
        ]

    # ----- discovering tools -------------------------------------------------------------

    async def list_tools(self, server: str | None = None, refresh: bool = False) -> ToolsListing:
        """The tools of one server, or of every enabled server. A server that cannot be reached is
        reported in `errors` and does not stop the others."""
        specs = (
            [self.get_spec(server)] if server else [s for s in self._registry.list() if s.enabled]
        )
        tools: list[ToolDescriptor] = []
        errors: dict[str, str] = {}
        for spec in specs:
            try:
                tools += await self._tools_for(spec, refresh)
            except _Failure as failure:
                errors[spec.name] = failure.message
        return ToolsListing(tools=tools, errors=errors)

    async def _tools_for(self, spec: ServerSpec, refresh: bool) -> list[ToolDescriptor]:
        cached = self._cache.get(spec.name)
        if cached and not refresh and self._clock() - cached[0] < self._s.tools_cache_ttl_s:
            return cached[1]
        try:
            async with asyncio.timeout(spec.timeout_s):
                async with self._client(spec) as client:
                    tools = await self._tools_from(client, spec.name)
        except _Failure as failure:
            self._record(spec, "error", failure.message)
            raise
        except AppError as exc:
            self._record(spec, "error", exc.message)
            raise _Failure("connection", exc.message) from None
        except TimeoutError:
            message = f"no answer within {spec.timeout_s:.0f}s"
            self._record(spec, "error", message)
            raise _Failure("timeout", message) from None
        except Exception as exc:
            message = f"could not connect ({_describe(exc)})"
            self._record(spec, "error", message)
            raise _Failure("connection", message) from None
        self._cache[spec.name] = (self._clock(), tools)
        self._record(spec, "ok", "", [t.name for t in tools])
        return tools

    def _record(self, spec: ServerSpec, state: str, error: str, names: list[str] | None = None):
        self._status[spec.name] = ServerStatus(
            spec=spec,
            state=state,  # type: ignore[arg-type]
            tool_names=names or [],
            error=error,
            checked_at=time.time(),
        )

    async def check(self, name: str) -> ServerStatus:
        """Connect now and report whether the server answers, and what it offers."""
        spec = self.get_spec(name)
        if not spec.enabled:
            return ServerStatus(spec=spec, state="disabled")
        try:
            await self._tools_for(spec, refresh=True)
        except _Failure:
            pass  # recorded in the status
        return self._status[name].model_copy(update={"spec": spec})

    # ----- calling a tool ----------------------------------------------------------------

    async def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        with span("mcp.call_tool", {"mcp.server": server, "mcp.tool": tool}) as sp:
            try:
                result = await self._call(server, tool, arguments, started)
            except _Failure as failure:
                result = self._error(server, tool, failure.kind, failure.message, started)
            sp.set_attributes(
                {
                    "mcp.ok": result.ok,
                    "mcp.error_kind": result.error_kind or "",
                    "mcp.truncated": result.truncated,
                    "mcp.argument_names": ",".join(sorted(arguments)),  # names, never values
                }
            )
            if not result.ok:
                sp.set_status("error", f"{result.error_kind}: {result.content}"[:200])
        return result

    def _error(
        self, server: str, tool: str, kind: ErrorKind, message: str, started: float
    ) -> ToolResult:
        return ToolResult(
            server=server, tool=tool, ok=False, is_error=True, content=message,
            duration_ms=(time.perf_counter() - started) * 1000, error_kind=kind,
        )  # fmt: skip

    async def _call(self, server: str, tool: str, arguments: dict, started: float) -> ToolResult:
        spec = self._registry.get(server)
        if spec is None:
            raise _Failure("not_found", f"There is no server '{server}'.")
        if not spec.enabled:
            raise _Failure("disabled", f"The server '{server}' is switched off.")
        if tool not in spec.tool_allowlist and "*" not in spec.tool_allowlist:
            raise _Failure("not_allowed", f"The tool '{tool}' is not allowed on server '{server}'.")

        cached = self._cache.get(server)
        known = (
            {t.name: t for t in cached[1]}
            if cached and self._clock() - cached[0] < self._s.tools_cache_ttl_s
            else None
        )
        if known is not None:  # check the arguments before connecting at all
            self._check_arguments(known, server, tool, arguments)

        failure: _Failure | None = None
        raw = None
        try:
            async with asyncio.timeout(spec.timeout_s):
                async with self._client(spec) as client:
                    if known is None:
                        tools = await self._tools_from(client, server)
                        self._cache[server] = (self._clock(), tools)
                        try:
                            self._check_arguments(
                                {t.name: t for t in tools}, server, tool, arguments
                            )
                        except _Failure as bad:
                            # Do not raise inside the connection: the SDK would wrap it in an
                            # ExceptionGroup. Leave it first, then report it.
                            failure = bad
                    if failure is None:
                        raw = await client.call_tool(tool, arguments)
        except _Failure:
            raise
        except AppError as exc:
            raise _Failure("connection", exc.message) from None
        except TimeoutError:
            raise _Failure(
                "timeout", f"The tool did not finish within {spec.timeout_s:.0f}s."
            ) from None
        except Exception as exc:
            for leaf in _flatten(exc):  # a _Failure may still arrive wrapped in a group
                if isinstance(leaf, _Failure):
                    raise leaf from None
            raise _Failure("connection", f"Could not use the server ({_describe(exc)}).") from None
        if failure is not None:
            raise failure
        return self._result(server, tool, raw, started)

    @staticmethod
    def _check_arguments(
        tools: dict[str, ToolDescriptor], server: str, tool: str, arguments: dict
    ) -> None:
        descriptor = tools.get(tool)
        if descriptor is None:
            raise _Failure("not_found", f"The server '{server}' has no tool '{tool}'.")
        errors = sorted(
            Draft202012Validator(descriptor.input_schema).iter_errors(arguments),
            key=lambda e: list(e.path),
        )
        if errors:
            first = "; ".join(
                (f"'{'.'.join(map(str, e.path))}': " if e.path else "") + e.message
                for e in errors[:3]
            )
            raise _Failure("invalid_arguments", f"The arguments do not match the tool: {first}")

    def _result(self, server: str, tool: str, raw: Any, started: float) -> ToolResult:
        limit = self._s.max_output_chars
        parts = []
        for block in raw.content:
            kind = getattr(block, "type", "")
            if kind == "text":
                parts.append(block.text)
            elif kind == "image":
                parts.append(f"[image omitted: {getattr(block, 'mime_type', 'image')}]")
            elif kind in ("resource", "resource_link"):
                uri = getattr(getattr(block, "resource", block), "uri", "")
                parts.append(f"[resource: {uri}]")
            else:
                parts.append("[unsupported content omitted]")
        text = "\n".join(parts)
        truncated = len(text) > limit
        if truncated:
            text = text[:limit] + f"\n[output cut at {limit} characters]"
        structured = raw.structured_content
        if structured is not None and len(repr(structured)) > limit:
            structured, truncated = None, True
        is_error = bool(raw.is_error)
        return ToolResult(
            server=server,
            tool=tool,
            ok=not is_error,
            is_error=is_error,
            content=text,
            structured=structured,
            duration_ms=(time.perf_counter() - started) * 1000,
            truncated=truncated,
            error_kind="tool_error" if is_error else None,
        )

    # ----- for the model -----------------------------------------------------------------

    async def llm_tools(self, refresh: bool = False) -> LlmToolSet:
        """Allowed tools of enabled servers, in the OpenAI function format."""
        listing = await self.list_tools(refresh=refresh)
        allowed = {s.name: s for s in self._registry.list() if s.enabled}
        tools = [
            t
            for t in listing.tools
            if t.server in allowed
            and (
                t.name in allowed[t.server].tool_allowlist
                or "*" in allowed[t.server].tool_allowlist
            )
        ]
        return build_llm_tools(tools, self._s.max_description_chars)
