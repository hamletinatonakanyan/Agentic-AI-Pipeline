"""Helpers for client tests: small in-process MCP servers that count how often they are reached."""

import asyncio
import sys

import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from mcp_tools.manager import MCPClientManager
from mcp_tools.models import ServerSpec
from mcp_tools.registry import ServerRegistry
from mcp_tools.settings import OWN_SERVER_COMMAND, McpSettings


class Probe:
    """A fake MCP server and a log of what reached it."""

    def __init__(self, name: str = "probe") -> None:
        self.server = MCPServer(name)
        self.calls: list[tuple[str, dict]] = []
        self.listings = 0
        probe = self

        @self.server.tool()
        def echo(text: str, times: int = 1) -> str:
            """Repeat the text."""
            probe.calls.append(("echo", {"text": text, "times": times}))
            return text * times

        @self.server.tool()
        def fail() -> str:
            """Always fails, with a message the model may see."""
            probe.calls.append(("fail", {}))
            raise ToolError("the database is down")

        @self.server.tool()
        async def slow(seconds: float = 5.0) -> str:
            """Sleeps."""
            probe.calls.append(("slow", {"seconds": seconds}))
            await asyncio.sleep(seconds)
            return "done"

        @self.server.tool()
        def big(n: int = 50_000) -> str:
            """A very long answer."""
            probe.calls.append(("big", {"n": n}))
            return "x" * n

        @self.server.tool()
        def secret() -> str:
            """Not on the allow-list in most tests."""
            probe.calls.append(("secret", {}))
            return "classified"

    def client(self) -> Client:
        self.listings += 1
        return Client(self.server)


def spec(name: str = "probe", allow=("echo", "fail", "slow", "big"), **kw) -> ServerSpec:
    """A registration that passes the default policy. The in-process factory ignores the command
    and picks the server by name."""
    command, *args = OWN_SERVER_COMMAND.split()
    fields = dict(
        name=name, transport="stdio", command=command, args=args, tool_allowlist=list(allow)
    )
    return ServerSpec(**{**fields, **kw})


class Rig:
    def __init__(self, **settings) -> None:
        self.settings = McpSettings(_env_file=None, **settings)
        self.registry = ServerRegistry(":memory:")
        self.probes: dict[str, Probe] = {"probe": Probe("probe")}
        self.now = 0.0
        self.manager = MCPClientManager(
            self.settings,
            self.registry,
            client_factory=lambda s: self.probes[s.name].client(),
            clock=lambda: self.now,
        )

    @property
    def probe(self) -> Probe:
        return self.probes["probe"]


@pytest.fixture
def rig() -> Rig:
    return Rig()


@pytest.fixture
def rig_factory():
    return Rig


@pytest.fixture
def make_spec():
    return spec


@pytest.fixture
def python_exe():
    return sys.executable


# ----- a real HTTP server in a thread, on a free port -------------------------------------------------

import threading  # noqa: E402
import time  # noqa: E402

import uvicorn  # noqa: E402


class LiveServer:
    def __init__(self, app) -> None:
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", lifespan="on")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> "LiveServer":
        self._thread.start()
        for _ in range(100):
            if self._server.started:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("the test server did not start")
        self.port = self._server.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        return self

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)


@pytest.fixture
def live():
    """`with live(app) as server:` serves the app over real HTTP for the duration of the block."""
    return LiveServer
