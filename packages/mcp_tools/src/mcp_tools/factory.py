"""Builds the real thing from settings. Nothing connects or starts a process at build time."""

from collections.abc import Callable
from dataclasses import dataclass

from mcp import Client
from mcp.server.mcpserver import MCPServer

from common import get_app_settings
from mcp_tools.corpus import CorpusProvider, HttpCorpusProvider
from mcp_tools.http import McpHttpEndpoint
from mcp_tools.manager import ClientFactory, MCPClientManager
from mcp_tools.models import ExposedInfo, ExposedTool
from mcp_tools.registry import ServerRegistry
from mcp_tools.server import build_server
from mcp_tools.settings import OWN_SERVER_COMMAND, McpSettings


@dataclass
class Mcp:
    settings: McpSettings
    registry: ServerRegistry
    manager: MCPClientManager
    server: MCPServer  # our own server
    endpoint: McpHttpEndpoint  # our own server over HTTP

    def close(self) -> None:
        self.registry.close()

    async def exposed(self) -> ExposedInfo:
        """What our own server offers, read the way any client would read it."""
        async with Client(self.server) as client:
            tools = (await client.list_tools()).tools
            resources = (await client.list_resources()).resources
            templates = (await client.list_resource_templates()).resource_templates
            prompts = (await client.list_prompts()).prompts
        return ExposedInfo(
            server_name=self.settings.server_name,
            tools=[
                ExposedTool(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.input_schema or {},
                    read_only=t.annotations.read_only_hint if t.annotations else None,
                )
                for t in tools
            ],
            resources=[str(r.uri) for r in resources],
            resource_templates=[t.uri_template for t in templates],
            prompts=[p.name for p in prompts],
            http_path="/mcp",
            stdio_command=OWN_SERVER_COMMAND,
        )


def build_mcp(
    settings: McpSettings,
    *,
    corpus: CorpusProvider | None = None,
    key_provider: Callable[[], str] | None = None,
    client_factory: ClientFactory | None = None,
    env: dict[str, str] | None = None,
) -> Mcp:
    """The real wiring. e2e passes an in-process `corpus`. Tests pass fakes."""
    registry = ServerRegistry(settings.db_path)
    registry.ensure_own_server(settings)
    manager = MCPClientManager(settings, registry, client_factory=client_factory, env=env)
    server = build_server(settings, corpus or HttpCorpusProvider(settings))
    endpoint = McpHttpEndpoint(
        server, settings, key_provider or (lambda: get_app_settings().app_api_key)
    )
    return Mcp(
        settings=settings, registry=registry, manager=manager, server=server, endpoint=endpoint
    )
