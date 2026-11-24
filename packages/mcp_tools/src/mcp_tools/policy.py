"""What a registration is allowed to do.

Registering a stdio server means "run this command on this machine". Registering an HTTP server
means "send requests, and these headers, to this URL". Without limits, anyone holding the API
key could run any command or send your secrets anywhere. So every registration is checked
against allow-lists from settings before it is saved, and again before it is used.
"""

from urllib.parse import urlparse

from common import AppError
from mcp_tools.models import ServerSpec
from mcp_tools.settings import McpSettings


def _refuse(code: str, message: str) -> AppError:
    return AppError(code, message, 403)


def check_spec(spec: ServerSpec, settings: McpSettings) -> None:
    if spec.transport == "stdio":
        if spec.command_line not in settings.allowed_commands:
            raise _refuse(
                "command_not_allowed",
                f"'{spec.command_line}' is not in MCP_ALLOWED_COMMANDS. A registration may only "
                "run a command that is listed there, exactly.",
            )
        for ref in spec.env_refs:
            if ref not in settings.allowed_env_refs:
                raise _refuse("env_ref_not_allowed", f"'{ref}' is not in MCP_ALLOWED_ENV_REFS.")
        return

    parsed = urlparse(spec.url or "")
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise AppError("invalid_server", "The URL must be http or https.", 422)
    if parsed.hostname not in settings.allowed_hosts:
        raise _refuse(
            "host_not_allowed",
            f"Host '{parsed.hostname}' is not in MCP_ALLOWED_HOSTS "
            f"({', '.join(settings.allowed_hosts) or 'empty'}).",
        )
    for ref in spec.headers_refs.values():
        if ref not in settings.allowed_header_refs:
            raise _refuse(
                "header_ref_not_allowed",
                f"'{ref}' is not in MCP_ALLOWED_HEADER_REFS, so it cannot be sent as a header.",
            )
