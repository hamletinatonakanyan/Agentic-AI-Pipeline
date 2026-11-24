"""The client side's contract: registrations, tools, results."""

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,31}$"
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_HEADER_NAME = re.compile(r"^[A-Za-z0-9-]{1,64}$")

Transport = Literal["stdio", "streamable-http"]
ErrorKind = Literal[
    "not_found",  # no such server or tool
    "disabled",  # the server is switched off
    "not_allowed",  # the tool is not on the server's allow-list
    "invalid_arguments",  # the arguments do not match the tool's JSON Schema
    "timeout",
    "connection",  # could not start or reach the server
    "protocol",  # the server answered in a way we cannot use
    "tool_error",  # the tool ran and reported an error
]


class ServerSpec(BaseModel):
    """A saved MCP server. Secrets are referenced by NAME, never stored."""

    name: str = Field(pattern=NAME_PATTERN)
    transport: Transport
    # stdio: a command to launch. It must match MCP_ALLOWED_COMMANDS exactly.
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env_refs: list[str] = Field(default_factory=list)  # environment variables to pass on, by name
    # streamable-http
    url: str | None = None
    headers_refs: dict[str, str] = Field(default_factory=dict)  # header name -> env variable name
    timeout_s: float = Field(default=30.0, gt=0, le=300)
    tool_allowlist: list[str] = Field(default_factory=list)  # empty = no tool may be called
    enabled: bool = True

    @field_validator("env_refs")
    @classmethod
    def _env_names(cls, v: list[str]) -> list[str]:
        for name in v:
            if not _ENV_NAME.match(name):
                raise ValueError(f"'{name}' is not a valid environment variable name")
        return v

    @field_validator("headers_refs")
    @classmethod
    def _header_names(cls, v: dict[str, str]) -> dict[str, str]:
        for header, ref in v.items():
            if not _HEADER_NAME.match(header):
                raise ValueError(f"'{header}' is not a valid header name")
            if not _ENV_NAME.match(ref):
                raise ValueError(f"'{ref}' is not a valid environment variable name")
        return v

    @model_validator(mode="after")
    def _fields_match_transport(self) -> "ServerSpec":
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("a stdio server needs a command")
            if self.url or self.headers_refs:
                raise ValueError("a stdio server takes no url or headers_refs")
        else:
            if not self.url:
                raise ValueError("a streamable-http server needs a url")
            if self.command or self.args or self.env_refs:
                raise ValueError("an HTTP server takes no command, args or env_refs")
        return self

    @property
    def command_line(self) -> str:
        return " ".join([self.command or "", *self.args]).strip()


class ToolDescriptor(BaseModel):
    server: str
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    read_only: bool | None = None


class ToolResult(BaseModel):
    server: str
    tool: str
    ok: bool
    is_error: bool
    content: str  # the text for the model (or the error message)
    structured: dict[str, Any] | None = None
    duration_ms: float = 0.0
    truncated: bool = False
    error_kind: ErrorKind | None = None


class ToolsListing(BaseModel):
    tools: list[ToolDescriptor]
    errors: dict[str, str] = Field(default_factory=dict)  # server -> why it could not be listed


class ServerStatus(BaseModel):
    spec: ServerSpec
    state: Literal["unknown", "ok", "error", "disabled"] = "unknown"
    tool_names: list[str] = Field(default_factory=list)
    error: str = ""
    checked_at: float | None = None


class CallRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class AllowlistUpdate(BaseModel):
    tool_allowlist: list[str]


class EnabledUpdate(BaseModel):
    enabled: bool


class ExposedTool(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool | None = None


class ExposedInfo(BaseModel):
    """What our own server offers, for the UI."""

    server_name: str
    tools: list[ExposedTool]
    resources: list[str]
    resource_templates: list[str]
    prompts: list[str]
    http_path: str
    stdio_command: str


class McpStatus(BaseModel):
    db_path: str
    servers: int
    allowed_commands: list[str]
    allowed_hosts: list[str]
    allowed_env_refs: list[str]
    allowed_header_refs: list[str]
