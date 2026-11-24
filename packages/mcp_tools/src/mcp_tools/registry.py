"""Saved server registrations, in SQLite. Secrets are never stored, only the names that
reference them."""

import sqlite3
import threading
import time
from pathlib import Path

from mcp_tools.models import ServerSpec
from mcp_tools.settings import OWN_SERVER_COMMAND, McpSettings

OWN_SERVER_NAME = "own"


OWN_TOOLS = ["calculate", "search_corpus", "list_tables", "query_table"]
# Default allow-lists of earlier versions. A stored list that is still exactly one of these was
# never edited, so it is upgraded to include new tools. Any other list is the user's choice.
_OLD_DEFAULT_ALLOWLISTS = [["calculate", "search_corpus"]]


def own_server_spec() -> ServerSpec:
    command, *args = OWN_SERVER_COMMAND.split()
    return ServerSpec(
        name=OWN_SERVER_NAME,
        transport="stdio",
        command=command,
        args=args,
        env_refs=["APP_API_KEY", "MCP_RETRIEVAL_URL", "MCP_INGEST_URL"],
        tool_allowlist=list(OWN_TOOLS),
    )


class ServerRegistry:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS servers ("
                "name TEXT PRIMARY KEY, spec TEXT NOT NULL, updated_at REAL NOT NULL)"
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def list(self) -> list[ServerSpec]:
        with self._lock:
            rows = self._conn.execute("SELECT spec FROM servers ORDER BY name").fetchall()
        return [ServerSpec.model_validate_json(r[0]) for r in rows]

    def get(self, name: str) -> ServerSpec | None:
        with self._lock:
            row = self._conn.execute("SELECT spec FROM servers WHERE name=?", (name,)).fetchone()
        return ServerSpec.model_validate_json(row[0]) if row else None

    def put(self, spec: ServerSpec) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO servers VALUES (?,?,?)",
                (spec.name, spec.model_dump_json(), time.time()),
            )

    def delete(self, name: str) -> bool:
        with self._lock, self._conn:
            return self._conn.execute("DELETE FROM servers WHERE name=?", (name,)).rowcount > 0

    def ensure_own_server(self, settings: McpSettings) -> None:
        """Our own server is registered once, on first start. Later edits (for example to its
        allow-list) are kept, never overwritten. The one exception: an allow-list that is still an
        earlier version's untouched default gets the new tools added."""
        existing = self.get(OWN_SERVER_NAME)
        if existing is None:
            if own_server_spec().command_line in settings.allowed_commands:
                self.put(own_server_spec())
        elif existing.tool_allowlist in _OLD_DEFAULT_ALLOWLISTS:
            self.put(existing.model_copy(update={"tool_allowlist": list(OWN_TOOLS)}))
