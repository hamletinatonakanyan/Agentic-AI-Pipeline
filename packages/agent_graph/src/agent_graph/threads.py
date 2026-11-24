"""The list of conversations. LangGraph saves each conversation's state, but it has no list of
threads, so this keeps one: who owns a thread, its title, and when it was last used."""

import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from agent_graph.models import ThreadInfo

USER_ID = re.compile(r"^[A-Za-z0-9_.@-]{1,100}$")


class ThreadStore:
    def __init__(self, path: str, clock=time.time) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._clock = clock
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS threads (thread_id TEXT PRIMARY KEY, user_id TEXT, "
                "title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL, "
                "messages INTEGER NOT NULL DEFAULT 0)"
            )
            # A database made before domains existed has no `collection` column: add it.
            columns = {r[1] for r in self._conn.execute("PRAGMA table_info(threads)")}
            if "collection" not in columns:
                self._conn.execute("ALTER TABLE threads ADD COLUMN collection TEXT")
            # What each answer showed (sources, tool calls, timings, trace id), by run.
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS runs (thread_id TEXT NOT NULL, run_id TEXT NOT NULL, "
                "created_at REAL NOT NULL, record TEXT NOT NULL, PRIMARY KEY (thread_id, run_id))"
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    _COLUMNS = "thread_id, user_id, title, created_at, updated_at, messages, collection"

    @staticmethod
    def _info(row) -> ThreadInfo:
        return ThreadInfo(
            thread_id=row[0], user_id=row[1], title=row[2],
            created_at=row[3], updated_at=row[4], messages=row[5], collection=row[6],
        )  # fmt: skip

    def create(self, user_id: str | None, title: str | None) -> ThreadInfo:
        thread_id, now = uuid.uuid4().hex[:16], self._clock()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO threads (thread_id, user_id, title, created_at, updated_at, messages) "
                "VALUES (?,?,?,?,?,0)",
                (thread_id, user_id, title or "", now, now),
            )
        return ThreadInfo(
            thread_id=thread_id, user_id=user_id, title=title or "", created_at=now, updated_at=now
        )

    def get(self, thread_id: str) -> ThreadInfo | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {self._COLUMNS} FROM threads WHERE thread_id=?", (thread_id,)
            ).fetchone()
        return self._info(row) if row else None

    def list(self, user_id: str | None = None, limit: int = 100) -> list[ThreadInfo]:
        sql, args = f"SELECT {self._COLUMNS} FROM threads", []
        if user_id:
            sql, args = sql + " WHERE user_id=?", [user_id]
        with self._lock:
            rows = self._conn.execute(
                sql + " ORDER BY updated_at DESC LIMIT ?", [*args, limit]
            ).fetchall()
        return [self._info(r) for r in rows]

    def touch(
        self,
        thread_id: str,
        messages: int,
        title_hint: str | None = None,
        collection: str | None = None,
    ) -> None:
        """Record activity. The first message becomes the title if there is none yet."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE threads SET updated_at=?, messages=?, "
                "title=CASE WHEN title='' AND ?<>'' THEN ? ELSE title END, "
                "collection=COALESCE(?, collection) WHERE thread_id=?",
                (
                    self._clock(),
                    messages,
                    title_hint or "",
                    (title_hint or "")[:60],
                    collection,
                    thread_id,
                ),
            )

    # ----- what each answer showed -------------------------------------------------------

    def save_run(self, thread_id: str, run_id: str, record_json: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (thread_id, run_id, created_at, record) "
                "VALUES (?,?,?,?)",
                (thread_id, run_id, self._clock(), record_json),
            )

    def runs(self, thread_id: str) -> dict[str, tuple[float, str]]:
        """run_id -> (created_at, record json) for one conversation."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, created_at, record FROM runs WHERE thread_id=?", (thread_id,)
            ).fetchall()
        return {r[0]: (r[1], r[2]) for r in rows}

    def delete(self, thread_id: str) -> bool:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM runs WHERE thread_id=?", (thread_id,))
            return (
                self._conn.execute("DELETE FROM threads WHERE thread_id=?", (thread_id,)).rowcount
                > 0
            )
