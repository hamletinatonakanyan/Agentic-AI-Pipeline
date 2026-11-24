"""The standalone adapters: reach retrieval and the MCP tools over HTTP.

Inside e2e these are replaced by in-process adapters (no network, no key). Both versions
satisfy the same two Protocols, so the graph cannot tell them apart.
"""

import time
from collections.abc import Callable
from typing import Any

import httpx

from agent_graph.models import Passage, RetrievalChoice
from agent_graph.protocols import AgentTool, RetrievedContext, ToolOutcome
from common import AppError


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return f"HTTP {response.status_code}"


class HttpRetriever:
    def __init__(
        self,
        base_url: str,
        key_provider: Callable[[], str],
        timeout_s: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/api/retrieval/retrieve"
        self._key = key_provider
        self._http = client or httpx.Client(timeout=timeout_s)

    def retrieve(self, query: str, choice: RetrievalChoice, top_k: int) -> RetrievedContext:
        body: dict[str, Any] = {
            "query": query, "mode": choice.mode, "rewrite": choice.rewrite,
            "rerank": choice.rerank, "top_k": top_k,
        }  # fmt: skip
        if choice.collection:
            body["collection"] = choice.collection
        try:
            response = self._http.post(self._url, json=body, headers={"X-API-Key": self._key()})
        except httpx.HTTPError as exc:
            raise AppError(
                "retrieval_unreachable", f"Cannot reach retrieval ({type(exc).__name__}).", 502
            ) from exc
        if response.status_code >= 400:
            raise AppError("retrieval_failed", _error_message(response)[:300], 502)
        data = response.json()
        return RetrievedContext(
            passages=[
                Passage(
                    chunk_id=h["chunk_id"],
                    text=h["text"],
                    source_name=str(h.get("payload", {}).get("source_name", "")),
                    page=h.get("payload", {}).get("page"),
                    score=float(h.get("rerank_score") or h.get("score") or 0.0),
                )
                for h in data["hits"]
            ],
            rewritten_queries=data.get("rewritten_queries", []),
            retrieval_ms=float(data.get("timings", {}).get("total_ms", 0.0)),
        )


class HttpToolProvider:
    """Lists tools from `/api/mcp/llm-tools` and calls them through the MCP management API."""

    def __init__(
        self,
        base_url: str,
        key_provider: Callable[[], str],
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/") + "/api/mcp"
        self._key = key_provider
        self._http = client or httpx.AsyncClient(timeout=timeout_s)
        self._meta: dict[str, dict[str, Any]] = {}

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._key()}

    async def tools(self) -> list[AgentTool]:
        response = await self._http.get(f"{self._base}/llm-tools", headers=self._headers())
        response.raise_for_status()
        data = response.json()
        self._meta = data.get("meta", {})
        return [
            AgentTool(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("parameters", {}),
                read_only=self._meta.get(t["name"], {}).get("read_only"),
            )
            for t in data.get("tools", [])
        ]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        started = time.perf_counter()
        meta = self._meta.get(name)
        if meta is None:
            return ToolOutcome(ok=False, content=f"Unknown tool '{name}'.", error_kind="not_found")
        url = f"{self._base}/servers/{meta['server']}/tools/{meta['tool']}/call"
        try:
            response = await self._http.post(
                url, json={"arguments": arguments}, headers=self._headers()
            )
        except httpx.HTTPError as exc:
            return ToolOutcome(
                ok=False,
                content=f"Cannot reach the tool service ({type(exc).__name__}).",
                error_kind="unreachable",
            )
        ms = (time.perf_counter() - started) * 1000
        if response.status_code >= 400:
            return ToolOutcome(
                ok=False,
                content=_error_message(response)[:500],
                error_kind="refused",
                duration_ms=ms,
            )
        r = response.json()
        return ToolOutcome(
            ok=bool(r["ok"]),
            content=r["content"],
            error_kind=r.get("error_kind"),
            truncated=bool(r.get("truncated")),
            duration_ms=float(r.get("duration_ms", ms)),
        )

    async def aclose(self) -> None:
        await self._http.aclose()
