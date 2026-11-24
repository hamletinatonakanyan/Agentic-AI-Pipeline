"""Where `search_corpus` and `corpus://sources` get their data.

A small interface with two implementations:
- `HttpCorpusProvider` calls the retrieval and ingest HTTP APIs. The stdio server runs as a
  separate process, so HTTP is its only option, and it keeps this package from importing either.
- an in-process one, which e2e injects into the mounted server so the app does not call itself.
"""

import os
from typing import Protocol

import httpx
from pydantic import BaseModel

from mcp_tools.settings import McpSettings


class Passage(BaseModel):
    rank: int
    chunk_id: str
    source_name: str
    page: int | None = None
    score: float
    text: str


class CorpusSearchOutput(BaseModel):
    query: str
    collection: str
    results: list[Passage]


class SourceItem(BaseModel):
    source_id: str
    source_name: str
    chunk_count: int
    pages: int


class CorpusChunk(BaseModel):
    chunk_id: str
    source_name: str
    page: int | None = None
    chunk_index: int = 0
    text: str


class CorpusChunks(BaseModel):
    chunks: list[CorpusChunk]
    truncated: bool = False  # stopped at the scan limit: there are more chunks than were read


class CorpusError(Exception):
    """A problem with a clear, safe message for the model and the user."""


class CorpusProvider(Protocol):
    async def search(
        self, query: str, top_k: int, collection: str | None
    ) -> CorpusSearchOutput: ...

    async def sources(self, collection: str | None) -> list[SourceItem]: ...

    async def chunks(self, collection: str | None, source_name: str | None) -> CorpusChunks:
        """Every stored chunk of a collection (or of one source), up to the scan limit. Used by
        the table tools, which need all the rows and not just the best few."""
        ...


def trim(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class HttpCorpusProvider:
    def __init__(
        self,
        settings: McpSettings,
        client: httpx.AsyncClient | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._s = settings
        self._http = client or httpx.AsyncClient(timeout=settings.call_timeout_s)
        self._env = os.environ if env is None else env

    def _headers(self) -> dict[str, str]:
        key = self._env.get("APP_API_KEY", "")
        if not key:
            raise CorpusError("APP_API_KEY is not set for this tool server.")
        return {"X-API-Key": key}

    async def _request(self, method: str, url: str, what: str, **kwargs) -> dict | list:
        try:
            response = await self._http.request(method, url, headers=self._headers(), **kwargs)
        except httpx.TimeoutException:
            raise CorpusError(f"the {what} service did not answer in time") from None
        except httpx.HTTPError:
            raise CorpusError(f"cannot reach the {what} service") from None
        if response.status_code >= 400:
            try:
                detail = response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                detail = f"HTTP {response.status_code}"
            raise CorpusError(f"the {what} service said: {detail}"[:300])
        try:
            return response.json()
        except ValueError:
            raise CorpusError(f"the {what} service sent an unreadable reply") from None

    async def search(self, query: str, top_k: int, collection: str | None) -> CorpusSearchOutput:
        name = collection or self._s.default_collection
        body = await self._request(
            "POST",
            f"{self._s.retrieval_url.rstrip('/')}/retrieve",
            "retrieval",
            json={"query": query, "collection": name, "top_k": top_k, "mode": "hybrid"},
        )
        try:
            results = [
                Passage(
                    rank=h["rank"],
                    chunk_id=h["chunk_id"],
                    source_name=str(h["payload"].get("source_name", "")),
                    page=h["payload"].get("page"),
                    score=h["score"],
                    text=trim(h["text"], self._s.passage_chars),
                )
                for h in body["hits"]
            ]
        except (KeyError, TypeError):
            raise CorpusError("the retrieval service sent an unexpected reply") from None
        return CorpusSearchOutput(query=query, collection=name, results=results)

    async def chunks(self, collection: str | None, source_name: str | None) -> CorpusChunks:
        name = collection or self._s.default_collection
        url = f"{self._s.retrieval_url.rstrip('/')}/collections/{name}/chunks"
        found: list[CorpusChunk] = []
        after: str | None = None
        while True:
            params: dict[str, str | int] = {"limit": 500}
            if source_name:
                params["source_name"] = source_name
            if after:
                params["after"] = after
            page = await self._request("GET", url, "retrieval", params=params)
            try:
                found += [
                    CorpusChunk(
                        chunk_id=c["chunk_id"],
                        source_name=c.get("source_name", ""),
                        page=c.get("page"),
                        chunk_index=c.get("chunk_index", 0),
                        text=c["text"],
                    )
                    for c in page["chunks"]
                ]
                after = page.get("next")
            except (KeyError, TypeError):
                raise CorpusError("the retrieval service sent an unexpected reply") from None
            if len(found) >= self._s.table_max_chunks:
                return CorpusChunks(chunks=found[: self._s.table_max_chunks], truncated=bool(after))
            if not after:
                return CorpusChunks(chunks=found)

    async def sources(self, collection: str | None) -> list[SourceItem]:
        name = collection or self._s.default_collection
        body = await self._request(
            "GET", f"{self._s.ingest_url.rstrip('/')}/collections/{name}/sources", "ingest"
        )
        try:
            return [
                SourceItem(
                    source_id=s["source_id"],
                    source_name=s["source_name"],
                    chunk_count=s["chunk_count"],
                    pages=s["pages"],
                )
                for s in body
            ]
        except (KeyError, TypeError):
            raise CorpusError("the ingest service sent an unexpected reply") from None
