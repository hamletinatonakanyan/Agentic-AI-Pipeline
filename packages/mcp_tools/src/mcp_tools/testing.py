"""Fakes for tests in any package."""

from mcp_tools.corpus import (
    CorpusChunk,
    CorpusChunks,
    CorpusError,
    CorpusSearchOutput,
    Passage,
    SourceItem,
)


class FakeCorpus:
    def __init__(self, fail: str | None = None, chunks: list[CorpusChunk] | None = None) -> None:
        self.fail = fail
        self.stored = chunks or []
        self.searches: list[tuple[str, int, str | None]] = []
        self.chunk_requests: list[tuple[str | None, str | None]] = []

    async def chunks(self, collection: str | None, source_name: str | None) -> CorpusChunks:
        self.chunk_requests.append((collection, source_name))
        if self.fail:
            raise CorpusError(self.fail)
        return CorpusChunks(chunks=[c for c in self.stored if source_name in (None, c.source_name)])

    async def search(self, query: str, top_k: int, collection: str | None) -> CorpusSearchOutput:
        self.searches.append((query, top_k, collection))
        if self.fail:
            raise CorpusError(self.fail)
        return CorpusSearchOutput(
            query=query,
            collection=collection or "docs",
            results=[
                Passage(
                    rank=i + 1, chunk_id=f"c{i + 1}", source_name="cats.txt", page=i + 1,
                    score=1.0 / (i + 1), text=f"Passage {i + 1} about {query}.",
                )
                for i in range(min(top_k, 3))
            ],
        )  # fmt: skip

    async def sources(self, collection: str | None) -> list[SourceItem]:
        if self.fail:
            raise CorpusError(self.fail)
        return [SourceItem(source_id="s1", source_name="cats.txt", chunk_count=4, pages=2)]
