"""RetrievalService: question in, ranked chunks out. Pure Python, no HTTP.

    rewrite (optional) -> search each query (dense, sparse or hybrid) -> merge -> rerank (optional)

Each stage is a span and a timing. Queries are encoded from the collection's own spec, so the
embedding model, the query prefix and the BM25 language always match how it was ingested.
"""

import time
from typing import Any

from common import AppError
from observability import current_trace_id, span
from retrieval.fusion import rrf_merge
from retrieval.models import (
    ChunkPage,
    CollectionInfo,
    CollectionSpec,
    Hit,
    RawHit,
    RerankedItem,
    RerankRequest,
    RerankResponse,
    RetrievalOptions,
    RetrievalResult,
    RetrievalStatus,
    RewriteResponse,
    RewriteStrategy,
    StrategiesResponse,
    Timings,
)
from retrieval.protocols import QueryEncoder, QueryRewriter, Reranker, SearchStore
from retrieval.settings import RetrievalSettings


def _ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000


class RetrievalService:
    def __init__(
        self,
        settings: RetrievalSettings,
        store: SearchStore,
        encoder: QueryEncoder,
        reranker: Reranker,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self._s = settings
        self._store = store
        self._encoder = encoder
        self._reranker = reranker
        self._rewriter = rewriter

    # ----- retrieve ---------------------------------------------------------------------------

    def retrieve(self, query: str, opts: RetrievalOptions) -> RetrievalResult:
        query = self._check_query(query)
        collection = opts.collection or self._s.collection
        top_k = opts.top_k or self._s.top_k
        candidates = opts.candidates or self._s.candidates
        if candidates < top_k:
            raise AppError("invalid_options", "candidates must be at least top_k.", 422)
        model = (opts.rerank_model or self._s.rerank_model) if opts.rerank else None

        started = time.perf_counter()
        warnings: list[str] = []
        with span(
            "retrieval.retrieve",
            {
                "retrieval.collection": collection,
                "retrieval.mode": opts.mode,
                "retrieval.rewrite": opts.rewrite,
                "retrieval.rerank": opts.rerank,
                "retrieval.top_k": top_k,
            },
        ) as root:
            trace_id = current_trace_id()
            spec = self._spec(collection)

            t = time.perf_counter()
            queries = self._rewrite(query, opts.rewrite)
            rewrite_ms = _ms(t) if opts.rewrite != "none" else 0.0

            t = time.perf_counter()
            # Fetch more per query when something later narrows the list (rerank, or merging).
            per_query_limit = candidates if (opts.rerank or len(queries) > 1) else top_k
            rankings = [
                self._search(collection, spec, q, i, opts, per_query_limit, candidates, warnings)
                for i, q in enumerate(queries)
            ]
            merged = rankings[0] if len(rankings) == 1 else rrf_merge(rankings, self._s.rrf_k)
            merged = merged[: candidates if opts.rerank else top_k]
            retrieve_ms = _ms(t)

            rerank_scores: dict[str, float] = {}
            rerank_ms = 0.0
            if opts.rerank and merged:
                t = time.perf_counter()
                merged, rerank_scores = self._rerank(query, merged, model, top_k)
                rerank_ms = _ms(t)
            merged = merged[:top_k]

            hits = [self._hit(i, h, rerank_scores.get(h.id)) for i, h in enumerate(merged, 1)]
            root.set_attributes({"retrieval.hits": len(hits), "retrieval.queries": len(queries)})

        return RetrievalResult(
            query=query,
            collection=collection,
            rewritten_queries=queries,
            hits=hits,
            timings=Timings(
                rewrite_ms=rewrite_ms,
                retrieve_ms=retrieve_ms,
                rerank_ms=rerank_ms,
                total_ms=_ms(started),
            ),
            mode=opts.mode,
            rewrite=opts.rewrite,
            rerank=opts.rerank,
            rerank_model=model,
            top_k=top_k,
            candidates=candidates,
            warnings=warnings,
            trace_id=trace_id,
        )

    def _check_query(self, query: str) -> str:
        query = query.strip()
        if not query:
            raise AppError("invalid_query", "The query is empty.", 422)
        if len(query) > self._s.max_query_chars:
            raise AppError(
                "invalid_query",
                f"The query is longer than {self._s.max_query_chars} characters.",
                422,
            )
        return query

    def _spec(self, collection: str) -> CollectionSpec:
        info = self._store.get_collection(collection)
        if info is None:
            raise AppError("collection_not_found", f"No collection '{collection}'.", 404)
        if info.spec is None:
            raise AppError(
                "collection_not_managed",
                f"Collection '{collection}' was not created by ingest, so its embedding model "
                "is unknown.",
                409,
            )
        return info.spec

    def _rewrite(self, query: str, strategy: RewriteStrategy) -> list[str]:
        if strategy == "none":
            return [query]
        if self._rewriter is None:
            raise AppError("rewriting_unavailable", "Query rewriting is not configured.", 503)
        with span("retrieval.rewrite", {"retrieval.strategy": strategy}) as sp:
            queries = self._rewriter.rewrite(query, strategy)
            sp.set_attribute("retrieval.queries", len(queries))
        return queries

    def _search(
        self,
        collection: str,
        spec: CollectionSpec,
        query: str,
        index: int,
        opts: RetrievalOptions,
        limit: int,
        candidates: int,
        warnings: list[str],
    ) -> list[RawHit]:
        with span("retrieval.search", {"retrieval.query_index": index, "retrieval.limit": limit}):
            dense = (
                self._encoder.encode_dense(query, spec)
                if opts.mode in ("dense", "hybrid")
                else None
            )
            sparse = None
            if opts.mode in ("sparse", "hybrid"):
                sparse = self._encoder.encode_sparse(query, spec)
                if not sparse.indices:  # only stop words, or nothing BM25 can use
                    note = "the query has no keyword terms for BM25"
                    fallback = "used dense search only" if dense is not None else "no results"
                    message = f"{note}: {fallback}"
                    if message not in warnings:
                        warnings.append(message)
                    sparse = None
            if dense is None and sparse is None:
                return []
            return self._store.search(
                collection,
                dense=dense,
                sparse=sparse,
                limit=limit,
                filters=opts.filters,
                prefetch_limit=candidates,
            )

    def _rerank(
        self, query: str, hits: list[RawHit], model: str | None, top_k: int
    ) -> tuple[list[RawHit], dict[str, float]]:
        with span(
            "retrieval.rerank", {"retrieval.candidates": len(hits), "retrieval.model": model}
        ):
            scores = self._reranker.score(
                query, [str(h.payload.get("text", "")) for h in hits], model
            )
        if len(scores) != len(hits):
            raise AppError(
                "rerank_failed", "The reranker returned the wrong number of scores.", 502
            )
        order = sorted(range(len(hits)), key=lambda i: -scores[i])  # stable: ties keep search order
        return [hits[i] for i in order], {hits[i].id: scores[i] for i in order}

    @staticmethod
    def _hit(rank: int, raw: RawHit, rerank_score: float | None) -> Hit:
        payload = dict(raw.payload)
        text = str(payload.pop("text", ""))
        return Hit(
            rank=rank,
            chunk_id=raw.id,
            score=raw.score,
            rerank_score=rerank_score,
            text=text,
            payload=payload,
        )

    # ----- the pieces on their own (for inspecting and testing) ---------------------------

    def rewrite(self, query: str, strategy: RewriteStrategy) -> RewriteResponse:
        query = self._check_query(query)
        with span("retrieval.rewrite_only", {"retrieval.strategy": strategy}):
            trace_id = current_trace_id()
            queries = self._rewrite(query, strategy)
        return RewriteResponse(
            query=query, strategy=strategy, rewritten_queries=queries, trace_id=trace_id
        )

    def rerank(self, request: RerankRequest) -> RerankResponse:
        query = self._check_query(request.query)
        model = request.model or self._s.rerank_model
        started = time.perf_counter()
        with span("retrieval.rerank_only", {"retrieval.candidates": len(request.candidates)}):
            trace_id = current_trace_id()
            scores = self._reranker.score(query, [c.text for c in request.candidates], model)
        if len(scores) != len(request.candidates):
            raise AppError(
                "rerank_failed", "The reranker returned the wrong number of scores.", 502
            )
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        items = [
            RerankedItem(
                rank=rank,
                chunk_id=request.candidates[i].chunk_id,
                text=request.candidates[i].text,
                rerank_score=scores[i],
                original_rank=i + 1,
            )
            for rank, i in enumerate(order[: request.top_n or len(order)], start=1)
        ]
        return RerankResponse(
            query=query, model=model, items=items, rerank_ms=_ms(started), trace_id=trace_id
        )

    # ----- information ------------------------------------------------------------------

    def list_collections(self) -> list[CollectionInfo]:
        return self._store.list_collections()

    def chunks_page(
        self,
        collection: str,
        source_name: str | None = None,
        after: str | None = None,
        limit: int = 256,
    ) -> ChunkPage:
        """Stored chunks, a page at a time. Read-only. A collection that does not exist is a
        404, not an empty list, so a typo is not mistaken for "no data"."""
        if self._store.get_collection(collection) is None:
            raise AppError("collection_not_found", f"No collection '{collection}'.", 404)
        chunks, nxt = self._store.scroll(
            collection, source_name=source_name, after=after, limit=max(1, min(limit, 1000))
        )
        # Pages come in storage order, NOT document order: a reader that needs the order of a
        # source sorts everything it collected by (page, chunk_index).
        return ChunkPage(chunks=chunks, next=nxt)

    def strategies(self) -> StrategiesResponse:
        models_fn: Any = getattr(self._reranker, "models", None)
        return StrategiesResponse(
            modes=["dense", "sparse", "hybrid"],
            rewrite_strategies=["none", "multi_query", "hyde", "step_back"],
            rerank_models=models_fn() if models_fn else [],
            defaults={
                "collection": self._s.collection,
                "mode": "hybrid",
                "top_k": self._s.top_k,
                "candidates": self._s.candidates,
                "rerank_model": self._s.rerank_model,
            },
            collections=self.list_collections(),
        )

    def status(self) -> RetrievalStatus:
        s = self._s
        return RetrievalStatus(
            qdrant_url=s.qdrant_url,
            default_collection=s.collection,
            rerank_model=s.rerank_model,
            top_k=s.top_k,
            candidates=s.candidates,
        )
