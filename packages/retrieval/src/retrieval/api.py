"""Routes. Three routers, so each app decides where each one is mounted:

- `health_router`: public liveness.
- `status_router`: protected `/status`.
- `router`:        protected, the retrieval routes, mounted at `/api/retrieval`.
"""

from fastapi import APIRouter, Depends, Query, Request

from common import require_api_key
from retrieval.factory import Retrieval
from retrieval.models import (
    ChunkPage,
    CollectionInfo,
    RerankRequest,
    RerankResponse,
    RetrievalResult,
    RetrievalStatus,
    RetrieveRequest,
    RewriteRequest,
    RewriteResponse,
    StrategiesResponse,
)

health_router = APIRouter()
status_router = APIRouter(dependencies=[Depends(require_api_key)])
router = APIRouter(dependencies=[Depends(require_api_key)])


def get_retrieval(request: Request) -> Retrieval:
    return request.app.state.retrieval


@health_router.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@status_router.get("/status", response_model=RetrievalStatus)
def status(retrieval: Retrieval = Depends(get_retrieval)) -> RetrievalStatus:
    return retrieval.service.status()


@router.post("/retrieve", response_model=RetrievalResult)
def retrieve(
    body: RetrieveRequest, retrieval: Retrieval = Depends(get_retrieval)
) -> RetrievalResult:
    return retrieval.service.retrieve(body.query, body)


@router.post("/rewrite", response_model=RewriteResponse)
def rewrite(body: RewriteRequest, retrieval: Retrieval = Depends(get_retrieval)) -> RewriteResponse:
    return retrieval.service.rewrite(body.query, body.strategy)


@router.post("/rerank", response_model=RerankResponse)
def rerank(body: RerankRequest, retrieval: Retrieval = Depends(get_retrieval)) -> RerankResponse:
    return retrieval.service.rerank(body)


@router.get("/strategies", response_model=StrategiesResponse)
def strategies(retrieval: Retrieval = Depends(get_retrieval)) -> StrategiesResponse:
    return retrieval.service.strategies()


@router.get("/collections", response_model=list[CollectionInfo])
def collections(retrieval: Retrieval = Depends(get_retrieval)) -> list[CollectionInfo]:
    return retrieval.service.list_collections()


@router.get("/collections/{name}/chunks", response_model=ChunkPage)
def chunks(
    name: str,
    source_name: str | None = None,
    after: str | None = None,
    limit: int = Query(default=256, ge=1, le=1000),
    retrieval: Retrieval = Depends(get_retrieval),
) -> ChunkPage:
    """Stored chunks, a page at a time (follow `next`). Read-only. Order is storage order."""
    return retrieval.service.chunks_page(name, source_name, after, limit)
