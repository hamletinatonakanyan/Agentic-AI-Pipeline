"""Routes. Three routers, so the app decides where each one is mounted:

- `health_router`: public liveness.
- `status_router`: protected, `/status`.
- `router`: protected, the trace routes. Mounted at `/api/obs`.
"""

from typing import Literal

from fastapi import APIRouter, Depends, Query, Request

from common import require_api_key
from observability.models import ObsStatus, TraceFilter, TraceSummary, TraceTree
from observability.service import ObservabilityService

health_router = APIRouter()
status_router = APIRouter(dependencies=[Depends(require_api_key)])
router = APIRouter(dependencies=[Depends(require_api_key)])


def get_service(request: Request) -> ObservabilityService:
    return request.app.state.obs_service


@health_router.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@status_router.get("/status", response_model=ObsStatus)
def status(service: ObservabilityService = Depends(get_service)) -> ObsStatus:
    return service.status()


@router.get("/traces", response_model=list[TraceSummary])
def list_traces(
    name: str | None = None,
    status: Literal["ok", "error"] | None = None,
    since: float | None = None,
    until: float | None = None,
    min_duration_ms: float | None = None,
    session_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: ObservabilityService = Depends(get_service),
) -> list[TraceSummary]:
    filters = TraceFilter(
        name=name,
        status=status,
        since=since,
        until=until,
        min_duration_ms=min_duration_ms,
        session_id=session_id,
    )
    return service.list_traces(filters, limit, offset)


@router.get("/traces/{trace_id}", response_model=TraceTree)
def get_trace(trace_id: str, service: ObservabilityService = Depends(get_service)) -> TraceTree:
    return service.get_trace(trace_id)
