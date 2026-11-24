from fastapi import APIRouter, Depends, Request

from common import AppSettings, get_app_settings, require_api_key
from e2e.models import StatusReport
from e2e.service import StatusService

# Public: liveness only. Exempt from the access key.
health_router = APIRouter()


@health_router.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


# Protected: every route on this router needs X-API-Key.
router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


def get_status_service(
    request: Request,
    app_settings: AppSettings = Depends(get_app_settings),
) -> StatusService:
    # Same `get_app_settings` the key check uses, so one override covers both.
    state = request.app.state
    return StatusService(app_settings, state.settings, state.qdrant, state.built)


@router.get("/status", response_model=StatusReport)
def status(service: StatusService = Depends(get_status_service)) -> StatusReport:
    return service.report()
