"""Routes. Four routers, so each app decides where each one is mounted:

- `health_router`: public liveness.
- `status_router`: protected `/status`.
- `jobs_router`:   protected `/jobs/{id}`.
- `router`:        protected, the evals routes, mounted at `/api/evals`.
"""

from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from common import AppError, require_api_key
from evals.catalog import METRICS, MetricInfo
from evals.factory import Evals
from evals.models import (
    Comparison,
    Dataset,
    DatasetItem,
    DatasetSummary,
    EstimateResponse,
    EvalStatus,
    GenerateRequest,
    ItemPatch,
    JobAccepted,
    JobStatus,
    Run,
    RunRequest,
    RunSummary,
    TargetsInfo,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

health_router = APIRouter()
status_router = APIRouter(dependencies=[Depends(require_api_key)])
jobs_router = APIRouter(dependencies=[Depends(require_api_key)])
router = APIRouter(dependencies=[Depends(require_api_key)])


def get_evals(request: Request) -> Evals:
    return request.app.state.evals


def _accepted(job: JobStatus) -> JSONResponse:
    return JSONResponse(
        JobAccepted(job_id=job.job_id, state=job.state).model_dump(), status_code=202
    )


@health_router.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@status_router.get("/status", response_model=EvalStatus)
def status(evals: Evals = Depends(get_evals)) -> EvalStatus:
    return evals.service.status()


@jobs_router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, evals: Evals = Depends(get_evals)) -> JobStatus:
    return evals.service.get_job(job_id)


# ----- datasets -------------------------------------------------------------------------------


@router.post("/datasets", response_model=DatasetSummary, status_code=201)
def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    evals: Evals = Depends(get_evals),
) -> DatasetSummary:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise AppError("file_too_large", "The dataset file is larger than 20 MB.", 413)
    label = (name or file.filename or "uploaded dataset").strip() or "uploaded dataset"
    return evals.service.upload_dataset(label, data)


@router.post("/datasets/generate", response_model=JobAccepted, status_code=202)
def generate_dataset(
    body: GenerateRequest,
    idempotency_key: str | None = Header(default=None),
    evals: Evals = Depends(get_evals),
) -> JSONResponse:
    return _accepted(evals.service.generate_dataset(body, idempotency_key))


@router.get("/datasets", response_model=list[DatasetSummary])
def list_datasets(evals: Evals = Depends(get_evals)) -> list[DatasetSummary]:
    return evals.service.list_datasets()


@router.get("/datasets/{dataset_id}", response_model=Dataset)
def get_dataset(dataset_id: str, evals: Evals = Depends(get_evals)) -> Dataset:
    return evals.service.get_dataset(dataset_id)


@router.patch("/datasets/{dataset_id}/items/{item_id}", response_model=DatasetItem)
def patch_item(
    dataset_id: str, item_id: str, body: ItemPatch, evals: Evals = Depends(get_evals)
) -> DatasetItem:
    return evals.service.patch_item(dataset_id, item_id, body)


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, evals: Evals = Depends(get_evals)) -> dict[str, str]:
    evals.service.delete_dataset(dataset_id)
    return {"deleted": dataset_id}


@router.get("/metrics", response_model=list[MetricInfo])
def metrics() -> list[MetricInfo]:
    """Every metric, with the dataset fields and target outputs it needs."""
    return METRICS


@router.get("/collections", response_model=list[str])
def list_collections(evals: Evals = Depends(get_evals)) -> list[str]:
    return evals.service.list_collections()


@router.get("/targets", response_model=TargetsInfo)
def targets(evals: Evals = Depends(get_evals)) -> TargetsInfo:
    return evals.service.targets_info()


# ----- runs -----------------------------------------------------------------------------------


@router.post("/runs/estimate", response_model=EstimateResponse)
def estimate_run(body: RunRequest, evals: Evals = Depends(get_evals)) -> EstimateResponse:
    return evals.service.estimate(body)


@router.post("/runs", response_model=JobAccepted, status_code=202)
def start_run(
    body: RunRequest,
    idempotency_key: str | None = Header(default=None),
    evals: Evals = Depends(get_evals),
) -> JSONResponse:
    return _accepted(evals.service.start_run(body, idempotency_key))


@router.get("/runs", response_model=list[RunSummary])
def list_runs(evals: Evals = Depends(get_evals)) -> list[RunSummary]:
    return evals.service.list_runs()


# Declared before /runs/{run_id}, or "compare" would be read as a run id.
@router.get("/runs/compare", response_model=Comparison)
def compare_runs(
    a: str = Query(min_length=1), b: str = Query(min_length=1), evals: Evals = Depends(get_evals)
) -> Comparison:
    return evals.service.compare(a, b)


@router.get("/runs/{run_id}", response_model=Run)
def get_run(run_id: str, evals: Evals = Depends(get_evals)) -> Run:
    return evals.service.get_run(run_id)
