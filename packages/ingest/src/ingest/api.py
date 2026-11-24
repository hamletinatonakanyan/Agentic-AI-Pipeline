"""Routes. Four routers, so each app decides where each one is mounted:

- `health_router`: public liveness.
- `status_router`: protected `/status`.
- `jobs_router`:   protected `/jobs/{id}`.
- `router`:        protected, the ingest routes, mounted at `/api/ingest`.
"""

from fastapi import APIRouter, Depends, File, Form, Header, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from common import AppError, require_api_key
from common.errors import jsonable
from ingest.factory import Ingest
from ingest.models import (
    SUPPORTED_EXTENSIONS,
    CollectionInfo,
    DeleteCollectionReport,
    DeleteReport,
    IngestOptions,
    IngestStatus,
    JobAccepted,
    JobStatus,
    PreviewResponse,
    SourceInfo,
    UploadedFile,
)
from ingest.parsers import clean_name, extension

health_router = APIRouter()
status_router = APIRouter(dependencies=[Depends(require_api_key)])
jobs_router = APIRouter(dependencies=[Depends(require_api_key)])
router = APIRouter(dependencies=[Depends(require_api_key)])


def get_ingest(request: Request) -> Ingest:
    return request.app.state.ingest


class PathRequest(BaseModel):
    path: str = Field(min_length=1)
    options: IngestOptions = Field(default_factory=IngestOptions)


def parse_options(raw: str | None) -> IngestOptions:
    """Multipart forms carry options as a JSON string."""
    if not raw or not raw.strip():
        return IngestOptions()
    try:
        return IngestOptions.model_validate_json(raw)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False)
        raise AppError(
            "invalid_options", "`options` is not valid.", 422, {"errors": jsonable(errors)}
        ) from exc


def read_uploads(files: list[UploadFile], max_mb: int) -> list[UploadedFile]:
    if not files:
        raise AppError("no_files", "Send at least one file.", 422)
    limit = max_mb * 1024 * 1024
    out = []
    for f in files:
        name = clean_name(f.filename or "")
        if extension(name) not in SUPPORTED_EXTENSIONS:
            raise AppError(
                "unsupported_format",
                f"'{name}' is not supported. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
                415,
            )
        data = f.file.read(limit + 1)
        if len(data) > limit:
            raise AppError("file_too_large", f"'{name}' is larger than {max_mb} MB.", 413)
        out.append(UploadedFile(name=name, data=data))
    return out


@health_router.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@status_router.get("/status", response_model=IngestStatus)
def status(ingest: Ingest = Depends(get_ingest)) -> IngestStatus:
    return ingest.service.status()


@jobs_router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, ingest: Ingest = Depends(get_ingest)) -> JobStatus:
    job = ingest.jobs.get(job_id)
    if job is None:
        raise AppError("job_not_found", f"No job {job_id}.", 404)
    return job


@router.post("/files", response_model=JobAccepted, status_code=202)
def ingest_files(
    files: list[UploadFile] = File(...),
    options: str | None = Form(default=None),
    idempotency_key: str | None = Header(default=None),
    ingest: Ingest = Depends(get_ingest),
) -> JSONResponse:
    opts = parse_options(options)
    uploads = read_uploads(files, ingest.service.max_upload_mb)
    job = ingest.jobs.submit(lambda: ingest.service.ingest_files(uploads, opts), idempotency_key)
    return JSONResponse(
        JobAccepted(job_id=job.job_id, state=job.state).model_dump(), status_code=202
    )


@router.post("/path", response_model=JobAccepted, status_code=202)
def ingest_path(
    body: PathRequest,
    idempotency_key: str | None = Header(default=None),
    ingest: Ingest = Depends(get_ingest),
) -> JSONResponse:
    ingest.service.collect_paths(body.path)  # refuse bad paths now, not inside the job
    job = ingest.jobs.submit(
        lambda: ingest.service.ingest_path(body.path, body.options), idempotency_key
    )
    return JSONResponse(
        JobAccepted(job_id=job.job_id, state=job.state).model_dump(), status_code=202
    )


@router.post("/chunks/preview", response_model=PreviewResponse)
def preview_chunks(
    file: UploadFile = File(...),
    options: str | None = Form(default=None),
    ingest: Ingest = Depends(get_ingest),
) -> PreviewResponse:
    (upload,) = read_uploads([file], ingest.service.max_upload_mb)
    return ingest.service.preview_chunks(upload, parse_options(options))


@router.get("/collections", response_model=list[CollectionInfo])
def list_collections(ingest: Ingest = Depends(get_ingest)) -> list[CollectionInfo]:
    return ingest.service.list_collections()


@router.get("/collections/{name}/sources", response_model=list[SourceInfo])
def list_sources(name: str, ingest: Ingest = Depends(get_ingest)) -> list[SourceInfo]:
    return ingest.service.list_sources(name)


@router.delete("/collections/{name}", response_model=DeleteCollectionReport)
def delete_collection(
    name: str, confirm: str | None = None, ingest: Ingest = Depends(get_ingest)
) -> DeleteCollectionReport:
    """Delete a whole collection. `confirm` must repeat the name."""
    return ingest.service.delete_collection(name, confirm)


@router.delete("/collections/{name}/sources/{source_id}", response_model=DeleteReport)
def delete_source(name: str, source_id: str, ingest: Ingest = Depends(get_ingest)) -> DeleteReport:
    return ingest.service.delete_source(name, source_id)
