"""One error shape for the whole system:

{"error": {"code": "...", "message": "...", "details": {...}}}
"""

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class AppError(Exception):
    """Raise this from services and routes. The handler turns it into the envelope."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def _envelope(code: str, message: str, status: int, details: dict[str, Any] | None = None):
    body = ErrorEnvelope(error=ErrorBody(code=code, message=message, details=details or {}))
    return JSONResponse(status_code=status, content=body.model_dump())


def install_error_handlers(app: FastAPI) -> None:
    """Make every error from this app use the envelope."""

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return _envelope(exc.code, exc.message, exc.status_code, exc.details)

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException):
        return _envelope(f"http_{exc.status_code}", str(exc.detail), exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        # Drop the raw input so a bad request never echoes secrets back.
        errors = [{k: v for k, v in e.items() if k != "input"} for e in exc.errors()]
        return _envelope(
            "validation_error", "Request validation failed", 422, {"errors": jsonable(errors)}
        )


def jsonable(value: Any) -> Any:
    """Make validation error contexts JSON-safe (they can hold exception objects)."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
