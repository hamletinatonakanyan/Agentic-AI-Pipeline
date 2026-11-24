"""Business logic behind the trace endpoints. No HTTP in here."""

from common import AppError
from observability.models import ObsStatus, TraceFilter, TraceSummary, TraceTree
from observability.protocols import TraceStore
from observability.tracer import Observability


class ObservabilityService:
    def __init__(self, handle: Observability) -> None:
        self._handle = handle

    def _store(self) -> TraceStore:
        if not self._handle.enabled:
            raise AppError(
                "observability_disabled", "Observability is off (OBS_ENABLED=false).", 503
            )
        if self._handle.store is None:
            raise AppError(
                "no_trace_store", "No local trace store. Add `local` to OBS_EXPORTERS.", 503
            )
        return self._handle.store

    def list_traces(
        self, filters: TraceFilter, limit: int = 50, offset: int = 0
    ) -> list[TraceSummary]:
        return self._store().list_traces(filters, limit, offset)

    def get_trace(self, trace_id: str) -> TraceTree:
        tree = self._store().get_trace(trace_id)
        if tree is None:
            raise AppError("trace_not_found", f"No trace {trace_id}.", 404)
        return tree

    def status(self) -> ObsStatus:
        s = self._handle.settings
        has_store = self._handle.store is not None
        return ObsStatus(
            enabled=s.enabled,
            exporters=s.exporter_names if s.enabled else [],
            sample_rate=s.sample_rate,
            store="sqlite" if has_store else "none",
            db_path=s.local_db_path if has_store else "",
        )
