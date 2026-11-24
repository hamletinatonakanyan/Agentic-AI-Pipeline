"""LlamaParse v2 REST client (httpx, no SDK).

The client follows LlamaIndex's published API. The live test `tests/test_ingest_live.py`
checks it against the real service. Details the documentation leaves open are handled
defensively:

- The EU base URL: configurable (`INGEST_LLAMAPARSE_BASE_URL`).
- Whether result `page_number` is 0- or 1-based: results are matched by position when the counts
  agree, otherwise by number in either base.
- Where the tables option lives: not sent. HTML tables in the result are converted to Markdown
  by `ingest.tables` instead, which handles either output.

Flow: POST /api/v2/parse/upload (file + JSON `configuration`) -> poll GET /api/v2/parse/{id}
?expand=markdown until COMPLETED -> read `markdown.pages[]`.
"""

import json
import time
from collections.abc import Callable
from typing import Any

import httpx

from ingest.errors import ParseError

_TERMINAL_FAILURES = {"FAILED", "CANCELLED"}


def page_ranges(pages: list[int]) -> str:
    """[1, 2, 3, 5, 7, 8] -> "1-3,5,7-8" (1-based, as the API expects)."""
    parts: list[str] = []
    ordered = sorted(set(pages))
    start = prev = ordered[0]
    for n in [*ordered[1:], None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ",".join(parts)


class LlamaParseClient:
    def __init__(
        self,
        api_key: str,
        tier: str = "fast",
        base_url: str = "https://api.cloud.llamaindex.ai",
        timeout_s: float = 300.0,
        poll_interval_s: float = 2.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._key = api_key
        self._tier = tier
        self._base = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._poll = poll_interval_s
        self._http = client or httpx.Client(timeout=60.0)
        self._sleep = sleep

    def parse(self, pdf: bytes, pages: list[int]) -> dict[int, str]:
        job_id = self._upload(pdf, pages)
        body = self._wait(job_id)
        return self._pages(body, sorted(set(pages)))

    # ----------------------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}"}

    def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._http.request(method, url, headers=self._headers(), **kwargs)
        except httpx.HTTPError as exc:
            raise ParseError(f"LlamaParser request failed: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            # Never echo headers (they hold the key). A short body excerpt is enough to debug.
            raise ParseError(
                f"LlamaParser returned HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ParseError("LlamaParser returned a response that is not JSON") from exc

    def _upload(self, pdf: bytes, pages: list[int]) -> str:
        configuration = {
            "tier": self._tier,
            "version": "latest",
            "page_ranges": {"target_pages": page_ranges(pages)},
        }
        body = self._request(
            "POST",
            f"{self._base}/api/v2/parse/upload",
            files={"file": ("document.pdf", pdf, "application/pdf")},
            data={"configuration": json.dumps(configuration)},
        )
        job_id = body.get("id") or (body.get("job") or {}).get("id")
        if not job_id:
            raise ParseError("LlamaParser did not return a job id")
        return str(job_id)

    def _wait(self, job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self._timeout_s
        while True:
            body = self._request(
                "GET", f"{self._base}/api/v2/parse/{job_id}", params={"expand": "markdown"}
            )
            job = body.get("job") or body  # the docs show the status nested under "job"
            status = str(job.get("status", "")).upper()
            if status == "COMPLETED":
                return body
            if status in _TERMINAL_FAILURES:
                detail = job.get("error_message") or status
                raise ParseError(f"LlamaParser job {status.lower()}: {detail}")
            if time.monotonic() >= deadline:
                raise ParseError(f"LlamaParser job did not finish within {self._timeout_s:.0f}s")
            self._sleep(self._poll)

    @staticmethod
    def _pages(body: dict[str, Any], requested: list[int]) -> dict[int, str]:
        pages = ((body.get("markdown") or {}).get("pages")) or []
        ok = [p for p in pages if p.get("success", True) is not False]
        if len(ok) == len(requested):
            # Same count as requested: trust the order, whatever base the numbers use.
            ordered = sorted(ok, key=lambda p: p.get("page_number", 0))
            return {n: p.get("markdown") or "" for n, p in zip(requested, ordered, strict=True)}

        wanted = set(requested)
        numbers = [p.get("page_number") for p in ok]
        if all(n in wanted for n in numbers):
            offset = 0
        elif all(isinstance(n, int) and n + 1 in wanted for n in numbers):
            offset = 1  # 0-based numbering
        else:
            raise ParseError("LlamaParser returned pages that do not match the request")
        return {p["page_number"] + offset: p.get("markdown") or "" for p in ok}
