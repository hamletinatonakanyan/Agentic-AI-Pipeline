"""PDFs, routed page by page.

For each page: read the text layer locally (free). A page with at least `threshold` characters
keeps it. Any other page is treated as scanned and goes to the remote parser, one request for all
scanned pages of the file, with each result cached by file hash, page number and tier.

Nothing here limits or confirms spending. The cache only prevents re-sending
unchanged pages.

Note: the local text layer is plain text. Tables in a text-layer PDF are flattened by `pypdf`.
LlamaParser returns tables, which are converted to Markdown tables.
"""

import io

from pypdf import PdfReader

from ingest.errors import ParseError
from ingest.models import PageResult, ParseStats
from ingest.protocols import ParseCache, PdfParser
from ingest.tables import html_tables_to_markdown
from observability import span


class PdfRouter:
    def __init__(
        self,
        threshold: int,
        tier: str,
        mode: str = "llamaparse",
        remote: PdfParser | None = None,
        cache: ParseCache | None = None,
        remote_unavailable_reason: str = "LLAMA_CLOUD_API_KEY is not set",
    ) -> None:
        self._threshold = threshold
        self._tier = tier
        self._mode = mode
        self._remote = remote
        self._cache = cache
        self._reason = remote_unavailable_reason

    def parse(
        self, data: bytes, content_hash: str, allow_remote: bool = True
    ) -> tuple[list[PageResult], list[str], ParseStats]:
        """`allow_remote=False` (used by preview) never calls the remote parser. Cached scanned
        pages are still used, and the others are skipped with a warning."""
        warnings: list[str] = []
        stats = ParseStats()
        reader = self._open(data)

        local: dict[int, str] = {}
        scanned: list[int] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # a damaged page should not fail the whole file
                text = ""
                warnings.append(f"page {number}: text extraction failed ({type(exc).__name__})")
            if len(text.strip()) >= self._threshold:
                local[number] = text
            else:
                scanned.append(number)
        stats.pages_local = len(local)

        results: dict[int, PageResult] = {
            n: PageResult(page_no=n, text=t, parser="local") for n, t in local.items()
        }
        if scanned:
            self._parse_scanned(data, content_hash, scanned, results, warnings, stats, allow_remote)
        pages = [results[n] for n in sorted(results)]
        if not pages and not warnings:
            warnings.append("PDF has no pages")
        return pages, warnings, stats

    # ----------------------------------------------------------------------------------------

    @staticmethod
    def _open(data: bytes) -> PdfReader:
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ParseError("PDF is password-protected")
            len(reader.pages)  # force the page tree to load, so a damaged file fails here
            return reader
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"could not open PDF: {exc}") from exc

    def _parse_scanned(
        self,
        data: bytes,
        content_hash: str,
        scanned: list[int],
        results: dict[int, PageResult],
        warnings: list[str],
        stats: ParseStats,
        allow_remote: bool = True,
    ) -> None:
        listed = ", ".join(map(str, scanned[:20])) + ("..." if len(scanned) > 20 else "")
        if self._mode == "local" or self._remote is None:
            why = "INGEST_PDF_PARSER=local" if self._mode == "local" else self._reason
            warnings.append(f"{len(scanned)} scanned page(s) skipped ({why}): {listed}")
            stats.pages_skipped = len(scanned)
            return

        texts: dict[int, str] = {}
        missing: list[int] = []
        for number in scanned:
            cached = self._cache.get(content_hash, number, self._tier) if self._cache else None
            if cached is None:
                missing.append(number)
            else:
                texts[number] = cached
        stats.pages_cached = len(texts)

        cached_pages = set(texts)
        if missing and not allow_remote:
            warnings.append(
                f"{len(missing)} scanned page(s) not previewed (preview never calls LlamaParser; "
                f"they are sent when you ingest): {', '.join(map(str, missing[:20]))}"
            )
            stats.pages_skipped += len(missing)
            missing = []
        if missing:
            with span("llamaparse.parse", {"pages": len(missing), "tier": self._tier}):
                fetched = self._remote.parse(data, missing)
            stats.pages_llamaparse = len(missing)
            for number in missing:
                text = fetched.get(number, "")
                if self._cache:
                    # Cache empty results too, so a blank scan is not sent again.
                    self._cache.put(content_hash, number, self._tier, text)
                texts[number] = text

        for number in sorted(texts):
            text = html_tables_to_markdown(texts[number])
            if not text.strip():
                warnings.append(f"page {number}: no text found")
                stats.pages_skipped += 1
                continue
            results[number] = PageResult(
                page_no=number, text=text, parser="llamaparse", cached=number in cached_pages
            )
