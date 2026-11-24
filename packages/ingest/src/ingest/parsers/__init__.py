"""Turn an uploaded file into a `Document` of Markdown pages."""

import hashlib
from pathlib import PurePosixPath

from ingest.errors import ParseError
from ingest.models import SUPPORTED_EXTENSIONS, Document, ParseStats, UploadedFile
from ingest.parsers.docx import parse_docx
from ingest.parsers.pdf import PdfRouter
from ingest.parsers.tabular import parse_csv, parse_xlsx
from ingest.parsers.text import parse_text


def clean_name(name: str) -> str:
    """The file name only, without any directory part (browsers and clients may send one)."""
    return PurePosixPath(name.replace("\\", "/")).name or "unnamed"


def extension(name: str) -> str:
    return PurePosixPath(clean_name(name)).suffix.lower()


class FileParser:
    def __init__(self, pdf: PdfRouter) -> None:
        self._pdf = pdf

    def parse(self, file: UploadedFile, allow_remote: bool = True) -> Document:
        name = clean_name(file.name)
        ext = extension(name)
        content_hash = hashlib.sha256(file.data).hexdigest()
        stats = ParseStats()

        if ext == ".pdf":
            pages, warnings, stats = self._pdf.parse(file.data, content_hash, allow_remote)
        elif ext in (".txt", ".md"):
            pages, warnings = parse_text(file.data, markdown=ext == ".md")
            stats.pages_local = len(pages)
        elif ext == ".docx":
            pages, warnings = parse_docx(file.data)
            stats.pages_local = len(pages)
        elif ext == ".csv":
            pages, warnings = parse_csv(file.data)
            stats.pages_local = len(pages)
        elif ext == ".xlsx":
            pages, warnings = parse_xlsx(file.data)
            stats.pages_local = len(pages)
        else:
            raise ParseError(
                f"unsupported file type '{ext or name}'. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        return Document(
            source_name=name,
            content_hash=content_hash,
            pages=pages,
            warnings=warnings,
            stats=stats,
        )
