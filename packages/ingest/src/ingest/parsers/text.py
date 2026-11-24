"""txt and md. One page."""

from ingest.models import PageResult
from ingest.tables import html_tables_to_markdown


def decode_text(data: bytes) -> tuple[str, list[str]]:
    """UTF-8 (with or without BOM). Falls back to Latin-1 with a warning, which never fails."""
    try:
        return data.decode("utf-8-sig"), []
    except UnicodeDecodeError:
        return data.decode("latin-1"), ["not valid UTF-8; decoded as Latin-1"]


def parse_text(data: bytes, markdown: bool) -> tuple[list[PageResult], list[str]]:
    text, warnings = decode_text(data)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if markdown:
        text = html_tables_to_markdown(text)
    if not text.strip():
        return [], [*warnings, "file is empty"]
    return [PageResult(page_no=1, text=text)], warnings
