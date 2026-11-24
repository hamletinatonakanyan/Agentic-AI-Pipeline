"""csv and xlsx -> Markdown tables.

csv: the whole file is one table on one page.
xlsx: each visible sheet is one "page" (numbered by its position), headed `## <sheet name>`,
and the sheet name is kept as the page's label. Formulas are read as their saved values.
"""

import csv
import io
from datetime import date, datetime, time
from typing import Any

from openpyxl import load_workbook

from ingest.errors import ParseError
from ingest.models import PageResult
from ingest.parsers.text import decode_text
from ingest.tables import markdown_table

csv.field_size_limit(10_000_000)  # allow very long cells


def _delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def parse_csv(data: bytes) -> tuple[list[PageResult], list[str]]:
    text, warnings = decode_text(data)
    if not text.strip():
        return [], [*warnings, "file is empty"]
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=_delimiter(text[:20_000])))
    except csv.Error as exc:
        raise ParseError(f"could not read csv: {exc}") from exc
    table = markdown_table(rows)
    if not table:
        return [], [*warnings, "file has no data"]
    return [PageResult(page_no=1, text=table)], warnings


def _value(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat(sep=" ")
    if isinstance(v, date | time):
        return v.isoformat()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _trim(rows: list[list[Any]]) -> list[list[Any]]:
    """Drop empty trailing rows and columns (spreadsheets often report a huge used range)."""
    while rows and not any(c not in (None, "") for c in rows[-1]):
        rows.pop()
    width = 0
    for row in rows:
        for i in range(len(row) - 1, -1, -1):
            if row[i] not in (None, ""):
                width = max(width, i + 1)
                break
    return [row[:width] for row in rows]


def parse_xlsx(data: bytes) -> tuple[list[PageResult], list[str]]:
    try:
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:
        raise ParseError(f"could not open xlsx: {exc}") from exc

    pages: list[PageResult] = []
    warnings: list[str] = []
    try:
        for position, ws in enumerate(wb.worksheets, start=1):
            if ws.sheet_state != "visible":
                warnings.append(f"sheet '{ws.title}' is hidden; skipped")
                continue
            rows = _trim([[_value(c) for c in row] for row in ws.iter_rows(values_only=True)])
            table = markdown_table(rows)
            if not table:
                warnings.append(f"sheet '{ws.title}' is empty; skipped")
                continue
            pages.append(
                PageResult(page_no=position, text=f"## {ws.title}\n\n{table}", label=ws.title)
            )
    finally:
        wb.close()
    if not pages and not warnings:
        warnings.append("workbook has no sheets")
    return pages, warnings
