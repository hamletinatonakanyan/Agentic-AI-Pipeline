"""docx -> Markdown, in document order: headings, lists, paragraphs and tables. One page."""

import io
import re

from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from ingest.errors import ParseError
from ingest.models import PageResult
from ingest.tables import markdown_table

_HEADING_STYLE = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)


def _paragraph(par: Paragraph) -> str:
    text = par.text.strip()
    if not text:
        return ""
    style = (par.style.name if par.style is not None else "") or ""
    if style.lower() == "title":
        return f"# {text}"
    m = _HEADING_STYLE.match(style)
    if m:
        return f"{'#' * min(int(m.group(1)), 6)} {text}"
    if style.lower().startswith("list"):
        return f"- {text}"
    return text


def _table(table: Table) -> str:
    rows = []
    for row in table.rows:
        seen: set[int] = set()
        cells = []
        for cell in row.cells:
            # A merged cell appears several times. Keep its text once, leave the rest empty.
            key = id(cell._tc)
            cells.append("" if key in seen else cell.text)
            seen.add(key)
        rows.append(cells)
    return markdown_table(rows)


def parse_docx(data: bytes) -> tuple[list[PageResult], list[str]]:
    try:
        doc = DocxDocument(io.BytesIO(data))
    except Exception as exc:  # python-docx raises several kinds for a bad file
        raise ParseError(f"could not open docx: {exc}") from exc

    parts: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            part = _paragraph(Paragraph(child, doc))
        elif tag == "tbl":
            part = _table(Table(child, doc))
        else:
            continue
        if part:
            parts.append(part)

    text = "\n\n".join(parts)
    if not text.strip():
        return [], ["document has no text"]
    return [PageResult(page_no=1, text=text)], []
