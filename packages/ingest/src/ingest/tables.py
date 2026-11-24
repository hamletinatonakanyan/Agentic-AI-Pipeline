"""Tables as Markdown pipe tables.

- `markdown_table`: rows -> a Markdown table (used for csv, xlsx and docx).
- `html_tables_to_markdown`: rewrites `<table>` blocks in text (LlamaParser may return them).
- `find_tables`: where Markdown tables sit in a text, so the chunker can split them by rows.
"""

import re
from html.parser import HTMLParser
from typing import Any

_SEPARATOR_LINE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_HTML_TABLE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)


def cell_text(value: Any) -> str:
    """One cell: single line, pipes escaped, extra spaces collapsed."""
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text.replace("|", "\\|")


def markdown_table(rows: list[list[Any]]) -> str:
    """The first row is the header. Rows are padded to the same width. Empty rows are dropped."""
    cleaned = [[cell_text(c) for c in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header = cleaned[0]
    if not any(header):
        header = [f"col{i + 1}" for i in range(width)]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in cleaned[1:]]
    return "\n".join(lines)


def find_tables(text: str) -> list[tuple[int, int]]:
    """Character ranges of Markdown tables: a header line, a `---` line, then `|` rows."""
    lines = text.splitlines(keepends=True)
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    found: list[tuple[int, int]] = []
    i = 0
    while i < len(lines) - 1:
        head, sep = lines[i], lines[i + 1]
        if (
            head.lstrip().startswith("|")
            and "|" in sep
            and _SEPARATOR_LINE.match(sep.rstrip("\r\n"))
        ):
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            found.append((offsets[i], offsets[j - 1] + len(lines[j - 1])))
            i = j
        else:
            i += 1
    return found


class _TableReader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._span = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
            try:
                self._span = max(int(dict(attrs).get("colspan") or 1), 1)
            except ValueError:
                self._span = 1
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell))
            self._row.extend([""] * (self._span - 1))  # a merged cell leaves empty neighbours
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def html_tables_to_markdown(text: str) -> str:
    """Replace every `<table>...</table>` in `text` with a Markdown table."""

    def convert(match: re.Match[str]) -> str:
        reader = _TableReader()
        reader.feed(match.group(0))
        reader.close()
        table = markdown_table(reader.rows)
        return f"\n\n{table}\n\n" if table else ""

    return _HTML_TABLE.sub(convert, text)
