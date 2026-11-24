"""Questions about whole tables: counts, sums, averages, filters, groups.

Search returns a handful of passages, so it cannot count the rows of a 300-row table. The rows
are all stored, though, as Markdown tables inside the chunks (ingest splits a table by rows and
repeats the header in every chunk, with no overlap). This module puts them back together and
answers structured questions over them.

It is NOT text-to-SQL. The model never writes a query string. It fills in fixed fields (a table,
filters of the form column / operator / value, one aggregate, optional group-by), the fields are
checked against the table's real header, and the answer is computed here with plain Python. There
is no SQL, no `eval`, and nothing is written anywhere.
"""

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

Operator = Literal["=", "!=", "<", "<=", ">", ">=", "contains"]
Aggregate = Literal["count", "count_distinct", "sum", "avg", "min", "max", "rows"]

_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


class TableError(Exception):
    """A mistake in the question, with a message the model can act on."""


# ----- the model's side: what it may ask --------------------------------------------------------


class TableFilter(BaseModel):
    column: str = Field(description="A column name, exactly as list_tables shows it.")
    op: Operator = Field(
        description="One of = != < <= > >= contains. Numbers compare as numbers, text ignores case."
    )
    value: str | int | float | bool = Field(description="What to compare with.")


class TableInfo(BaseModel):
    name: str
    source: str
    columns: list[str]
    rows: int
    first_row: dict[str, str] = Field(default_factory=dict)


class TablesListing(BaseModel):
    collection: str
    tables: list[TableInfo]
    truncated: bool = False  # the scan stopped early because the collection is very large
    note: str = ""


class QueryResult(BaseModel):
    table: str
    total_rows: int
    matched_rows: int
    aggregate: str
    column: str | None = None
    group_by: list[str] = Field(default_factory=list)
    # A number for a plain aggregate; for `rows` and for groups, a list of objects.
    result: Any = None
    groups_total: int | None = None
    skipped_non_numeric: int = 0
    note: str = ""


# ----- reading tables out of chunks ---------------------------------------------------------------


@dataclass
class Table:
    name: str
    source: str
    columns: list[str]
    rows: list[list[str]] = field(default_factory=list)

    def info(self) -> TableInfo:
        first = dict(zip(self.columns, self.rows[0], strict=False)) if self.rows else {}
        return TableInfo(
            name=self.name,
            source=self.source,
            columns=self.columns,
            rows=len(self.rows),
            first_row=first,
        )


def _cells(line: str) -> list[str]:
    parts = _UNESCAPED_PIPE.split(line.strip())
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return [p.strip().replace("\\|", "|") for p in parts]


def _unique(names: list[str]) -> list[str]:
    seen: Counter[str] = Counter()
    out = []
    for i, n in enumerate(names):
        n = n or f"col{i + 1}"
        seen[n] += 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def tables_in_text(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """(header, rows) for each Markdown pipe table in a text."""
    lines = text.splitlines()
    found = []
    i = 0
    while i < len(lines) - 1:
        if lines[i].lstrip().startswith("|") and _SEPARATOR.match(lines[i + 1].rstrip()):
            header = _unique(_cells(lines[i]))
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                row = _cells(lines[j])
                row = (row + [""] * len(header))[: len(header)]  # pad or cut to the header
                rows.append(row)
                j += 1
            found.append((header, rows))
            i = j
        else:
            i += 1
    return found


def build_tables(chunks: Iterable[Any], max_rows: int = 200_000) -> tuple[list[Table], bool]:
    """Put the tables back together from stored chunks. Each chunk needs `source_name`, `page`,
    `chunk_index` and `text`. Returns the tables and whether `max_rows` cut the reading short."""
    ordered = sorted(chunks, key=lambda c: (c.source_name, c.page or 0, c.chunk_index))
    per_source: dict[str, list[Table]] = {}
    total = 0
    truncated = False
    for chunk in ordered:
        for header, rows in tables_in_text(chunk.text):
            tables = per_source.setdefault(chunk.source_name, [])
            table = next((t for t in tables if t.columns == header), None)
            if table is None:
                table = Table(name="", source=chunk.source_name, columns=header)
                tables.append(table)
            room = max_rows - total
            if len(rows) > room:
                rows, truncated = rows[: max(room, 0)], True
            table.rows.extend(rows)
            total += len(rows)
    result: list[Table] = []
    for source, tables in per_source.items():
        for n, table in enumerate(tables, start=1):
            table.name = source if len(tables) == 1 else f"{source}#{n}"
            result.append(table)
    return result, truncated


def source_of(table_name: str) -> str:
    """`report.md#2` -> `report.md`."""
    return table_name.rsplit("#", 1)[0] if re.search(r"#\d+$", table_name) else table_name


def find_table(tables: list[Table], name: str) -> Table:
    wanted = name.strip()
    for t in tables:
        if t.name == wanted:
            return t
    for t in tables:  # forgive the case, and a missing extension
        if t.name.casefold() == wanted.casefold() or t.name.casefold().startswith(
            wanted.casefold() + "."
        ):
            return t
    names = ", ".join(t.name for t in tables) or "none"
    raise TableError(f"No table named '{name}'. Tables available: {names}.")


# ----- answering questions -----------------------------------------------------------------------


def _number(text: str) -> float | None:
    t = text.strip()
    if not t:
        return None
    try:
        value = float(t)
    except ValueError:
        return None
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def _column(table: Table, name: str) -> int:
    for i, c in enumerate(table.columns):
        if c == name:
            return i
    for i, c in enumerate(table.columns):
        if c.casefold() == name.strip().casefold():
            return i
    raise TableError(
        f"Table '{table.name}' has no column '{name}'. Columns: {', '.join(table.columns)}."
    )


def _matches(cell: str, op: str, value: str) -> bool:
    a, b = _number(cell), _number(value)
    numeric = a is not None and b is not None
    if op == "contains":
        return value.casefold() in cell.casefold()
    if op in ("=", "!="):
        same = a == b if numeric else cell.strip().casefold() == value.strip().casefold()
        return same if op == "=" else not same
    if not cell.strip():
        return False  # an empty cell is neither smaller nor larger
    if b is not None and a is None:
        return False  # a number was asked for and this cell is not one (for example "n/a")
    left, right = (a, b) if numeric else (cell.strip().casefold(), value.strip().casefold())
    return {"<": left < right, "<=": left <= right, ">": left > right, ">=": left >= right}[op]  # type: ignore[operator]


def _round(x: float) -> float | int:
    return int(x) if float(x).is_integer() else round(x, 6)


def _aggregate(rows: list[list[str]], aggregate: str, col: int | None) -> tuple[Any, int]:
    """(value, number of non-numeric cells skipped)."""
    if aggregate == "count":
        return len(rows), 0
    assert col is not None
    cells = [r[col] for r in rows]
    if aggregate == "count_distinct":
        return len({c.strip().casefold() for c in cells if c.strip()}), 0
    numbers = [n for c in cells if (n := _number(c)) is not None]
    skipped = sum(1 for c in cells if c.strip() and _number(c) is None)
    if not numbers:
        return None, skipped
    return {
        "sum": lambda: _round(sum(numbers)),
        "avg": lambda: _round(sum(numbers) / len(numbers)),
        "min": lambda: _round(min(numbers)),
        "max": lambda: _round(max(numbers)),
    }[aggregate](), skipped


def run_query(
    table: Table,
    *,
    aggregate: Aggregate = "count",
    column: str | None = None,
    filters: list[TableFilter] | None = None,
    group_by: list[str] | None = None,
    select: list[str] | None = None,
    order: Literal["desc", "asc"] = "desc",
    limit: int = 20,
    max_groups: int = 50,
) -> QueryResult:
    limit = max(1, min(limit, max_groups))
    conditions = [
        (
            _column(table, f.column),
            f.op,
            str(f.value).lower() if isinstance(f.value, bool) else str(f.value),
        )
        for f in (filters or [])
    ]
    group_cols = [_column(table, g) for g in (group_by or [])]
    if len(group_cols) > 2:
        raise TableError("group_by accepts at most 2 columns.")
    col = _column(table, column) if column else None
    if aggregate in ("sum", "avg", "min", "max", "count_distinct") and col is None:
        raise TableError(
            f"aggregate '{aggregate}' needs `column`. Columns: {', '.join(table.columns)}."
        )
    if aggregate == "rows" and group_cols:
        raise TableError("aggregate 'rows' cannot be combined with group_by.")

    matched = [r for r in table.rows if all(_matches(r[i], op, v) for i, op, v in conditions)]
    base = QueryResult(
        table=table.name,
        total_rows=len(table.rows),
        matched_rows=len(matched),
        aggregate=aggregate,
        column=table.columns[col] if col is not None else None,
        group_by=[table.columns[i] for i in group_cols],
    )

    if aggregate == "rows":
        picked = [_column(table, s) for s in select] if select else list(range(len(table.columns)))
        base.result = [{table.columns[i]: r[i] for i in picked} for r in matched[:limit]]
        if len(matched) > limit:
            base.note = f"Showing the first {limit} of {len(matched)} matching rows."
        return base

    if not group_cols:
        value, skipped = _aggregate(matched, aggregate, col)
        base.result, base.skipped_non_numeric = value, skipped
        if value is None:
            base.note = (
                f"No numeric values in '{base.column}' among the {len(matched)} matching rows."
            )
        elif skipped:
            base.note = f"{skipped} non-numeric cell(s) in '{base.column}' were ignored."
        return base

    groups: dict[tuple[str, ...], list[list[str]]] = {}
    for r in matched:
        groups.setdefault(tuple(r[i] for i in group_cols), []).append(r)
    scored = []
    skipped_total = 0
    for key, rows in groups.items():
        value, skipped = _aggregate(rows, aggregate, col)
        skipped_total += skipped
        scored.append((key, value, len(rows)))
    scored.sort(
        key=lambda t: (t[1] is None, -(t[1] or 0) if order == "desc" else (t[1] or 0), t[0])
    )
    base.groups_total = len(scored)
    base.skipped_non_numeric = skipped_total
    base.result = [
        {**dict(zip(base.group_by, key, strict=True)), "value": value, "rows": n}
        for key, value, n in scored[:limit]
    ]
    if len(scored) > limit:
        base.note = f"Showing {limit} of {len(scored)} groups."
    return base
