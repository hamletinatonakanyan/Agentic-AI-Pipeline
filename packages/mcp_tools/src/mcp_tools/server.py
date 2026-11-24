"""Our own MCP server: a calculator, a corpus search, the list of ingested sources, and a prompt.

    uv run python -m mcp_tools.server            # over stdio (what a client launches)

Over stdio, stdout IS the protocol channel. Anything printed there corrupts it, so this module
logs only to stderr.
"""

import json
import logging
import sys
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from mcp_tools.calculator import CalculatorError
from mcp_tools.calculator import calculate as safe_calculate
from mcp_tools.corpus import CorpusError, CorpusProvider, CorpusSearchOutput, HttpCorpusProvider
from mcp_tools.settings import McpSettings
from mcp_tools.tables import (
    Aggregate,
    QueryResult,
    Table,
    TableError,
    TableFilter,
    TablesListing,
    build_tables,
    find_table,
    run_query,
    source_of,
)
from observability import span

INSTRUCTIONS = (
    "Tools for working with a private document collection. Use search_corpus to find passages "
    "and cite them by source and page. Use calculate for arithmetic instead of guessing. For "
    "how many, total or average questions over a table (csv, xlsx), call list_tables and then "
    "query_table: search only returns a few passages and cannot count rows."
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


def build_server(settings: McpSettings, provider: CorpusProvider) -> MCPServer:
    server = MCPServer(settings.server_name, instructions=INSTRUCTIONS)

    @server.tool(annotations=READ_ONLY)
    def calculate(
        expression: Annotated[
            str,
            Field(description="Arithmetic only: numbers, + - * / // % ** and parentheses."),
        ],
    ) -> float:
        """Evaluate an arithmetic expression exactly, for example `2 * (3 + 4)`."""
        with span("mcp.tool.calculate", {"mcp.tool": "calculate"}):
            try:
                return safe_calculate(expression)
            except CalculatorError as exc:
                raise ToolError(str(exc)) from None

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def search_corpus(
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=settings.max_query_chars,
                description="What to look for, in natural language.",
            ),
        ],
        top_k: Annotated[int, Field(ge=1, le=20, description="How many passages to return.")] = 5,
        collection: Annotated[
            str | None, Field(description="Collection to search. Leave out for the default.")
        ] = None,
    ) -> CorpusSearchOutput:
        """Search the ingested documents and return the best passages with their sources."""
        with span("mcp.tool.search_corpus", {"mcp.tool": "search_corpus", "mcp.top_k": top_k}):
            try:
                return await provider.search(query, top_k, collection)
            except CorpusError as exc:
                raise ToolError(str(exc)) from None

    async def _load_table(name: str, collection: str | None) -> tuple[Table, bool]:
        """Read just the table's own source. If that finds no such table (a typo, a different
        case), read the whole collection, so the answer is found or the error names the real
        tables instead of saying there are none."""
        got = await provider.chunks(collection, source_of(name))
        tables, cut = build_tables(got.chunks, settings.table_max_rows)
        try:
            return find_table(tables, name), got.truncated or cut
        except TableError:
            got = await provider.chunks(collection, None)
            tables, cut = build_tables(got.chunks, settings.table_max_rows)
            return find_table(tables, name), got.truncated or cut

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def list_tables(
        collection: Annotated[
            str | None, Field(description="Collection to look in. Leave out for the default.")
        ] = None,
    ) -> TablesListing:
        """List the tables (csv, xlsx, tables inside documents) with their columns and row counts.
        Call this first, to see the exact table and column names query_table needs."""
        with span("mcp.tool.list_tables", {"mcp.tool": "list_tables"}) as sp:
            try:
                got = await provider.chunks(collection, None)
            except CorpusError as exc:
                raise ToolError(str(exc)) from None
            tables, cut = build_tables(got.chunks, settings.table_max_rows)
            sp.set_attribute("mcp.tables", len(tables))
            note = ""
            if not tables:
                note = (
                    "No tables found. Tables appear after csv or xlsx files (or documents "
                    "with tables) are ingested."
                )
            elif got.truncated or cut:
                note = (
                    "The collection is large: only part of it was scanned, so some tables "
                    "or rows may be missing."
                )
            return TablesListing(
                collection=collection or settings.default_collection,
                tables=[t.info() for t in tables],
                truncated=got.truncated or cut,
                note=note,
            )

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def query_table(
        table: Annotated[str, Field(description="Table name from list_tables, e.g. tickets.csv.")],
        aggregate: Annotated[
            Aggregate,
            Field(
                description="count rows, count_distinct / sum / avg / min / max of `column`, "
                "or rows to list the matching rows."
            ),
        ] = "count",
        column: Annotated[
            str | None, Field(description="The column to sum, average, min, max or count_distinct.")
        ] = None,
        filters: Annotated[
            list[TableFilter] | None,
            Field(description="Keep only rows matching ALL of these (column, op, value)."),
        ] = None,
        group_by: Annotated[
            list[str] | None, Field(max_length=2, description="Compute the aggregate per group.")
        ] = None,
        select: Annotated[
            list[str] | None, Field(description="With aggregate=rows: which columns to show.")
        ] = None,
        order: Annotated[
            str, Field(description="Order groups by value: desc (default) or asc.")
        ] = "desc",
        limit: Annotated[
            int, Field(ge=1, le=500, description="Most groups or rows to return.")
        ] = 20,
        collection: Annotated[
            str | None, Field(description="Collection to look in. Leave out for the default.")
        ] = None,
    ) -> QueryResult:
        """Count, sum, average, min/max or list rows of a whole table, across ALL its rows.
        Use it for how-many, total and average questions. Search cannot count."""
        with span(
            "mcp.tool.query_table",
            {"mcp.tool": "query_table", "mcp.table": table, "mcp.aggregate": aggregate},
        ) as sp:
            if order not in ("asc", "desc"):
                raise ToolError("order must be 'asc' or 'desc'.")
            try:
                found, incomplete = await _load_table(table, collection)
                result = run_query(
                    found,
                    aggregate=aggregate,
                    column=column,
                    filters=filters,
                    group_by=group_by,
                    select=select,
                    order="asc" if order == "asc" else "desc",
                    limit=limit,
                    max_groups=settings.table_max_groups,
                )
            except CorpusError as exc:
                raise ToolError(str(exc)) from None
            except TableError as exc:
                raise ToolError(str(exc)) from None
            if incomplete:
                result.note = (result.note + " " if result.note else "") + (
                    "WARNING: the table is larger than the read limit, so this result is "
                    "incomplete."
                )
            sp.set_attributes({"mcp.matched_rows": result.matched_rows})
            return result

    @server.resource(
        "corpus://sources",
        name="sources",
        description="The documents ingested into the default collection.",
        mime_type="application/json",
    )
    async def sources() -> str:
        return await _sources_json(provider, None)

    @server.resource(
        "corpus://sources/{collection}",
        name="sources_of_collection",
        description="The documents ingested into one collection.",
        mime_type="application/json",
    )
    async def sources_of(collection: str) -> str:
        return await _sources_json(provider, collection)

    @server.prompt()
    def summarize_with_citations(topic: str) -> str:
        """Ask the model to summarize a topic from the corpus, citing its sources."""
        return (
            f"Summarize what the documents say about: {topic}\n\n"
            "First call search_corpus (a few times with different wording if needed). Use only "
            "what the passages say. After each claim, cite it as [source_name, page N]. If the "
            "passages do not cover the topic, say so instead of guessing."
        )

    return server


async def _sources_json(provider: CorpusProvider, collection: str | None) -> str:
    try:
        items = await provider.sources(collection)
    except CorpusError as exc:
        raise ValueError(str(exc)) from None
    return json.dumps([i.model_dump() for i in items], indent=2)


def main() -> None:
    # stderr only: stdout belongs to the protocol.
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    settings = McpSettings()
    build_server(settings, HttpCorpusProvider(settings)).run("stdio")


if __name__ == "__main__":
    main()
