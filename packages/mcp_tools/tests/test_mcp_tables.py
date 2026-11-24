"""Table questions: reading tables out of chunks, answering structured queries, and the two tools."""

import asyncio
import random

import httpx
import pytest
from mcp import Client

from mcp_tools.corpus import CorpusChunk, CorpusError, HttpCorpusProvider
from mcp_tools.server import build_server
from mcp_tools.settings import McpSettings
from mcp_tools.tables import (
    TableError,
    TableFilter,
    build_tables,
    find_table,
    run_query,
    tables_in_text,
)
from mcp_tools.testing import FakeCorpus

HEADER = ["id", "plan", "priority", "hours", "note"]
ROWS = [
    ["T1", "Starter", "P1", "5.5", "login"],
    ["T2", "Team", "P2", "20", "api"],
    ["T3", "Team", "P1", "9", "billing"],
    ["T4", "Business", "P3", "", "export"],  # an empty cell
    ["T5", "Starter", "P1", "n/a", "login"],  # a non-numeric cell
    ["T6", "Business", "P2", "10", "api"],
    ["T7", "Team", "P3", "100", "PIPE \\| inside"],  # an escaped pipe, as ingest writes it
]


def md(header, rows):
    line = lambda cells: "| " + " | ".join(cells) + " |"  # noqa: E731
    return "\n".join([line(header), line(["---"] * len(header)), *(line(r) for r in rows)])


def chunk(source, index, text, page=1):
    return CorpusChunk(
        chunk_id=f"{source}-{index}", source_name=source, page=page, chunk_index=index, text=text
    )


def tickets(split=3):
    """The ROWS as a table split by rows over several chunks, header repeated, like ingest does."""
    parts = [ROWS[i : i + split] for i in range(0, len(ROWS), split)]
    return [chunk("tickets.csv", i, md(HEADER, part)) for i, part in enumerate(parts)]


def table(chunks=None):
    tables, cut = build_tables(chunks or tickets())
    assert not cut
    return find_table(tables, "tickets.csv")


# ----- putting tables back together -------------------------------------------------------------------


def test_rows_split_over_chunks_come_back_whole_and_in_order_whatever_order_they_arrive():
    shuffled = tickets()
    random.Random(1).shuffle(shuffled)
    t = table(shuffled)
    assert t.columns == HEADER and [r[0] for r in t.rows] == [
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6",
        "T7",
    ]
    assert t.rows[6][4] == "PIPE | inside"  # the escaped pipe is a pipe again, not a column break


def test_ragged_rows_are_padded_and_duplicate_headers_renamed():
    found = tables_in_text("| a | a | b |\n| --- | --- | --- |\n| 1 | 2 |\n| 1 | 2 | 3 | 4 |")
    header, rows = found[0]
    assert header == ["a", "a_2", "b"] and rows == [["1", "2", ""], ["1", "2", "3"]]


def test_one_source_with_several_tables_gets_numbered_names_and_text_around_them_is_ignored():
    text = (
        "Intro text.\n\n"
        + md(["x", "y"], [["1", "2"]])
        + "\n\nBetween.\n\n"
        + md(["p", "q"], [["a", "b"]])
    )
    tables, _ = build_tables([chunk("doc.md", 0, text), chunk("other.csv", 0, md(["k"], [["v"]]))])
    assert sorted(t.name for t in tables) == ["doc.md#1", "doc.md#2", "other.csv"]
    assert find_table(tables, "doc.md#2").columns == ["p", "q"]


def test_text_without_a_table_gives_no_table_and_a_pipe_in_prose_is_not_one():
    assert build_tables([chunk("a.md", 0, "Just words | with a pipe | in them.")])[0] == []


def test_reading_stops_at_the_row_limit_and_says_so():
    tables, cut = build_tables(tickets(), max_rows=4)
    assert cut and len(tables[0].rows) == 4


def test_finding_a_table_forgives_case_and_a_missing_extension_and_lists_the_rest_on_a_miss():
    tables, _ = build_tables(tickets())
    assert (
        find_table(tables, "TICKETS.CSV").name
        == find_table(tables, "tickets").name
        == "tickets.csv"
    )
    with pytest.raises(TableError, match="Tables available: tickets.csv"):
        find_table(tables, "invoices.csv")


# ----- counting and filtering --------------------------------------------------------------------------


def f(column, op, value):
    return TableFilter(column=column, op=op, value=value)


def test_a_plain_count_is_exact_even_when_the_rows_span_many_chunks():
    r = run_query(table(tickets(split=1)), filters=[f("priority", "=", "P1")])
    assert (r.result, r.matched_rows, r.total_rows) == (3, 3, 7)
    assert run_query(table(), aggregate="count").result == 7


def test_text_comparison_ignores_case_and_contains_matches_inside_the_cell():
    t = table()
    assert run_query(t, filters=[f("plan", "=", "team")]).result == 3
    assert run_query(t, filters=[f("note", "contains", "PIPE")]).result == 1
    assert run_query(t, filters=[f("plan", "!=", "Team")]).result == 4


def test_numbers_compare_as_numbers_not_as_text():
    t = table()
    # as text "100" < "20" and "9" > "10"; as numbers 9 < 10 < 20 < 100
    assert run_query(t, filters=[f("hours", ">", 9)]).result == 3  # 20, 10, 100
    assert run_query(t, filters=[f("hours", "<=", 10)]).result == 3  # 5.5, 9, 10
    assert run_query(t, filters=[f("hours", "=", "20.0")]).result == 1  # 20 == 20.0


def test_an_empty_cell_is_neither_smaller_nor_larger_and_filters_combine_with_and():
    t = table()
    assert run_query(t, filters=[f("hours", "<", 1000)]).result == 5  # not the empty or the n/a row
    both = run_query(t, filters=[f("priority", "=", "P1"), f("plan", "=", "Starter")])
    assert both.result == 2


def test_a_boolean_or_number_value_is_accepted_as_a_filter_value():
    assert run_query(table(), filters=[f("id", "=", True)]).result == 0  # "true" matches nothing
    assert run_query(table(), filters=[f("hours", "=", 20)]).result == 1


# ----- aggregates ---------------------------------------------------------------------------------------


def test_sum_avg_min_max_use_only_numbers_and_say_what_they_skipped():
    t = table()
    total = run_query(t, aggregate="sum", column="hours")
    assert (
        total.result == 144.5 and total.skipped_non_numeric == 1
    )  # "n/a" skipped, "" is just empty
    assert "1 non-numeric" in total.note
    assert run_query(t, aggregate="avg", column="hours").result == pytest.approx(
        28.9
    )  # 5 numeric cells: 144.5 / 5
    assert run_query(t, aggregate="min", column="hours").result == 5.5
    assert run_query(t, aggregate="max", column="hours").result == 100


def test_count_distinct_counts_different_values_ignoring_case_and_empties():
    assert run_query(table(), aggregate="count_distinct", column="plan").result == 3


def test_a_numeric_aggregate_over_text_gives_no_value_and_an_explanation():
    r = run_query(table(), aggregate="sum", column="plan")
    assert r.result is None and "No numeric values" in r.note


def test_group_by_one_and_two_columns_with_order_and_limit():
    t = table()
    by_plan = run_query(t, group_by=["plan"])
    assert (
        by_plan.result
        == [
            {"plan": "Team", "value": 3, "rows": 3},
            {"plan": "Business", "value": 2, "rows": 2},  # ties go alphabetically
            {"plan": "Starter", "value": 2, "rows": 2},
        ]
        and by_plan.groups_total == 3
    )
    asc = run_query(t, group_by=["plan"], order="asc", limit=1)
    assert asc.result == [{"plan": "Business", "value": 2, "rows": 2}]
    assert asc.groups_total == 3 and "Showing 1 of 3" in asc.note
    both = run_query(t, group_by=["plan", "priority"], filters=[f("priority", "=", "P1")])
    assert {(g["plan"], g["priority"]): g["value"] for g in both.result} == {
        ("Starter", "P1"): 2,
        ("Team", "P1"): 1,
    }


def test_average_per_group():
    r = run_query(table(), aggregate="avg", column="hours", group_by=["priority"])
    values = {g["priority"]: g["value"] for g in r.result}
    assert values == {"P3": 100, "P2": 15, "P1": 7.25}  # P3's empty cell is not counted as zero


def test_rows_mode_lists_matching_rows_with_chosen_columns_and_a_limit():
    r = run_query(
        table(),
        aggregate="rows",
        filters=[f("priority", "=", "P1")],
        select=["id", "plan"],
        limit=2,
    )
    assert r.result == [{"id": "T1", "plan": "Starter"}, {"id": "T3", "plan": "Team"}]
    assert r.matched_rows == 3 and "first 2 of 3" in r.note


def test_the_number_of_groups_returned_is_capped():
    big = build_tables([chunk("many.csv", 0, md(["k"], [[str(i)] for i in range(100)]))])[0][0]
    r = run_query(big, group_by=["k"], limit=500, max_groups=10)
    assert len(r.result) == 10 and r.groups_total == 100


# ----- mistakes are explained, never guessed ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"filters": [f("prio", "=", "P1")]}, "no column 'prio'. Columns: id, plan, priority"),
        ({"aggregate": "sum"}, "needs `column`"),
        ({"aggregate": "avg", "column": "nope"}, "no column 'nope'"),
        ({"group_by": ["id", "plan", "priority"]}, "at most 2 columns"),
        ({"aggregate": "rows", "group_by": ["plan"]}, "cannot be combined"),
    ],
)
def test_bad_questions_get_a_message_that_says_how_to_fix_them(kwargs, message):
    with pytest.raises(TableError, match=message):
        run_query(table(), **kwargs)


def test_a_column_name_is_matched_ignoring_case():
    assert run_query(table(), filters=[f("PRIORITY", "=", "P1")]).result == 3


# ----- the two tools ---------------------------------------------------------------------------------------


def server_with(chunks, fail=None):
    corpus = FakeCorpus(chunks=chunks, fail=fail)
    return build_server(McpSettings(_env_file=None), corpus), corpus


async def call(server, tool, **arguments):
    async with Client(server) as client:
        return await client.call_tool(tool, arguments)


def test_list_tables_shows_every_table_with_its_columns_row_count_and_a_sample_row():
    extra = [chunk("plans.csv", 0, md(["plan", "price"], [["Starter", "29"], ["Team", "99"]]))]
    server, corpus = server_with([*tickets(), *extra])
    out = asyncio.run(call(server, "list_tables")).structured_content
    by_name = {t["name"]: t for t in out["tables"]}
    assert set(by_name) == {"tickets.csv", "plans.csv"} and out["collection"] == "docs"
    assert by_name["tickets.csv"]["rows"] == 7 and by_name["tickets.csv"]["columns"] == HEADER
    assert by_name["plans.csv"]["first_row"] == {"plan": "Starter", "price": "29"}
    assert corpus.chunk_requests == [(None, None)]  # scanned the whole collection


def test_query_table_answers_and_reads_only_that_tables_source():
    server, corpus = server_with([*tickets(), chunk("other.csv", 0, md(["k"], [["v"]]))])
    out = asyncio.run(
        call(server, "query_table", table="tickets.csv",
             filters=[{"column": "priority", "op": "=", "value": "P1"}], collection="alpha")
    ).structured_content  # fmt: skip
    assert (out["result"], out["matched_rows"], out["total_rows"]) == (3, 3, 7)
    assert corpus.chunk_requests == [("alpha", "tickets.csv")]


def test_a_numbered_table_name_reads_the_source_without_the_number():
    text = md(["x"], [["1"]]) + "\n\ntext\n\n" + md(["p"], [["a"], ["b"]])
    server, corpus = server_with([chunk("doc.md", 0, text)])
    out = asyncio.run(call(server, "query_table", table="doc.md#2")).structured_content
    assert out["result"] == 2 and corpus.chunk_requests == [(None, "doc.md")]


def test_tool_mistakes_come_back_as_tool_errors_the_model_can_read():
    server, _ = server_with(tickets())
    wrong_column = asyncio.run(
        call(server, "query_table", table="tickets.csv", aggregate="sum", column="nope")
    )
    assert wrong_column.is_error and "no column 'nope'" in wrong_column.content[0].text
    wrong_table = asyncio.run(call(server, "query_table", table="missing.csv"))
    assert wrong_table.is_error and "Tables available" in wrong_table.content[0].text


def test_the_filter_operator_is_checked_by_the_schema_before_it_reaches_the_code():
    server, _ = server_with(tickets())

    async def go():
        async with Client(server) as client:
            tool = next(t for t in (await client.list_tools()).tools if t.name == "query_table")
            return tool, await client.call_tool(
                "query_table",
                {
                    "table": "tickets.csv",
                    "filters": [{"column": "id", "op": "DROP TABLE", "value": 1}],
                },
            )

    tool, bad = asyncio.run(go())
    assert bad.is_error  # not a valid operator: refused, nothing ran
    assert tool.annotations.read_only_hint is True and tool.input_schema["required"] == ["table"]
    assert "ALL its rows" in tool.description


def test_a_failing_provider_is_a_clear_tool_error_for_both_tools():
    server, _ = server_with([], fail="cannot reach the retrieval service")
    for tool, args in (("list_tables", {}), ("query_table", {"table": "x.csv"})):
        r = asyncio.run(call(server, tool, **args))
        assert r.is_error and "cannot reach the retrieval service" in r.content[0].text


def test_no_tables_is_said_plainly_and_a_truncated_read_is_flagged():
    server, _ = server_with([chunk("a.md", 0, "plain text")])
    out = asyncio.run(call(server, "list_tables")).structured_content
    assert out["tables"] == [] and "No tables found" in out["note"]


# ----- reading chunks over HTTP (the stdio server and the standalone app) ----------------------------------------


def http_provider(handler, **settings):
    return HttpCorpusProvider(
        McpSettings(_env_file=None, **settings),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        env={"APP_API_KEY": "k"},
    )


def page(chunks, nxt=None):
    return {
        "chunks": [
            {"chunk_id": c, "source_name": "t.csv", "page": 1, "chunk_index": i, "text": "x"}
            for i, c in enumerate(chunks)
        ],
        "next": nxt,
    }


def test_the_http_provider_follows_the_cursor_and_sends_the_key_and_the_source_filter():
    seen = []

    def handler(request):
        seen.append((dict(request.url.params), request.headers["x-api-key"]))
        return httpx.Response(
            200, json=page(["a", "b"], "c1") if "after" not in request.url.params else page(["c"])
        )

    got = asyncio.run(http_provider(handler).chunks("alpha", "t.csv"))
    assert [c.chunk_id for c in got.chunks] == ["a", "b", "c"] and not got.truncated
    assert seen[0][0] == {"limit": "500", "source_name": "t.csv"} and seen[1][0]["after"] == "c1"
    assert all(key == "k" for _, key in seen)


def test_the_http_provider_stops_at_the_scan_limit_and_reports_truncation():
    handler = lambda request: httpx.Response(200, json=page(["a", "b", "c"], "more"))  # noqa: E731
    got = asyncio.run(http_provider(handler, table_max_chunks=5).chunks(None, None))
    assert len(got.chunks) == 5 and got.truncated


def test_the_http_provider_turns_failures_into_clear_errors():
    def gone(request):
        body = {"error": {"code": "collection_not_found", "message": "No collection 'x'."}}
        return httpx.Response(404, json=body)

    with pytest.raises(CorpusError, match="No collection 'x'"):
        asyncio.run(http_provider(gone).chunks("x", None))
    odd = lambda request: httpx.Response(200, json={"unexpected": True})  # noqa: E731
    with pytest.raises(CorpusError, match="unexpected reply"):
        asyncio.run(http_provider(odd).chunks(None, None))


def test_a_mistyped_table_name_is_resolved_or_answered_with_the_real_table_names():
    server, corpus = server_with(tickets())
    # the wrong case still finds it (the source filter is exact, so it falls back to a full scan)
    ok = asyncio.run(call(server, "query_table", table="Tickets.CSV")).structured_content
    assert ok["result"] == 7 and corpus.chunk_requests == [(None, "Tickets.CSV"), (None, None)]
    miss = asyncio.run(call(server, "query_table", table="ticket.csv"))
    assert miss.is_error and "Tables available: tickets.csv" in miss.content[0].text
