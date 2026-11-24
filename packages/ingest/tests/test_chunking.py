import pytest

from ingest.chunking import chunk_document, chunk_text
from ingest.models import ChunkOptions, Document, PageResult
from ingest.tables import find_tables, html_tables_to_markdown, markdown_table
from ingest.tokenizer import ApproxTokenizer

TOK = ApproxTokenizer()


def opts(strategy="recursive", size=40, overlap=8) -> ChunkOptions:
    return ChunkOptions(strategy=strategy, chunk_size=size, overlap=overlap)


def prose(paragraphs=12, sentences=4) -> str:
    out = []
    for p in range(paragraphs):
        out.append(
            " ".join(
                f"Sentence {p}-{s} talks about topic number {p * 10 + s}." for s in range(sentences)
            )
        )
    return "\n\n".join(out)


# ----- tokenizer ---------------------------------------------------------------------------


def test_tokenizer_is_about_four_characters_per_token_for_english():
    text = "The quick brown fox jumps over the lazy dog. " * 20
    n = TOK.count(text)
    assert 0.2 * len(text) < n < 0.4 * len(text)


def test_tokenizer_counts_each_cjk_character():
    assert TOK.count("日本語のテキスト") == 8


def test_token_spans_slice_the_original_text():
    text = "Hello, wörld! 你好"
    assert [text[a:b] for a, b in TOK.spans(text)][:3] == ["Hell", "o", ","]


# ----- recursive / fixed -----------------------------------------------------------------------


def test_empty_and_blank_text_give_no_chunks():
    assert chunk_text("", opts()) == []
    assert chunk_text("  \n\n  ", opts()) == []


def test_short_text_is_one_chunk():
    pieces = chunk_text("A short note.", opts(size=100))
    assert [p.text for p in pieces] == ["A short note."]


@pytest.mark.parametrize("strategy", ["recursive", "fixed", "markdown"])
def test_chunks_fit_the_size_and_offsets_slice_the_source(strategy):
    text = prose()
    pieces = chunk_text(text, opts(strategy))
    assert len(pieces) > 3
    for p in pieces:
        assert p.tokens <= 40
        assert text[p.start : p.end] == p.text


@pytest.mark.parametrize("strategy", ["recursive", "fixed"])
def test_chunks_cover_all_text_with_overlap_and_no_gaps(strategy):
    text = prose()
    pieces = chunk_text(text, opts(strategy))
    for prev, nxt in zip(pieces, pieces[1:], strict=False):
        assert nxt.start <= prev.end, "gap between chunks"
        assert nxt.start > prev.start, "no progress"
    assert pieces[0].start == 0
    assert pieces[-1].end == len(text.rstrip())


def test_recursive_overlap_repeats_text_from_the_previous_chunk():
    pieces = chunk_text(prose(), opts("recursive", size=40, overlap=10))
    assert any(nxt.start < prev.end for prev, nxt in zip(pieces, pieces[1:], strict=False))


def test_no_overlap_means_chunks_do_not_repeat_text():
    pieces = chunk_text(prose(), opts("recursive", size=40, overlap=0))
    assert all(nxt.start >= prev.end for prev, nxt in zip(pieces, pieces[1:], strict=False))


def test_recursive_prefers_paragraph_boundaries():
    text = "Alpha one two three four five six.\n\nBeta one two three four five six."
    pieces = chunk_text(text, opts("recursive", size=16, overlap=0))
    assert [p.text for p in pieces] == [
        "Alpha one two three four five six.",
        "Beta one two three four five six.",
    ]


def test_overlap_does_not_start_in_the_middle_of_a_word():
    pieces = chunk_text("internationalization " * 60, opts("recursive", size=20, overlap=6))
    for p in pieces:
        assert p.text.startswith("internationalization")


def test_long_text_without_separators_is_hard_split():
    text = "日本語" * 100  # 300 tokens, no spaces or punctuation
    pieces = chunk_text(text, opts("recursive", size=40, overlap=0))
    assert len(pieces) == 8
    assert all(p.tokens <= 40 for p in pieces)
    assert "".join(p.text for p in pieces) == text


def test_cjk_sentences_split_on_fullwidth_punctuation():
    text = "これは最初の文です。これは二番目の文です。これは三番目の文です。"
    pieces = chunk_text(text, opts("recursive", size=16, overlap=0))
    assert all(p.text.endswith("。") for p in pieces)


def test_fixed_makes_windows_of_the_requested_size():
    text = " ".join(f"w{i}" for i in range(200))
    pieces = chunk_text(text, opts("fixed", size=30, overlap=10))
    assert all(p.tokens <= 30 for p in pieces)
    assert pieces[0].tokens == 30


# ----- markdown ---------------------------------------------------------------------------------


MD = """# Guide

Intro text for the guide.

## Setup

Install the thing and run it.

### Details

Fine print goes here.

## Usage

Use the thing every day.
"""


def test_markdown_never_crosses_a_heading_and_records_the_path():
    pieces = chunk_text(MD, opts("markdown", size=200, overlap=0))
    paths = [p.section_path for p in pieces]
    assert paths == ["Guide", "Guide > Setup", "Guide > Setup > Details", "Guide > Usage"]
    for p in pieces:
        assert p.text.count("#") == p.text.count("# ") + p.text.count("## ") + p.text.count("### ")


def test_non_markdown_strategies_still_report_the_section_path():
    pieces = chunk_text(MD, opts("recursive", size=200, overlap=0))
    assert len(pieces) == 1 and pieces[0].section_path == "Guide"


def test_headings_inside_code_fences_are_ignored():
    text = "# Real\n\n```\n# not a heading\n```\n\nBody."
    pieces = chunk_text(text, opts("markdown", size=200, overlap=0))
    assert [p.section_path for p in pieces] == ["Real"]


# ----- tables ---------------------------------------------------------------------------------------


def make_table(rows=30) -> str:
    return markdown_table(
        [["id", "name", "note"], *[[i, f"item {i}", "some longer note text"] for i in range(rows)]]
    )


def test_markdown_table_escapes_pipes_and_newlines_and_pads_rows():
    table = markdown_table([["a", "b"], ["x|y", "line1\nline2"], ["only"]])
    assert table.splitlines() == [
        "| a | b |",
        "| --- | --- |",
        "| x\\|y | line1 line2 |",
        "| only |  |",
    ]


def test_empty_table_is_empty_text():
    assert markdown_table([]) == ""
    assert markdown_table([[None, ""], ["", None]]) == ""


def test_find_tables_locates_the_block():
    text = "Before.\n\n" + make_table(3) + "\n\nAfter."
    ((a, b),) = find_tables(text)
    assert text[a:b].startswith("| id | name | note |")
    assert text[a:b].rstrip().endswith("| 2 | item 2 | some longer note text |")


def test_a_pipe_line_without_a_separator_is_not_a_table():
    assert find_tables("| just | a line |\nnot a separator\n") == []


@pytest.mark.parametrize("strategy", ["recursive", "fixed", "markdown"])
def test_big_tables_split_by_rows_with_the_header_repeated(strategy):
    table = make_table(30)
    pieces = chunk_text(table, opts(strategy, size=60, overlap=8))
    header = "| id | name | note |\n| --- | --- | --- |"
    assert len(pieces) > 3
    for p in pieces:
        assert p.text.startswith(header)
        assert p.tokens <= 60
    body_rows = [line for p in pieces for line in p.text.splitlines()[2:]]
    assert body_rows == table.splitlines()[2:]  # every row exactly once, in order


def test_table_chunk_offsets_cover_its_rows():
    text = "Intro paragraph.\n\n" + make_table(30) + "\n\nOutro."
    pieces = [p for p in chunk_text(text, opts(size=60)) if p.text.startswith("| id")]
    for p in pieces:
        last_row = p.text.splitlines()[-1]
        assert text[p.start : p.end].endswith(last_row)


def test_prose_around_a_table_is_chunked_separately():
    text = "Intro paragraph.\n\n" + make_table(3) + "\n\nOutro paragraph."
    texts = [p.text for p in chunk_text(text, opts(size=200))]
    assert texts[0] == "Intro paragraph."
    assert texts[-1] == "Outro paragraph."
    assert texts[1].startswith("| id")


def test_one_oversized_row_still_gets_its_own_chunk():
    table = markdown_table([["a"], ["word " * 100], ["small"]])
    pieces = chunk_text(table, opts(size=30, overlap=4))
    assert any("word word" in p.text for p in pieces)
    assert pieces[-1].text.endswith("| small |")


def test_header_only_table_is_one_chunk():
    pieces = chunk_text("| a | b |\n| --- | --- |\n", opts())
    assert [p.text for p in pieces] == ["| a | b |\n| --- | --- |"]


def test_html_tables_become_markdown_tables():
    html = (
        "Intro\n<table><thead><tr><th>Name</th><th>Qty</th></tr></thead>"
        "<tbody><tr><td>Apple &amp; pear</td><td>3</td></tr>"
        '<tr><td colspan="2">total<br>row</td></tr></tbody></table>\nOutro'
    )
    out = html_tables_to_markdown(html)
    assert "<table" not in out
    assert "| Name | Qty |" in out
    assert "| Apple & pear | 3 |" in out
    assert "| total row |  |" in out
    assert out.startswith("Intro") and out.rstrip().endswith("Outro")
    assert len(find_tables(out)) == 1


# ----- documents ----------------------------------------------------------------------------------------


def test_chunk_document_carries_page_label_and_parser():
    doc = Document(
        source_name="x.pdf",
        content_hash="h",
        pages=[
            PageResult(page_no=1, text="First page text.", parser="local"),
            PageResult(page_no=2, text="Scanned page text.", parser="llamaparse"),
            PageResult(page_no=3, text="# Sheet\n\n| a |\n| --- |\n| 1 |", label="Sheet"),
        ],
    )
    chunks = chunk_document(doc, opts(size=200))
    assert [(c.page_no, c.parser) for c in chunks] == [
        (1, "local"),
        (2, "llamaparse"),
        (3, "local"),
        (3, "local"),
    ]
    assert chunks[2].label == "Sheet"


# ----- stability: why editing one paragraph leaves most chunks alone ---------------------------------


def test_chunks_far_from_an_edit_are_identical_text():
    paras = [
        f"Paragraph {i}: " + f"this one is about subject number {i} in detail. " * 3
        for i in range(12)
    ]
    before = [p.text for p in chunk_text("\n\n".join(paras), opts("recursive", 40, 0))]
    paras[6] = "Paragraph 6 was replaced by something short."
    after = [p.text for p in chunk_text("\n\n".join(paras), opts("recursive", 40, 0))]
    assert before[:6] == after[:6]  # everything before the edit
    assert before[-6:] == after[-6:]  # and after it
    assert before != after


def test_an_oversized_paragraph_is_split_on_its_own_not_packed_with_neighbours():
    big = " ".join(f"Sentence {i} of the big one." for i in range(20))
    text = f"Short intro.\n\n{big}\n\nShort outro."
    pieces = chunk_text(text, opts("recursive", 40, 0))
    assert pieces[0].text == "Short intro."
    assert pieces[-1].text == "Short outro."


def test_overlap_never_repeats_a_whole_previous_chunk():
    text = "tiny\n\n" + "A longer paragraph with enough words to need its own chunk here. " * 4
    pieces = chunk_text(text, opts("recursive", 40, 30))
    for prev, nxt in zip(pieces, pieces[1:], strict=False):
        assert nxt.start > prev.start
