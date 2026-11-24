import pytest

from ingest.errors import ParseError
from ingest.models import UploadedFile
from ingest.parsers import FileParser, clean_name
from ingest.parsers.pdf import PdfRouter
from ingest.tables import find_tables


@pytest.fixture
def parser() -> FileParser:
    return FileParser(PdfRouter(threshold=50, tier="fast", mode="local"))


def test_clean_name_drops_directories():
    assert clean_name("a/b/report.pdf") == "report.pdf"
    assert clean_name("C:\\docs\\report.pdf") == "report.pdf"
    assert clean_name("") == "unnamed"


def test_txt_and_md_are_one_page_and_keep_text(parser, make_upload):
    doc = parser.parse(make_upload("notes.txt", "Hello\r\nworld"))
    assert [(p.page_no, p.text) for p in doc.pages] == [(1, "Hello\nworld")]
    assert doc.stats.pages_local == 1
    assert len(doc.content_hash) == 64


def test_non_utf8_text_falls_back_with_a_warning(parser, make_upload):
    doc = parser.parse(UploadedFile(name="old.txt", data="café".encode("latin-1")))
    assert doc.pages[0].text == "café"
    assert "Latin-1" in doc.warnings[0]


def test_empty_text_gives_no_pages_and_a_warning(parser, make_upload):
    doc = parser.parse(make_upload("empty.txt", "  \n"))
    assert doc.pages == [] and "empty" in doc.warnings[0]


def test_markdown_html_tables_become_markdown_tables(parser, make_upload):
    doc = parser.parse(make_upload("t.md", "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"))
    assert "<table" not in doc.pages[0].text
    assert len(find_tables(doc.pages[0].text)) == 1


def test_same_bytes_same_hash_different_bytes_different_hash(parser, make_upload):
    a = parser.parse(make_upload("a.txt", "same"))
    b = parser.parse(make_upload("b.txt", "same"))
    c = parser.parse(make_upload("c.txt", "different"))
    assert a.content_hash == b.content_hash != c.content_hash


def test_csv_becomes_a_markdown_table(parser, make_upload):
    doc = parser.parse(make_upload("data.csv", 'id,name,note\n1,Ann,"has, comma"\n2,Bob,\n'))
    assert doc.pages[0].text.splitlines() == [
        "| id | name | note |",
        "| --- | --- | --- |",
        "| 1 | Ann | has, comma |",
        "| 2 | Bob |  |",
    ]


def test_csv_detects_semicolon_and_tab_delimiters(parser, make_upload):
    semi = parser.parse(make_upload("s.csv", "a;b\n1;2\n"))
    tab = parser.parse(make_upload("t.csv", "a\tb\n1\t2\n"))
    assert semi.pages[0].text.splitlines()[0] == "| a | b |"
    assert tab.pages[0].text.splitlines()[0] == "| a | b |"


def test_empty_csv_has_no_pages(parser, make_upload):
    assert parser.parse(make_upload("e.csv", "")).pages == []


def test_xlsx_each_visible_nonempty_sheet_is_a_page(parser, xlsx_bytes, make_upload):
    doc = parser.parse(make_upload("book.xlsx", xlsx_bytes))
    assert [(p.page_no, p.label) for p in doc.pages] == [(1, "Sales"), (2, "Notes")]
    assert doc.pages[0].text.splitlines() == [
        "## Sales",
        "",
        "| Region | Units | Price |",
        "| --- | --- | --- |",
        "| North | 10 | 2.5 |",
        "| South | 20 | 3 |",  # 20.0 is shown as 20
    ]
    assert "pipe \\| inside" in doc.pages[1].text
    assert any("Empty" in w for w in doc.warnings)
    assert any("Secret" in w and "hidden" in w for w in doc.warnings)


def test_docx_keeps_headings_paragraphs_and_tables_in_order(parser, docx_bytes, make_upload):
    text = parser.parse(make_upload("r.docx", docx_bytes)).pages[0].text
    assert text.index("# Quarterly report") < text.index("Intro paragraph")
    assert text.index("Intro paragraph") < text.index("## Numbers")
    assert text.index("## Numbers") < text.index("| Region | Sales |")
    assert text.index("| South | 20 |") < text.index("Closing remarks.")
    assert len(find_tables(text)) == 1


def test_corrupt_files_raise_parse_error(parser, make_upload):
    for name in ("bad.docx", "bad.xlsx", "bad.pdf"):
        with pytest.raises(ParseError):
            parser.parse(make_upload(name, b"this is not a real file"))


def test_unsupported_type_is_a_parse_error(parser, make_upload):
    with pytest.raises(ParseError, match="unsupported"):
        parser.parse(make_upload("page.html", "<p>x</p>"))
    with pytest.raises(ParseError, match="unsupported"):
        parser.parse(make_upload("README", "x"))


def test_extension_is_case_insensitive(parser, make_upload):
    assert parser.parse(make_upload("NOTES.TXT", "hi")).pages[0].text == "hi"
