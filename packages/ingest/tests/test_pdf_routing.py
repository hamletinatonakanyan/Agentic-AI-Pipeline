import pytest

from ingest.errors import ParseError
from ingest.models import UploadedFile
from ingest.parsers import FileParser
from ingest.parsers.pdf import PdfRouter

LONG = "This page has a real text layer with plenty of characters to pass the threshold."


def parse(pdf_bytes, router):
    return FileParser(router).parse(UploadedFile(name="doc.pdf", data=pdf_bytes))


def router(remote=None, cache=None, mode="llamaparse", threshold=50):
    return PdfRouter(threshold=threshold, tier="fast", mode=mode, remote=remote, cache=cache)


def test_text_pages_stay_local_and_never_touch_the_remote(make_pdf, remote_cls, cache):
    remote = remote_cls()
    doc = parse(make_pdf([LONG, LONG + " Second."]), router(remote, cache))
    assert [(p.page_no, p.parser) for p in doc.pages] == [(1, "local"), (2, "local")]
    assert remote.calls == []
    assert (doc.stats.pages_local, doc.stats.pages_llamaparse) == (2, 0)
    assert "real text layer" in doc.pages[0].text


def test_only_scanned_pages_are_sent_in_one_request(make_pdf, remote_cls, cache):
    remote = remote_cls()
    doc = parse(make_pdf([LONG, None, LONG, None, "tiny"]), router(remote, cache))
    assert remote.calls == [[2, 4, 5]]  # blank pages and a page under the threshold
    assert [(p.page_no, p.parser) for p in doc.pages] == [
        (1, "local"),
        (2, "llamaparse"),
        (3, "local"),
        (4, "llamaparse"),
        (5, "llamaparse"),
    ]
    assert doc.pages[1].text == "Scanned text of page 2"
    assert (doc.stats.pages_local, doc.stats.pages_llamaparse) == (2, 3)


def test_a_scanned_page_is_sent_once_then_served_from_cache(make_pdf, remote_cls, cache):
    remote = remote_cls()
    pdf = make_pdf([LONG, None])
    first = parse(pdf, router(remote, cache))
    second = parse(pdf, router(remote, cache))
    assert len(remote.calls) == 1
    assert (first.stats.pages_llamaparse, first.stats.pages_cached) == (1, 0)
    assert (second.stats.pages_llamaparse, second.stats.pages_cached) == (0, 1)
    assert second.pages[1].cached is True
    assert second.pages[1].text == first.pages[1].text


def test_cache_is_per_tier(make_pdf, remote_cls, cache):
    remote = remote_cls()
    pdf = make_pdf([None])
    parse(pdf, PdfRouter(50, "fast", remote=remote, cache=cache))
    parse(pdf, PdfRouter(50, "agentic", remote=remote, cache=cache))
    assert len(remote.calls) == 2


def test_only_the_new_pages_are_sent_when_some_are_cached(make_pdf, remote_cls, cache):
    remote = remote_cls()
    pdf = make_pdf([None, None, None])
    parse(pdf, router(remote, cache))
    # Forget page 2 only.
    cache.data = {k: v for k, v in cache.data.items() if k[1] != 2}
    remote.calls.clear()
    doc = parse(pdf, router(remote, cache))
    assert remote.calls == [[2]]
    assert (doc.stats.pages_llamaparse, doc.stats.pages_cached) == (1, 2)


def test_blank_scans_are_cached_too_and_reported(make_pdf, remote_cls, cache):
    remote = remote_cls({1: "   "})
    pdf = make_pdf([None])
    doc = parse(pdf, router(remote, cache))
    assert doc.pages == [] and doc.stats.pages_skipped == 1
    assert "page 1: no text found" in doc.warnings
    parse(pdf, router(remote, cache))
    assert len(remote.calls) == 1  # the empty result was cached, so no second send


def test_local_mode_skips_scanned_pages_with_a_warning(make_pdf, remote_cls, cache):
    remote = remote_cls()
    doc = parse(make_pdf([LONG, None]), router(remote, cache, mode="local"))
    assert remote.calls == []
    assert [p.page_no for p in doc.pages] == [1]
    assert doc.stats.pages_skipped == 1
    assert "INGEST_PDF_PARSER=local" in doc.warnings[0]


def test_no_remote_configured_skips_with_a_warning_instead_of_failing(make_pdf, cache):
    doc = parse(make_pdf([LONG, None]), router(None, cache))
    assert [p.page_no for p in doc.pages] == [1]
    assert "LLAMA_CLOUD_API_KEY is not set" in doc.warnings[0]


def test_remote_failure_fails_the_file_but_not_other_work(make_pdf, remote_cls, cache):
    remote = remote_cls(fail=ParseError("LlamaParser returned HTTP 500"))
    with pytest.raises(ParseError, match="HTTP 500"):
        parse(make_pdf([None]), router(remote, cache))
    assert cache.data == {}  # nothing half-cached


def test_threshold_decides_what_counts_as_scanned(make_pdf, remote_cls, cache):
    remote = remote_cls()
    parse(make_pdf(["short text"]), router(remote, cache, threshold=5))
    assert remote.calls == []
    parse(make_pdf(["short text"]), router(remote, cache, threshold=500))
    assert remote.calls == [[1]]


def test_html_tables_from_the_remote_become_markdown(make_pdf, remote_cls, cache):
    remote = remote_cls({1: "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"})
    doc = parse(make_pdf([None]), router(remote, cache))
    assert "<table" not in doc.pages[0].text
    assert "| A |" in doc.pages[0].text


def test_preview_mode_never_calls_the_remote_but_uses_the_cache(make_pdf, remote_cls, cache):
    remote = remote_cls()
    pdf = make_pdf([LONG, None, None])
    r = router(remote, cache)
    # Nothing cached yet: preview skips both scanned pages and sends nothing.
    first = FileParser(r).parse(UploadedFile(name="d.pdf", data=pdf), allow_remote=False)
    assert remote.calls == []
    assert [p.page_no for p in first.pages] == [1]
    assert first.stats.pages_skipped == 2
    assert "not previewed" in first.warnings[0]
    # After a real run the cache is warm, and preview shows those pages without sending.
    parse(pdf, r)
    remote.calls.clear()
    again = FileParser(r).parse(UploadedFile(name="d.pdf", data=pdf), allow_remote=False)
    assert remote.calls == [] and [p.page_no for p in again.pages] == [1, 2, 3]
