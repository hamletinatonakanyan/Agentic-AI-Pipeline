import json

import httpx
import pytest

from ingest.errors import ParseError
from ingest.parsers.llamaparse import LlamaParseClient, page_ranges


def client_for(handler, **kwargs) -> LlamaParseClient:
    return LlamaParseClient(
        api_key="llx-secret-key-123456",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
        **kwargs,
    )


def test_page_ranges_compresses_runs():
    assert page_ranges([1, 2, 3, 5, 7, 8]) == "1-3,5,7-8"
    assert page_ranges([4]) == "4"
    assert page_ranges([3, 1, 2, 2]) == "1-3"


def test_full_flow_uploads_polls_and_returns_markdown_by_page():
    seen = {}
    polls = iter(["PENDING", "RUNNING", "COMPLETED"])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer llx-secret-key-123456"
        if request.method == "POST":
            seen["url"] = str(request.url)
            body = request.content.decode("latin-1")
            seen["has_pdf"] = "%PDF" in body
            cfg = body.split('name="configuration"')[1].split("\r\n\r\n")[1].split("\r\n")[0]
            seen["config"] = json.loads(cfg)
            return httpx.Response(200, json={"id": "pjb-1", "status": "PENDING"})
        assert request.url.params["expand"] == "markdown"
        status = next(polls)
        job = {"id": "pjb-1", "status": status}
        if status != "COMPLETED":
            return httpx.Response(200, json={"job": job})
        pages = [
            {"page_number": 2, "markdown": "Page two", "success": True},
            {"page_number": 5, "markdown": "Page five", "success": True},
        ]
        return httpx.Response(200, json={"job": job, "markdown": {"pages": pages}})

    result = client_for(handler, tier="cost_effective").parse(b"%PDF-fake", [2, 5])
    assert result == {2: "Page two", 5: "Page five"}
    assert seen["url"] == "https://api.cloud.llamaindex.ai/api/v2/parse/upload"
    assert seen["has_pdf"]
    assert seen["config"] == {
        "tier": "cost_effective",
        "version": "latest",
        "page_ranges": {"target_pages": "2,5"},
    }


def result_with(pages):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "j"})
        return httpx.Response(
            200, json={"job": {"status": "COMPLETED"}, "markdown": {"pages": pages}}
        )

    return handler


@pytest.mark.parametrize("numbers", [[1, 2], [0, 1], [10, 11]])
def test_pages_match_by_position_whatever_numbering_the_api_uses(numbers):
    pages = [{"page_number": n, "markdown": f"text {i}"} for i, n in enumerate(numbers)]
    result = client_for(result_with(pages)).parse(b"x", [1, 2])
    assert result == {1: "text 0", 2: "text 1"}


def test_zero_based_numbers_are_shifted_when_a_page_is_missing():
    pages = [{"page_number": 0, "markdown": "a"}]  # asked for pages 1 and 2, got one back
    assert client_for(result_with(pages)).parse(b"x", [1, 2]) == {1: "a"}


def test_failed_pages_are_left_out():
    pages = [
        {"page_number": 1, "markdown": "ok", "success": True},
        {"page_number": 2, "success": False, "error": "boom"},
    ]
    assert client_for(result_with(pages)).parse(b"x", [1, 2]) == {1: "ok"}


def test_pages_that_do_not_match_the_request_are_an_error():
    pages = [{"page_number": 99, "markdown": "?"}]
    with pytest.raises(ParseError, match="do not match"):
        client_for(result_with(pages)).parse(b"x", [1, 2])


def test_job_failure_surfaces_the_error_message():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"id": "j"})
        return httpx.Response(200, json={"job": {"status": "FAILED", "error_message": "bad pdf"}})

    with pytest.raises(ParseError, match="failed: bad pdf"):
        client_for(handler).parse(b"x", [1])


def test_http_errors_never_leak_the_key():
    def handler(request):
        return httpx.Response(401, text="invalid token")

    with pytest.raises(ParseError) as err:
        client_for(handler).parse(b"x", [1])
    assert "401" in str(err.value)
    assert "llx-secret-key-123456" not in str(err.value)


def test_network_errors_become_parse_errors():
    def handler(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(ParseError, match="ConnectError"):
        client_for(handler).parse(b"x", [1])


def test_a_job_that_never_finishes_times_out():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"id": "j"})
        return httpx.Response(200, json={"job": {"status": "RUNNING"}})

    with pytest.raises(ParseError, match="did not finish"):
        client_for(handler, timeout_s=0.0).parse(b"x", [1])


def test_missing_job_id_is_an_error():
    def handler(request):
        return httpx.Response(200, json={"status": "PENDING"})

    with pytest.raises(ParseError, match="job id"):
        client_for(handler).parse(b"x", [1])
