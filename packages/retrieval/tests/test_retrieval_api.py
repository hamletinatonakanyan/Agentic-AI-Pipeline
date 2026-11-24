import pytest
from fastapi.testclient import TestClient

from common import AppSettings, get_app_settings
from observability import shutdown_observability
from observability.settings import ObsSettings
from retrieval.app import create_app
from retrieval.factory import Retrieval
from retrieval.testing import DOCS, point_id

KEY = "test-key-0123456789"
H = {"X-API-Key": KEY}


@pytest.fixture
def client(rig) -> TestClient:
    shutdown_observability()
    app = create_app(
        retrieval=Retrieval(service=rig.service),
        obs_settings=ObsSettings(_env_file=None, local_db_path=":memory:"),
    )
    app.dependency_overrides[get_app_settings] = lambda: AppSettings(app_api_key=KEY)
    yield TestClient(app, follow_redirects=False)
    shutdown_observability()


def test_health_is_public_and_the_rest_needs_the_key(client):
    assert client.get("/health").status_code == 200
    for method, path in [
        ("GET", "/status"), ("GET", "/api/retrieval/status"), ("GET", "/api/retrieval/strategies"),
        ("GET", "/api/retrieval/collections"), ("POST", "/api/retrieval/retrieve"),
        ("POST", "/api/retrieval/rewrite"), ("POST", "/api/retrieval/rerank"),
    ]:  # fmt: skip
        assert client.request(method, path).status_code == 401, path


def test_ui_is_public_and_has_no_secrets(client):
    assert client.get("/").headers["location"] == "/ui/"
    for path in ("/ui/", "/ui/app.js", "/ui/kit/kit.js"):
        r = client.get(path)
        assert r.status_code == 200 and KEY not in r.text
    assert "Retrieval" in client.get("/ui/").text


def test_retrieve(client):
    body = {"query": DOCS[2][3], "mode": "hybrid", "top_k": 2}
    r = client.post("/api/retrieval/retrieve", json=body, headers=H)
    assert r.status_code == 200
    data = r.json()
    assert data["hits"][0]["chunk_id"] == point_id(3) and len(data["hits"]) == 2
    assert data["mode"] == "hybrid" and data["timings"]["total_ms"] > 0
    assert data["trace_id"]


def test_retrieve_with_everything_on(client):
    body = {
        "query": "dog daily walk", "mode": "hybrid", "rewrite": "multi_query", "rerank": True,
        "top_k": 3, "candidates": 8, "filters": {"metadata.year": {"gte": 2022}},
    }  # fmt: skip
    data = client.post("/api/retrieval/retrieve", json=body, headers=H).json()
    assert len(data["rewritten_queries"]) == 3 and data["rerank"] is True
    assert all(h["rerank_score"] is not None for h in data["hits"])


def test_validation_errors_use_the_envelope(client):
    for body in (
        {},
        {"query": ""},
        {"query": "x", "mode": "fuzzy"},
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 10, "candidates": 5},
        {"query": "x", "rewrite": "magic"},
    ):
        r = client.post("/api/retrieval/retrieve", json=body, headers=H)
        assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error", body


def test_domain_errors_use_the_envelope(client):
    r = client.post("/api/retrieval/retrieve", json={"query": "x", "collection": "nope"}, headers=H)
    assert r.status_code == 404 and r.json()["error"]["code"] == "collection_not_found"
    r = client.post(
        "/api/retrieval/retrieve", json={"query": "x", "filters": {"a b": 1}}, headers=H
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_filter"


def test_rewrite_endpoint(client):
    r = client.post("/api/retrieval/rewrite", json={"query": "why", "strategy": "hyde"}, headers=H)
    assert r.status_code == 200 and r.json()["rewritten_queries"][0] == "why"


def test_rerank_endpoint(client):
    body = {
        "query": "cat purr",
        "candidates": [{"text": "nothing"}, {"text": "cat purr loudly", "chunk_id": "x"}],
        "top_n": 1,
    }
    items = client.post("/api/retrieval/rerank", json=body, headers=H).json()["items"]
    assert len(items) == 1 and items[0]["chunk_id"] == "x" and items[0]["original_rank"] == 2
    bad = client.post("/api/retrieval/rerank", json={"query": "q", "candidates": []}, headers=H)
    assert bad.status_code == 422


def test_strategies_and_collections_and_status(client):
    s = client.get("/api/retrieval/strategies", headers=H).json()
    assert s["modes"] == ["dense", "sparse", "hybrid"] and s["collections"][0]["name"] == "docs"
    assert client.get("/api/retrieval/collections", headers=H).json()[0]["points"] == len(DOCS)
    assert (
        client.get("/status", headers=H).json()
        == client.get("/api/retrieval/status", headers=H).json()
    )


# ----- reading stored chunks (for tools that need every row of a table) ---------------------------------


def test_chunks_are_listed_page_by_page_until_the_cursor_ends(client):
    seen, after, pages = [], None, 0
    while True:
        url = "/api/retrieval/collections/docs/chunks?limit=3" + (
            f"&after={after}" if after else ""
        )
        page = client.get(url, headers=H).json()
        seen += page["chunks"]
        pages += 1
        after = page["next"]
        if not after:
            break
    assert pages == 3 and len(seen) == len(DOCS)
    assert len({c["chunk_id"] for c in seen}) == len(DOCS)  # no chunk twice, none missing
    assert {c["source_name"] for c in seen} == {s for s, *_ in DOCS}
    cat = next(c for c in seen if c["source_name"] == "cats.txt" and c["page"] == 2)
    assert cat["text"].startswith("A cat sleeps") and cat["chunk_index"] == 2


def test_chunks_can_be_limited_to_one_source_and_need_the_key(client):
    page = client.get(
        "/api/retrieval/collections/docs/chunks?source_name=dogs.txt", headers=H
    ).json()
    assert [c["page"] for c in sorted(page["chunks"], key=lambda c: c["page"])] == [1, 2]
    assert page["next"] is None
    nothing = client.get("/api/retrieval/collections/docs/chunks?source_name=nope.txt", headers=H)
    assert nothing.json() == {"chunks": [], "next": None}
    assert client.get("/api/retrieval/collections/docs/chunks").status_code == 401


def test_chunks_of_a_missing_collection_is_404_not_an_empty_list(client):
    r = client.get("/api/retrieval/collections/typo/chunks", headers=H)
    assert r.status_code == 404 and r.json()["error"]["code"] == "collection_not_found"
    assert (
        client.get("/api/retrieval/collections/docs/chunks?limit=0", headers=H).status_code == 422
    )
