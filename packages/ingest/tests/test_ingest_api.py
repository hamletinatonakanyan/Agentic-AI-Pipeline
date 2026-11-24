import json

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from common import AppSettings, get_app_settings
from ingest.app import create_app
from ingest.factory import build_ingest
from ingest.settings import IngestSettings
from ingest.testing import FakeDense, FakeSparse
from observability import shutdown_observability
from observability.settings import ObsSettings

KEY = "test-key-0123456789"
H = {"X-API-Key": KEY}
TEXT = "\n\n".join(
    f"Paragraph {i} about topic {i} and more words to fill it." * 3 for i in range(8)
)


def make_client(client=None, **settings) -> TestClient:
    ingest = build_ingest(
        IngestSettings(
            _env_file=None, manifest_path=":memory:", chunk_size=40, chunk_overlap=8, **settings
        ),
        client=client or QdrantClient(":memory:"),
        dense=FakeDense(),
        sparse=FakeSparse(),
        inline_jobs=True,
    )
    app = create_app(
        ingest=ingest, obs_settings=ObsSettings(_env_file=None, local_db_path=":memory:")
    )
    app.dependency_overrides[get_app_settings] = lambda: AppSettings(app_api_key=KEY)
    return TestClient(app, follow_redirects=False)


@pytest.fixture(autouse=True)
def _obs_reset():
    shutdown_observability()
    yield
    shutdown_observability()


@pytest.fixture
def client() -> TestClient:
    return make_client()


def upload(client, files, options=None, headers=None):
    parts = [("files", (name, data)) for name, data in files.items()]
    data = {"options": json.dumps(options)} if options is not None else {}
    return client.post(
        "/api/ingest/files", files=parts, data=data, headers={**H, **(headers or {})}
    )


def job(client, job_id):
    return client.get(f"/api/ingest/jobs/{job_id}", headers=H).json()


# ----- access ----------------------------------------------------------------------------------------


def test_health_is_public_and_everything_else_needs_the_key(client):
    assert client.get("/health").status_code == 200
    protected = [
        ("GET", "/status"), ("GET", "/api/ingest/status"), ("GET", "/jobs/x"),
        ("GET", "/api/ingest/jobs/x"), ("GET", "/api/ingest/collections"),
        ("GET", "/api/ingest/collections/docs/sources"),
        ("DELETE", "/api/ingest/collections/docs/sources/x"),
        ("POST", "/api/ingest/files"), ("POST", "/api/ingest/path"),
        ("POST", "/api/ingest/chunks/preview"),
    ]  # fmt: skip
    for method, path in protected:
        assert client.request(method, path).status_code == 401, path


def test_ui_is_public_and_has_no_secrets(client):
    assert client.get("/").headers["location"] == "/ui/"
    for path in ("/ui/", "/ui/app.js", "/ui/kit/kit.js"):
        r = client.get(path)
        assert r.status_code == 200 and KEY not in r.text
    assert "Ingest" in client.get("/ui/").text


def test_status_describes_defaults_for_the_ui(client):
    body = client.get("/api/ingest/status", headers=H).json()
    assert body == client.get("/status", headers=H).json()
    assert body["embed_model"] == "google/embeddinggemma-300m"
    assert "none" in body["languages"] and "english" in body["languages"]
    assert ".xlsx" in body["supported_extensions"] and ".html" not in body["supported_extensions"]
    assert body["llamaparse_configured"] is False
    assert "llx" not in json.dumps(body)


# ----- the main flow ----------------------------------------------------------------------------------


def test_upload_ingest_rerun_and_delete(client):
    r = upload(client, {"a.txt": TEXT.encode(), "b.md": b"# Title\n\nSome markdown."})
    assert r.status_code == 202
    accepted = r.json()
    assert accepted["state"] in ("queued", "completed")

    done = job(client, accepted["job_id"])
    assert done["state"] == "completed"
    assert (done["report"]["added"], done["report"]["failed"]) == (2, [])
    assert client.get(f"/jobs/{accepted['job_id']}", headers=H).json() == done  # standard path too

    cols = client.get("/api/ingest/collections", headers=H).json()
    assert cols[0]["name"] == "docs" and cols[0]["points"] == done["report"]["chunks_written"]
    assert cols[0]["spec"]["embedding_model"] == "fake-dense"

    sources = client.get("/api/ingest/collections/docs/sources", headers=H).json()
    assert {s["source_name"] for s in sources} == {"a.txt", "b.md"}

    again = job(
        client,
        upload(client, {"a.txt": TEXT.encode(), "b.md": b"# Title\n\nSome markdown."}).json()[
            "job_id"
        ],
    )
    assert (again["report"]["added"], again["report"]["skipped"]) == (0, 2)

    a = next(s for s in sources if s["source_name"] == "a.txt")
    gone = client.delete(f"/api/ingest/collections/docs/sources/{a['source_id']}", headers=H)
    assert gone.json()["chunks_deleted"] == a["chunk_count"]
    assert (
        client.delete(
            f"/api/ingest/collections/docs/sources/{a['source_id']}", headers=H
        ).status_code
        == 404
    )


def test_options_are_applied(client):
    opts = {
        "collection": "custom",
        "chunking": {"strategy": "fixed", "chunk_size": 20, "overlap": 0},
        "sparse_language": "german",
        "metadata": {"year": 2025},
    }
    done = job(client, upload(client, {"a.txt": TEXT.encode()}, opts).json()["job_id"])
    assert done["report"]["collection"] == "custom"
    spec = client.get("/api/ingest/collections", headers=H).json()[0]["spec"]
    assert spec["sparse_language"] == "german"
    src = client.get("/api/ingest/collections/custom/sources", headers=H).json()[0]
    assert src["metadata"] == {"year": 2025} and src["chunk_count"] > 5


def test_same_idempotency_key_runs_once(client):
    one = upload(client, {"a.txt": TEXT.encode()}, headers={"Idempotency-Key": "abc"}).json()
    two = upload(client, {"a.txt": TEXT.encode()}, headers={"Idempotency-Key": "abc"}).json()
    assert one["job_id"] == two["job_id"]
    assert job(client, one["job_id"])["report"]["added"] == 1


def test_a_corrupt_file_is_reported_in_the_job_not_as_an_http_error(client):
    done = job(
        client, upload(client, {"bad.docx": b"nope", "ok.txt": b"fine text"}).json()["job_id"]
    )
    assert done["state"] == "completed"
    assert done["report"]["added"] == 1
    assert done["report"]["failed"][0]["source"] == "bad.docx"


def test_changing_language_on_an_existing_collection_fails_the_job_clearly(client):
    upload(client, {"a.txt": TEXT.encode()})
    done = job(
        client, upload(client, {"b.txt": b"more"}, {"sparse_language": "german"}).json()["job_id"]
    )
    assert done["state"] == "failed"
    assert "collection_mismatch" in done["error"]


# ----- validation -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [{"sparse_language": "klingon"}, {"collection": "bad name"}, {"chunking": {"chunk_size": 4}},
     {"chunking": {"chunk_size": 50, "overlap": 50}}],
)  # fmt: skip
def test_bad_options_are_a_422_envelope(client, options):
    r = upload(client, {"a.txt": b"x"}, options)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_options"
    assert client.get("/api/ingest/collections", headers=H).json() == []  # nothing was started


def test_options_that_are_not_json_are_a_422(client):
    r = client.post(
        "/api/ingest/files",
        files=[("files", ("a.txt", b"x"))],
        data={"options": "{oops"},
        headers=H,
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_options"


def test_unsupported_extension_is_415_and_html_is_not_supported(client):
    for name in ("page.html", "pic.png", "README"):
        r = upload(client, {name: b"x"})
        assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_format"


def test_no_files_is_422(client):
    assert client.post("/api/ingest/files", headers=H).status_code == 422


def test_oversized_upload_is_413():
    client = make_client(max_upload_mb=1)
    r = upload(client, {"big.txt": b"x" * (1024 * 1024 + 1)})
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_unknown_job_is_404(client):
    assert client.get("/api/ingest/jobs/nope", headers=H).json()["error"]["code"] == "job_not_found"


# ----- preview ------------------------------------------------------------------------------------------------


def test_preview_returns_chunks_and_creates_nothing(client):
    r = client.post(
        "/api/ingest/chunks/preview",
        files=[("file", ("a.txt", TEXT.encode()))],
        data={
            "options": json.dumps(
                {"chunking": {"strategy": "recursive", "chunk_size": 30, "overlap": 5}}
            )
        },
        headers=H,
    )
    body = r.json()
    assert r.status_code == 200 and body["source_name"] == "a.txt"
    assert len(body["chunks"]) > 3 and body["chunks"][0]["tokens"] <= 30
    assert client.get("/api/ingest/collections", headers=H).json() == []
    assert client.get("/api/ingest/collections/docs/sources", headers=H).json() == []


# ----- server paths ----------------------------------------------------------------------------------------------


def test_path_ingest_inside_the_allowed_folder(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text(TEXT)
    client = make_client(allowed_paths=str(tmp_path / "docs"))
    r = client.post("/api/ingest/path", json={"path": str(tmp_path / "docs")}, headers=H)
    assert r.status_code == 202
    assert job(client, r.json()["job_id"])["report"]["added"] == 1


def test_path_outside_the_allowed_folder_is_403_before_any_job(tmp_path):
    (tmp_path / "docs").mkdir()
    client = make_client(allowed_paths=str(tmp_path / "docs"))
    r = client.post("/api/ingest/path", json={"path": str(tmp_path)}, headers=H)
    assert r.status_code == 403 and r.json()["error"]["code"] == "path_not_allowed"


def test_paths_are_off_without_allowed_folders(client, tmp_path):
    r = client.post("/api/ingest/path", json={"path": str(tmp_path)}, headers=H)
    assert r.status_code == 403 and "INGEST_ALLOWED_PATHS" in r.json()["error"]["message"]


# ----- when Qdrant is down ----------------------------------------------------------------------------------------


def test_qdrant_down_gives_clear_errors():
    down = QdrantClient(url="http://127.0.0.1:9", timeout=1, check_compatibility=False)
    client = make_client(client=down)
    r = client.get("/api/ingest/collections", headers=H)
    assert r.status_code == 503 and r.json()["error"]["code"] == "vector_store_unavailable"
    failed = job(client, upload(client, {"a.txt": b"hello there"}).json()["job_id"])
    assert failed["state"] == "failed" and "make up" in failed["error"]
    # Preview needs no Qdrant at all.
    ok = client.post(
        "/api/ingest/chunks/preview", files=[("file", ("a.txt", b"hello there"))], headers=H
    )
    assert ok.status_code == 200


# ----- deleting a whole collection (a "domain") -----------------------------------------------------------


def test_deleting_a_collection_removes_its_chunks_and_lets_the_same_files_be_ingested_again(client):
    first = job(
        client, upload(client, {"a.txt": TEXT.encode()}, {"collection": "beta"}).json()["job_id"]
    )
    other = job(
        client, upload(client, {"a.txt": TEXT.encode()}, {"collection": "acme"}).json()["job_id"]
    )
    written = first["report"]["chunks_written"]
    assert written and other["report"]["added"] == 1

    # the name must be repeated, in the request, to confirm
    for query in ("", "?confirm=wrong"):
        r = client.delete(f"/api/ingest/collections/beta{query}", headers=H)
        assert r.status_code == 422 and r.json()["error"]["code"] == "confirmation_required"
    assert {c["name"] for c in client.get("/api/ingest/collections", headers=H).json()} == {
        "beta",
        "acme",
    }

    r = client.delete("/api/ingest/collections/beta?confirm=beta", headers=H)
    assert r.status_code == 200
    assert r.json() == {"collection": "beta", "chunks_deleted": written, "sources_forgotten": 1}

    cols = {c["name"]: c for c in client.get("/api/ingest/collections", headers=H).json()}
    assert set(cols) == {"acme"} and cols["acme"]["points"] > 0  # the other domain is untouched
    assert client.get("/api/ingest/collections/beta/sources", headers=H).json() == []

    # the manifest forgot it, so the same file is ADDED again, not skipped as "unchanged"
    again = job(
        client, upload(client, {"a.txt": TEXT.encode()}, {"collection": "beta"}).json()["job_id"]
    )
    assert (again["report"]["added"], again["report"]["skipped"]) == (1, 0)


def test_deleting_an_unknown_collection_is_404_and_needs_the_key(client):
    r = client.delete("/api/ingest/collections/nothing?confirm=nothing", headers=H)
    assert r.status_code == 404 and r.json()["error"]["code"] == "collection_not_found"
    assert client.delete("/api/ingest/collections/docs?confirm=docs").status_code == 401


def test_a_collection_that_ingest_did_not_create_is_never_deleted():
    from qdrant_client import models

    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        "someone_elses", vectors_config=models.VectorParams(size=4, distance="Cosine")
    )
    client = make_client(qdrant)
    r = client.delete("/api/ingest/collections/someone_elses?confirm=someone_elses", headers=H)
    assert r.status_code == 409 and r.json()["error"]["code"] == "collection_not_managed"
    assert qdrant.collection_exists("someone_elses")
