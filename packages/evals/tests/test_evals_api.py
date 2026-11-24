import json

import pytest
from fastapi.testclient import TestClient

from common import AppSettings, get_app_settings
from evals.app import create_app
from evals.factory import Evals
from observability import shutdown_observability
from observability.settings import ObsSettings

KEY = "test-key-0123456789"
H = {"X-API-Key": KEY}


@pytest.fixture
def client(rig) -> TestClient:
    shutdown_observability()
    app = create_app(
        evals=Evals(service=rig.service),
        obs_settings=ObsSettings(_env_file=None, local_db_path=":memory:"),
    )
    app.dependency_overrides[get_app_settings] = lambda: AppSettings(app_api_key=KEY)
    yield TestClient(app, follow_redirects=False)
    shutdown_observability()


def upload(client, *items, name="demo"):
    data = "\n".join(json.dumps(i) for i in items).encode()
    return client.post(
        "/api/evals/datasets",
        files={"file": ("golden.jsonl", data)},
        data={"name": name},
        headers=H,
    )


QA = [
    {"id": "q1", "question": "Q1?", "relevant_chunk_ids": ["c1"], "reference_answer": "R1"},
    {"id": "q2", "question": "Q2?", "relevant_chunk_ids": ["c2"], "reference_answer": "R2"},
]


def script(rig):
    from evals.models import TargetResponse

    rig.script.replies = {
        "Q1?": TargetResponse(retrieved_chunk_ids=["c9", "c1"], timings={"retrieval_ms": 10.0}),
        "Q2?": TargetResponse(retrieved_chunk_ids=["c2"], timings={"retrieval_ms": 30.0}),
    }


def test_health_is_public_and_the_rest_needs_the_key(client):
    assert client.get("/health").status_code == 200
    protected = [
        ("GET", "/status"), ("GET", "/api/evals/status"), ("GET", "/jobs/x"),
        ("GET", "/api/evals/jobs/x"), ("GET", "/api/evals/datasets"), ("POST", "/api/evals/datasets"),
        ("POST", "/api/evals/datasets/generate"), ("GET", "/api/evals/datasets/x"),
        ("PATCH", "/api/evals/datasets/x/items/y"), ("DELETE", "/api/evals/datasets/x"),
        ("GET", "/api/evals/collections"), ("GET", "/api/evals/targets"),
        ("POST", "/api/evals/runs"), ("POST", "/api/evals/runs/estimate"),
        ("GET", "/api/evals/runs"), ("GET", "/api/evals/runs/x"), ("GET", "/api/evals/runs/compare"),
    ]  # fmt: skip
    for method, path in protected:
        assert client.request(method, path).status_code == 401, path


def test_ui_is_public_and_has_no_secrets(client):
    assert client.get("/").headers["location"] == "/ui/"
    for path in ("/ui/", "/ui/app.js", "/ui/kit/kit.js"):
        r = client.get(path)
        assert r.status_code == 200 and KEY not in r.text
    assert "Evals" in client.get("/ui/").text


def test_upload_review_run_and_read_the_report(client, rig):
    created = upload(client, *QA)
    assert created.status_code == 201
    ds = created.json()
    assert ds["name"] == "demo" and ds["status"] == "ready" and ds["counts"]["accepted"] == 2
    assert client.get(f"/api/evals/datasets/{ds['id']}", headers=H).json()["items"][0]["id"] == "q1"

    script(rig)
    body = {
        "dataset_id": ds["id"],
        "target": {"kind": "preset", "name": "ids_only", "options": {"mode": "dense"}},
        "config": {"ks": [1, 3]},
    }
    est = client.post("/api/evals/runs/estimate", json=body, headers=H).json()
    assert est["questions"] == 2 and est["max_judge_calls"] == 0

    r = client.post("/api/evals/runs", json=body, headers={**H, "Idempotency-Key": "once"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert (
        client.post("/api/evals/runs", json=body, headers={**H, "Idempotency-Key": "once"}).json()[
            "job_id"
        ]
        == job_id
    )

    job = client.get(f"/api/evals/jobs/{job_id}", headers=H).json()
    assert job["state"] == "completed" and job["progress"] == job["total"] == 2
    assert client.get(f"/jobs/{job_id}", headers=H).json() == job  # the standard path too

    run = client.get(f"/api/evals/runs/{job_id}", headers=H).json()
    assert run["report"]["metrics"]["mrr"]["mean"] == 0.75 and len(run["results"]) == 2
    assert run["config"]["judge"]["model"] == "fake-judge"
    assert [x["id"] for x in client.get("/api/evals/runs", headers=H).json()] == [job_id]


def test_compare_endpoint_and_its_route_order(client, rig):
    ds = upload(client, *QA).json()
    script(rig)
    body = {
        "dataset_id": ds["id"],
        "target": {"kind": "preset", "name": "ids_only"},
        "config": {"ks": [1]},
    }
    a = client.post("/api/evals/runs", json=body, headers=H).json()["job_id"]
    rig.script.replies["Q1?"].retrieved_chunk_ids = ["c1"]
    b = client.post("/api/evals/runs", json=body, headers=H).json()["job_id"]
    r = client.get(f"/api/evals/runs/compare?a={a}&b={b}", headers=H)
    assert r.status_code == 200  # "compare" was not mistaken for a run id
    cmp = r.json()
    assert {m["metric"]: m["delta"] for m in cmp["metrics"]}["recall@1"] == 0.5
    assert cmp["changed_questions"][0]["item_id"] == "q1"
    assert client.get("/api/evals/runs/compare", headers=H).status_code == 422
    missing = client.get(f"/api/evals/runs/compare?a={a}&b=nope", headers=H)
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "run_not_found"


def test_generate_review_and_patch(client, rig):
    r = client.post("/api/evals/datasets/generate", json={"collection": "docs", "n": 2}, headers=H)
    assert r.status_code == 202
    job = client.get(f"/api/evals/jobs/{r.json()['job_id']}", headers=H).json()
    ds = client.get(f"/api/evals/datasets/{job['result']['dataset_id']}", headers=H).json()
    assert ds["status"] == "draft" and ds["items"][0]["evidence"][0]["text"]
    item = ds["items"][0]
    patched = client.patch(
        f"/api/evals/datasets/{ds['id']}/items/{item['id']}",
        json={"status": "accepted", "question": "Edited?"}, headers=H,
    )  # fmt: skip
    assert patched.json()["status"] == "accepted" and patched.json()["question"] == "Edited?"
    assert client.get("/api/evals/datasets", headers=H).json()[0]["counts"]["accepted"] == 1
    assert client.delete(f"/api/evals/datasets/{ds['id']}", headers=H).json() == {
        "deleted": ds["id"]
    }
    assert client.get(f"/api/evals/datasets/{ds['id']}", headers=H).status_code == 404


def test_errors_use_the_envelope(client, rig):
    bad = upload(client, {"question": ""})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_dataset"
    ds = upload(client, *QA).json()
    cases = [
        ({"dataset_id": ds["id"], "target": {"kind": "preset", "name": "magic"}}, 422, "unknown_target"),
        ({"dataset_id": "nope", "target": {"kind": "preset", "name": "ids_only"}}, 404, "dataset_not_found"),
        ({"dataset_id": ds["id"], "target": {"kind": "http", "url": "https://evil.example.com"}}, 403, "target_not_allowed"),
        ({"dataset_id": ds["id"], "target": {"kind": "preset", "name": "ids_only"}, "config": {"ks": [0]}}, 422, "validation_error"),
        ({"dataset_id": ds["id"], "target": {"kind": "ftp"}}, 422, "validation_error"),
    ]  # fmt: skip
    for body, status, code in cases:
        r = client.post("/api/evals/runs", json=body, headers=H)
        assert (r.status_code, r.json()["error"]["code"]) == (status, code), body
    assert client.get("/api/evals/runs/nope", headers=H).json()["error"]["code"] == "run_not_found"
    assert client.get("/api/evals/jobs/nope", headers=H).json()["error"]["code"] == "job_not_found"
    huge = client.post(
        "/api/evals/datasets/generate", json={"collection": "docs", "n": 100000}, headers=H
    )
    assert huge.status_code == 422


def test_info_endpoints(client):
    assert client.get("/api/evals/collections", headers=H).json() == ["docs"]
    t = client.get("/api/evals/targets", headers=H).json()
    assert {"scripted", "ids_only"} <= set(t["presets"]) and t["allowed_target_hosts"]
    assert (
        client.get("/status", headers=H).json() == client.get("/api/evals/status", headers=H).json()
    )


def test_the_metric_catalog_says_what_each_metric_needs(client):
    assert client.get("/api/evals/metrics").status_code == 401
    metrics = {m["id"]: m for m in client.get("/api/evals/metrics", headers=H).json()}
    assert {"recall", "precision", "hit_rate", "mrr", "ndcg", "faithfulness", "completeness",
            "answer_accuracy", "source_accuracy"} == set(metrics)  # fmt: skip
    assert metrics["recall"]["needs_dataset"] == ["relevant_chunk_ids"]
    assert (
        metrics["completeness"]["needs_dataset"] == ["reference_answer"]
        and metrics["completeness"]["judged"]
    )
    assert (
        metrics["faithfulness"]["needs_dataset"] == []
        and "contexts" in metrics["faithfulness"]["needs_target"]
    )
    assert metrics["source_accuracy"]["needs_target"] == ["cited_chunk_ids"]


def test_a_run_with_an_unknown_metric_is_a_422_before_it_starts(client):
    body = {
        "dataset_id": "x",
        "target": {"kind": "preset", "name": "ids_only"},
        "config": {"metrics": ["recall", "bogus"]},
    }
    r = client.post("/api/evals/runs", json=body, headers=H)
    assert r.status_code == 422
