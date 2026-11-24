import json

from fastapi.testclient import TestClient
from mcp import Client as McpClient
from qdrant_client import QdrantClient

from agent_graph.runtime import build_agent
from agent_graph.settings import AgentSettings
from agent_graph.testing import Brain, FakeEmbedder
from common import AppSettings, get_app_settings
from e2e.app import create_app
from e2e.settings import E2ESettings
from e2e.wiring import InProcessCorpus, InProcessRetriever, McpToolProvider, retrieval_preset
from evals.factory import build_evals
from evals.settings import EvalSettings
from ingest.factory import build_ingest
from ingest.settings import IngestSettings
from ingest.testing import FakeDense, FakeSparse
from llm import LLMSettings
from llm.testing import FakeLLM
from mcp_tools.factory import build_mcp
from mcp_tools.settings import McpSettings
from observability import span
from observability.settings import ObsSettings
from retrieval.factory import build_retrieval
from retrieval.settings import RetrievalSettings
from retrieval.testing import FakeEncoder, FakeReranker, FakeRewriter

KEY = "test-key-0123456789"


class FakeProbe:
    def __init__(self, up: bool) -> None:
        self.up = up

    def is_reachable(self) -> bool:
        return self.up


def make_ingest(client=None, **settings):
    """The real ingest code (HTTP, jobs, manifest, Qdrant adapter) on in-memory Qdrant and fake
    embedders: no Docker, no model download, no files."""
    return build_ingest(
        IngestSettings(
            _env_file=None, manifest_path=":memory:", chunk_size=40, chunk_overlap=8, **settings
        ),
        client=client or QdrantClient(":memory:"),
        dense=FakeDense(),
        sparse=FakeSparse(),
        inline_jobs=True,
    )


def make_retrieval(client=None):
    """The real retrieval code on the given in-memory Qdrant, with fake encoder and reranker."""
    return build_retrieval(
        RetrievalSettings(_env_file=None, top_k=3, candidates=10),
        client=client or QdrantClient(":memory:"),
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        rewriter=FakeRewriter(),
    )


def passage_to_question(prompt: str) -> str:
    """A fake LLM that 'drafts' a question equal to the passage, so the fake dense encoder finds
    exactly the chunk it came from."""
    passage = prompt.split("PASSAGE:\n", 1)[1]
    return json.dumps({"question": passage, "answer": "A reference answer."})


def make_evals(retrieval, client=None, judge=None):
    """The real evals code with an in-memory database, a fake LLM, and the in-process retrieval
    preset: no network, no model."""
    return build_evals(
        EvalSettings(_env_file=None, db_path=":memory:"),
        presets={"retrieval": retrieval_preset(retrieval.service)},
        client=client or QdrantClient(":memory:"),
        llm=FakeLLM(passage_to_question),
        judge=judge,
        inline=True,
    )


def make_mcp(ingest, retrieval, key: str):
    """Our MCP server with the in-process corpus, and a manager that reaches the 'own' server
    in-process (no subprocess), so these tests stay fast. The key guard uses the test key."""
    holder = {}
    mcp = build_mcp(
        McpSettings(_env_file=None, db_path=":memory:"),
        corpus=InProcessCorpus(retrieval.service, ingest.service),
        key_provider=lambda: key,
        client_factory=lambda spec: McpClient(holder["mcp"].server),
    )
    holder["mcp"] = mcp
    return mcp


def make_agent(retrieval, mcp, brain=None, **settings):
    """The real agent graph and service, with a scripted model, in-memory persistence, the real
    in-process adapters over the real retrieval and MCP manager, and a fake embedder."""
    return build_agent(
        AgentSettings(_env_file=None, checkpoint_backend="memory", **settings),
        llm=brain or Brain(),
        retriever=InProcessRetriever(retrieval.service),
        tools=McpToolProvider(mcp.manager),
        embedder=FakeEmbedder(),
    )


def make_client(
    key: str = KEY,
    qdrant_up: bool = True,
    llm_key: str = "",
    shared=None,
    brain=None,
    judge=None,
    **flags,
) -> TestClient:
    """`shared` is one in-memory Qdrant used by ingest, retrieval and evals, for contract tests."""
    retrieval = make_retrieval(shared)
    ingest = make_ingest(shared)
    mcp = make_mcp(ingest, retrieval, key)
    app = create_app(
        settings=E2ESettings(_env_file=None, **flags),
        qdrant=FakeProbe(qdrant_up),
        obs_settings=ObsSettings(_env_file=None, local_db_path=":memory:"),  # never a real file
        ingest=ingest,
        mcp=mcp,
        retrieval=retrieval,
        evals=make_evals(retrieval, shared, judge),
        agent=make_agent(retrieval, mcp, brain),
        llm_settings=LLMSettings(_env_file=None, openai_api_key=llm_key),
    )
    app.dependency_overrides[get_app_settings] = lambda: AppSettings(app_api_key=key)
    return TestClient(app)


def by_name(body: dict) -> dict:
    return {c["name"]: c for c in body["components"]}


def test_health_is_public():
    assert make_client().get("/health").json() == {"ok": True}


def test_status_requires_key():
    client = make_client()
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"X-API-Key": "nope"}).status_code == 401


def test_status_is_503_when_key_not_configured():
    r = make_client(key="").get("/api/status", headers={"X-API-Key": "anything"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "not_configured"


def test_status_reports_qdrant_down_without_failing():
    r = make_client(qdrant_up=False).get("/api/status", headers={"X-API-Key": KEY})
    assert r.status_code == 200
    qdrant = by_name(r.json())["qdrant"]
    assert qdrant["reachable"] is False
    assert "make up" in qdrant["detail"]


def test_status_reports_qdrant_up():
    r = make_client(qdrant_up=True).get("/api/status", headers={"X-API-Key": KEY})
    assert by_name(r.json())["qdrant"]["reachable"] is True


def test_status_never_returns_the_full_key():
    r = make_client().get("/api/status", headers={"X-API-Key": KEY})
    assert KEY not in r.text


def test_every_component_is_listed_and_the_agent_reports_the_missing_key():
    r = make_client().get("/api/status", headers={"X-API-Key": KEY})
    comps = by_name(r.json())
    assert set(comps) >= {
        "access_key",
        "qdrant",
        "ingest",
        "retrieval",
        "evals",
        "mcp",
        "agent",
        "llm",
    }
    # The agent is built, but it needs the model, and the test app has no OPENAI_API_KEY.
    assert comps["agent"]["configured"] is False
    assert comps["agent"]["missing"] == ["OPENAI_API_KEY"]


def test_enable_flag_is_reflected():
    r = make_client(enable_agent=False).get("/api/status", headers={"X-API-Key": KEY})
    assert by_name(r.json())["agent"]["enabled"] is False


def test_ui_shell_and_kit_are_public():
    client = make_client()
    page = client.get("/")
    assert page.status_code == 200
    assert "Agentic Blocks" in page.text
    for path in ("/app.js", "/ui/kit/kit.js", "/ui/kit/kit.css"):
        assert client.get(path).status_code == 200


def test_ui_files_contain_no_secrets():
    client = make_client()
    for path in ("/", "/app.js", "/ui/kit/kit.js"):
        assert KEY not in client.get(path).text


# ----- observability inside e2e -----------------------------------------------------------


def test_observability_is_built_and_has_a_ui_tab():
    comps = by_name(make_client().get("/api/status", headers={"X-API-Key": KEY}).json())
    obs = comps["observability"]
    assert obs["configured"] is True
    assert obs["ui_path"] == "/ui/observability/"
    assert "store: sqlite" in obs["detail"]
    assert comps["agent"]["ui_path"] == "/ui/agent/"


def test_obs_health_is_public_but_status_and_traces_need_the_key():
    client = make_client()
    assert client.get("/api/obs/health").status_code == 200
    assert client.get("/api/obs/status").status_code == 401
    assert client.get("/api/obs/traces").status_code == 401
    assert client.get("/api/obs/traces", headers={"X-API-Key": KEY}).json() == []


def test_a_span_made_anywhere_shows_up_in_the_trace_endpoint():
    client = make_client()
    with span("demo.request"):
        with span("demo.step"):
            pass
    traces = client.get("/api/obs/traces", headers={"X-API-Key": KEY}).json()
    assert [t["root_name"] for t in traces] == ["demo.request"]
    assert traces[0]["span_count"] == 2


def test_obs_ui_is_public_and_has_no_secrets():
    client = make_client()
    for path in ("/ui/observability/", "/ui/observability/app.js"):
        r = client.get(path)
        assert r.status_code == 200
        assert KEY not in r.text


def test_disabling_observability_unmounts_it():
    client = make_client(enable_obs=False)
    assert client.get("/api/obs/health").status_code == 404
    comps = by_name(client.get("/api/status", headers={"X-API-Key": KEY}).json())
    assert comps["observability"]["enabled"] is False
    assert comps["observability"]["ui_path"] is None
    assert "E2E_ENABLE_OBS" in comps["observability"]["detail"]


# ----- ingest inside e2e ---------------------------------------------------------------------------------


def upload_to(client, files, options=None):
    parts = [("files", (n, d)) for n, d in files.items()]
    data = {"options": json.dumps(options)} if options else {}
    return client.post("/api/ingest/files", files=parts, data=data, headers={"X-API-Key": KEY})


def test_ingest_is_built_and_has_a_ui_tab():
    comps = by_name(make_client().get("/api/status", headers={"X-API-Key": KEY}).json())
    assert comps["ingest"]["configured"] is True
    assert comps["ingest"]["ui_path"] == "/ui/ingest/"
    assert "embedding:" in comps["ingest"]["detail"]
    assert "LLAMA_CLOUD_API_KEY not set" in comps["ingest"]["detail"]


def test_ingest_routes_are_mounted_and_protected():
    client = make_client()
    assert client.get("/api/ingest/health").status_code == 200
    for path in ("/api/ingest/status", "/api/ingest/collections", "/api/ingest/jobs/x"):
        assert client.get(path).status_code == 401, path
    assert client.get("/api/ingest/collections", headers={"X-API-Key": KEY}).json() == []


def test_ingest_ui_is_public_and_has_no_secrets():
    client = make_client()
    for path in ("/ui/ingest/", "/ui/ingest/app.js"):
        r = client.get(path)
        assert r.status_code == 200 and KEY not in r.text


def test_ingest_a_file_then_see_its_trace_in_observability():
    client = make_client()
    h = {"X-API-Key": KEY}
    text = "\n\n".join(f"Paragraph {i} about topic {i} with extra words." * 3 for i in range(8))
    job_id = upload_to(client, {"a.txt": text.encode()}).json()["job_id"]
    done = client.get(f"/api/ingest/jobs/{job_id}", headers=h).json()
    assert done["state"] == "completed" and done["report"]["added"] == 1

    # One run is one trace, with its steps as spans.
    traces = client.get("/api/obs/traces?name=ingest", headers=h).json()
    assert [t["root_name"] for t in traces] == ["ingest.run"]
    assert traces[0]["trace_id"] == done["report"]["trace_id"]
    tree = client.get(f"/api/obs/traces/{traces[0]['trace_id']}", headers=h).json()
    run = tree["roots"][0]
    file_span = run["children"][0]
    assert file_span["name"] == "ingest.file"
    assert [c["name"] for c in file_span["children"]] == [
        "ingest.parse", "ingest.chunk", "ingest.embed", "ingest.upsert",
    ]  # fmt: skip

    # Re-running the same file is a no-op (skipped) and is traced too.
    again = upload_to(client, {"a.txt": text.encode()}).json()["job_id"]
    report = client.get(f"/api/ingest/jobs/{again}", headers=h).json()["report"]
    assert (report["added"], report["updated"], report["skipped"]) == (0, 0, 1)
    assert len(client.get("/api/obs/traces?name=ingest", headers=h).json()) == 2


def test_disabling_ingest_unmounts_it():
    client = make_client(enable_ingest=False)
    assert client.get("/api/ingest/health").status_code == 404
    comps = by_name(client.get("/api/status", headers={"X-API-Key": KEY}).json())
    assert comps["ingest"]["enabled"] is False and comps["ingest"]["ui_path"] is None
    assert "E2E_ENABLE_INGEST" in comps["ingest"]["detail"]


# ----- the documented .env.example must actually load ------------------------------------------


def test_root_env_example_loads_into_every_settings_class(monkeypatch):
    """`cp .env.example .env` is the first thing a user does. It must not crash startup."""
    from pathlib import Path

    from common import AppSettings
    from observability.settings import ObsSettings as Obs

    example = Path(__file__).resolve().parents[2] / ".env.example"
    for name in ("INGEST_CHUNK_SIZE", "QDRANT_URL", "OBS_ENABLED", "E2E_ENABLE_OBS"):
        monkeypatch.delenv(name, raising=False)

    ingest = IngestSettings(_env_file=example)
    assert (ingest.chunk_size, ingest.chunk_overlap) == (512, 64)  # empty values: defaults
    assert ingest.embed_model == "google/embeddinggemma-300m"
    assert ingest.pdf_parser == "llamaparse" and ingest.llamaparse_tier == "fast"
    assert ingest.allowed_paths == [] and ingest.qdrant_url == "http://localhost:6333"
    assert ingest.qdrant_api_key == ""  # the inline comment did not leak into the value

    from evals.settings import EvalSettings
    from llm import LLMSettings as LLMS
    from retrieval.settings import RetrievalSettings as RetS

    ev = EvalSettings(_env_file=example)
    assert ev.default_ks == [1, 3, 5, 10] and ev.judge_model is None
    assert ev.allowed_target_hosts == ["localhost", "127.0.0.1"]
    assert ev.allowed_header_refs == ["APP_API_KEY"]
    rt = RetS(_env_file=example)
    assert (rt.top_k, rt.candidates) == (
        5,
        50,
    ) and rt.rerank_model == "BAAI/bge-reranker-v2-m3-int8"
    llm = LLMS(_env_file=example)
    assert llm.model == "gpt-6-luna" and llm.use_reasoning is True and llm.max_retries == 2
    from mcp_tools.settings import McpSettings as Mcps

    mcp = Mcps(_env_file=example)
    assert mcp.allowed_commands == ["python -m mcp_tools.server"]
    assert mcp.allowed_hosts == ["localhost", "127.0.0.1"] and "APP_API_KEY" in mcp.allowed_env_refs
    assert mcp.retrieval_url == "http://localhost:8000/api/retrieval"
    assert (mcp.table_max_chunks, mcp.table_max_rows, mcp.table_max_groups) == (5000, 200000, 50)
    from agent_graph.settings import AgentSettings as AgentS

    agent = AgentS(_env_file=example)
    assert (agent.checkpoint_backend, agent.approval, agent.max_check_retries) == (
        "sqlite",
        "sensitive",
        2,
    )
    assert agent.auto_remember is False and agent.memory_embed_model == "google/embeddinggemma-300m"
    assert (
        agent.retrieval_url == "http://localhost:8002" and agent.mcp_url == "http://localhost:8004"
    )
    assert agent.trace_content is True and agent.trace_text_chars == 4000
    assert llm.price_input_per_m is None and llm.price_output_per_m is None  # empty: no estimate
    assert Obs(_env_file=example).exporter_names == ["local"]
    assert E2ESettings(_env_file=example).enable_ingest is True
    assert AppSettings(_env_file=example).app_api_key == ""


def test_env_example_holds_no_secret_values():
    """Only `.env` (git-ignored) holds real values. The template is committed."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    secret_name = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD)")
    for path in [
        root / ".env.example",
        *root.glob("packages/*/.env.example"),
        root / "e2e/.env.example",
    ]:
        for line in path.read_text().splitlines():
            m = re.match(r"^([A-Z0-9_]+)=([^#]*)", line)
            if m and secret_name.search(m.group(1)):
                assert m.group(2).strip() == "", f"{path.name}: {m.group(1)} has a value"


# ----- llm status card ---------------------------------------------------------------------------


def test_llm_card_reports_a_missing_key_and_never_shows_a_key():
    comps = by_name(make_client().get("/api/status", headers={"X-API-Key": KEY}).json())
    assert comps["llm"]["configured"] is False and comps["llm"]["missing"] == ["OPENAI_API_KEY"]
    assert "gpt-6-luna" in comps["llm"]["detail"] and "low" in comps["llm"]["detail"]

    secret = "sk-test-key-0123456789abcdef"
    r = make_client(llm_key=secret).get("/api/status", headers={"X-API-Key": KEY})
    assert by_name(r.json())["llm"]["configured"] is True
    assert secret not in r.text


# ----- retrieval inside e2e, and the ingest <-> retrieval contract ------------------------------------

DOC_TEXT = (
    "Cats are small domestic animals that purr when content.\n\n"
    "Dogs are loyal companions that bark at strangers.\n\n"
    "The zebrafinch is a small bird from Australia.\n\n"
    "Jupiter is the largest planet in the solar system."
)


def test_retrieval_is_built_and_has_a_ui_tab():
    comps = by_name(make_client().get("/api/status", headers={"X-API-Key": KEY}).json())
    assert (
        comps["retrieval"]["configured"] is True
        and comps["retrieval"]["ui_path"] == "/ui/retrieval/"
    )
    assert "rerank model" in comps["retrieval"]["detail"]


def test_retrieval_routes_are_mounted_and_protected():
    client = make_client()
    assert client.get("/api/retrieval/health").status_code == 200
    for path in (
        "/api/retrieval/status",
        "/api/retrieval/strategies",
        "/api/retrieval/collections",
    ):
        assert client.get(path).status_code == 401, path
    assert client.get("/ui/retrieval/").status_code == 200
    assert client.get("/ui/retrieval/app.js").status_code == 200


def test_disabling_retrieval_unmounts_it():
    client = make_client(enable_retrieval=False)
    assert client.get("/api/retrieval/health").status_code == 404
    comps = by_name(client.get("/api/status", headers={"X-API-Key": KEY}).json())
    assert comps["retrieval"]["enabled"] is False and comps["retrieval"]["ui_path"] is None


def test_the_two_packages_agree_on_the_vector_names():
    import ingest.store as writer
    import retrieval.store as reader

    assert (writer.DENSE, writer.SPARSE, writer.SPEC_KEY) == (
        reader.DENSE,
        reader.SPARSE,
        reader.SPEC_KEY,
    )


def ingest_through_http(client, name="animals.txt", text=DOC_TEXT, options=None):
    h = {"X-API-Key": KEY}
    parts = [("files", (name, text.encode()))]
    data = {
        "options": json.dumps(
            options or {"chunking": {"strategy": "recursive", "chunk_size": 16, "overlap": 0}}
        )
    }
    job_id = client.post("/api/ingest/files", files=parts, data=data, headers=h).json()["job_id"]
    job = client.get(f"/api/ingest/jobs/{job_id}", headers=h).json()
    assert job["state"] == "completed", job
    return job["report"]


def retrieve(client, query, **options):
    body = {"query": query, **options}
    r = client.post("/api/retrieval/retrieve", json=body, headers={"X-API-Key": KEY})
    assert r.status_code == 200, r.text
    return r.json()


def test_what_ingest_writes_retrieval_can_find():
    client = make_client(shared=QdrantClient(":memory:"))
    report = ingest_through_http(
        client,
        options={
            "chunking": {"strategy": "recursive", "chunk_size": 16, "overlap": 0},
            "metadata": {"year": 2025},
        },
    )
    assert report["chunks_written"] >= 4

    # Dense: the exact chunk text is the nearest vector.
    dense = retrieve(
        client, "The zebrafinch is a small bird from Australia.", mode="dense", top_k=1
    )
    hit = dense["hits"][0]
    assert hit["text"] == "The zebrafinch is a small bird from Australia."
    assert hit["payload"]["source_name"] == "animals.txt" and hit["payload"]["page"] == 1
    assert hit["payload"]["metadata"] == {"year": 2025} and "text" not in hit["payload"]

    # Sparse: a word only one chunk has. Hybrid: both together.
    assert retrieve(client, "Jupiter", mode="sparse")["hits"][0]["text"].startswith("Jupiter")
    assert retrieve(client, "Jupiter", mode="hybrid")["hits"][0]["text"].startswith("Jupiter")

    # Filters use the payload ingest wrote, including nested metadata.
    by_year = retrieve(client, "bird", filters={"metadata.year": 2025}, top_k=3)
    assert by_year["hits"]
    assert retrieve(client, "bird", filters={"metadata.year": 1999})["hits"] == []
    assert retrieve(client, "bird", filters={"source_name": "nope.txt"})["hits"] == []


def test_retrieval_reads_the_spec_ingest_stored():
    client = make_client(shared=QdrantClient(":memory:"))
    ingest_through_http(client)
    h = {"X-API-Key": KEY}
    spec = client.get("/api/retrieval/strategies", headers=h).json()["collections"][0]["spec"]
    assert spec["embedding_model"] == "fake-dense" and spec["sparse_language"] == "english"
    assert spec["query_prefix"] == "" and spec["dense_dim"] == 4


def test_after_an_edit_retrieval_sees_the_new_text_and_never_the_old():
    client = make_client(shared=QdrantClient(":memory:"))
    ingest_through_http(client)
    edited = DOC_TEXT.replace(
        "Jupiter is the largest planet in the solar system.", "Saturn has bright rings."
    )
    report = ingest_through_http(client, text=edited)
    assert report["chunks_reused"] > 0 and report["chunks_written"] >= 1
    assert retrieve(client, "Saturn", mode="sparse")["hits"][0]["text"].startswith("Saturn")
    assert retrieve(client, "Jupiter", mode="sparse")["hits"] == []
    # Reused chunks keep working, with refreshed positions.
    assert (
        retrieve(
            client, "Cats are small domestic animals that purr when content.", mode="dense", top_k=1
        )["hits"][0]["payload"]["char_start"]
        == 0
    )


def test_a_retrieval_request_is_traced_alongside_ingest():
    client = make_client(shared=QdrantClient(":memory:"))
    ingest_through_http(client)
    result = retrieve(client, "zebrafinch", rerank=True, rewrite="multi_query")
    h = {"X-API-Key": KEY}
    tree = client.get(f"/api/obs/traces/{result['trace_id']}", headers=h).json()
    root = tree["roots"][0]
    assert root["name"] == "retrieval.retrieve"
    assert [c["name"] for c in root["children"]].count("retrieval.search") == 3
    assert {"retrieval.rewrite", "retrieval.rerank"} <= {c["name"] for c in root["children"]}
    names = {t["root_name"] for t in client.get("/api/obs/traces", headers=h).json()}
    assert {"ingest.run", "retrieval.retrieve"} <= names


# ----- evals inside e2e ---------------------------------------------------------------------------


def test_evals_is_built_and_has_a_ui_tab():
    comps = by_name(make_client().get("/api/status", headers={"X-API-Key": KEY}).json())
    assert comps["evals"]["configured"] is True and comps["evals"]["ui_path"] == "/ui/evals/"
    assert "retrieval" in comps["evals"]["detail"] and "retrieval_http" in comps["evals"]["detail"]


def test_evals_routes_are_mounted_and_protected():
    client = make_client()
    assert client.get("/api/evals/health").status_code == 200
    for path in (
        "/api/evals/status",
        "/api/evals/datasets",
        "/api/evals/runs",
        "/api/evals/jobs/x",
    ):
        assert client.get(path).status_code == 401, path
    for path in ("/ui/evals/", "/ui/evals/app.js"):
        r = client.get(path)
        assert r.status_code == 200 and KEY not in r.text


def test_disabling_evals_unmounts_it():
    client = make_client(enable_evals=False)
    assert client.get("/api/evals/health").status_code == 404
    comps = by_name(client.get("/api/status", headers={"X-API-Key": KEY}).json())
    assert comps["evals"]["enabled"] is False and comps["evals"]["ui_path"] is None


def wait_job(client, job_id, prefix="/api/evals"):
    job = client.get(f"{prefix}/jobs/{job_id}", headers={"X-API-Key": KEY}).json()
    assert job["state"] == "completed", job
    return job


def test_end_to_end_ingest_draft_review_evaluate_degrade_compare():
    """The whole chain over HTTP, on in-memory Qdrant and fakes: the same workflow a person follows in the UI."""
    client = make_client(shared=QdrantClient(":memory:"))
    h = {"X-API-Key": KEY}
    ingest_through_http(client)

    # Draft questions from the ingested chunks, then review them.
    gen = client.post(
        "/api/evals/datasets/generate",
        json={"collection": "docs", "n": 3, "min_chars": 0},
        headers=h,
    )
    dataset_id = wait_job(client, gen.json()["job_id"])["result"]["dataset_id"]
    dataset = client.get(f"/api/evals/datasets/{dataset_id}", headers=h).json()
    assert dataset["status"] == "draft" and len(dataset["items"]) == 3
    assert all(i["evidence"][0]["text"] and i["relevant_chunk_ids"] for i in dataset["items"])
    for item in dataset["items"]:
        client.patch(
            f"/api/evals/datasets/{dataset_id}/items/{item['id']}",
            json={"status": "accepted"},
            headers=h,
        )

    def run(options):
        body = {
            "dataset_id": dataset_id,
            "target": {"kind": "preset", "name": "retrieval", "options": options},
            "config": {"ks": [1, 3]},
        }
        r = client.post("/api/evals/runs", json=body, headers=h)
        assert r.status_code == 202, r.text
        return wait_job(client, r.json()["job_id"])["job_id"]

    good = run({"mode": "dense", "top_k": 3})
    degraded = run({"mode": "dense", "top_k": 3, "filters": {"source_name": "nobody.txt"}})

    report = client.get(f"/api/evals/runs/{good}", headers=h).json()["report"]
    assert report["metrics"]["recall@1"]["mean"] == 1.0 and report["metrics"]["mrr"]["mean"] == 1.0
    assert report["errors"] == 0 and report["latency"]["retrieval_ms"]["n"] == 3
    assert "faithfulness" in report["not_computed"]  # retrieval returns no answer

    bad = client.get(f"/api/evals/runs/{degraded}", headers=h).json()["report"]
    assert bad["metrics"]["recall@3"]["mean"] == 0.0

    cmp = client.get(f"/api/evals/runs/compare?a={good}&b={degraded}", headers=h).json()
    assert {m["metric"]: m["delta"] for m in cmp["metrics"]}["recall@1"] == -1.0
    assert len(cmp["changed_questions"]) >= 3 and cmp["changed_questions"][0]["delta"] == -1.0

    # The run is one trace, and the retrieval calls it made sit inside it.
    traces = client.get("/api/obs/traces?name=eval.run", headers=h).json()
    assert len(traces) == 2
    tree = client.get(f"/api/obs/traces/{traces[-1]['trace_id']}", headers=h).json()
    questions = tree["roots"][0]["children"]
    assert [q["name"] for q in questions] == ["eval.question"] * 3
    assert questions[0]["children"][0]["name"] == "retrieval.retrieve"


def test_a_bad_retrieval_option_is_refused_before_the_run_starts():
    client = make_client(shared=QdrantClient(":memory:"))
    h = {"X-API-Key": KEY}
    ds = client.post(
        "/api/evals/datasets", files={"file": ("g.jsonl", b'{"question": "Q?"}')}, headers=h
    ).json()
    body = {
        "dataset_id": ds["id"],
        "target": {"kind": "preset", "name": "retrieval", "options": {"mode": "fuzzy"}},
    }
    r = client.post("/api/evals/runs", json=body, headers=h)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_target"


def test_every_ui_script_is_valid_javascript():
    """No browser here, so at least make sure each UI file parses. Skipped if node is missing."""
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if not node:
        import pytest

        pytest.skip("node is not installed")
    root = Path(__file__).resolve().parents[2]
    scripts = [*root.glob("packages/*/src/*/ui/app.js"), root / "e2e/src/e2e/ui/app.js"]
    scripts.append(root / "e2e/src/e2e/ui/chat.js")
    scripts += list(root.glob("packages/common/src/common/ui_kit/kit.js"))
    assert len(scripts) >= 6
    for script in scripts:
        done = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
        assert done.returncode == 0, f"{script.name}: {done.stderr}"


# ----- MCP inside e2e --------------------------------------------------------------------------------


def test_mcp_is_built_and_has_a_ui_tab():
    comps = by_name(make_client().get("/api/status", headers={"X-API-Key": KEY}).json())
    assert comps["mcp"]["configured"] is True and comps["mcp"]["ui_path"] == "/ui/mcp/"
    assert "/mcp" in comps["mcp"]["detail"] and "1 registered" in comps["mcp"]["detail"]


def test_mcp_management_routes_are_mounted_and_protected():
    client = make_client()
    assert client.get("/api/mcp/health").status_code == 200
    for path in ("/api/mcp/status", "/api/mcp/servers", "/api/mcp/exposed", "/api/mcp/llm-tools"):
        assert client.get(path).status_code == 401, path
    for path in ("/ui/mcp/", "/ui/mcp/app.js"):
        r = client.get(path)
        assert r.status_code == 200 and KEY not in r.text


def test_the_mcp_endpoint_itself_needs_the_key():
    client = make_client()
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    headers = {"Accept": "application/json, text/event-stream"}
    for extra in ({}, {"X-API-Key": "wrong"}):
        r = client.post("/mcp/", json=body, headers={**headers, **extra})
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"


def test_disabling_mcp_unmounts_it():
    client = make_client(enable_mcp=False)
    assert client.get("/api/mcp/health").status_code == 404
    # Nothing answers as MCP: the static mount at "/" refuses a POST, it does not serve a tool.
    assert client.post("/mcp", json={}).status_code in (404, 405)
    assert client.post("/mcp/", json={}).status_code in (404, 405)
    comps = by_name(client.get("/api/status", headers={"X-API-Key": KEY}).json())
    assert comps["mcp"]["enabled"] is False and comps["mcp"]["ui_path"] is None


def test_a_tool_call_through_the_api_reaches_retrieval_and_is_one_trace():
    client = make_client(shared=QdrantClient(":memory:"))
    h = {"X-API-Key": KEY}
    ingest_through_http(client)
    r = client.post(
        "/api/mcp/servers/own/tools/search_corpus/call",
        json={"arguments": {"query": "zebrafinch", "top_k": 2}},
        headers=h,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["structured"]["results"][0]["text"].startswith("The zebrafinch")
    assert body["structured"]["results"][0]["source_name"] == "animals.txt"

    traces = client.get("/api/obs/traces?name=mcp.call_tool", headers=h).json()
    assert len(traces) == 1
    tree = client.get(f"/api/obs/traces/{traces[0]['trace_id']}", headers=h).json()

    def names(node):
        return [node["name"]] + [n for c in node["children"] for n in names(c)]

    seen = names(tree["roots"][0])
    assert seen[0] == "mcp.call_tool"
    assert (
        "mcp.tool.search_corpus" in seen and "retrieval.retrieve" in seen
    )  # one trace, end to end


def test_a_real_mcp_client_calls_our_tool_over_http_and_finds_an_ingested_document(live):
    """MCP client -> /mcp (key checked) -> tool -> retrieval -> vector store -> your document."""
    import asyncio

    import httpx
    import httpx2
    from mcp.client.streamable_http import streamable_http_client

    app = create_app_for_live()
    with live(app) as server:
        h = {"X-API-Key": KEY}
        files = [("files", ("animals.txt", DOC_TEXT.encode()))]
        job = httpx.post(server.url + "/api/ingest/files", files=files, headers=h).json()["job_id"]
        done = httpx.get(f"{server.url}/api/ingest/jobs/{job}", headers=h).json()
        assert done["state"] == "completed"

        async def go():
            http = httpx2.AsyncClient(headers=h)
            async with McpClient(
                streamable_http_client(server.url + "/mcp", http_client=http)
            ) as c:
                tools = sorted(t.name for t in (await c.list_tools()).tools)
                found = await c.call_tool("search_corpus", {"query": "Jupiter", "top_k": 1})
                sources = await c.read_resource("corpus://sources")
                return tools, found, sources

        tools, found, sources = asyncio.run(go())
    assert tools == ["calculate", "list_tables", "query_table", "search_corpus"]
    top = found.structured_content["results"][0]
    assert "Jupiter is the largest planet" in top["text"] and top["source_name"] == "animals.txt"
    assert "animals.txt" in sources.contents[0].text


def create_app_for_live():
    shared = QdrantClient(":memory:")
    ingest, retrieval = make_ingest(shared), make_retrieval(shared)
    mcp = make_mcp(ingest, retrieval, KEY)
    app = create_app(
        settings=E2ESettings(_env_file=None),
        qdrant=FakeProbe(True),
        obs_settings=ObsSettings(_env_file=None, local_db_path=":memory:"),
        ingest=ingest,
        retrieval=retrieval,
        evals=make_evals(retrieval, shared),
        mcp=mcp,
        agent=make_agent(retrieval, mcp),
        llm_settings=LLMSettings(_env_file=None, openai_api_key=""),
    )
    app.dependency_overrides[get_app_settings] = lambda: AppSettings(app_api_key=KEY)
    return app


def test_the_model_would_be_offered_our_tools_in_the_openai_format():
    client = make_client()
    tools = client.get("/api/mcp/llm-tools", headers={"X-API-Key": KEY}).json()["tools"]
    assert [t["name"] for t in tools] == [
        "own__calculate",
        "own__search_corpus",
        "own__list_tables",
        "own__query_table",
    ]
    assert tools[0]["parameters"]["required"] == ["expression"]
