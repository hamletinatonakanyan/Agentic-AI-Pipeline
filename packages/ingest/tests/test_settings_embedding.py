import pytest
from pydantic import ValidationError

from ingest.embedding import resolve_prefixes
from ingest.models import ChunkOptions, IngestOptions
from ingest.settings import IngestSettings


def settings(**kw) -> IngestSettings:
    return IngestSettings(_env_file=None, **kw)


def test_defaults_match_the_documented_values():
    s = settings()
    assert s.embed_model == "google/embeddinggemma-300m"
    assert (s.pdf_parser, s.llamaparse_tier, s.scanned_text_threshold) == ("llamaparse", "fast", 50)
    assert s.sparse_model == "Qdrant/bm25"


def test_shared_names_are_read_without_the_prefix(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "llx-abc")
    s = settings()
    assert s.qdrant_url == "http://qdrant:6333" and s.llama_cloud_api_key == "llx-abc"
    assert "llx-abc" not in repr(s)  # secrets stay out of logs


def test_empty_values_from_env_example_use_the_defaults(monkeypatch):
    for name in ("INGEST_CHUNK_SIZE", "INGEST_EMBED_MODEL", "INGEST_ALLOWED_PATHS"):
        monkeypatch.setenv(name, "")
    s = settings()
    assert s.chunk_size == 512 and s.embed_model == "google/embeddinggemma-300m"
    assert s.allowed_paths == []


def test_allowed_paths_are_comma_separated(monkeypatch):
    monkeypatch.setenv("INGEST_ALLOWED_PATHS", "/data/a, /data/b")
    assert settings().allowed_paths == ["/data/a", "/data/b"]


def test_bad_values_are_rejected_at_startup():
    for bad in (
        {"chunk_overlap": 600},
        {"sparse_language": "klingon"},
        {"pdf_parser": "magic"},
        {"llamaparse_tier": "free"},
    ):
        with pytest.raises(ValidationError):
            settings(**bad)


def test_language_options_accept_none_and_normalise_case():
    assert settings(sparse_language="German").sparse_language == "german"
    assert IngestOptions(sparse_language="NONE").sparse_language == "none"
    assert IngestOptions().sparse_language is None
    with pytest.raises(ValidationError):
        IngestOptions(sparse_language="chinese")  # no bundled stemmer: must be "none"


def test_collection_names_are_restricted():
    assert IngestOptions(collection="my-docs_2.0").collection == "my-docs_2.0"
    for bad in ("a b", "a/b", "../x", ""):
        with pytest.raises(ValidationError):
            IngestOptions(collection=bad)


def test_chunk_options_validate_overlap():
    with pytest.raises(ValidationError):
        ChunkOptions(chunk_size=100, overlap=100)
    with pytest.raises(ValidationError):
        ChunkOptions(chunk_size=8)


def test_gemma_prefixes_are_built_in():
    query, doc = resolve_prefixes("google/embeddinggemma-300m")
    assert query == "task: search result | query: " and doc == "title: none | text: "


def test_e5_models_get_the_e5_prefixes_and_unknown_models_none():
    assert resolve_prefixes("intfloat/multilingual-e5-large") == ("query: ", "passage: ")
    assert resolve_prefixes("BAAI/bge-small-en-v1.5") == ("", "")


def test_prefix_settings_override_built_ins_even_with_empty_text():
    assert resolve_prefixes("google/embeddinggemma-300m", "Q: ", "D: ") == ("Q: ", "D: ")
    assert resolve_prefixes("google/embeddinggemma-300m", "", "") == ("", "")
