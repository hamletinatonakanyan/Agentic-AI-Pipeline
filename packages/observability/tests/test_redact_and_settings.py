import pytest

from observability import init_observability, span
from observability.exporters import InMemoryExporter
from observability.redact import REDACTED, Redactor
from observability.settings import DEFAULT_REDACT_PATTERNS, ObsSettings


def test_patterns_redact_keys_and_emails_in_values():
    r = Redactor(DEFAULT_REDACT_PATTERNS)
    out = r.mapping(
        {"q": "mail me at jane@example.com", "k": "sk-abcdefghijklmnopqrstuv", "n": 3, "ok": "hi"}
    )
    assert "jane@example.com" not in out["q"]
    assert out["k"] == REDACTED
    assert out["n"] == 3
    assert out["ok"] == "hi"


def test_sensitive_keys_are_replaced_whatever_the_value():
    r = Redactor([])
    out = r.mapping({"api_key": "abc", "Authorization": "Bearer x", "x-api-key": "y", "name": "n"})
    assert out["api_key"] == out["Authorization"] == out["x-api-key"] == REDACTED
    assert out["name"] == "n"


def test_token_counts_are_not_mistaken_for_secrets():
    out = Redactor([]).mapping({"gen_ai.usage.input_tokens": 12})
    assert out["gen_ai.usage.input_tokens"] == 12


def test_nested_values_are_redacted():
    r = Redactor(DEFAULT_REDACT_PATTERNS)
    out = r.mapping({"a": {"b": ["x@y.io"]}})
    assert out["a"]["b"] == [REDACTED]


def test_redaction_happens_before_export():
    exporter = InMemoryExporter()
    init_observability(ObsSettings(_env_file=None), exporters=[exporter])
    with pytest.raises(ValueError):
        with span("x", {"password": "hunter2", "note": "contact a@b.co"}):
            raise ValueError("failed for a@b.co")
    data = exporter.spans[0]
    assert data.attributes["password"] == REDACTED
    assert "a@b.co" not in data.attributes["note"]
    assert "a@b.co" not in data.status_message
    assert "a@b.co" not in str(data.events)


def test_unknown_exporter_is_rejected():
    with pytest.raises(ValueError):
        ObsSettings(_env_file=None, exporters="local,otlp")


def test_settings_read_env(monkeypatch):
    monkeypatch.setenv("OBS_SAMPLE_RATE", "0.5")
    monkeypatch.setenv("OBS_EXPORTERS", "local,console")
    s = ObsSettings(_env_file=None)
    assert s.sample_rate == 0.5
    assert s.exporter_names == ["local", "console"]
