import pytest

from model_fingerprint.config import normalize_base_url, resolve_key


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://api.openai.com", "https://api.openai.com/v1"),
        ("https://api.openai.com/", "https://api.openai.com/v1"),
        ("https://api.openai.com/v1", "https://api.openai.com/v1"),
        ("https://api.openai.com/v1/", "https://api.openai.com/v1"),
        ("https://api.openai.com/v1/chat/completions", "https://api.openai.com/v1"),
        ("api.deepseek.com", "https://api.deepseek.com/v1"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1"),
        ("http://localhost:11434/v1/", "http://localhost:11434/v1"),
    ],
)
def test_normalize_base_url(raw, expected):
    assert normalize_base_url(raw) == expected


def test_resolve_key_prefers_cli_arg(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_FINGERPRINT_KEY", "env-key")
    (tmp_path / ".model_fingerprint_key").write_text("file-key", encoding="utf-8")
    assert resolve_key("cli-key", search_dir=tmp_path) == "cli-key"
    assert resolve_key(None, search_dir=tmp_path) == "env-key"


def test_resolve_key_falls_back_to_file(monkeypatch, tmp_path):
    monkeypatch.delenv("MODEL_FINGERPRINT_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CLINE_API_KEY", raising=False)
    (tmp_path / ".model_fingerprint_key").write_text("  file-key\n", encoding="utf-8")
    assert resolve_key(None, search_dir=tmp_path) == "file-key"


def test_resolve_key_raises_when_missing(monkeypatch, tmp_path):
    for name in ("MODEL_FINGERPRINT_KEY", "OPENAI_API_KEY", "CLINE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError):
        resolve_key(None, search_dir=tmp_path)
