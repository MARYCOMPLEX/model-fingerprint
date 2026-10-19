"""Configuration: where the endpoint is, which model, and the credentials."""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROVIDER_ENDPOINTS = {
    "cline": "https://api.cline.bot/api/v1",
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "vllm": "http://localhost:8000/v1",
}

KEY_ENV_VARS = ("MODEL_FINGERPRINT_KEY", "OPENAI_API_KEY", "CLINE_API_KEY")
KEY_FILES = (".model_fingerprint_key", ".cline_key", ".openai_key")


def normalize_base_url(url):
    """Accept ``host``, ``.../v1`` or a full ``.../chat/completions`` URL."""
    url = (url or "").strip().rstrip("/")
    if not url:
        raise ValueError("empty base URL")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")]
    if not url.endswith("/v1") and "/v1/" not in url:
        url += "/v1"
    return url


def resolve_key(cli_key=None, search_dir=None):
    if cli_key:
        return cli_key.strip()
    for name in KEY_ENV_VARS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    search_dir = Path(search_dir or Path.cwd())
    for name in KEY_FILES:
        candidate = search_dir / name
        if candidate.exists():
            value = candidate.read_text(encoding="utf-8").strip()
            if value:
                return value
    raise ValueError(
        "no API key: pass --key, set "
        + " / ".join(KEY_ENV_VARS)
        + ", or put one in "
        + ", ".join(KEY_FILES)
    )


@dataclass
class Endpoint:
    base_url: str
    model: str
    key: str = field(repr=False, default="")
    extra_headers: dict = field(default_factory=dict)
    retries: int = 4
    timeout: int = 180

    @property
    def chat_url(self):
        return self.base_url.rstrip("/") + "/chat/completions"

    @classmethod
    def for_provider(cls, provider, model, key=None, **kwargs):
        if provider not in PROVIDER_ENDPOINTS:
            raise ValueError(f"unknown provider {provider!r}; known: {', '.join(PROVIDER_ENDPOINTS)}")
        return cls(
            base_url=normalize_base_url(PROVIDER_ENDPOINTS[provider]),
            model=model,
            key=key or "",
            **kwargs,
        )
