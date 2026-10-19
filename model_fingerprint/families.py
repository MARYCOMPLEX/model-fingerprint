"""Model-family definitions and the tokenizer references used to recognise them.

Two kinds of reference are supported:

* ``tokenizers`` assets — a ``tokenizer.json`` (or its gzip) loaded through the
  HuggingFace ``tokenizers`` library.  Three DeepSeek generations and Qwen3 are
  bundled so the tool works offline.
* ``tiktoken`` encodings — used for OpenAI models, optional dependency.
"""

import gzip
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

DATA_DIR = Path(__file__).resolve().parent / "data"

# Sources are only needed by ``--download-data`` to fetch fresh references.
TOKENIZER_SOURCES = {
    "deepseek_v41": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "deepseek_v4": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "deepseek_v3": "deepseek-ai/DeepSeek-V3",
    "qwen3": "Qwen/Qwen3.8-27B",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "kimi_k2": "moonshotai/Kimi-K2-Instruct",
}


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    vendor: str
    tokenizer: Optional[str] = None       # asset name in model_fingerprint/data
    tiktoken: Optional[str] = None        # tiktoken encoding name
    template_module: Optional[str] = None  # vendored encoder module
    markers: tuple = field(default_factory=tuple)
    notes: str = ""
    group: str = ""                       # generations sharing one text tokenizer

    @property
    def tokenizer_group(self):
        return self.group or self.key


FAMILIES = (
    Family(
        key="deepseek_v41",
        label="DeepSeek V4.1",
        vendor="DeepSeek",
        tokenizer="deepseek_v41",
        template_module="deepseek_encoding",
        markers=("<｜System｜>", "<｜deepseek_image｜>", "<｜begin▁of▁sentence｜>"),
        group="deepseek",
    ),
    Family(
        key="deepseek_v4",
        label="DeepSeek V4",
        vendor="DeepSeek",
        tokenizer="deepseek_v4",
        template_module="deepseek_encoding",
        markers=("<｜User｜>", "<｜Assistant｜>"),
        group="deepseek",
    ),
    Family(
        key="deepseek_v3",
        label="DeepSeek V3 / R1",
        vendor="DeepSeek",
        tokenizer="deepseek_v3",
        template_module="deepseek_encoding",
        markers=("<｜User｜>", "<｜latest_reminder｜>"),
        group="deepseek",
    ),
    Family(
        key="qwen3",
        label="Qwen 3",
        vendor="Alibaba",
        tokenizer="qwen3",
        markers=("<|im_start|>", "<|im_end|>"),
        notes="no bundled chat template; fragment deltas only",
        group="qwen",
    ),
    Family(key="openai", label="OpenAI GPT", vendor="OpenAI", tiktoken="cl100k_base",
           group="openai_cl100k"),
    Family(key="openai_o200k", label="OpenAI GPT-4o/5", vendor="OpenAI", tiktoken="o200k_base",
           group="openai_o200k"),
    Family(
        key="llama3",
        label="Llama 3",
        vendor="Meta",
        tokenizer="llama3",
        markers=("<|start_header_id|>",),
        group="llama",
    ),
    Family(
        key="kimi_k2",
        label="Kimi K2",
        vendor="Moonshot",
        tokenizer="kimi_k2",
        markers=("<|im_start|>", "<|im_end|>"),
        group="kimi",
    ),
)

BY_KEY = {family.key: family for family in FAMILIES}


def _temp_tokenizer_json(asset_name: str) -> Path:
    """Decompress a bundled tokenizer into a temp file ``tokenizers`` can open."""
    gz_path = DATA_DIR / f"{asset_name}.json.gz"
    plain_path = DATA_DIR / f"{asset_name}.json"
    if plain_path.exists():
        return plain_path
    if not gz_path.exists():
        raise FileNotFoundError(
            f"no tokenizer asset {asset_name!r}; run `model-fingerprint --download-data`"
        )
    handle = tempfile.NamedTemporaryFile(
        prefix=f"mf_{asset_name}_", suffix=".json", delete=False
    )
    with gzip.open(gz_path, "rb") as source, open(handle.name, "wb") as target:
        shutil.copyfileobj(source, target)
    return Path(handle.name)


def load_tokenizer(family: Family) -> Optional[Callable[[str], int]]:
    """Return a ``str -> token count`` function, or None when unavailable."""
    if family.tokenizer:
        from tokenizers import Tokenizer

        path = _temp_tokenizer_json(family.tokenizer)
        tokenizer = Tokenizer.from_file(str(path))
        return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    if family.tiktoken:
        try:
            import tiktoken
        except ImportError:
            return None
        try:
            encoding = tiktoken.get_encoding(family.tiktoken)
        except Exception:
            return None
        return lambda text: len(encoding.encode(text))
    return None


def available_families(include_optional=True):
    available = []
    for family in FAMILIES:
        if family.tokenizer:
            if (DATA_DIR / f"{family.tokenizer}.json.gz").exists() or (
                DATA_DIR / f"{family.tokenizer}.json"
            ).exists():
                available.append(family)
        elif family.tiktoken and include_optional:
            try:
                import tiktoken  # noqa: F401
            except ImportError:
                continue
            available.append(family)
    return available


def guess_family(model_id: str) -> Optional[Family]:
    """Best-effort family from a model id, used to pick the template."""
    lowered = (model_id or "").lower()
    if "deepseek" in lowered:
        if "v4.1" in lowered or "v41" in lowered:
            return BY_KEY["deepseek_v41"]
        if "v4" in lowered:
            return BY_KEY["deepseek_v4"]
        return BY_KEY["deepseek_v3"]
    if "qwen" in lowered:
        return BY_KEY["qwen3"]
    if "kimi" in lowered:
        return BY_KEY["kimi_k2"]
    if "llama" in lowered:
        return BY_KEY["llama3"]
    if any(tag in lowered for tag in ("gpt", "o1", "o3", "o4", "chatgpt")):
        return BY_KEY["openai_o200k"]
    return None


def download_reference_tokenizers(force=False):
    """Fetch every reference tokenizer from HuggingFace into data/."""
    import requests

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for asset, repo in TOKENIZER_SOURCES.items():
        target = DATA_DIR / f"{asset}.json.gz"
        if target.exists() and not force:
            print(f"  skip {asset} (already present)")
            continue
        print(f"  fetch {asset} <- {repo}")
        response = requests.get(
            f"https://huggingface.co/{repo}/resolve/main/tokenizer.json", timeout=300
        )
        response.raise_for_status()
        with gzip.open(target, "wb", compresslevel=9) as handle:
            handle.write(response.content)
        print(f"    {len(response.content):,} bytes -> {target.name}")


def cache_summary():
    rows = []
    for family in FAMILIES:
        if family.tokenizer:
            path = DATA_DIR / f"{family.tokenizer}.json.gz"
            size = path.stat().st_size if path.exists() else 0
            rows.append((family.key, f"{size / 1e6:.1f} MB" if size else "missing"))
        elif family.tiktoken:
            try:
                import tiktoken  # noqa: F401
                status = "tiktoken"
            except ImportError:
                status = "needs pip install tiktoken"
            rows.append((family.key, status))
    return rows


__all__ = [
    "BY_KEY",
    "DATA_DIR",
    "FAMILIES",
    "Family",
    "available_families",
    "cache_summary",
    "download_reference_tokenizers",
    "guess_family",
    "load_tokenizer",
]
