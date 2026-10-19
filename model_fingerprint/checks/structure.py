"""Absolute prompt-structure check against a known chat template.

For a family whose template we ship (DeepSeek, via the official
``encode_messages``), the *absolute* ``prompt_tokens`` value can be reproduced
exactly.  On the Cline gateway this matched 15/15 prompts for V4.1 and 0/15 for
every other reference, which is the tightest signal available.

Endpoints that prepend their own system prompt will not match, so this check
reports the residual rather than failing hard.
"""

import sys
from dataclasses import dataclass, field

from ..families import DATA_DIR, BY_KEY, guess_family

PROMPTS = (
    "Say OK",
    "Count from one to ten.",
    "The quick brown fox jumps over the lazy dog.",
    "Explain recursion with a simple example.",
    "中国的首都是北京。",
    "请解释一下什么是数据库索引。",
    "Write a Python function to check if a string is a palindrome.",
    "SELECT id, name FROM users WHERE age > 30 ORDER BY name;",
    "List three primary colors.",
    "Translate 'good morning' into Japanese.",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
    "{\"a\": 1, \"b\": [2, 3], \"c\": {\"d\": true}}",
    "🌟 emoji test 🚀 and numbers 12345",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What is 17 * 23?",
)

# DeepSeek generations are close enough in layout that each one's template is
# worth trying: whatever reproduces exactly names the generation.
DEEPSEEK_GENERATIONS = ("deepseek_v41", "deepseek_v4", "deepseek_v3")


@dataclass
class TemplateMatch:
    family_key: str = ""
    label: str = ""
    exact: int = 0
    total: int = 0
    offset: object = None
    residuals: list = field(default_factory=list)

    def to_dict(self):
        return {
            "family": self.family_key,
            "label": self.label,
            "exact": f"{self.exact}/{self.total}",
            "constant_offset": self.offset,
        }


@dataclass
class StructureResult:
    chosen: TemplateMatch = None
    attempts: list = field(default_factory=list)
    skipped: str = ""

    @property
    def template(self):
        return self.chosen.label if self.chosen else ""

    @property
    def family_key(self):
        return self.chosen.family_key if self.chosen else ""

    @property
    def exact(self):
        return self.chosen.exact if self.chosen else 0

    @property
    def total(self):
        return self.chosen.total if self.chosen else 0

    @property
    def residuals(self):
        return self.chosen.residuals if self.chosen else []

    @property
    def constant_offset(self):
        return self.chosen.offset if self.chosen else None

    def summary(self):
        return {
            "template": self.template,
            "family": self.family_key,
            "exact": f"{self.exact}/{self.total}",
            "constant_offset": self.constant_offset,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "skipped": self.skipped,
        }


def _load_encoder():
    if str(DATA_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_DIR))
    import deepseek_encoding
    return deepseek_encoding


def _residuals(api_counts, rendered):
    return [None if api is None else api - local for api, local in zip(api_counts, rendered)]


def _constant(values):
    present = [value for value in values if value is not None]
    if not present or len(present) != len(values):
        return None
    first = present[0]
    return first if all(value == first for value in present) else None


def run(client, log, model_id=None):
    declared = guess_family(model_id or client.endpoint.model)
    if declared is None or not declared.template_module:
        return StructureResult(
            skipped=f"no bundled chat template matches {client.endpoint.model!r}"
        )

    from ..families import load_tokenizer

    encoder = _load_encoder()
    candidates = [BY_KEY[key] for key in DEEPSEEK_GENERATIONS if key in BY_KEY]

    api_counts = []
    log(f"  {'api':>5}  prompt")
    for prompt in PROMPTS:
        api = client.prompt_tokens(prompt)
        api_counts.append(api)
        log(f"  {str(api):>5}  {prompt[:44]!r}")
    if all(value is None for value in api_counts):
        return StructureResult(skipped="endpoint does not report usage.prompt_tokens")

    result = StructureResult()
    for family in candidates:
        counter = load_tokenizer(family)
        if counter is None:
            continue
        rendered = [
            counter(
                encoder.encode_messages(
                    [{"role": "user", "content": prompt}], thinking_mode="thinking"
                )
            )
            for prompt in PROMPTS
        ]
        residuals = _residuals(api_counts, rendered)
        exact = sum(1 for value in residuals if value == 0)
        match = TemplateMatch(
            family_key=family.key,
            label=family.label,
            exact=exact,
            total=len(PROMPTS),
            offset=_constant(residuals),
            residuals=residuals,
        )
        result.attempts.append(match)
        log(f"  {family.label:<16} exact={exact}/{len(PROMPTS)} "
            f"offset={match.offset}")

    # Prefer an exact reproduction; otherwise accept a match with a constant offset.
    exact_matches = [match for match in result.attempts if match.exact == match.total]
    if exact_matches:
        result.chosen = exact_matches[0]
    else:
        offset_matches = [match for match in result.attempts if match.offset is not None]
        if offset_matches:
            result.chosen = offset_matches[0]
    if result.chosen is None and result.attempts:
        result.chosen = max(result.attempts, key=lambda match: match.exact)
    return result
