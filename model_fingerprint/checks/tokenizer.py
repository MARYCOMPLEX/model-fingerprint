"""Tokenizer fingerprint by *delta* measurement.

The server reports ``usage.prompt_tokens`` for whatever chat template it uses.
Rendering that template locally only works when we know the exact template and
system prompt, which non-Cline providers will not match.  Measuring the token
cost of an appended fragment cancels the template out::

    delta(fragment) = tokens(prompt + fragment) - tokens(prompt)

and that delta depends only on the tokenizer.

Special tokens are also measured, but with a guard: some gateways rewrite the
fullwidth ``｜`` and ``▁`` characters in user content, which makes every
vocabulary agree.  When that rewrite is detected the special-token rows are
excluded from scoring instead of being counted as evidence.
"""

from dataclasses import dataclass, field

from ..families import available_families, load_tokenizer

BASE = "Alpha beta gamma delta."

FRAGMENTS = (
    " 1234567890",
    " 中国的首都是北京。",
    " def fibonacci(n):\n    return n if n < 2 else 1",
    " SELECT * FROM users WHERE id = 42;",
    " 🌟🚀💡 emoji",
    " Explain the difference between TCP and UDP.",
    " The quick brown fox jumps over the lazy dog.",
    " 请把下面这句话翻译成英文：人工智能正在改变世界。",
    " C++ std::vector<int> vs Rust Vec<i32>",
    " x=1+2*3-4/5%6**7",
)

# A token that only a specific vocabulary encodes as one unit.
SPECIALS = (
    "<｜System｜>",
    "<｜deepseek_image｜>",
    "<｜User｜>",
    "<｜latest_reminder｜>",
    "<｜begin▁of▁thinking｜>",
    "<｜begin▁of▁sentence｜>",
)


def ascii_variant(text):
    """The rewrite gateways apply to user content: fullwidth ``｜`` -> ASCII ``|``."""
    return text.replace("｜", "|")


@dataclass
class FamilyScore:
    key: str
    label: str
    fragment_hits: int = 0
    fragment_total: int = 0
    special_hits: int = 0
    special_total: int = 0

    @property
    def score(self):
        total = self.fragment_total + self.special_total
        if not total:
            return 0.0
        return (self.fragment_hits + self.special_hits) / total

    def to_dict(self):
        return {
            "label": self.label,
            "fragments": f"{self.fragment_hits}/{self.fragment_total}",
            "specials": f"{self.special_hits}/{self.special_total}",
            "score": round(self.score, 3),
        }


@dataclass
class TokenizerResult:
    api_counts: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    best: FamilyScore = None
    tied: list = field(default_factory=list)
    conclusive: bool = False
    sanitised_specials: bool = False
    skipped: str = ""

    def summary(self):
        return {
            "best": self.best.key if self.best else None,
            "tied": self.tied,
            "conclusive": self.conclusive,
            "sanitised_specials": self.sanitised_specials,
            "scores": {key: score.to_dict() for key, score in self.scores.items()},
            "skipped": self.skipped,
        }


def _measure(client, prompt):
    return client.prompt_tokens(prompt)


def run(client, log, families=None):
    families = families or available_families()
    counters = {}
    labels = {}
    for family in families:
        counter = load_tokenizer(family)
        if counter is None:
            continue
        counters[family.key] = counter
        labels[family.key] = family.label
    if not counters:
        return TokenizerResult(skipped="no reference tokenizers available")

    base_tokens = _measure(client, BASE)
    if base_tokens is None:
        return TokenizerResult(skipped="endpoint does not report usage.prompt_tokens")

    result = TokenizerResult(api_counts={"__base__": base_tokens})
    scores = {key: FamilyScore(key=key, label=labels[key]) for key in counters}

    log(f"  base prompt: {base_tokens} tokens (template cost included)")
    log("  fragment deltas (template-independent)")
    log(f"  {'api':>4} " + " ".join(f"{key:>12}" for key in counters) + "  fragment")
    for fragment in FRAGMENTS:
        api_total = _measure(client, BASE + fragment)
        if api_total is None:
            continue
        api_delta = api_total - base_tokens
        result.api_counts[fragment] = api_delta
        cells = []
        for key, counter in counters.items():
            delta = counter(BASE + fragment) - counter(BASE)
            scores[key].fragment_total += 1
            scores[key].fragment_hits += delta == api_delta
            cells.append(f"{delta:>12d}")
        log(f"  {api_delta:>4d} " + " ".join(cells) + f"  {fragment[:34]!r}")

    log("\n  special-token deltas")
    log(f"  {'api':>4} " + " ".join(f"{key:>12}" for key in counters) + "  token")
    rows = []
    for special in SPECIALS:
        api_total = _measure(client, BASE + special)
        if api_total is None:
            continue
        api_delta = api_total - base_tokens
        result.api_counts[special] = api_delta
        raw_deltas = {key: counter(BASE + special) - counter(BASE) for key, counter in counters.items()}
        ascii_deltas = {
            key: counter(BASE + ascii_variant(special)) - counter(BASE)
            for key, counter in counters.items()
        }
        rows.append((special, api_delta, raw_deltas, ascii_deltas))

    # Decide whether the endpoint escaped the special characters by comparing,
    # per vocabulary, how often raw and ASCII-escaped forms explain the
    # observed deltas.  A verbatim endpoint matches raw forms on nearly every
    # row; a sanitising one matches escaped forms instead.
    raw_hits = {key: 0 for key in counters}
    ascii_hits = {key: 0 for key in counters}
    for _, api_delta, raw_deltas, ascii_deltas in rows:
        for key in counters:
            raw_hits[key] += raw_deltas[key] == api_delta
            ascii_hits[key] += ascii_deltas[key] == api_delta
    majority = (len(rows) + 1) // 2
    result.sanitised_specials = any(
        ascii_hits[key] > raw_hits[key] and ascii_hits[key] >= majority
        for key in counters
    )

    for special, api_delta, raw_deltas, _ in rows:
        if result.sanitised_specials:
            log(f"  {api_delta:>4d} " + " ".join(f"{raw_deltas[k]:>12d}" for k in counters)
                + f"  {special}  [escaped by gateway, not scored]")
            continue
        for key in counters:
            scores[key].special_total += 1
            scores[key].special_hits += raw_deltas[key] == api_delta
        log(f"  {api_delta:>4d} " + " ".join(f"{raw_deltas[k]:>12d}" for k in counters)
            + f"  {special}")

    if result.sanitised_specials:
        log("\n  note: gateway escapes fullwidth '｜' in user content, so special-token")
        log("        rows cannot separate vocabularies that share a regular-text tokenizer.")

    result.scores = scores
    ranked = sorted(scores.values(), key=lambda item: item.score, reverse=True)
    result.best = ranked[0]
    top = result.best.score
    result.tied = [item.key for item in ranked if abs(item.score - top) < 1e-9]
    result.conclusive = top >= 0.9 and len(result.tied) == 1
    return result
