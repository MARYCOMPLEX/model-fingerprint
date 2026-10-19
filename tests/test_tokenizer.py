import pytest

from model_fingerprint.checks import tokenizer as tokenizer_check
from model_fingerprint.checks.tokenizer import ascii_variant


def test_ascii_variant_only_rewrites_fullwidth_pipe():
    assert ascii_variant("<｜User｜>") == "<|User|>"
    # The block character must survive: DeepSeek's thinking token uses it and
    # replacing it would fake a match for the wrong row.
    assert ascii_variant("<｜begin▁of▁thinking｜>") == "<|begin▁of▁thinking|>"


class FakeClient:
    """Serves prompt_tokens from a lookup table, mimicking a tokenizer."""

    def __init__(self, counter, template_tokens):
        self.counter = counter
        self.template_tokens = template_tokens
        self.endpoint = type("E", (), {"model": "test"})()

    def prompt_tokens(self, prompt, **_):
        return self.counter(prompt) + self.template_tokens


def fake_family(key, counter, label=None):
    family = type("F", (), {"key": key, "label": label or key, "tokenizer_group": key})()
    return family, counter


def test_deltas_cancel_out_the_template(monkeypatch):
    # Two vocabularies: one splits " 1234567890" into 5 tokens, the other 6.
    def matching(text):
        return len(text.split()) * 2

    def wrong(text):
        return len(text) * 2

    families = []
    counters = {}
    for key, counter in (("match", matching), ("wrong", wrong)):
        family, _ = fake_family(key, counter)
        families.append(family)
        counters[key] = counter

    monkeypatch.setattr(tokenizer_check, "load_tokenizer", lambda f: counters[f.key])
    monkeypatch.setattr(tokenizer_check, "available_families", lambda: families)

    client = FakeClient(matching, template_tokens=17)
    result = tokenizer_check.run(client, log=lambda *_: None, families=families)

    assert result.best.key == "match"
    assert result.best.fragment_hits == result.best.fragment_total
    assert result.conclusive


def test_special_tokens_detected_as_sanitised(monkeypatch):
    def plain(text):
        # Rough character counter; fullwidth pipes cost extra so a raw form can
        # never coincide with its ASCII escape.
        return len(text) + 2 * text.count("｜")

    def merges_fullwidth(text):
        # Pretend this vocabulary encodes '<｜User｜>' as a single token.
        return len(text.replace("<｜User｜>", "x")) + 2 * text.count("｜")

    families = []
    counters = {}
    for key, counter in (("plain", plain), ("fullwidth", merges_fullwidth)):
        family, _ = fake_family(key, counter)
        families.append(family)
        counters[key] = counter

    monkeypatch.setattr(tokenizer_check, "load_tokenizer", lambda f: counters[f.key])

    class SanitisingClient(FakeClient):
        def prompt_tokens(self, prompt, **_):
            # The gateway rewrites '｜' to '|' before tokenizing, so the served
            # count always looks like the ASCII form.
            return plain(prompt.replace("｜", "|"))

    client = SanitisingClient(plain, template_tokens=0)
    result = tokenizer_check.run(client, log=lambda *_: None, families=families)

    assert result.sanitised_specials
    # Structural rows carry no information, so no family may claim a special hit.
    assert all(score.special_hits == 0 for score in result.scores.values())
    assert all(score.special_total == 0 for score in result.scores.values())


def test_skips_without_usage(monkeypatch):
    class NoUsage(FakeClient):
        def prompt_tokens(self, prompt, **_):
            return None

    monkeypatch.setattr(tokenizer_check, "load_tokenizer", lambda f: len)
    family, _ = fake_family("x", len)
    result = tokenizer_check.run(NoUsage(len, 0), log=lambda *_: None, families=[family])
    assert "usage.prompt_tokens" in result.skipped
