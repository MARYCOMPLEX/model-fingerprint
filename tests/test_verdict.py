from model_fingerprint.checks.behaviour import BehaviourResult
from model_fingerprint.checks.capabilities import CapabilityResult
from model_fingerprint.checks.structure import StructureResult, TemplateMatch
from model_fingerprint.checks.tokenizer import FamilyScore, TokenizerResult
from model_fingerprint.verdict import identify

ALL_DEEPSEEK = ["deepseek_v41", "deepseek_v4", "deepseek_v3"]


def make_tokenizer(best_key="deepseek_v41", tied=None, sanitised=True):
    keys = list(ALL_DEEPSEEK)
    if best_key not in keys:
        keys.append(best_key)
    scores = {}
    for key in keys:
        score = FamilyScore(key=key, label=key)
        score.fragment_hits = 10
        score.fragment_total = 10
        scores[key] = score
    return TokenizerResult(
        scores=scores,
        best=scores[best_key],
        tied=tied or [best_key],
        conclusive=not tied or len(tied) == 1,
        sanitised_specials=sanitised,
    )


def make_structure(family_key="deepseek_v41", exact=15, total=15, offset=None):
    match = TemplateMatch(
        family_key=family_key,
        label="DeepSeek V4.1" if family_key == "deepseek_v41" else family_key,
        exact=exact,
        total=total,
        offset=offset,
        residuals=[offset] * total if offset is not None else [0] * exact,
    )
    return StructureResult(chosen=match, attempts=[match])


def test_template_picks_generation_when_all_deepseek_tied():
    tok = make_tokenizer(tied=ALL_DEEPSEEK)
    verdict = identify("cline-pass/deepseek-v4.1-flash", tok, make_structure(), None)
    assert verdict.family_key == "deepseek_v41"
    assert verdict.vocabulary == "deepseek"
    assert verdict.confidence > 0.9


def test_constant_offset_still_identifies_generation():
    tok = make_tokenizer(tied=ALL_DEEPSEEK)
    struct = make_structure(exact=0, total=5, offset=53)
    verdict = identify("deepseek-v4.1", tok, struct, None)
    assert verdict.generation == "deepseek_v41"
    assert any("constant +53" in item for item in verdict.evidence)


def test_self_report_contradiction_is_a_caveat():
    tok = make_tokenizer(tied=ALL_DEEPSEEK)
    behaviour = BehaviourResult(reports={"vendor": "OpenAI"})
    verdict = identify(
        "cline-pass/deepseek-v4.1-flash", tok, make_structure(), behaviour
    )
    assert any("contradicts" in item for item in verdict.caveats)


def test_unknown_vocabulary_falls_back_to_template():
    tok = TokenizerResult(skipped="no usage")
    struct = make_structure(exact=10, total=10)
    verdict = identify("deepseek-v4.1", tok, struct, None)
    assert verdict.family_key == "deepseek_v41"


def test_only_capabilities_yields_no_family_claim():
    caps = CapabilityResult(
        vision={"supported": True},
        tools={"supported": True},
        long_context={"supported": True, "prompt_tokens": 1000},
    )
    verdict = identify("some-model", None, None, None, caps)
    assert verdict.family_key is None
    assert verdict.confidence > 0


def test_template_contradicting_tokenizer_raises_a_caveat():
    tok = make_tokenizer("qwen3", tied=["qwen3"])
    struct = make_structure("deepseek_v41")
    verdict = identify("vendor/mystery", tok, struct, None)
    assert any("disagree" in item for item in verdict.caveats)


def test_unconfirmed_generation_is_not_asserted():
    tok = make_tokenizer(tied=ALL_DEEPSEEK)
    verdict = identify("cline-pass/deepseek-v4-pro", tok, None, None)
    assert "not confirmed" in verdict.family_label
    assert verdict.confidence <= 0.5
    assert any("merely echoes the model id" in item for item in verdict.caveats)
