"""Combine probe results into an identification verdict with confidence.

Evidence is combined hierarchically:

1. **Vocabulary family** — fragment token deltas say which tokenizer serves the
   endpoint (DeepSeek vs Qwen vs OpenAI vs Llama).  All DeepSeek generations
   share one regular-text tokenizer, so this level cannot name a generation.
2. **Template reproduction** — absolute ``prompt_tokens`` reproduced against
   the official chat template identify the exact generation *within* the
   vocabulary family.
3. **Capabilities** — vision and context length corroborate the generation.
4. **Self-report** — recorded, deliberately low weight.
"""

from dataclasses import dataclass, field

from .checks import WEIGHTS
from .families import BY_KEY, guess_family


@dataclass
class Verdict:
    family_key: str = None
    family_label: str = "unknown"
    vendor: str = "unknown"
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
    caveats: list = field(default_factory=list)
    vocabulary: str = None
    generation: str = None

    def to_dict(self):
        return {
            "family": self.family_key,
            "label": self.family_label,
            "vendor": self.vendor,
            "confidence": round(self.confidence, 3),
            "vocabulary_family": self.vocabulary,
            "generation": self.generation,
            "evidence": self.evidence,
            "caveats": self.caveats,
        }


def _capability_score(capabilities):
    if capabilities is None:
        return None
    signals = []
    for attr in ("vision", "tools", "long_context"):
        value = getattr(capabilities, attr, None)
        if isinstance(value, dict) and "supported" in value:
            signals.append(1.0 if value["supported"] else 0.0)
    return sum(signals) / len(signals) if signals else None


def _tokenizer_evidence(tok_result, verdict, evidence, caveats):
    """Level 1: which vocabulary family is serving?"""
    if not tok_result or tok_result.skipped or not tok_result.best:
        if tok_result and tok_result.skipped:
            caveats.append(f"tokenizer probes skipped: {tok_result.skipped}")
        return None, 0.0

    weight = WEIGHTS["tokenizer"]
    best = tok_result.best
    if tok_result.tied and len(tok_result.tied) > 1:
        label = BY_KEY[best.key].vendor + "-family (generations share one text tokenizer)"
    else:
        label = best.label
    evidence.append(
        f"fragment token deltas match the {label} vocabulary on "
        f"{best.fragment_hits}/{best.fragment_total} probes"
    )
    if best.special_total and best.special_hits == best.special_total:
        evidence.append(
            f"special-token deltas agree on {best.special_hits}/{best.special_total}"
        )
    if tok_result.sanitised_specials:
        caveats.append(
            "gateway escapes fullwidth special characters in user content, so "
            "special-token probes cannot separate generations that share a text tokenizer"
        )
    if len(tok_result.tied) > 1:
        caveats.append(
            "vocabularies score identically ("
            + ", ".join(tok_result.tied)
            + "); the chat-template probe decides between generations"
        )
        verdict.vocabulary = BY_KEY[best.key].tokenizer_group
        # Same vocabulary, different generations: partial credit only.
        return None, weight * best.score * 0.7
    verdict.vocabulary = BY_KEY[best.key].tokenizer_group
    verdict.family_key = best.key
    verdict.family_label = best.label
    verdict.vendor = BY_KEY[best.key].vendor
    return best.key, weight * best.score


def _structure_evidence(struct_result, verdict, evidence, caveats):
    """Level 2: which generation within the vocabulary family?"""
    if not struct_result or struct_result.skipped or not struct_result.total:
        if struct_result and struct_result.skipped:
            caveats.append(f"chat-template probe skipped: {struct_result.skipped}")
        return None, 0.0

    weight = WEIGHTS["structure"]
    template_family = BY_KEY.get(struct_result.family_key)
    if template_family is None:
        return None, 0.0

    if struct_result.exact == struct_result.total:
        evidence.append(
            f"server prompt_tokens reproduce the {struct_result.template} chat "
            f"template exactly on {struct_result.exact}/{struct_result.total} prompts"
        )
        verdict.generation = template_family.key
        return template_family.key, weight

    if struct_result.constant_offset is not None:
        evidence.append(
            f"the {struct_result.template} template explains every prompt with a "
            f"constant {struct_result.constant_offset:+d}-token offset "
            f"({struct_result.total}/{struct_result.total} prompts)"
        )
        caveats.append(
            "template not reproduced exactly (the endpoint prepends its own prompt); "
            "the constant offset only that generation's tokenizer produces is what identifies it"
        )
        verdict.generation = template_family.key
        return template_family.key, weight * 0.6

    evidence.append(
        f"the {struct_result.template} template reproduces on "
        f"{struct_result.exact}/{struct_result.total} prompts; the endpoint injects "
        "variable context"
    )
    caveats.append("absolute prompt counts are template-dependent on this endpoint")
    return None, weight * 0.2


def identify(model_id, tok_result, struct_result, behaviour_result,
             capability_result=None, endpoint_result=None):
    verdict = Verdict()
    evidence = []
    caveats = []
    weighted = 0.0
    available = 0.0

    declared = guess_family(model_id)
    if declared:
        verdict.family_key = declared.key
        verdict.family_label = declared.label
        verdict.vendor = declared.vendor

    # Level 1 and 2: vocabulary, then generation.
    tok_weight_used = WEIGHTS["tokenizer"] if tok_result and not tok_result.skipped else 0.0
    struct_weight_used = WEIGHTS["structure"] if struct_result and not struct_result.skipped else 0.0
    vocab_key, vocab_score = _tokenizer_evidence(tok_result, verdict, evidence, caveats)
    available += tok_weight_used
    weighted += vocab_score

    gen_key, gen_score = _structure_evidence(struct_result, verdict, evidence, caveats)
    available += struct_weight_used
    weighted += gen_score

    if gen_key:
        family = BY_KEY[gen_key]
        consistent = (
            verdict.vocabulary is None
            or family.tokenizer_group == verdict.vocabulary
        )
        if consistent:
            verdict.family_key = family.key
            verdict.family_label = family.label
            verdict.vendor = family.vendor
            # An exact template reproduction resolves the ambiguity that the
            # tokenizer level had to discount; restore the withheld credit.
            if struct_result.exact == struct_result.total:
                weighted += WEIGHTS["tokenizer"] * 0.3
        else:
            caveats.append(
                f"template probe points to {family.label} but the tokenizer points to "
                f"{verdict.vocabulary}; the two disagree"
            )
    elif vocab_key:
        verdict.family_key = vocab_key
        verdict.family_label = BY_KEY[vocab_key].label
        verdict.vendor = BY_KEY[vocab_key].vendor

    # Level 3: capabilities.
    cap_score = _capability_score(capability_result)
    if cap_score is not None:
        available += WEIGHTS["capabilities"]
        weighted += WEIGHTS["capabilities"] * cap_score
        vision = getattr(capability_result, "vision", None) or {}
        tools = getattr(capability_result, "tools", None) or {}
        context = getattr(capability_result, "long_context", None) or {}
        if vision.get("supported"):
            evidence.append("accepts image input and reads a solid colour correctly")
        if tools.get("supported"):
            evidence.append("emits OpenAI-style tool calls")
        if context.get("supported"):
            tokens = context.get("prompt_tokens")
            evidence.append(
                f"retrieves a needle at ~{tokens:,} prompt tokens"
                if tokens else "retrieves a needle in a long padded context"
            )

    # Level 4: behaviour.
    if behaviour_result and getattr(behaviour_result, "reports", None):
        available += WEIGHTS["behaviour"]
        weighted += WEIGHTS["behaviour"] * 0.5
        claimed = (behaviour_result.reports.get("vendor") or "").strip()
        if claimed:
            evidence.append(f"self-reports vendor {claimed!r} (weak evidence)")
            if verdict.vendor.lower() not in claimed.lower():
                caveats.append(
                    f"self-report {claimed!r} contradicts the tokenizer/template evidence "
                    f"for {verdict.vendor}; DeepSeek V4 models frequently claim to be "
                    "OpenAI or Anthropic, so trust the measurements"
                )

    # Level 5: endpoint metadata.
    if endpoint_result is not None:
        available += WEIGHTS["endpoint"]
        weighted += WEIGHTS["endpoint"] * 0.8
        routing = getattr(endpoint_result, "routing", {}) or {}
        backend = routing.get("backend_model")
        if backend:
            evidence.append(f"backend model field: {backend}")
        provider = routing.get("final_provider")
        if provider:
            evidence.append(f"gateway routed to provider {provider}")

    verdict.confidence = min(1.0, weighted / available) if available else 0.0

    # Do not present a model-id guess as a measurement.
    if verdict.family_key and verdict.generation != verdict.family_key:
        declared_only = verdict.family_key == getattr(declared, "key", None)
        if declared_only:
            verdict.family_label += " (name from the model id, not confirmed by probes)"
            verdict.confidence *= 0.5
            caveats.append(
                "no probe confirmed the generation; the label merely echoes the model id"
            )

    verdict.evidence = evidence
    verdict.caveats = caveats
    return verdict
