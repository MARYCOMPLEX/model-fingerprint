"""Endpoint hygiene: model listing, pricing, caching, refusal semantics."""

import re
import statistics
import time
from dataclasses import dataclass, field

from ..client import ProbeError


@dataclass
class EndpointResult:
    listed_models: list = field(default_factory=list)
    listed_in_model: object = None
    pricing: dict = field(default_factory=dict)
    cache_hits: list = field(default_factory=list)
    latency_tps: list = field(default_factory=list)
    small_budget_error: object = None
    routing: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)

    def summary(self):
        return {
            "listed_models": len(self.listed_models),
            "listed_in_model": self.listed_in_model,
            "pricing": self.pricing,
            "cache_hits": self.cache_hits,
            "tokens_per_second": self.latency_tps,
            "small_budget_error": self.small_budget_error,
            "routing": self.routing,
            "errors": self.errors,
        }


def check_models(client, log):
    try:
        listed = client.list_models()
    except Exception as exc:  # noqa: BLE001 - endpoint may not implement /models
        log(f"  /models unavailable: {exc}")
        return []
    log(f"  /models returned {len(listed)} ids")
    return listed


def check_pricing(client, log):
    probes = (
        ("Reply with OK.", 32),
        ("Write exactly 300 words about the moon.", 2000),
        ("Explain photosynthesis in detail, about 500 words.", 3000),
        ("Give me 20 numbered facts about octopuses.", 3000),
    )
    points = []
    for prompt, limit in probes:
        try:
            result = client.chat([{"role": "user", "content": prompt}], max_tokens=limit)
        except ProbeError:
            continue
        usage = result["usage"] or {}
        cost = usage.get("cost")
        if cost is None:
            continue
        points.append((usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), cost))
    if len(points) < 2:
        log("  usage.cost not reported; skipping price regression")
        return {}
    try:
        import numpy as np
    except ImportError:
        return {}
    matrix = np.array([[p, c] for p, c, _ in points], dtype=float)
    costs = np.array([c for _, _, c in points], dtype=float)
    solution, *_ = np.linalg.lstsq(matrix, costs, rcond=None)
    pricing = {
        "input_per_million": round(float(solution[0]) * 1e6, 4),
        "output_per_million": round(float(solution[1]) * 1e6, 4),
    }
    log(f"  estimated ${pricing['input_per_million']}/M in, ${pricing['output_per_million']}/M out")
    return pricing


def check_caching(client, log):
    prefix = (
        "You are a meticulous assistant. "
        + "The following is reference material. " * 60
        + "Remember: the secret word is PINEAPPLE. " * 30
    )
    messages = [{"role": "user", "content": prefix + "\nQuestion: What is the capital of France?"}]
    hits = []
    for _ in range(3):
        try:
            result = client.chat(messages, max_tokens=100)
        except ProbeError:
            break
        details = (result["usage"] or {}).get("prompt_tokens_details") or {}
        hits.append(details.get("cached_tokens") or 0)
        time.sleep(0.3)
    log(f"  cached prompt tokens per repeat: {hits}")
    return hits


def check_latency(client, log):
    speeds = []
    for _ in range(3):
        try:
            result = client.chat(
                [{"role": "user", "content": "Write about 600 words on the history of tea."}],
                max_tokens=2000,
            )
        except ProbeError:
            break
        completion = (result["usage"] or {}).get("completion_tokens", 0)
        if result["elapsed"] > 0:
            speeds.append(round(completion / result["elapsed"], 1))
    if speeds:
        log(f"  ~{statistics.mean(speeds):.0f} tokens/s  samples={speeds}")
    return speeds


def check_small_budget(client, log):
    """Some gateways (Cline) report an exhausted budget as HTTP 500."""
    prompt = "Explain the difference between TCP and UDP in exactly 50 words."
    outcome = {}
    for limit in (16, 200):
        successes = 0
        for _ in range(2):
            try:
                client.chat([{"role": "user", "content": prompt}], max_tokens=limit)
                successes += 1
            except ProbeError:
                pass
        outcome[limit] = successes
        log(f"  max_tokens={limit:<4} {successes}/2 succeeded")
    return outcome


def run(client, log):
    result = EndpointResult()
    log("  Model listing")
    result.listed_models = check_models(client, log)
    model = client.endpoint.model
    result.listed_in_model = model in result.listed_models if result.listed_models else None

    log("  Pricing")
    result.pricing = check_pricing(client, log)

    log("  Context caching")
    result.cache_hits = check_caching(client, log)

    log("  Throughput")
    result.latency_tps = check_latency(client, log)

    log("  Small-budget semantics")
    result.small_budget_error = check_small_budget(client, log)

    try:
        first = client.chat([{"role": "user", "content": "Say OK."}], max_tokens=200)
        result.routing = {
            "backend_model": first.get("backend_model"),
            "final_provider": (first.get("routing") or {}).get("finalProvider"),
            "canonical": (first.get("routing") or {}).get("canonicalSlug"),
        }
        log(f"  backend model field: {result.routing['backend_model']!r}")
        if result.routing["final_provider"]:
            log(f"  provider: {result.routing['final_provider']}")
    except ProbeError as exc:
        result.errors["routing"] = str(exc)[:200]
    return result
