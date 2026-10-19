"""Rendering: console output, JSON and Markdown."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def setup_stdout():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


class Console:
    def __init__(self, quiet=False):
        self.quiet = quiet

    def __call__(self, text=""):
        if not self.quiet:
            print(text)

    def section(self, title):
        if self.quiet:
            return
        print()
        print("=" * 78)
        print(title)
        print("=" * 78)


def print_verdict(console, verdict, model_id, base_url):
    console.section("IDENTIFICATION")
    print(f"  endpoint   {base_url}")
    print(f"  model id   {model_id}")
    print(f"  verdict    {verdict.family_label} ({verdict.vendor})")
    print(f"  confidence {verdict.confidence:.0%}")
    if verdict.evidence:
        print("\n  evidence")
        for item in verdict.evidence:
            print(f"    + {item}")
    if verdict.caveats:
        print("\n  caveats")
        for item in verdict.caveats:
            print(f"    ! {item}")


def build_document(endpoint, verdict, results):
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": {
            "base_url": endpoint.base_url,
            "model": endpoint.model,
        },
        "verdict": verdict.to_dict(),
        "checks": {name: _summarize(value) for name, value in results.items()},
    }


def _summarize(value):
    if hasattr(value, "summary"):
        return value.summary()
    if isinstance(value, dict):
        return value
    return value


def write_json(path, document):
    Path(path).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def render_markdown(document):
    verdict = document["verdict"]
    endpoint = document["endpoint"]
    lines = [
        "# Model fingerprint report",
        "",
        f"- endpoint: `{endpoint['base_url']}`",
        f"- model id: `{endpoint['model']}`",
        f"- verdict: **{verdict['label']}** ({verdict['vendor']})",
        f"- confidence: **{verdict['confidence']:.0%}**",
        f"- generated: `{document['generated_at']}`",
        "",
        "## Evidence",
        "",
    ]
    lines += [f"- {item}" for item in verdict["evidence"]] or ["- (none)"]
    lines += ["", "## Caveats", ""]
    lines += [f"- {item}" for item in verdict["caveats"]] or ["- (none)"]

    checks = document["checks"]

    tokenizer = checks.get("tokenizer") or {}
    if tokenizer.get("scores"):
        lines += ["", "## Tokenizer delta matrix", "",
                  "| family | fragments | special tokens | score |", "|---|---|---|---|"]
        for key, value in tokenizer["scores"].items():
            lines.append(
                f"| {value.get('label', key)} | {value.get('fragments', '-')} "
                f"| {value.get('specials', '-')} | {value.get('score', '-')} |"
            )

    structure = checks.get("structure") or {}
    if structure.get("template"):
        lines += ["", "## Template reproduction", "",
                  f"- template: {structure['template']}",
                  f"- exact: {structure['exact']}",
                  f"- constant offset: {structure['constant_offset']}"]

    caps = checks.get("capabilities") or {}
    if caps:
        lines += ["", "## Capabilities", "",
                  f"- vision: {caps.get('vision')}",
                  f"- tools: {caps.get('tools')}",
                  f"- long context: {caps.get('long_context')}"]

    ep = checks.get("endpoint") or {}
    if ep:
        lines += ["", "## Endpoint", "",
                  f"- listed models: {ep.get('listed_models')}",
                  f"- pricing: {ep.get('pricing')}",
                  f"- tokens/s: {ep.get('tokens_per_second')}",
                  f"- backend model: {(ep.get('routing') or {}).get('backend_model')}"]
    return "\n".join(lines) + "\n"


def write_markdown(path, document):
    Path(path).write_text(render_markdown(document), encoding="utf-8")
