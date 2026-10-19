"""Command-line entry point.

One command line is enough::

    model-fingerprint --url https://api.example.com/v1 --key sk-... --model <id>

Missing pieces fall back to environment variables (``MODEL_FINGERPRINT_URL``,
``MODEL_FINGERPRINT_MODEL``, ``MODEL_FINGERPRINT_KEY`` / ``OPENAI_API_KEY``)
and key files, so a CI job only needs ``model-fingerprint`` plus three env vars.
"""

import argparse
import os
import sys

from .checks import CHECKS, DEFAULT_ORDER
from .client import Client, ProbeError
from .config import Endpoint, normalize_base_url, resolve_key
from .families import (
    FAMILIES,
    cache_summary,
    download_reference_tokenizers,
    guess_family,
)
from .report import (
    Console,
    build_document,
    print_verdict,
    setup_stdout,
    write_json,
    write_markdown,
)
from .verdict import identify

ENV_URL = "MODEL_FINGERPRINT_URL"
ENV_MODEL = "MODEL_FINGERPRINT_MODEL"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="model-fingerprint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Identify which model an OpenAI-compatible endpoint actually serves.",
        epilog=(
            "examples:\n"
            "  model-fingerprint --url https://api.example.com/v1 --key sk-... "
            "--model my-model\n"
            "  model-fingerprint --provider openai --key sk-... --model gpt-5.5\n"
            "  model-fingerprint --url http://localhost:8000/v1 --model deepseek-ai/DeepSeek-V4\n"
            "  model-fingerprint --quick            # skip long-context and endpoint probes\n"
            "  model-fingerprint --list-models --url https://api.deepseek.com/v1 --key sk-...\n"
            "  MODEL_FINGERPRINT_URL=... MODEL_FINGERPRINT_MODEL=... "
            "MODEL_FINGERPRINT_KEY=... model-fingerprint\n"
        ),
    )
    parser.add_argument("--url", "--base-url", dest="url", help="base URL, with or without /v1")
    parser.add_argument("--provider", choices=sorted({
        "cline", "openai", "deepseek", "openrouter", "siliconflow", "moonshot",
        "dashscope", "zhipu", "groq", "together", "ollama", "lmstudio", "vllm",
    }), help="use a known provider's base URL")
    parser.add_argument("--key", "--api-key", dest="key", help="API key")
    parser.add_argument("--model", "-m", help="model id as the endpoint expects it")
    parser.add_argument("--header", action="append", default=[], metavar="NAME:VALUE",
                        help="extra HTTP header (repeatable)")
    parser.add_argument("--only", action="append", choices=sorted(CHECKS),
                        help="run only the named check (repeatable)")
    parser.add_argument("--skip", action="append", default=[], choices=sorted(CHECKS),
                        help="skip the named check (repeatable)")
    parser.add_argument("--no-long-context", action="store_true",
                        help="skip the ~150k-token needle test")
    parser.add_argument("--quick", action="store_true",
                        help="tokenizer + structure + behaviour only (fast, cheap)")
    parser.add_argument("--json", dest="json_out", metavar="PATH", help="write results to PATH")
    parser.add_argument("--markdown", dest="md_out", metavar="PATH", help="write a report to PATH")
    parser.add_argument("--quiet", "-q", action="store_true", help="verdict only")
    parser.add_argument("--list-models", action="store_true", help="print the endpoint's model ids and exit")
    parser.add_argument("--download-data", action="store_true",
                        help="fetch reference tokenizers from HuggingFace")
    parser.add_argument("--list-families", action="store_true",
                        help="show supported families and their reference assets")
    parser.add_argument("--retries", type=int, default=4, help="retries per request (default 4)")
    parser.add_argument("--timeout", type=int, default=180, help="request timeout in seconds")
    return parser


def _parse_headers(items):
    headers = {}
    for item in items:
        if ":" not in item:
            raise SystemExit(f"error: --header expects NAME:VALUE, got {item!r}")
        name, value = item.split(":", 1)
        headers[name.strip()] = value.strip()
    return headers


def _resolve_target(args):
    url = args.url or os.environ.get(ENV_URL)
    if args.provider and not url:
        from .config import PROVIDER_ENDPOINTS
        url = PROVIDER_ENDPOINTS[args.provider]
    if not url:
        raise SystemExit(
            "error: no endpoint. Pass --url https://api.example.com/v1, "
            f"--provider <name>, or set {ENV_URL}."
        )

    model = args.model or os.environ.get(ENV_MODEL) or ""
    if not model and not args.list_models:
        raise SystemExit(
            "error: no model id. Pass --model <id> or set " f"{ENV_MODEL}."
        )

    return normalize_base_url(url), model


def main(argv=None):
    setup_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_families:
        print(f"{'key':<14} {'label':<18} {'vendor':<10} reference")
        for family in FAMILIES:
            reference = family.tokenizer or family.tiktoken or "-"
            print(f"{family.key:<14} {family.label:<18} {family.vendor:<10} {reference}")
        print("\nbundled assets:")
        for key, status in cache_summary():
            print(f"  {key:<14} {status}")
        return 0

    if args.download_data:
        download_reference_tokenizers(force=True)
        return 0

    try:
        key = resolve_key(args.key)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    base_url, model = _resolve_target(args)

    endpoint = Endpoint(
        base_url=base_url,
        model=model,
        key=key,
        extra_headers=_parse_headers(args.header),
        retries=args.retries,
        timeout=args.timeout,
    )
    client = Client(endpoint)
    console = Console(quiet=args.quiet)

    if args.list_models:
        try:
            for model_id in client.list_models():
                print(model_id)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.quick:
        order = ["tokenizer", "structure", "behaviour"]
        args.skip = [name for name in CHECKS if name not in order]
        args.no_long_context = True

    order = [name for name in (args.only or DEFAULT_ORDER) if name not in args.skip]
    console(f"probing {endpoint.chat_url}")
    console(f"  model: {endpoint.model}")

    results = {}
    for name in order:
        console.section(name.upper())
        module = CHECKS[name]
        try:
            if name == "tokenizer":
                results[name] = module.run(client, console)
            elif name == "structure":
                results[name] = module.run(client, console, model_id=endpoint.model)
            elif name == "capabilities":
                results[name] = module.run(client, console, long_context=not args.no_long_context)
            else:
                results[name] = module.run(client, console)
        except ProbeError as exc:
            console(f"  [error] {exc}")
            results[name] = {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - a bad response should not abort the run
            console(f"  [error] {exc!r}")
            results[name] = {"error": repr(exc)}

    verdict = identify(
        endpoint.model,
        results.get("tokenizer"),
        results.get("structure"),
        results.get("behaviour"),
        results.get("capabilities"),
        results.get("endpoint"),
    )
    print_verdict(console, verdict, endpoint.model, endpoint.base_url)

    document = build_document(endpoint, verdict, results)
    if args.json_out:
        write_json(args.json_out, document)
        console(f"\n  json written to {args.json_out}")
    if args.md_out:
        write_markdown(args.md_out, document)
        console(f"  markdown written to {args.md_out}")

    if not args.json_out and not args.md_out and not args.quiet:
        console("\n  tip: --json report.json --markdown report.md to save this")

    return 0 if verdict.family_key else 1
