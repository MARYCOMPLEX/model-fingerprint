# Contributing

## Setup

```bash
git clone https://github.com/MARYCOMPLEX/model-fingerprint.git
cd model-fingerprint
pip install -r requirements.txt
python -m pytest tests -q
```

The test suite is fully offline. Any test that would need a live endpoint
belongs behind an environment guard and must not run by default.

## Adding a model family

1. Add a `Family(...)` entry to `FAMILIES` in
   `model_fingerprint/families.py`.
2. Provide a reference tokenizer, one of:
   - `tokenizer.json` from HuggingFace, compressed to
     `model_fingerprint/data/<name>.json.gz`; or
   - a `tiktoken` encoding name (`tiktoken="o200k_base"`).
3. If the family publishes an `encode_messages` implementation, vendor it into
   `model_fingerprint/data/` and set `template_module`, `group`, and add the key
   to `DEEPSEEK_GENERATIONS`-style lists in `checks/structure.py`. That enables
   the generation-level check.
4. Run `model-fingerprint --list-families` to confirm the asset is picked up.

## Measurement rules

- Never score a probe whose observable the gateway rewrites. The special-token
  rows are excluded automatically when the escape signature is detected.
- Prefer *deltas* over absolute counts. Absolute counts depend on the chat
  template and on any system prompt the provider injects.
- Capabilities never identify a family on their own; they only corroborate.
- Self-reports are recorded but weighted at 3% and always called out when they
  contradict measurements.

## Style

- `black` + `ruff` if available.
- Type annotations on public functions.
- One probe group per module in `model_fingerprint/checks/`.
