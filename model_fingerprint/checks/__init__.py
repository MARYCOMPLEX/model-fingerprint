"""Probe groups.  Each module exposes ``run(client, log, **kwargs)``."""

from . import behaviour, capabilities, endpoint, structure, tokenizer

CHECKS = {
    "tokenizer": tokenizer,
    "structure": structure,
    "behaviour": behaviour,
    "capabilities": capabilities,
    "endpoint": endpoint,
}

DEFAULT_ORDER = ("tokenizer", "structure", "behaviour", "capabilities", "endpoint")

# How much each group contributes to the final identification verdict.
WEIGHTS = {
    "tokenizer": 0.55,
    "structure": 0.25,
    "capabilities": 0.12,
    "behaviour": 0.03,
    "endpoint": 0.05,
}

__all__ = ["CHECKS", "DEFAULT_ORDER", "WEIGHTS"]
