"""Capability probes: vision, tool calling, long context, reasoning output."""

import base64
import struct
import zlib
from dataclasses import dataclass, field

from ..client import ProbeError


@dataclass
class CapabilityResult:
    vision: object = None
    tools: object = None
    long_context: object = None
    reasoning_field: object = None
    skipped: dict = field(default_factory=dict)

    def summary(self):
        return {
            "vision": self.vision,
            "tools": self.tools,
            "long_context": self.long_context,
            "reasoning_field": self.reasoning_field,
            "skipped": self.skipped,
        }


def _solid_png(width=48, height=48, rgb=(255, 0, 0)):
    def chunk(tag, payload):
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def check_vision(client):
    image = _solid_png()
    try:
        result = client.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? One word."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image}"},
                        },
                    ],
                }
            ],
            max_tokens=800,
        )
    except ProbeError as exc:
        return {"supported": False, "error": str(exc)[:160]}
    answer = result["content"].strip()
    return {"supported": "red" in answer.lower(), "answer": answer[:80]}


def check_tools(client):
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    try:
        result = client.chat(
            [{"role": "user", "content": "What's the weather in Tokyo? Use the tool."}],
            max_tokens=1500,
            tools=[tool],
        )
    except ProbeError as exc:
        return {"supported": False, "error": str(exc)[:160]}
    calls = result["tool_calls"]
    return {
        "supported": bool(calls),
        "finish_reason": result["finish_reason"],
        "call": calls[0].get("function") if calls else None,
    }


def _needle_prompt(target_chars, needle, position=0.5):
    filler = "This is unrelated reference text used only to pad the context window. " * 20
    blocks = []
    size = 0
    index = 0
    while size < target_chars:
        block = f"[block {index}] {filler}\n"
        blocks.append(block)
        size += len(block)
        index += 1
    text = "".join(blocks)
    cut = int(len(text) * position)
    text = text[:cut] + f"\nIMPORTANT FACT: the access code is {needle}.\n" + text[cut:]
    return text + "\nQuestion: What is the access code mentioned in the text? Reply with just the code."


def check_long_context(client, chars=620_000, needle="ZEBRA-4471"):
    prompt = _needle_prompt(chars, needle)
    try:
        result = client.chat([{"role": "user", "content": prompt}], max_tokens=2000)
    except ProbeError as exc:
        return {"supported": False, "error": str(exc)[:160], "chars": chars}
    usage = result["usage"] or {}
    return {
        "supported": needle in result["content"],
        "prompt_tokens": usage.get("prompt_tokens"),
        "answer": result["content"].strip()[:60],
        "chars": chars,
    }


def check_reasoning_field(client):
    try:
        result = client.chat(
            [{"role": "user", "content": "What is 17 * 23? Think step by step."}],
            max_tokens=2000,
        )
    except ProbeError as exc:
        return {"present": False, "error": str(exc)[:160]}
    details = (result["usage"] or {}).get("completion_tokens_details") or {}
    return {
        "present": bool(result["reasoning"]),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "finish_reason": result["finish_reason"],
    }


def run(client, log, long_context=True):
    result = CapabilityResult()

    log("  Vision")
    result.vision = check_vision(client)
    log(f"    {result.vision}")

    log("  Tool calling")
    result.tools = check_tools(client)
    log(f"    {result.tools}")

    log("  Reasoning output field")
    result.reasoning_field = check_reasoning_field(client)
    log(f"    {result.reasoning_field}")

    if long_context:
        log("  Long context (~150k tokens)")
        result.long_context = check_long_context(client)
        log(f"    {result.long_context}")
    else:
        result.skipped["long_context"] = "--no-long-context"
    return result
