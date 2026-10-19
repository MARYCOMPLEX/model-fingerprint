"""HTTP client for OpenAI-compatible chat endpoints."""

import json
import time

import requests

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
TRANSIENT_MESSAGES = (
    "empty response content",   # Cline: upstream produced nothing
    "overloaded",
    "rate limit",
    "timed out",
    "timeout",
)


class ProbeError(RuntimeError):
    """The endpoint rejected or failed the request in a way we cannot use."""


class Client:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.session = requests.Session()
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            **(endpoint.extra_headers or {}),
        }
        if endpoint.key:
            headers["Authorization"] = f"Bearer {endpoint.key}"
        self.session.headers.update(headers)
        self.last_status = None

    def _should_retry(self, status, error_text):
        if status in RETRYABLE_STATUS:
            return True
        lowered = str(error_text).lower()
        return any(marker in lowered for marker in TRANSIENT_MESSAGES)

    def chat(self, messages, **options):
        """POST ``/chat/completions`` and return a normalized result dict.

        Normalized shape::

            {"content", "reasoning", "tool_calls", "finish_reason",
             "usage", "backend_model", "routing", "raw"}
        """
        payload = {"model": options.pop("model", self.endpoint.model), "messages": messages}
        payload.update({k: v for k, v in options.items() if v is not None})
        body_text = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        last_error = "no attempt made"
        for attempt in range(self.endpoint.retries):
            started = time.perf_counter()
            try:
                response = self.session.post(
                    self.endpoint.chat_url,
                    data=body_text,
                    timeout=self.endpoint.timeout,
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                time.sleep(1.0 + attempt)
                continue
            self.last_status = response.status_code
            elapsed = time.perf_counter() - started
            try:
                body = response.json()
            except ValueError:
                last_error = f"HTTP {response.status_code}, non-JSON body: {response.text[:200]!r}"
                if self._should_retry(response.status_code, response.text):
                    time.sleep(1.0 + attempt)
                    continue
                raise ProbeError(last_error)

            if response.status_code == 200 and body.get("error") is None:
                return self._normalize(body, elapsed)

            # Cline reports failures with a 200-shaped envelope; OpenAI-style
            # endpoints use {"error": {...}}.  Handle both.
            error = body.get("error") or body.get("message") or body
            last_error = f"HTTP {response.status_code}: {error}"
            if not self._should_retry(response.status_code, error):
                raise ProbeError(last_error)
            time.sleep(1.5 * (attempt + 1))

        raise ProbeError(f"failed after {self.endpoint.retries} attempts: {last_error}")

    def _normalize(self, body, elapsed):
        envelope = body.get("data") if isinstance(body.get("data"), dict) and "choices" in body["data"] else body
        if body.get("success") is False:
            raise ProbeError(str(body.get("error")))
        choices = envelope.get("choices") or []
        if not choices:
            raise ProbeError(f"no choices in response: {json.dumps(body)[:200]}")
        choice = choices[0]
        message = choice.get("message") or {}
        metadata = message.get("provider_metadata") or {}
        gateway = metadata.get("gateway") or {}
        return {
            "content": message.get("content") or "",
            "reasoning": message.get("reasoning") or "",
            "tool_calls": message.get("tool_calls") or [],
            "finish_reason": choice.get("finish_reason"),
            "usage": envelope.get("usage") or {},
            "backend_model": envelope.get("model"),
            "routing": gateway.get("routing") or {},
            "elapsed": elapsed,
            "raw": body,
        }

    # -- convenience wrappers -------------------------------------------------

    def text(self, prompt, system=None, max_tokens=1200, **options):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, max_tokens=max_tokens, **options)["content"].strip()

    def prompt_tokens(self, prompt, max_tokens=400, allow_retry=True):
        """Token count the server reports for a prompt.

        Endpoints that only count after generating (or refuse tiny budgets) may
        fail; retrying with a larger budget keeps the measurement usable.
        """
        budgets = (max_tokens, 2000) if allow_retry else (max_tokens,)
        for budget in budgets:
            try:
                result = self.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=budget,
                )
            except ProbeError:
                continue
            tokens = (result["usage"] or {}).get("prompt_tokens")
            if isinstance(tokens, int) and tokens > 0:
                return tokens
        return None

    def list_models(self):
        url = self.endpoint.base_url.rstrip("/") + "/models"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        body = response.json()
        entries = body.get("data") if isinstance(body.get("data"), list) else body
        return [item.get("id") for item in entries if isinstance(item, dict) and item.get("id")]
