import json

import pytest

from model_fingerprint.client import Client, ProbeError
from model_fingerprint.config import Endpoint


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    @property
    def text(self):
        return json.dumps(self._payload) if not isinstance(self._payload, Exception) else str(self._payload)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}
        self.post_calls = 0

    def post(self, url, data=None, timeout=None):
        self.post_calls += 1
        self.calls.append((url, data))
        return self.responses.pop(0)


def make_client(responses, retries=1):
    endpoint = Endpoint(base_url="https://example.com/v1", model="test-model", key="k", retries=retries)
    client = Client(endpoint)
    client.session = FakeSession(responses)
    return client


def test_openai_envelope_is_normalized():
    payload = {
        "model": "served-model",
        "choices": [{"message": {"content": "hi", "reasoning": "r"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2},
    }
    client = make_client([FakeResponse(200, payload)])
    result = client.chat([{"role": "user", "content": "hello"}])
    assert result["content"] == "hi"
    assert result["reasoning"] == "r"
    assert result["usage"]["prompt_tokens"] == 7
    assert result["backend_model"] == "served-model"


def test_nested_data_envelope_is_unwrapped():
    payload = {
        "success": True,
        "data": {
            "model": "deepseek/deepseek-v4",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3},
        },
    }
    client = make_client([FakeResponse(200, payload)])
    result = client.chat([{"role": "user", "content": "hello"}])
    assert result["content"] == "ok"
    assert result["backend_model"] == "deepseek/deepseek-v4"


def test_openai_error_is_raised_without_retry():
    payload = {"error": {"message": "invalid api key"}}
    client = make_client([FakeResponse(401, payload)], retries=3)
    with pytest.raises(ProbeError, match="401"):
        client.chat([{"role": "user", "content": "hello"}])
    assert client.session.post_calls == 1


def test_cline_style_transient_error_is_retried():
    bad = FakeResponse(500, {"error": "empty response content", "success": False})
    good = FakeResponse(200, {
        "success": True,
        "data": {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 5}},
    })
    client = make_client([bad, good], retries=2)
    result = client.chat([{"role": "user", "content": "hello"}])
    assert result["content"] == "ok"
    assert client.session.post_calls == 2


def test_prompt_tokens_returns_none_without_usage():
    payload = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    client = make_client([FakeResponse(200, payload), FakeResponse(200, payload)])
    assert client.prompt_tokens("hello") is None


def test_request_body_is_utf8_encoded():
    payload = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1}}
    client = make_client([FakeResponse(200, payload)])
    client.chat([{"role": "user", "content": "中文"}])
    _, body = client.session.calls[0]
    assert isinstance(body, bytes)
    assert "中文".encode("utf-8") in body
