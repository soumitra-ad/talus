"""Offline fake of NVIDIA's hosted OpenAI-compatible API, for driving the real ``openai`` SDK
through ``httpx.MockTransport``. Error responses echo the Authorization header on purpose so
tests can prove the key never leaks into messages or logs."""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx

MODEL = "nvidia/test-tool-model"


def final(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def tool_calls(*calls: tuple[str, dict[str, Any]], raw_args: str | None = None) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{i}_{name}",
                "type": "function",
                "function": {"name": name, "arguments": raw_args if raw_args is not None else json.dumps(args)},
            }
            for i, (name, args) in enumerate(calls)
        ],
    }


Step = dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]]


class FakeNvidia:
    """Records every request; answers /models and /chat/completions from a script."""

    def __init__(
        self,
        script: list[Step] | None = None,
        *,
        models: tuple[str, ...] = (MODEL,),
        models_status: int = 200,
        chat_status: int | None = None,
        chat_exc: Exception | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.script = list(script or [])
        self.models = models
        self.models_status = models_status
        self.chat_status = chat_status
        self.chat_exc = chat_exc
        self.fail_after = fail_after
        self.requests: list[dict[str, Any]] = []

    @property
    def chat_bodies(self) -> list[dict[str, Any]]:
        return [r["body"] for r in self.requests if r["path"].endswith("/chat/completions")]

    def _echo_error(self, status: int, request: httpx.Request) -> httpx.Response:
        # A hostile/buggy upstream that echoes the credential: must never surface anywhere.
        return httpx.Response(status, json={"error": {"message": f"bad request with {request.headers.get('authorization')}"}})

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append({
            "method": request.method, "host": request.url.host, "path": request.url.path,
            "auth": request.headers.get("authorization"), "body": body,
        })
        if request.url.path.endswith("/models"):
            if self.models_status != 200:
                return self._echo_error(self.models_status, request)
            return httpx.Response(200, json={"object": "list", "data": [{"id": m, "object": "model"} for m in self.models]})
        if self.chat_exc is not None:
            raise self.chat_exc
        chat_count = len(self.chat_bodies)
        if self.chat_status is not None and (self.fail_after is None or chat_count > self.fail_after):
            return self._echo_error(self.chat_status, request)
        step = self.script.pop(0) if self.script else final("OK")
        message = step(body) if callable(step) else step
        return httpx.Response(200, json={
            "id": f"chatcmpl-{chat_count}", "object": "chat.completion", "created": 1, "model": body["model"],
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
        })

    def http_client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))
