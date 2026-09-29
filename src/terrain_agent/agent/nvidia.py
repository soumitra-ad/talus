"""NVIDIA hosted NIM provider for the TALUS agent (OpenAI-compatible API).

TALUS talks to NVIDIA's *hosted* API only (``https://integrate.api.nvidia.com/v1``) through the
OpenAI-compatible ``/chat/completions`` endpoint. No local model, GPU, CUDA or container is
involved.

API-key handling (per-session, user-entered)
--------------------------------------------
* The key is supplied by the person using the app, typed into a masked input. It is never read
  from the environment, ``.env``, Streamlit secrets or any file -- this module has no code path
  that looks for one.
* It is held in a :class:`SessionSecret`, whose ``repr``/``str``/``format`` are redacted, so
  printing session state, a log line or an exception that happens to include the object never
  reveals it.
* A client is built per session from that secret. Nothing in this module is a module-level
  client or key, so one Streamlit session can never reach another session's key.
* The key travels only in the ``Authorization`` header to the fixed NVIDIA host above. It is
  never placed in a prompt, message, tool argument or tool result.
* Errors are mapped to fixed user-facing messages. SDK exception text is never shown or logged,
  because it can echo the raw response body or request details.

The LLM only interprets questions, selects tools and explains results. Every terrain number
comes from the deterministic ``dispatch_tool_call`` bridge (AGENTS.md §1).
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

try:
    import openai as _openai
except ImportError:  # pragma: no cover - openai is a declared dependency
    _openai = None

#: NVIDIA's hosted API. Fixed in code (not configurable) so configuration can never redirect a
#: user's key to another host.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: Per-request timeout for agent turns, and bounded retries for transient failures. The SDK
#: retries 408/429/5xx and connection errors with backoff.
NVIDIA_TIMEOUT_S = 90.0
NVIDIA_MAX_RETRIES = 2
#: The connect-time check is kept short and not retried, so a bad key fails fast.
NVIDIA_VALIDATION_TIMEOUT_S = 20.0

#: Upper bound on a single serialised tool result sent back to the model.
MAX_TOOL_RESULT_CHARS = 100_000
#: Upper bound on an accepted key's length; real keys are far shorter.
MAX_KEY_CHARS = 512

#: Fixed user-facing messages. Never built from exception text.
ERROR_MESSAGES: dict[str, str] = {
    "empty_key": "Please enter your NVIDIA API key.",
    "invalid_key_format": "That does not look like a valid NVIDIA API key. Please check it and try again.",
    "auth": "NVIDIA API authentication failed. Please check your API key.",
    "rate_limited": "NVIDIA API rate limit reached. Please try again later.",
    "timeout": "NVIDIA request timed out. Please try again.",
    "unavailable": "NVIDIA AI service is temporarily unavailable. Please try again.",
    "network": "Unable to reach NVIDIA AI service.",
    "model_not_found": "The configured NVIDIA model is not available on NVIDIA's hosted API. The operator must check NVIDIA_MODEL.",
    "tools_unsupported": "The configured NVIDIA model did not accept tool calling, which TALUS requires. The operator must choose another NVIDIA_MODEL.",
    "bad_request": "NVIDIA AI could not process this request. Please rephrase the question and retry.",
    "sdk_missing": "The NVIDIA client library is not installed on this server.",
}


# ---------------------------------------------------------------------------
# Session secret
# ---------------------------------------------------------------------------


class SessionSecret:
    """Holds one session's API key; every textual representation is redacted.

    ``reveal()`` is the only way to read the value, and is called only when building the
    HTTP client. The object is deliberately not a ``str`` subclass, so it cannot be
    concatenated into a prompt or log message by accident.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def clear(self) -> None:
        self._value = ""

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return "SessionSecret(<redacted>)"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return repr(self)

    def __eq__(self, other: object) -> bool:  # identity only: never compare secret values
        return self is other

    __hash__ = object.__hash__


def normalize_key(raw: Any) -> tuple[str | None, str | None]:
    """Return ``(key, None)`` for a usable key or ``(None, error_category)``."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "empty_key"
    key = raw.strip()
    # A header value cannot carry whitespace or control characters; reject rather than send.
    if len(key) > MAX_KEY_CHARS or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in key):
        return None, "invalid_key_format"
    return key, None


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def classify_nvidia_error(exc: BaseException) -> tuple[str, str]:
    """Map an SDK/transport exception to ``(category, fixed user-facing message)``.

    Only the exception *type* and HTTP status code are inspected; its text is never used.
    """
    import httpx

    status = getattr(exc, "status_code", None)
    if _openai is not None:
        if isinstance(exc, _openai.APITimeoutError):
            category = "timeout"
        elif isinstance(exc, _openai.APIConnectionError):
            category = "network"
        elif isinstance(exc, _openai.APIStatusError):
            category = _category_for_status(status)
        else:
            category = None
    else:  # pragma: no cover
        category = None
    if category is None:
        if isinstance(status, int):
            category = _category_for_status(status)
        elif isinstance(exc, (TimeoutError, httpx.TimeoutException)):
            category = "timeout"
        elif isinstance(exc, (ConnectionError, httpx.TransportError, OSError)):
            category = "network"
        else:
            category = "unavailable"
    return category, ERROR_MESSAGES[category]


def _category_for_status(status: Any) -> str:
    if status in (401, 403):
        return "auth"
    if status == 408:
        return "timeout"
    if status == 429:
        return "rate_limited"
    if status == 404:
        return "model_not_found"
    if status in (400, 422):
        return "bad_request"
    return "unavailable"


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def create_client(
    secret: SessionSecret,
    *,
    timeout_s: float = NVIDIA_TIMEOUT_S,
    max_retries: int = NVIDIA_MAX_RETRIES,
    http_client: Any | None = None,
) -> Any:
    """Build an OpenAI-compatible client bound to NVIDIA's hosted API and this session's key.

    ``http_client`` (an ``httpx.Client``) exists so tests can substitute a mock transport.
    """
    if _openai is None:  # pragma: no cover
        raise RuntimeError("openai package not installed")
    return _openai.OpenAI(
        api_key=secret.reveal(),
        base_url=NVIDIA_BASE_URL,
        timeout=timeout_s,
        max_retries=max_retries,
        http_client=http_client,
    )


# ---------------------------------------------------------------------------
# Connection check (runs only when the user clicks "Connect NVIDIA AI")
# ---------------------------------------------------------------------------


@dataclass
class ConnectionReport:
    """Outcome of a connect attempt. Contains no key material and no exception text."""

    ok: bool
    category: str | None
    message: str
    model: str
    model_listed: bool | None = None
    tool_calling: bool | None = None
    latency_s: float | None = None


_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "talus_connection_probe",
        "description": "Connection check. Do not call.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def validate_nvidia_key(
    raw_key: Any,
    model: str,
    *,
    http_client: Any | None = None,
    timeout_s: float = NVIDIA_VALIDATION_TIMEOUT_S,
) -> tuple[ConnectionReport, "NvidiaAgentClient | None"]:
    """Check a user-entered key against NVIDIA's hosted API and the configured model.

    1. Reject an empty or malformed key without any network call.
    2. List hosted models (if the listing is reachable) and confirm ``model`` is among them.
    3. Send one minimal chat completion that includes a tool definition, proving that the key
       authenticates and that the model accepts tool calling.

    No NASA request or terrain analysis happens here. On success the returned client is ready
    for the agent; on failure it is ``None`` and the report carries a fixed message.
    """
    if _openai is None:  # pragma: no cover
        return ConnectionReport(False, "sdk_missing", ERROR_MESSAGES["sdk_missing"], model), None

    key, problem = normalize_key(raw_key.reveal() if isinstance(raw_key, SessionSecret) else raw_key)
    if problem:
        return ConnectionReport(False, problem, ERROR_MESSAGES[problem], model), None

    secret = SessionSecret(key)
    probe = create_client(secret, timeout_s=timeout_s, max_retries=0, http_client=http_client)
    report = ConnectionReport(False, None, "", model)
    started = time.monotonic()
    try:
        try:
            listed = {m.id for m in probe.models.list()}
            report.model_listed = model in listed
        except _openai.APIStatusError as exc:
            if exc.status_code in (401, 403):
                raise
            report.model_listed = None  # listing unavailable: the chat probe below decides
        except _openai.APIConnectionError:
            raise
        if report.model_listed is False:
            report.category = "model_not_found"
            report.message = ERROR_MESSAGES["model_not_found"]
            return report, None

        try:
            probe.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word OK."}],
                tools=[_PROBE_TOOL],
                tool_choice="auto",
                max_tokens=32,
                temperature=0,
            )
        except _openai.BadRequestError:
            report.category = "tools_unsupported"
            report.message = ERROR_MESSAGES["tools_unsupported"]
            return report, None
        report.tool_calling = True
    except Exception as exc:  # noqa: BLE001 - mapped to a fixed message; text never shown
        report.category, report.message = classify_nvidia_error(exc)
        log.warning("NVIDIA connect failed: error_category=%s exception=%s", report.category, type(exc).__name__)
        return report, None
    finally:
        report.latency_s = round(time.monotonic() - started, 2)
        if http_client is None:  # an injected transport is shared with the agent client
            probe.close()

    report.ok = True
    report.message = "NVIDIA NIM Connected"
    log.info("NVIDIA connect succeeded: model=%s latency_s=%.2f", model, report.latency_s)
    agent_client = NvidiaAgentClient(create_client(secret, http_client=http_client), model)
    secret.clear()
    return report, agent_client


# ---------------------------------------------------------------------------
# Agent chat session over /chat/completions with tool calling
# ---------------------------------------------------------------------------


@dataclass
class NvidiaFunctionCall:
    """A tool call requested by the model. ``args`` is ``None`` when the model sent
    arguments that were not a JSON object."""

    id: str
    name: str
    args: dict[str, Any] | None


@dataclass
class NvidiaTurn:
    """One model reply, exposing the same two attributes the agent loop reads."""

    function_calls: list[NvidiaFunctionCall] | None = None
    text: str | None = None


@dataclass
class ToolResultPart:
    """A deterministic tool result to send back for a given tool call id."""

    tool_call_id: str
    name: str
    result: dict[str, Any]


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def openai_tool_definitions(declarations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert TALUS tool declarations to the OpenAI ``tools`` format."""
    return [
        {
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d.get("description", ""),
                "parameters": d.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for d in declarations
    ]


def _history_to_messages(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert normalised ``{"role": "user"|"model", "parts": [{"text"}]}`` turns."""
    messages: list[dict[str, Any]] = []
    for turn in history or []:
        role = "assistant" if turn.get("role") == "model" else "user"
        text = "\n".join(p.get("text", "") for p in turn.get("parts", []) if isinstance(p, dict))
        if text.strip():
            messages.append({"role": role, "content": text})
    return messages


def _serialise_result(result: dict[str, Any]) -> str:
    text = json.dumps({"result": result}, default=str)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = json.dumps({
            "result": {
                "status": result.get("status"),
                "note": "The full tool result was too large to send; only its status is shown.",
            }
        })
    return text


class NvidiaChatSession:
    """A single agent turn's conversation with NVIDIA, kept in OpenAI message format."""

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        system_prompt: str,
        tools: list[dict[str, Any]],
        history: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._tools = tools
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self.messages += _history_to_messages(history)

    def send_message(self, message: str | list[ToolResultPart]) -> NvidiaTurn:
        if isinstance(message, str):
            self.messages.append({"role": "user", "content": message})
        else:
            for part in message:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": part.tool_call_id,
                    "content": _serialise_result(part.result),
                })

        response = self._client.chat.completions.create(
            model=self._model,
            messages=self.messages,
            tools=self._tools,
            tool_choice="auto",
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            return NvidiaTurn()
        msg = choices[0].message
        content = msg.content if isinstance(msg.content, str) else None
        if content:
            content = _THINK_RE.sub("", content).strip()

        calls: list[NvidiaFunctionCall] = []
        raw_calls: list[dict[str, Any]] = []
        for index, tc in enumerate(msg.tool_calls or []):
            fn = getattr(tc, "function", None)
            name = str(getattr(fn, "name", "") or "")
            arguments = getattr(fn, "arguments", None) or "{}"
            call_id = str(getattr(tc, "id", "") or f"call_{uuid.uuid4().hex[:12]}_{index}")
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            except (TypeError, ValueError):
                parsed = None
            calls.append(NvidiaFunctionCall(call_id, name, parsed if isinstance(parsed, dict) else None))
            raw_calls.append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)},
            })

        assistant: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if raw_calls:
            assistant["tool_calls"] = raw_calls
        self.messages.append(assistant)
        return NvidiaTurn(function_calls=calls or None, text=content)


@dataclass
class NvidiaAgentClient:
    """A session's NVIDIA connection: an OpenAI-compatible client plus the verified model.

    Created only by :func:`validate_nvidia_key` (or tests). Holds the only reference to the
    session's key, inside the SDK client. ``close()`` releases it.
    """

    client: Any
    model: str
    max_tokens: int = 4096
    _closed: bool = field(default=False, repr=False)

    def __repr__(self) -> str:
        return f"NvidiaAgentClient(model={self.model!r})"

    def start_chat(
        self,
        *,
        system_prompt: str,
        tools: list[dict[str, Any]],
        history: list[dict[str, Any]] | None,
        temperature: float,
    ) -> NvidiaChatSession:
        if self._closed:
            raise RuntimeError("NVIDIA client has been disconnected")
        return NvidiaChatSession(
            self.client, self.model, system_prompt=system_prompt, tools=tools,
            history=history, temperature=temperature, max_tokens=self.max_tokens,
        )

    def close(self) -> None:
        """Drop the SDK client (and with it the key) for this session."""
        self._closed = True
        client, self.client = self.client, None
        try:
            if client is not None and hasattr(client, "close"):
                client.close()
        except Exception:  # noqa: BLE001
            log.debug("NVIDIA client close failed", exc_info=False)


__all__ = [
    "ConnectionReport",
    "ERROR_MESSAGES",
    "NVIDIA_BASE_URL",
    "NvidiaAgentClient",
    "NvidiaChatSession",
    "NvidiaFunctionCall",
    "NvidiaTurn",
    "SessionSecret",
    "ToolResultPart",
    "classify_nvidia_error",
    "create_client",
    "normalize_key",
    "openai_tool_definitions",
    "validate_nvidia_key",
]
