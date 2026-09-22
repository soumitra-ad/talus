"""A deterministic, offline stand-in for the Gemini client used to test the TALUS agent loop.

``TALUSAgent`` only ever calls ``client.chats.create(...)`` and then, repeatedly,
``chat.send_message(...)``, reading ``response.function_calls`` and ``response.text`` off the
result. This module implements exactly that surface with a scripted sequence of responses, so
the full agent loop (tool selection, multi-step tool sequencing, clarification questions, error
handling, bounded iteration) can be exercised in tests without network access, an API key, or
any real Gemini call. Nothing here performs terrain calculations; it only stands in for the
language model's turn-taking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Union


@dataclass
class MockFunctionCall:
    """Duck-compatible stand-in for ``google.genai.types.FunctionCall``."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockResponse:
    """Duck-compatible stand-in for ``google.genai.types.GenerateContentResponse``.

    Exposes only the two attributes ``TALUSAgent`` reads: ``function_calls`` (a list of
    ``MockFunctionCall``, or ``None``/empty for a final answer) and ``text``.
    """

    function_calls: list[MockFunctionCall] | None = None
    text: str | None = None


# A script step is either a fixed MockResponse, or a callable that receives the message just
# sent to the chat (the prompt, or the list of function-response parts) and returns one. The
# callable form lets a test branch on what the agent actually sent, e.g. to only return a
# specific tool result if that is what the mock previously asked for.
ScriptStep = Union[MockResponse, Callable[[Any], MockResponse]]


class ScriptExhaustedError(RuntimeError):
    """Raised when the agent asks the mock model for another turn than was scripted.

    This is a test bug (the script under-specifies the conversation), not a simulated model
    failure — use :class:`FailingGeminiClient` to simulate an actual model/API failure.
    """


class MockChat:
    """Stand-in for ``google.genai.chats.Chat``."""

    def __init__(self, script: list[ScriptStep], *, received_history: Any = None) -> None:
        self._script = list(script)
        self.sent_messages: list[Any] = []
        #: The ``history`` the agent passed to ``chats.create`` for this chat -- useful for
        #: tests that check history bounding/truncation without a real model.
        self.received_history = received_history

    def send_message(self, message: Any) -> MockResponse:
        self.sent_messages.append(message)
        if not self._script:
            raise ScriptExhaustedError(
                "MockChat script exhausted before the agent produced a final response. "
                "Add another scripted turn."
            )
        step = self._script.pop(0)
        response = step(message) if callable(step) and not isinstance(step, MockResponse) else step
        if not isinstance(response, MockResponse):
            raise TypeError(f"Scripted step must return a MockResponse, got {type(response)!r}")
        return response


class _MockChats:
    def __init__(self, script: list[ScriptStep]) -> None:
        self._script = script
        #: The most recent chat this created, so a caller of MockGeminiClient can inspect
        #: what was passed to chats.create (e.g. ``client.chats.last_chat.received_history``).
        self.last_chat: MockChat | None = None

    def create(self, *, model: str, config: Any = None, history: Any = None) -> MockChat:
        chat = MockChat(list(self._script), received_history=history)
        self.last_chat = chat
        return chat


class MockGeminiClient:
    """Drop-in stand-in for ``google.genai.Client``, injected via ``TALUSAgent(client=...)``.

    Every call to ``chats.create`` starts a fresh chat replaying the same scripted turn
    sequence from the start, matching how each ``TALUSAgent.chat()`` call starts a new
    ``google.genai`` chat session.
    """

    def __init__(self, script: list[ScriptStep]) -> None:
        self.chats = _MockChats(script)


def text_response(text: str) -> MockResponse:
    """A scripted final answer with no further tool calls."""
    return MockResponse(function_calls=None, text=text)


def tool_call_response(*calls: tuple[str, dict[str, Any]]) -> MockResponse:
    """A scripted turn where the model requests one or more tool calls."""
    return MockResponse(function_calls=[MockFunctionCall(name=n, args=a) for n, a in calls])


class FailingChat:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def send_message(self, message: Any) -> MockResponse:
        raise self._exc


class FailingGeminiClient:
    """Simulates a Gemini API/model failure (timeout, quota exhaustion, transport error, ...)
    on every call in the chat, so tests can verify ``TALUSAgent.chat()`` degrades to a
    structured error response instead of raising or exposing internal detail to the user."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("simulated model failure")
        self.chats = self._Chats(self._exc)

    class _Chats:
        def __init__(self, exc: Exception) -> None:
            self._exc = exc

        def create(self, *, model: str, config: Any = None, history: Any = None) -> FailingChat:
            return FailingChat(self._exc)
