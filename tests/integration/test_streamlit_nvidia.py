"""Streamlit NVIDIA AI setup flow: user-entered key, per-session isolation, disconnect, no leaks.

Runs the real app with Streamlit's ``AppTest``. The NVIDIA side is the real
``validate_nvidia_key`` + ``openai`` SDK talking to an offline fake of NVIDIA's hosted API
(``tests.nvidia_fakes.FakeNvidia``); only the HTTP transport is substituted. No real key, no
network.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.nvidia_fakes import FakeNvidia, final

APP_PATH = "../../app/streamlit_app.py"
KEY_A = "nvapi-STREAMLIT-TEST-KEY-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
KEY_B = "nvapi-STREAMLIT-TEST-KEY-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"


@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    st.cache_data.clear()
    st.cache_resource.clear()
    monkeypatch.delenv("TALUS_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from terrain_agent.config.settings import settings

    monkeypatch.setattr(settings.model, "provider", "nvidia")
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def fake(monkeypatch) -> FakeNvidia:
    """Route every NVIDIA client the app builds to one recording fake server."""
    from terrain_agent.agent import nvidia as nvidia_mod

    server = FakeNvidia(models=("nvidia/nemotron-3-super-120b-a12b",))
    real_create = nvidia_mod.create_client

    def create_client(secret, *, timeout_s=nvidia_mod.NVIDIA_TIMEOUT_S, max_retries=0, http_client=None):
        return real_create(secret, timeout_s=timeout_s, max_retries=0, http_client=http_client or server.http_client())

    monkeypatch.setattr(nvidia_mod, "create_client", create_client)
    from terrain_agent.config.settings import settings

    monkeypatch.setattr(settings.model, "nvidia_model", "nvidia/nemotron-3-super-120b-a12b")
    return server


def _app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=60)


def _key_input(app: AppTest):
    return next(w for w in app.text_input if w.label == "NVIDIA API Key")


def _connect(app: AppTest, key: str) -> AppTest:
    _key_input(app).set_value(key)
    next(b for b in app.button if b.label == "Connect NVIDIA AI").click().run()
    return app


def _page_text(app: AppTest) -> str:
    parts: list[str] = []
    for kind in ("markdown", "caption", "text", "info", "success", "warning", "error", "title", "header", "subheader"):
        parts.extend(str(getattr(e, "value", "")) for e in getattr(app, kind))
    parts.extend(str(w.value) for w in app.text_input)
    return "\n".join(parts)


def _session_dump(app: AppTest) -> str:
    out = []
    for k, v in app.session_state._state.filtered_state.items():
        out.append(f"{k}={v!r} {v!s}")
        try:
            out.append(json.dumps(v, default=repr))
        except Exception:
            pass
    return "\n".join(out)


def _assert_no_key(app: AppTest, key: str) -> None:
    assert key not in _page_text(app)
    assert key not in _session_dump(app)


# ---------------------------------------------------------------------------
# First screen
# ---------------------------------------------------------------------------


def test_initial_page_shows_nvidia_setup_and_disables_the_agent(fake, monkeypatch):
    import openai

    def _no_client(*a, **k):
        raise AssertionError("no NVIDIA client may be created at startup")

    monkeypatch.setattr(openai, "OpenAI", _no_client)
    app = _app()
    app.run()

    assert not app.exception
    page = _page_text(app)
    assert "TALUS AI" in page and "Lunar Terrain Intelligence Agent" in page
    assert "🔐 NVIDIA AI Setup" in page
    assert "Enter your NVIDIA API key to activate the TALUS AI Agent." in page
    assert "NVIDIA AI Not Connected" in page
    key_input = _key_input(app)
    assert key_input.proto.type == key_input.proto.PASSWORD  # masked input
    assert any(b.label == "Connect NVIDIA AI" for b in app.button)
    assert app.chat_input[0].proto.disabled is True
    assert not any("Shackleton" in b.label for b in app.button)  # no suggested prompts yet
    assert len(app.tabs) == 6  # the deterministic tools stay available
    assert fake.requests == []  # startup made no NVIDIA call
    assert app.session_state["nvidia_state"] == "not_connected"
    assert app.session_state["agent"] is None
    assert any("NVIDIA NIM: ○ NVIDIA AI Not Connected" in m.value for m in app.markdown)


def test_empty_key_is_rejected_without_a_request(fake):
    app = _app()
    app.run()
    _connect(app, "   ")

    assert not app.exception
    assert any("Please enter your NVIDIA API key." in e.value for e in app.error)
    assert app.session_state["nvidia_state"] == "not_connected"
    assert fake.requests == []
    assert app.chat_input[0].proto.disabled is True


# ---------------------------------------------------------------------------
# Connect / invalid key / errors
# ---------------------------------------------------------------------------


def test_valid_key_connects_and_enables_the_agent(fake, caplog):
    caplog.set_level(logging.DEBUG)
    app = _app()
    app.run()
    _connect(app, KEY_A)

    assert not app.exception
    assert app.session_state["nvidia_state"] == "connected"
    page = _page_text(app)
    assert "🟢 NVIDIA NIM Connected" in page and "AI Agent Ready" in page
    assert any("NVIDIA NIM: 🟢 NVIDIA NIM Connected" in m.value for m in app.markdown)
    assert app.chat_input[0].proto.disabled is False
    assert not any(w.label == "NVIDIA API Key" for w in app.text_input)  # setup screen gone
    assert any(b.label == "Disconnect NVIDIA AI" for b in app.button)
    assert [(r["method"], r["path"]) for r in fake.requests] == [("GET", "/v1/models"), ("POST", "/v1/chat/completions")]
    assert all(r["auth"] == f"Bearer {KEY_A}" for r in fake.requests)
    assert app.session_state["nvidia_api_key_input"] == ""  # widget value cleared
    _assert_no_key(app, KEY_A)
    assert KEY_A not in caplog.text


@pytest.mark.parametrize(
    "status, expected, state",
    [
        (401, "NVIDIA API authentication failed", "auth_failed"),
        (403, "NVIDIA API authentication failed", "auth_failed"),
        (408, "NVIDIA request timed out. Please try again.", "error"),
        (429, "NVIDIA API rate limit reached. Please try again later.", "error"),
        (500, "NVIDIA AI service is temporarily unavailable. Please try again.", "error"),
        (503, "NVIDIA AI service is temporarily unavailable. Please try again.", "error"),
    ],
)
def test_failed_connect_keeps_the_agent_disabled_with_a_safe_error(fake, caplog, status, expected, state):
    caplog.set_level(logging.DEBUG)
    fake.chat_status = status  # the fake echoes the Authorization header in its error body
    app = _app()
    app.run()
    _connect(app, KEY_A)

    assert not app.exception
    assert app.session_state["nvidia_state"] == state
    assert app.session_state["agent"] is None
    assert any(expected in e.value for e in app.error)
    assert app.chat_input[0].proto.disabled is True
    assert any(w.label == "NVIDIA API Key" for w in app.text_input)  # can retry
    page = _page_text(app)
    assert "Bearer" not in page and "Traceback" not in page
    _assert_no_key(app, KEY_A)
    assert KEY_A not in caplog.text
    if state == "auth_failed":
        assert "NVIDIA NIM Authentication Failed" in page


def test_network_failure_on_connect(fake, monkeypatch):
    import httpx

    fake.chat_exc = httpx.ConnectError("unreachable")
    app = _app()
    app.run()
    _connect(app, KEY_A)
    assert any("Unable to reach NVIDIA AI service." in e.value for e in app.error)
    assert app.session_state["agent"] is None


def test_timeout_on_connect(fake):
    import httpx

    fake.chat_exc = httpx.ReadTimeout("slow")
    app = _app()
    app.run()
    _connect(app, KEY_A)
    assert any("NVIDIA request timed out. Please try again." in e.value for e in app.error)


def test_retry_after_a_failed_key(fake):
    fake.chat_status = 401
    app = _app()
    app.run()
    _connect(app, "nvapi-wrong")
    assert app.session_state["nvidia_state"] == "auth_failed"
    fake.chat_status = None
    _connect(app, KEY_A)
    assert app.session_state["nvidia_state"] == "connected"


# ---------------------------------------------------------------------------
# Chat through NVIDIA, disconnect
# ---------------------------------------------------------------------------


def test_chat_after_connect_goes_to_nvidia_with_history(fake, caplog):
    caplog.set_level(logging.DEBUG)
    app = _app()
    app.run()
    _connect(app, KEY_A)
    fake.script += [final("First answer. Research/demo, not certified."), final("Second answer. Research/demo, not certified.")]

    app.chat_input[0].set_value("first question").run()
    app.chat_input[0].set_value("second question").run()

    assert not app.exception
    assert any("Second answer." in m.value for m in app.markdown)
    last = fake.chat_bodies[-1]["messages"]
    assert [(m["role"], m["content"]) for m in last[1:3]] == [("user", "first question"), ("assistant", last[2]["content"])]
    assert last[2]["content"].startswith("First answer.")
    assert all(r["auth"] == f"Bearer {KEY_A}" for r in fake.requests)
    assert KEY_A not in json.dumps(fake.chat_bodies)
    assert KEY_A not in json.dumps(app.session_state["messages"], default=str)  # chat history
    _assert_no_key(app, KEY_A)
    assert KEY_A not in caplog.text


def test_disconnect_clears_the_key_client_and_agent(fake):
    app = _app()
    app.run()
    _connect(app, KEY_A)
    agent = app.session_state["agent"]
    nv_client = agent._client

    next(b for b in app.button if b.label == "Disconnect NVIDIA AI").click().run()

    assert not app.exception
    assert app.session_state["nvidia_state"] == "not_connected"
    assert app.session_state["agent"] is None
    assert "_nvidia_pending_secret" not in app.session_state
    assert agent._client is None and nv_client.client is None  # SDK client (and key) dropped
    assert any(w.label == "NVIDIA API Key" for w in app.text_input)
    assert app.chat_input[0].proto.disabled is True
    _assert_no_key(app, KEY_A)
    requests_before = len(fake.requests)
    app.run()
    assert len(fake.requests) == requests_before  # nothing reconnects by itself


def test_key_revoked_mid_session_returns_to_setup(fake):
    app = _app()
    app.run()
    _connect(app, KEY_A)
    fake.chat_status = 401
    app.chat_input[0].set_value("hello").run()

    assert not app.exception
    assert app.session_state["nvidia_state"] == "auth_failed"
    assert app.session_state["agent"] is None
    assert any(w.label == "NVIDIA API Key" for w in app.text_input)


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------


def test_two_sessions_never_share_a_key_client_or_agent(fake):
    app_a, app_b = _app(), _app()
    app_a.run()
    _connect(app_a, KEY_A)

    app_b.run()  # a second visitor, while A is connected
    assert app_b.session_state["nvidia_state"] == "not_connected"
    assert app_b.session_state["agent"] is None
    assert app_b.chat_input[0].proto.disabled is True
    _assert_no_key(app_b, KEY_A)

    _connect(app_b, KEY_B)
    agent_a, agent_b = app_a.session_state["agent"], app_b.session_state["agent"]
    assert agent_a is not agent_b and agent_a._client is not agent_b._client

    fake.requests.clear()
    fake.script += [final("A answer. Research/demo, not certified.")]
    app_a.chat_input[0].set_value("question A").run()
    assert {r["auth"] for r in fake.requests} == {f"Bearer {KEY_A}"}
    fake.requests.clear()
    fake.script += [final("B answer. Research/demo, not certified.")]
    app_b.chat_input[0].set_value("question B").run()
    assert {r["auth"] for r in fake.requests} == {f"Bearer {KEY_B}"}

    # B never sees A's conversation, and disconnecting A leaves B connected.
    assert "question A" not in json.dumps(app_b.session_state["messages"])
    next(b for b in app_a.button if b.label == "Disconnect NVIDIA AI").click().run()
    app_b.run()
    assert app_a.session_state["agent"] is None
    assert app_b.session_state["nvidia_state"] == "connected" and app_b.session_state["agent"] is agent_b
    _assert_no_key(app_b, KEY_A)
    _assert_no_key(app_a, KEY_B)


# ---------------------------------------------------------------------------
# Operator-supplied keys are never used
# ---------------------------------------------------------------------------


def test_environment_or_secrets_nvidia_key_is_never_used(fake, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", KEY_A)
    app = _app()
    app.secrets["NVIDIA_API_KEY"] = KEY_B
    app.run()

    assert not app.exception
    assert app.session_state["nvidia_state"] == "not_connected"
    assert app.chat_input[0].proto.disabled is True
    assert fake.requests == []
    assert os.environ.get("NVIDIA_API_KEY") == KEY_A  # untouched; secrets value not mirrored
    _assert_no_key(app, KEY_A)
    _assert_no_key(app, KEY_B)


def test_gemini_key_is_not_required_for_nvidia_operation(fake, monkeypatch):
    from terrain_agent.config.settings import settings

    monkeypatch.setattr(settings.model, "api_key", None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = _app()
    app.run()
    _connect(app, KEY_A)
    assert app.session_state["nvidia_state"] == "connected"
    assert "GEMINI_API_KEY" not in _page_text(app)


def test_health_panel_reports_nvidia_without_a_request(fake):
    app = _app()
    app.run()
    _connect(app, KEY_A)
    before = len(fake.requests)
    next(b for b in app.button if b.label == "Run full health check").click().run()
    assert not app.exception
    captions = " ".join(c.value for c in app.caption)
    assert "NVIDIA NIM" in captions and "NVIDIA NIM Connected" in captions
    assert len(fake.requests) == before
    _assert_no_key(app, KEY_A)
