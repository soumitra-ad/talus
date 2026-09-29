"""NVIDIA hosted NIM provider: key handling, connection check, error mapping, tool calling.

Every test drives the real ``openai`` SDK against a fake NVIDIA server (``httpx.MockTransport``)
so the actual wire format -- Authorization header, /models, /chat/completions, tool calls, tool
results -- is exercised offline, with no real key and no network. Fake error responses echo the
key back in their bodies on purpose, to prove it can never reach a user-facing message or log.
"""

from __future__ import annotations

import json
import logging
import types
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from terrain_agent.agent import TALUSAgent
from terrain_agent.agent import agent as agent_module
from terrain_agent.agent import nvidia
from terrain_agent.agent.nvidia import (
    ERROR_MESSAGES,
    NVIDIA_BASE_URL,
    NvidiaAgentClient,
    SessionSecret,
    classify_nvidia_error,
    normalize_key,
    validate_nvidia_key,
)

KEY = "nvapi-UNIT-TEST-KEY-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
KEY_B = "nvapi-UNIT-TEST-KEY-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
from tests.nvidia_fakes import MODEL, FakeNvidia, Step, final, tool_calls  # noqa: E402


def agent_client(fake: FakeNvidia, key: str = KEY) -> NvidiaAgentClient:
    return NvidiaAgentClient(
        nvidia.create_client(SessionSecret(key), max_retries=0, http_client=fake.http_client()), MODEL
    )


def last_tool_result(body: dict[str, Any]) -> dict[str, Any]:
    msg = [m for m in body["messages"] if m["role"] == "tool"][-1]
    return json.loads(msg["content"])["result"]


def tool_results(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [json.loads(m["content"])["result"] for m in body["messages"] if m["role"] == "tool"]


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   ", "\n\t"])
def test_empty_key_is_rejected(raw):
    assert normalize_key(raw) == (None, "empty_key")


@pytest.mark.parametrize("raw", ["nvapi abc", "nvapi-\x00abc", "nvapi-\r\nX-Injected: 1", "x" * 600])
def test_malformed_key_is_rejected_before_any_request(raw):
    assert normalize_key(raw) == (None, "invalid_key_format")


def test_key_is_trimmed():
    assert normalize_key(f"  {KEY}\n") == (KEY, None)


def test_session_secret_never_renders_its_value():
    secret = SessionSecret(KEY)
    for text in (repr(secret), str(secret), f"{secret}", "%s" % (secret,), "{}".format(secret), repr([secret]), repr({"k": secret})):
        assert KEY not in text and "redacted" in text
    assert secret.reveal() == KEY
    assert secret != KEY  # never compares equal to the raw string
    secret.clear()
    assert not secret and secret.reveal() == ""


def test_agent_client_repr_hides_the_sdk_client():
    fake = FakeNvidia()
    client = agent_client(fake)
    assert KEY not in repr(client) and MODEL in repr(client)


# ---------------------------------------------------------------------------
# Connection check
# ---------------------------------------------------------------------------


def test_empty_key_makes_no_request():
    fake = FakeNvidia()
    report, client = validate_nvidia_key("   ", MODEL, http_client=fake.http_client())
    assert not report.ok and report.category == "empty_key" and client is None
    assert report.message == "Please enter your NVIDIA API key."
    assert fake.requests == []


def test_valid_key_connects_and_checks_model_and_tool_calling():
    fake = FakeNvidia()
    report, client = validate_nvidia_key(KEY, MODEL, http_client=fake.http_client())

    assert report.ok and report.message == "NVIDIA NIM Connected"
    assert report.model_listed is True and report.tool_calling is True
    assert isinstance(client, NvidiaAgentClient) and client.model == MODEL
    assert [(r["method"], r["path"]) for r in fake.requests] == [("GET", "/v1/models"), ("POST", "/v1/chat/completions")]
    assert all(r["host"] == "integrate.api.nvidia.com" for r in fake.requests)
    assert all(r["auth"] == f"Bearer {KEY}" for r in fake.requests)
    probe = fake.chat_bodies[0]
    assert probe["model"] == MODEL and probe["tools"] and probe["max_tokens"] <= 64
    # The key travels only in the Authorization header, never in a request body.
    assert KEY not in json.dumps([r["body"] for r in fake.requests])


def test_validation_accepts_a_session_secret():
    fake = FakeNvidia()
    report, _ = validate_nvidia_key(SessionSecret(KEY), MODEL, http_client=fake.http_client())
    assert report.ok and fake.requests[0]["auth"] == f"Bearer {KEY}"


def test_unlisted_model_is_reported_without_a_chat_request():
    fake = FakeNvidia(models=("someone/else",))
    report, client = validate_nvidia_key(KEY, MODEL, http_client=fake.http_client())
    assert not report.ok and report.category == "model_not_found" and client is None
    assert "NVIDIA_MODEL" in report.message
    assert fake.chat_bodies == []


def test_model_listing_outage_falls_back_to_the_chat_probe():
    fake = FakeNvidia(models_status=503)
    report, client = validate_nvidia_key(KEY, MODEL, http_client=fake.http_client())
    assert report.ok and report.model_listed is None and client is not None


@pytest.mark.parametrize(
    "status, category, message",
    [
        (401, "auth", "NVIDIA API authentication failed. Please check your API key."),
        (403, "auth", "NVIDIA API authentication failed. Please check your API key."),
        (408, "timeout", "NVIDIA request timed out. Please try again."),
        (429, "rate_limited", "NVIDIA API rate limit reached. Please try again later."),
        (500, "unavailable", "NVIDIA AI service is temporarily unavailable. Please try again."),
        (502, "unavailable", "NVIDIA AI service is temporarily unavailable. Please try again."),
        (503, "unavailable", "NVIDIA AI service is temporarily unavailable. Please try again."),
        (504, "unavailable", "NVIDIA AI service is temporarily unavailable. Please try again."),
    ],
)
def test_http_errors_map_to_fixed_safe_messages(status, category, message, caplog):
    caplog.set_level(logging.DEBUG)
    fake = FakeNvidia(chat_status=status)
    report, client = validate_nvidia_key(KEY, MODEL, http_client=fake.http_client())

    assert not report.ok and client is None
    assert (report.category, report.message) == (category, message)
    # The upstream body echoed the Authorization header; none of it may leak.
    assert KEY not in repr(report) and KEY not in caplog.text
    assert "Bearer" not in report.message


def test_auth_failure_on_the_model_listing_is_reported_as_auth():
    fake = FakeNvidia(models_status=401)
    report, _ = validate_nvidia_key(KEY, MODEL, http_client=fake.http_client())
    assert report.category == "auth" and fake.chat_bodies == []


def test_model_rejecting_tools_is_reported():
    fake = FakeNvidia(chat_status=400)
    report, client = validate_nvidia_key(KEY, MODEL, http_client=fake.http_client())
    assert report.category == "tools_unsupported" and client is None


@pytest.mark.parametrize(
    "exc, category, message",
    [
        (httpx.ReadTimeout("timed out"), "timeout", "NVIDIA request timed out. Please try again."),
        (httpx.ConnectError(f"cannot connect Bearer {KEY}"), "network", "Unable to reach NVIDIA AI service."),
    ],
)
def test_transport_failures_map_to_fixed_messages(exc, category, message, caplog):
    caplog.set_level(logging.DEBUG)
    fake = FakeNvidia(chat_exc=exc)
    report, client = validate_nvidia_key(KEY, MODEL, http_client=fake.http_client())
    assert (report.category, report.message) == (category, message) and client is None
    assert KEY not in caplog.text and KEY not in repr(report)


@pytest.mark.parametrize(
    "exc, category",
    [
        (TimeoutError(), "timeout"),
        (ConnectionError(f"Bearer {KEY}"), "network"),
        (RuntimeError(f"weird {KEY}"), "unavailable"),
        (types.SimpleNamespace(status_code=401), "auth"),
    ],
)
def test_classifier_never_uses_exception_text(exc, category):
    if not isinstance(exc, BaseException):
        exc = type("StatusError", (Exception,), {"status_code": 401})()
    got, message = classify_nvidia_error(exc)
    assert got == category and message == ERROR_MESSAGES[category] and KEY not in message


# ---------------------------------------------------------------------------
# No environment / secrets key is ever used
# ---------------------------------------------------------------------------


def test_environment_nvidia_key_is_never_used(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", KEY)
    fake = FakeNvidia()
    report, client = validate_nvidia_key("", MODEL, http_client=fake.http_client())
    assert report.category == "empty_key" and client is None and fake.requests == []

    agent = TALUSAgent(provider="nvidia")
    assert agent.provider == "nvidia" and not agent.is_live and agent.api_key is None


def test_no_source_file_reads_an_nvidia_key_from_configuration():
    """Static guard: the only reference to NVIDIA_API_KEY in shipped code is the app's list of
    secrets it refuses to mirror into the environment."""
    root = Path(__file__).resolve().parents[2]
    hits = []
    for path in list((root / "src").rglob("*.py")) + list((root / "app").rglob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "NVIDIA_API_KEY" in line:
                hits.append((path.name, line.strip()))
    assert hits == [("streamlit_app.py", '_NEVER_MIRROR = frozenset({"NVIDIA_API_KEY"})')]


def test_settings_default_to_the_nvidia_provider(monkeypatch):
    from terrain_agent.config.settings import DEFAULT_NVIDIA_MODEL, ModelConfig, TerrainSettings

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    s = TerrainSettings()
    assert s.llm_provider == "nvidia" and s.model.nvidia_model == DEFAULT_NVIDIA_MODEL
    assert not hasattr(ModelConfig(), "nvidia_api_key")
    monkeypatch.setenv("NVIDIA_MODEL", "vendor/other-model")
    assert TerrainSettings().model.nvidia_model == "vendor/other-model"
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    assert TerrainSettings().llm_provider == "nvidia"


# ---------------------------------------------------------------------------
# Agent over NVIDIA: chat, multi-turn, tool calling
# ---------------------------------------------------------------------------


def test_agent_infers_nvidia_from_its_client():
    agent = TALUSAgent(client=agent_client(FakeNvidia()))
    assert agent.provider == "nvidia" and agent.is_live and agent.model_name == MODEL


def test_plain_chat_request_and_response():
    fake = FakeNvidia([final("TALUS answers questions about lunar terrain. Research/demo, not certified.")])
    result = TALUSAgent(client=agent_client(fake)).chat("What can you do?")

    assert result["status"] == "ok" and "lunar terrain" in result["text"]
    body = fake.chat_bodies[0]
    assert body["model"] == MODEL and body["tool_choice"] == "auto"
    assert {t["function"]["name"] for t in body["tools"]} == {d["name"] for d in agent_module.TALUS_TOOL_DECLARATIONS}
    assert body["messages"][0] == {"role": "system", "content": agent_module.TALUS_SYSTEM_PROMPT}
    assert body["messages"][-1]["role"] == "user" and "What can you do?" in body["messages"][-1]["content"]


def test_multi_turn_history_is_sent_with_openai_roles():
    fake = FakeNvidia([final("Second answer. Research/demo, not certified.")])
    history = [
        {"role": "user", "parts": [{"text": "first question"}]},
        {"role": "model", "parts": [{"text": "first answer"}]},
    ]
    TALUSAgent(client=agent_client(fake)).chat("second question", history=history)
    roles = [(m["role"], m["content"]) for m in fake.chat_bodies[0]["messages"][1:3]]
    assert roles == [("user", "first question"), ("assistant", "first answer")]


def test_tool_call_round_trip():
    fake = FakeNvidia([
        tool_calls(("resolve_lunar_feature", {"name": "Shackleton Crater"})),
        lambda body: final(f"Resolved {last_tool_result(body)['feature']['name']}. Research/demo, not certified."),
    ])
    events: list[tuple[str, Any]] = []
    result = TALUSAgent(client=agent_client(fake)).chat(
        "Where is Shackleton?", on_event=lambda kind, info: events.append((kind, info.get("tool")))
    )

    assert result["status"] == "ok"
    assert result["text"].startswith("Resolved Shackleton Crater.")
    [call] = result["tool_calls"]
    assert call["tool"] == "resolve_lunar_feature" and call["result_status"] == "ok"
    second = fake.chat_bodies[1]["messages"]
    assistant = next(m for m in second if m["role"] == "assistant" and m.get("tool_calls"))
    tool_msg = next(m for m in second if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == assistant["tool_calls"][0]["id"] == "call_0_resolve_lunar_feature"
    assert json.loads(tool_msg["content"])["result"]["feature"]["name"] == "Shackleton Crater"
    assert ("tool_end", "resolve_lunar_feature") in events


def test_parallel_tool_calls_each_get_their_own_result():
    fake = FakeNvidia([
        tool_calls(("resolve_lunar_feature", {"name": "Shackleton Crater"}), ("resolve_lunar_feature", {"name": "Haworth"})),
        final("Both resolved. Research/demo, not certified."),
    ])
    TALUSAgent(client=agent_client(fake)).chat("Shackleton and Haworth?")
    ids = [m["tool_call_id"] for m in fake.chat_bodies[1]["messages"] if m["role"] == "tool"]
    assert ids == ["call_0_resolve_lunar_feature", "call_1_resolve_lunar_feature"]


def test_invalid_tool_arguments_become_a_tool_error_not_a_crash():
    fake = FakeNvidia([
        tool_calls(("resolve_lunar_feature", {}), raw_args="{not json"),
        final("I could not resolve that. Research/demo, not certified."),
    ])
    result = TALUSAgent(client=agent_client(fake)).chat("Where?")
    assert result["status"] == "ok"
    assert result["tool_calls"][0]["result"]["error_type"] == "MissingArgument"


def test_reasoning_tags_are_stripped_from_the_answer():
    fake = FakeNvidia([final("<think>private chain of thought</think>Final answer. Research/demo, not certified.")])
    text = TALUSAgent(client=agent_client(fake)).chat("hi")["text"]
    assert "private chain" not in text and text.startswith("Final answer.")


def test_model_failure_mid_turn_keeps_tool_results_and_shows_a_fixed_message(caplog):
    caplog.set_level(logging.DEBUG)
    fake = FakeNvidia([tool_calls(("resolve_lunar_feature", {"name": "Nowhere Crater"}))], chat_status=503, fail_after=1)
    result = TALUSAgent(client=agent_client(fake)).chat("Tell me about Nowhere Crater")

    assert result["status"] == "model_error" and result["error_category"] == "unavailable"
    assert "NVIDIA AI service is temporarily unavailable" in result["text"]
    assert [c["tool"] for c in result["tool_calls"]] == ["resolve_lunar_feature"]
    assert KEY not in json.dumps(result, default=str) and KEY not in caplog.text


def test_auth_failure_mid_session_blocks_the_model():
    fake = FakeNvidia(chat_status=401)
    agent = TALUSAgent(client=agent_client(fake))
    result = agent.chat("hello there")
    assert result["error_category"] == "auth"
    assert agent.model_block["message"] == ERROR_MESSAGES["auth"]


def test_rate_limit_mid_session_falls_back_to_deterministic_answer_for_a_named_place(monkeypatch):
    monkeypatch.setattr(
        "terrain_agent.agent.fallback.deterministic_analysis",
        lambda msg, **kw: {"status": "fallback", "notice": kw["notice"], "text": "deterministic", "tool_calls": [], "is_demo": False},
    )
    fake = FakeNvidia(chat_status=429)
    result = TALUSAgent(client=agent_client(fake)).chat("elevation around Shackleton Crater")
    assert result["status"] == "fallback" and result["notice"] == ERROR_MESSAGES["rate_limited"]


def test_empty_model_reply_is_reported_with_the_provider_name():
    fake = FakeNvidia([final("")])
    result = TALUSAgent(client=agent_client(fake)).chat("hi")
    assert result["status"] == "model_error" and "NVIDIA AI returned an empty response" in result["text"]


def test_close_drops_the_client():
    client = agent_client(FakeNvidia())
    agent = TALUSAgent(client=client)
    agent.close()
    assert not agent.is_live and client.client is None
    with pytest.raises(RuntimeError):
        client.start_chat(system_prompt="s", tools=[], history=None, temperature=0.1)


def test_gemini_remains_available_as_an_explicit_opt_in():
    from terrain_agent.agent.mock_model import MockGeminiClient, text_response

    agent = TALUSAgent(client=MockGeminiClient([text_response("Hi. Research/demo, not certified.")]))
    assert agent.provider == "gemini" and agent.chat("hi")["status"] == "ok"


# ---------------------------------------------------------------------------
# Key never reaches prompts, messages, tool arguments, results, history or logs
# ---------------------------------------------------------------------------


def test_key_never_appears_outside_the_authorization_header(caplog):
    caplog.set_level(logging.DEBUG)
    fake = FakeNvidia([
        tool_calls(("resolve_lunar_feature", {"name": "Shackleton Crater"})),
        final("Shackleton resolved. Research/demo, not certified."),
    ])
    report, client = validate_nvidia_key(KEY, MODEL, http_client=fake.http_client())
    agent = TALUSAgent(client=client)
    history = [{"role": "user", "parts": [{"text": "earlier"}]}, {"role": "model", "parts": [{"text": "reply"}]}]
    result = agent.chat("Where is Shackleton Crater?", history=history)

    assert report.ok and result["status"] == "ok"
    assert all(r["auth"] == f"Bearer {KEY}" and r["host"] == "integrate.api.nvidia.com" for r in fake.requests)
    bodies = json.dumps([r["body"] for r in fake.requests])
    assert KEY not in bodies                                    # prompts, model messages, tool results
    assert KEY not in json.dumps([c["args"] for c in result["tool_calls"]])  # tool arguments
    assert KEY not in json.dumps(result, default=str)           # what the UI stores as chat history
    assert KEY not in caplog.text                               # application and SDK logs
    assert KEY not in repr(agent.__dict__)
    assert NVIDIA_BASE_URL == "https://integrate.api.nvidia.com/v1"


def test_two_sessions_never_share_a_key():
    fake_a, fake_b = FakeNvidia([final("A. Research/demo, not certified.")]), FakeNvidia([final("B. Research/demo, not certified.")])
    _, client_a = validate_nvidia_key(KEY, MODEL, http_client=fake_a.http_client())
    _, client_b = validate_nvidia_key(KEY_B, MODEL, http_client=fake_b.http_client())
    agent_a, agent_b = TALUSAgent(client=client_a), TALUSAgent(client=client_b)

    assert agent_a._client is not agent_b._client
    agent_a.chat("question from A")
    agent_b.chat("question from B")
    assert {r["auth"] for r in fake_a.requests} == {f"Bearer {KEY}"}
    assert {r["auth"] for r in fake_b.requests} == {f"Bearer {KEY_B}"}

    agent_a.close()  # A disconnects: B is unaffected
    assert not agent_a.is_live and agent_b.is_live
    fake_b.script.append(final("B again. Research/demo, not certified."))
    assert agent_b.chat("B still works")["status"] == "ok"
    # Nothing module-level holds a client or key.
    assert not any(isinstance(v, (NvidiaAgentClient, SessionSecret)) for v in vars(nvidia).values())
    assert not any(isinstance(v, (NvidiaAgentClient, SessionSecret)) for v in vars(agent_module).values())


# ---------------------------------------------------------------------------
# End to end: NVIDIA -> resolve -> NASA ODE -> DEM -> terrain tools -> NVIDIA answer
# ---------------------------------------------------------------------------


def _shackleton_script(stats_tools: tuple[str, ...] = ("get_elevation_stats", "get_slope_stats"), search: bool = False) -> list[Step]:
    """What a tool-calling model does for 'terrain around Shackleton': every argument is taken
    from the previous deterministic tool result, never invented."""

    def after_resolve(body):
        area = last_tool_result(body)["analysis_area"]
        feature = last_tool_result(body)["feature"]
        if search:
            return tool_calls(("search_dem_products", {k: area[k] for k in ("min_lat", "max_lat", "min_lon", "max_lon")}))
        return tool_calls(("fetch_nasa_dem", {"lat": feature["center_lat"], "lon": feature["center_lon"], "radius_km": area["radius_km"]}))

    def after_search(body):
        feature = tool_results(body)[0]["feature"]
        area = tool_results(body)[0]["analysis_area"]
        return tool_calls(("fetch_nasa_dem", {"lat": feature["center_lat"], "lon": feature["center_lon"], "radius_km": area["radius_km"]}))

    def after_fetch(body):
        dem = last_tool_result(body)["dem_path"]
        area = tool_results(body)[0]["analysis_area"]
        box = {"dem_path": dem, **{k: area[k] for k in ("min_lat", "max_lat", "min_lon", "max_lon")}}
        return tool_calls(*[(t, box) for t in stats_tools])

    def answer(body):
        results = tool_results(body)
        elev = next(r for r in results if (r.get("analysis") or {}).get("elevation"))["analysis"]["elevation"]
        return final(f"Mean elevation {elev['mean_m']:.2f} m from the NASA DEM. Research/demo, not certified.")

    return (
        [tool_calls(("resolve_lunar_feature", {"name": "Shackleton Crater"})), after_resolve]
        + ([after_search] if search else [])
        + [after_fetch, answer]
    )


def test_shackleton_via_nvidia_with_nasa_ode_discovery_and_download(mock_nasa, make_service, tmp_path, monkeypatch):
    """NASA discovery + download against the offline ODE/PDS fixtures, driven by NVIDIA tool calls."""
    from tests.acquisition_fixtures import write_product

    label, data = write_product(tmp_path / "srv", "ldem_75s_240m", kind="polar", lines=200, samples=200)
    mock_nasa.add_product(product_id="ldem_75s_240m", label=label.read_bytes(), data=data.read_bytes())
    service = make_service(cache_dir=tmp_path / "cache")
    monkeypatch.setattr(agent_module, "_get_nasa_service", lambda cache_dir: service)
    record = types.SimpleNamespace(
        product_id="ldem_75s_240m", mission="LRO", instrument="LOLA", dataset="LOLA GDR DEM", product_type="GDRDEM",
        min_lat=-90.0, max_lat=-75.0, min_lon=0.0, max_lon=360.0, center_lat=-90.0, center_lon=0.0,
        file_url="https://pds-geosciences.wustl.edu/x.img", files_page_url=None, description="LOLA polar DEM",
    )
    monkeypatch.setattr("terrain_agent.tools.ode_search.search_lunar_dem", lambda **kw: [record])

    fake = FakeNvidia(_shackleton_script(search=True))
    result = TALUSAgent(client=agent_client(fake), dem_cache_dir=str(tmp_path / "cache")).chat(
        "What is the terrain around Shackleton crater?"
    )

    tools = [c["tool"] for c in result["tool_calls"]]
    assert tools == ["resolve_lunar_feature", "search_dem_products", "fetch_nasa_dem", "get_elevation_stats", "get_slope_stats"]
    assert all(c["result_status"] == "ok" for c in result["tool_calls"]), [c["result"] for c in result["tool_calls"]]
    fetch = result["tool_calls"][2]["result"]
    assert fetch["from_cache"] is False and fetch["provenance"]["product_id"] == "ldem_75s_240m"
    mean = result["tool_calls"][3]["result"]["analysis"]["elevation"]["mean_m"]
    assert f"{mean:.2f} m" in result["text"]  # the final NVIDIA answer quotes the tool's own number
    assert len(fake.chat_bodies) == 5


REAL_DEM_CACHE = Path(__file__).resolve().parents[1] / "fixtures" / "nasa_real_dem_cache"


@pytest.fixture
def real_dem_service(monkeypatch):
    from terrain_agent.acquisition.models import CoverageRequest
    from terrain_agent.acquisition.service import build_default_service

    service = build_default_service(cache_dir=REAL_DEM_CACHE, enabled=False)
    if service.find_cached(CoverageRequest.from_point(-89.9, 0.0, 5000.0), ["GDRDEM"]) is None:
        pytest.skip("Real NASA DEM fixture absent: run tests/fixtures/_download_real_dem.py once.")
    monkeypatch.setattr(agent_module, "_get_nasa_service", lambda cache_dir: service)
    return str(REAL_DEM_CACHE)


@pytest.mark.real_dem
def test_shackleton_then_rover_follow_up_on_the_real_nasa_dem(real_dem_service):
    """The two-turn conversation from the release checklist, on the real LRO/LOLA ldem_75s_240m
    DEM: 'What is the terrain around Shackleton crater?' then 'Is it safe for a rover?'."""
    fake = FakeNvidia(_shackleton_script(("get_elevation_stats", "get_slope_stats", "get_roughness_stats")))
    agent = TALUSAgent(client=agent_client(fake), dem_cache_dir=real_dem_service)
    first = agent.chat("What is the terrain around Shackleton crater?")

    assert [c["tool"] for c in first["tool_calls"]] == [
        "resolve_lunar_feature", "fetch_nasa_dem", "get_elevation_stats", "get_slope_stats", "get_roughness_stats",
    ]
    assert all(c["result_status"] == "ok" for c in first["tool_calls"])
    fetch = first["tool_calls"][1]["result"]
    assert fetch["provenance"]["product_id"] == "ldem_75s_240m" and fetch["provenance"]["instrument"] == "LOLA"
    elevation = first["tool_calls"][2]["result"]["analysis"]["elevation"]
    slope = first["tool_calls"][3]["result"]["analysis"]["slope"]
    assert -10_000 < elevation["min_m"] <= elevation["mean_m"] <= elevation["max_m"] < 10_000
    assert 0.0 <= slope["mean_slope_deg"] <= slope["max_slope_deg"] <= 90.0
    assert f"{elevation['mean_m']:.2f} m" in first["text"]

    # Follow-up: the model sees the previous turn and runs the deterministic rover tool.
    def rover_from_context(body):
        context = " ".join(m["content"] for m in body["messages"] if m["role"] in ("user", "assistant") and m.get("content"))
        assert "Shackleton" in context and "Mean elevation" in context
        return tool_calls(("fetch_nasa_dem", {"lat": -89.9, "lon": 0.0, "radius_km": 5}))

    def rover_route(body):
        dem = last_tool_result(body)["dem_path"]
        return tool_calls(("evaluate_traverse_route", {
            "waypoints": [[-89.9, 0.0], [-89.85, 30.0], [-89.8, 60.0]], "dem_path": dem, "max_slope_deg": 15.0,
        }))

    def rover_answer(body):
        r = last_tool_result(body)
        return final(f"Overall status {r['overall_status']}. Configured analysis threshold: {r['configured_threshold_deg']}°. "
                     "This is NOT a certified safety limit. Research/demo, not certified.")

    fake.script.extend([rover_from_context, rover_route, rover_answer])
    history = [
        {"role": "user", "parts": [{"text": "What is the terrain around Shackleton crater?"}]},
        {"role": "model", "parts": [{"text": first["text"]}]},
    ]
    second = agent.chat("Is it safe for a rover?", history=history)

    assert [c["tool"] for c in second["tool_calls"]] == ["fetch_nasa_dem", "evaluate_traverse_route"]
    assert second["tool_calls"][0]["result"]["from_cache"] is True  # no second download
    rover = second["tool_calls"][1]["result"]
    assert rover["status"] == "ok" and rover["overall_status"] in ("PASS", "REVIEW_REQUIRED", "FAIL")
    assert rover["configured_threshold_deg"] == 15.0
    assert f"Overall status {rover['overall_status']}" in second["text"]
    assert "Configured analysis threshold: 15.0°" in second["text"]
    assert "Note from TALUS" not in second["text"]  # the status is grounded in the tool result
