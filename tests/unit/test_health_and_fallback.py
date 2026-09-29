"""Deployment hardening: system health checks, Gemini quota handling, and the deterministic
fallback that answers from NASA DEM data when Gemini is unavailable.

All offline: NASA is served by the synthetic ODE/PDS server fixtures, Gemini by mock clients.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from terrain_agent import health
from terrain_agent.agent import TALUSAgent
from terrain_agent.agent import agent as agent_module
from terrain_agent.agent.agent import QUOTA_DAILY_MESSAGE, classify_model_error
from terrain_agent.agent.mock_model import FailingGeminiClient, MockGeminiClient, text_response
from terrain_agent.data.lunar_features import covering_circle, find_feature_in_text
from terrain_agent.terrain.resource_safety import OversizedRequestError
from tests.acquisition_fixtures import write_product


class _ApiError(Exception):
    def __init__(self, code: int, text: str = ""):
        super().__init__(f"{code} {text}")
        self.code = code


DAILY_QUOTA = _ApiError(429, "RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
PER_MINUTE = _ApiError(429, "RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerMinutePerProjectPerModel")


# ---------------------------------------------------------------------------
# Gemini quota classification and retry policy
# ---------------------------------------------------------------------------


def test_daily_quota_is_reported_with_the_required_message():
    assert classify_model_error(DAILY_QUOTA) == ("quota_daily", QUOTA_DAILY_MESSAGE)
    assert QUOTA_DAILY_MESSAGE == "Gemini daily quota reached. Terrain tools still available."


def test_per_minute_rate_limit_is_distinct_from_the_daily_quota():
    assert classify_model_error(PER_MINUTE)[0] == "rate_limited"


def test_429_is_not_retried_by_the_sdk():
    options = agent_module._gemini_http_options()
    assert 429 not in options.retry_options.http_status_codes
    assert 503 in options.retry_options.http_status_codes


# ---------------------------------------------------------------------------
# Feature detection and covering circles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, name",
    [
        ("What is the average elevation around Shackleton Crater?", "Shackleton Crater"),
        ("check rover safety near shackleton", "Shackleton Crater"),
        ("roughness around Haworth crater please", "Haworth Crater"),
        ("slopes on Malapert Mountain", "Malapert Mountain"),
    ],
)
def test_features_are_found_in_free_text(text, name):
    assert find_feature_in_text(text)["name"] == name


@pytest.mark.parametrize("text", ["what is the weather", "shackletonish terrain", "", None, 42])
def test_no_feature_is_invented_from_unrelated_text(text):
    assert find_feature_in_text(text) is None


def test_covering_circle_of_one_point_is_just_the_margin():
    lat, lon, radius = covering_circle([(-89.9, 0.0)], margin_km=3.0)
    assert (lat, lon) == pytest.approx((-89.9, 0.0), abs=1e-6)
    assert radius == pytest.approx(3.0)


def test_covering_circle_contains_every_point():
    lat, lon, radius = covering_circle([(-89.9, 0.0), (-89.85, 30.0), (-89.8, 60.0)])
    assert -90.0 < lat < -89.8 and 1.0 < radius < 12.0


def test_covering_circle_refuses_an_oversized_area():
    with pytest.raises(OversizedRequestError):
        covering_circle([(0.0, 0.0), (10.0, 10.0)])


# ---------------------------------------------------------------------------
# Deterministic fallback when Gemini fails
# ---------------------------------------------------------------------------


@pytest.fixture
def wired(mock_nasa, make_service, tmp_path, monkeypatch):
    label, data = write_product(tmp_path / "srv", "ldem_75s_240m", kind="polar", lines=200, samples=200)
    mock_nasa.add_product(product_id="ldem_75s_240m", label=label.read_bytes(), data=data.read_bytes())
    service = make_service(cache_dir=tmp_path / "cache")
    monkeypatch.setattr(agent_module, "_get_nasa_service", lambda cache_dir: service)
    return str(tmp_path / "cache")


class _CountingFailingClient(FailingGeminiClient):
    def __init__(self, exc):
        super().__init__(exc)
        self.creates = 0
        original = self.chats.create

        def create(**kwargs):
            self.creates += 1
            return original(**kwargs)

        self.chats.create = create


def test_quota_exhaustion_falls_back_to_real_deterministic_analysis(wired):
    client = _CountingFailingClient(DAILY_QUOTA)
    agent = TALUSAgent(client=client, dem_cache_dir=wired)
    events = []

    result = agent.chat("What is the average elevation around Shackleton Crater?",
                        on_event=lambda kind, info: events.append((kind, info.get("tool"))))

    assert result["status"] == "fallback"
    assert result["notice"] == QUOTA_DAILY_MESSAGE
    assert result["error_category"] == "quota_daily"
    tools = [c["tool"] for c in result["tool_calls"]]
    assert tools == ["resolve_lunar_feature", "fetch_nasa_dem", "get_elevation_stats", "get_slope_stats", "get_roughness_stats"]
    assert all(c["result_status"] == "ok" for c in result["tool_calls"])
    mean = result["tool_calls"][2]["result"]["analysis"]["elevation"]["mean_m"]
    assert f"{float(mean):.2f} m" in result["text"]  # the answer quotes the tool's own number
    assert "research and demonstration system" in result["text"].lower()
    assert ("tool_end", "get_elevation_stats") in events


def test_after_a_daily_quota_error_gemini_is_skipped_until_the_cooldown_ends(wired, monkeypatch):
    client = _CountingFailingClient(DAILY_QUOTA)
    agent = TALUSAgent(client=client, dem_cache_dir=wired)
    agent.chat("elevation around Shackleton Crater?")
    assert client.creates == 1

    second = agent.chat("roughness around Shackleton?")
    assert client.creates == 1  # no second doomed call
    assert second["status"] == "fallback" and second["notice"] == QUOTA_DAILY_MESSAGE
    assert agent.model_block["category"] == "quota_daily"

    # Once the cooldown has passed, Gemini is tried again.
    agent._model_blocked["until"] = 0.0
    assert agent.model_block is None
    agent.chat("elevation around Shackleton?")
    assert client.creates == 2


def test_safety_questions_add_a_deterministic_safe_region_search(wired):
    agent = TALUSAgent(client=FailingGeminiClient(DAILY_QUOTA), dem_cache_dir=wired)
    result = agent.chat("Check rover safety for a terrain region near Shackleton Crater.", max_slope_deg=12.0)
    assert result["tool_calls"][-1]["tool"] == "find_safe_regions"
    assert result["tool_calls"][-1]["args"]["max_slope_deg"] == 12.0
    assert "Configured analysis threshold: 12°" in result["text"]
    assert "NOT a certified safety limit" in result["text"]


def test_quota_exhaustion_without_a_known_place_says_so_plainly(tmp_path):
    agent = TALUSAgent(client=FailingGeminiClient(DAILY_QUOTA), dem_cache_dir=str(tmp_path))
    result = agent.chat("is the terrain over there flat?")
    assert result["status"] == "model_error"
    assert result["text"].startswith(QUOTA_DAILY_MESSAGE)
    assert "Shackleton Crater" in result["text"] and result["tool_calls"] == []


def test_fallback_reports_a_nasa_outage_without_inventing_numbers(mock_nasa, make_service, tmp_path, monkeypatch):
    service = make_service(cache_dir=tmp_path / "cache", retries=1)
    monkeypatch.setattr(agent_module, "_get_nasa_service", lambda cache_dir: service)
    mock_nasa.ode_script.extend([lambda request: httpx.Response(503)] * 4)
    agent = TALUSAgent(client=FailingGeminiClient(DAILY_QUOTA), dem_cache_dir=str(tmp_path / "cache"))

    result = agent.chat("average elevation around Shackleton Crater?")

    assert result["status"] == "fallback"
    assert [c["tool"] for c in result["tool_calls"]] == ["resolve_lunar_feature", "fetch_nasa_dem"]
    assert "could not be completed" in result["text"] and "No terrain values were produced" in result["text"]
    assert " m" not in result["text"].split("could not be completed")[0][-20:]


def test_demo_mode_still_answers_a_named_place_deterministically(wired):
    agent = TALUSAgent(api_key=None, dem_cache_dir=wired)
    assert not agent.is_live
    result = agent.chat("What is the average elevation around Shackleton Crater?")
    assert result["status"] == "fallback" and result["is_demo"] is True
    # NVIDIA is the default provider: the notice says it is not connected.
    assert "NVIDIA AI is not connected" in result["notice"]
    gemini = TALUSAgent(api_key=None, dem_cache_dir=wired, provider="gemini")
    assert "not configured" in gemini.chat("What is the average elevation around Shackleton Crater?")["notice"]


def test_demo_mode_without_a_place_keeps_the_capabilities_message(tmp_path):
    result = TALUSAgent(api_key=None, dem_cache_dir=str(tmp_path)).chat("hello")
    assert result["is_demo"] is True and "Demo Mode" in result["text"]


def test_a_working_model_is_unaffected_by_the_fallback(tmp_path):
    agent = TALUSAgent(client=MockGeminiClient([text_response("Hi. Research/demo, not certified.")]), dem_cache_dir=str(tmp_path))
    assert agent.chat("hello")["status"] == "ok"


# ---------------------------------------------------------------------------
# System health checks
# ---------------------------------------------------------------------------


def test_secrets_check_reports_names_never_values(monkeypatch):
    from terrain_agent.config.settings import settings

    monkeypatch.setattr(settings.model, "provider", "gemini")
    monkeypatch.setattr(settings.model, "api_key", None)
    monkeypatch.setattr(settings.model, "vertex_project_id", None)
    monkeypatch.setattr(settings.nasa, "downloads_enabled", False)
    result = health.check_secrets()
    assert result["state"] == "warn" and result["missing"] == ["GEMINI_API_KEY"]
    assert "TALUS_NASA_DOWNLOADS" in result["detail"]

    monkeypatch.setattr(settings.model, "api_key", "value-that-must-not-leak")
    monkeypatch.setattr(settings.nasa, "downloads_enabled", True)
    result = health.check_secrets()
    assert result["state"] == "ok" and "value-that-must-not-leak" not in str(result)


def test_gemini_check_states(monkeypatch):
    from terrain_agent.config.settings import settings

    monkeypatch.setattr(settings.model, "api_key", None)
    monkeypatch.setattr(settings.model, "vertex_project_id", None)
    assert health.check_gemini()["state"] == "warn"

    monkeypatch.setattr(settings.model, "api_key", "k")
    assert health.check_gemini()["state"] == "ok"  # configuration only: no request spent
    blocked = health.check_gemini(observed_block={"category": "quota_daily", "message": QUOTA_DAILY_MESSAGE})
    assert blocked["state"] == "fail" and blocked["detail"] == QUOTA_DAILY_MESSAGE

    monkeypatch.setattr(agent_module, "check_gemini_health", lambda: {"request": "FAIL", "error_category": "quota_daily"})
    live = health.check_gemini(live=True)
    assert live["state"] == "fail" and live["detail"] == QUOTA_DAILY_MESSAGE


def _transport(handler):
    return httpx.MockTransport(handler)


def test_nasa_ode_probe_ok():
    result = health.check_nasa_ode(transport=_transport(lambda r: httpx.Response(200, text='{"ODEResults": {"Count": "3"}}')))
    assert result["state"] == "ok"


def test_nasa_ode_probe_uses_only_the_fixed_nasa_endpoint():
    seen = []

    def handler(request):
        seen.append(request.url)
        return httpx.Response(200, text='{"ODEResults": {}}')

    health.check_nasa_ode(transport=_transport(handler))
    assert seen[0].host == "oderest.rsl.wustl.edu" and seen[0].params["results"] == "c"


@pytest.mark.parametrize(
    "handler, fragment",
    [
        (lambda r: httpx.Response(503), "HTTP 503"),
        (lambda r: (_ for _ in ()).throw(httpx.ConnectTimeout("slow")), "No response"),
        (lambda r: (_ for _ in ()).throw(httpx.ConnectError("down")), "Unreachable"),
    ],
)
def test_nasa_ode_probe_failures(handler, fragment):
    result = health.check_nasa_ode(transport=_transport(handler))
    assert result["state"] == "fail" and fragment in result["detail"]


def test_nasa_ode_probe_unrecognised_body_is_a_warning():
    assert health.check_nasa_ode(transport=_transport(lambda r: httpx.Response(200, text="<html>maintenance</html>")))["state"] == "warn"


def test_internet_check_dns_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise socket.gaierror("no dns")

    monkeypatch.setattr(health.socket, "create_connection", fail)
    assert health.check_internet()["state"] == "fail"


def test_cache_checks(tmp_path, monkeypatch):
    from terrain_agent.config.settings import settings

    monkeypatch.setattr(settings.paths, "cache_dir", tmp_path / "cache")
    monkeypatch.setattr(settings.paths, "sample_dir", tmp_path / "sample")
    assert health.check_dem_cache()["state"] == "warn"
    assert health.check_cache_writable()["state"] == "ok"
    (tmp_path / "cache" / "nasa").mkdir(parents=True)
    (tmp_path / "cache" / "nasa" / "x.tif").write_bytes(b"x")
    assert health.check_dem_cache()["count"] == 1

    blocker = tmp_path / "file-not-dir"
    blocker.write_text("x")
    monkeypatch.setattr(settings.paths, "cache_dir", blocker / "cache")
    assert health.check_cache_writable()["state"] == "fail"


def test_system_health_never_raises_and_can_skip_the_network(monkeypatch):
    monkeypatch.setattr(health, "check_dem_cache", lambda: 1 / 0)
    results = health.check_system_health(network=False)
    names = [r["name"] for r in results]
    assert "NASA ODE" not in names and "Internet access" not in names
    assert any(r["name"] == "Diagnostic" and r["state"] == "warn" for r in results)
    assert all(r["state"] in ("ok", "warn", "fail") for r in results)


def test_network_check_switch(monkeypatch):
    monkeypatch.setenv("TALUS_HEALTH_NETWORK_CHECK", "false")
    assert health.network_checks_enabled() is False
    monkeypatch.setenv("TALUS_HEALTH_NETWORK_CHECK", "true")
    assert health.network_checks_enabled() is True


def test_secrets_check_with_nvidia_requires_no_ai_secret(monkeypatch):
    """NVIDIA mode: the key is entered per session in the UI, so no AI secret is required."""
    from terrain_agent.config.settings import settings

    monkeypatch.setattr(settings.model, "provider", "nvidia")
    monkeypatch.setattr(settings.model, "api_key", None)
    monkeypatch.setattr(settings.model, "vertex_project_id", None)
    monkeypatch.setattr(settings.nasa, "downloads_enabled", True)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-ENV-KEY-MUST-NOT-BE-USED")
    result = health.check_secrets()
    assert result["state"] == "ok" and result["missing"] == []
    assert "GEMINI_API_KEY" not in result["detail"] and "nvapi-ENV" not in str(result)


def test_nvidia_health_check_states_make_no_request():
    assert health.check_nvidia()["state"] == "warn"
    assert "Not Connected" in health.check_nvidia("not_connected")["detail"]
    assert health.check_nvidia("connecting")["detail"] == "NVIDIA AI Connecting..."
    ok = health.check_nvidia("connected", model="nvidia/some-model")
    assert ok["state"] == "ok" and ok["detail"] == "NVIDIA NIM Connected (nvidia/some-model)"
    assert health.check_nvidia("auth_failed")["detail"] == "NVIDIA NIM Authentication Failed"
    blocked = health.check_nvidia("connected", observed_block={"category": "rate_limited", "message": "limit"})
    assert blocked["state"] == "fail" and blocked["detail"] == "limit"


def test_system_health_reports_nvidia_by_default(monkeypatch):
    from terrain_agent.config.settings import settings

    monkeypatch.setattr(settings.model, "provider", "nvidia")
    names = [r["name"] for r in health.check_system_health(network=False)]
    assert "NVIDIA NIM" in names and "Gemini API" not in names
    custom = health.check_nvidia("connected", model="m")
    assert custom in health.check_system_health(network=False, ai_check=custom)
