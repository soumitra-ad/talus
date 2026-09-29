"""Phase 7 UI tests for the TALUS Streamlit application.

Uses Streamlit's ``AppTest`` framework with a mocked deterministic-tool backend
(``terrain_agent.agent.agent.dispatch_tool_call``, the sole bridge ``app/streamlit_app.py``
uses to reach the terrain engine) and a mocked managed-DEM listing
(``ui_helpers.list_managed_dems``). This exercises the UI end to end -- widget interactions,
structured status-badge rendering, error handling -- without needing real DEM files, NASA
network access, or a live Gemini credential, per the Phase 7 instruction to test the app with
mocked backend responses.

Every status badge assertion below only ever sets a structured field (``overall_status``,
``status``) on the mocked result; none of these tests exercise a code path that parses a
status out of response text, because ``app/streamlit_app.py`` has no such code path (see
``ui_helpers.status_badge_html``).
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = "../../app/streamlit_app.py"


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    """``st.cache_data``/``st.cache_resource`` are process-global, so a stale cached result
    (or cached agent) from one test could leak into the next test's assertions otherwise."""
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=60)


def _mock_dispatch(monkeypatch, table_or_fn: dict[str, dict[str, Any]] | Callable[..., dict[str, Any]]) -> None:
    """Replace the deterministic tool bridge the UI calls, with a fixed table of results
    keyed by tool name, or a callable ``(tool_name, args, dem_cache_dir=None) -> dict``."""
    import terrain_agent.agent.agent as agent_module

    if callable(table_or_fn):
        fn = table_or_fn
    else:
        def fn(tool_name, args, dem_cache_dir=None, _table=table_or_fn):
            return _table[tool_name]

    monkeypatch.setattr(agent_module, "dispatch_tool_call", fn)


def _mock_managed_dems(monkeypatch, names: list[str]) -> None:
    import ui_helpers

    monkeypatch.setattr(ui_helpers, "list_managed_dems", lambda: list(names))


DISCLAIMER = "TALUS is a research and demonstration system. Not certified for flight safety."


TEST_NVIDIA_KEY = "nvapi-TEST-ONLY-not-a-real-key-0123456789"


def _connect(app: AppTest, monkeypatch, key: str = TEST_NVIDIA_KEY) -> AppTest:
    """Run the app and connect NVIDIA AI through the real setup form, with the network check
    replaced by a stub that accepts the key (no NVIDIA request is made)."""
    from terrain_agent.agent import nvidia as nvidia_mod

    def fake_validate(secret, model, **_):
        assert isinstance(secret, nvidia_mod.SessionSecret) and secret.reveal() == key
        return (
            nvidia_mod.ConnectionReport(True, None, "NVIDIA NIM Connected", model, True, True, 0.1),
            nvidia_mod.NvidiaAgentClient(client=None, model=model),
        )

    monkeypatch.setattr(nvidia_mod, "validate_nvidia_key", fake_validate)
    app.run()
    next(w for w in app.text_input if w.label == "NVIDIA API Key").set_value(key)
    next(b for b in app.button if b.label == "Connect NVIDIA AI").click().run()
    assert app.session_state["nvidia_state"] == "connected"
    return app


# ---------------------------------------------------------------------------
# 1. Basic load / layout
# ---------------------------------------------------------------------------


def test_app_loads_without_exceptions(app):
    app.run()
    assert not app.exception
    assert any("TALUS" in getattr(m, "value", "") for m in app.markdown)


def test_research_demo_disclaimer_is_always_shown(app):
    app.run()
    assert any("RESEARCH" in getattr(m, "value", "") and "DEMO" in getattr(m, "value", "") for m in app.markdown)


def test_no_dems_cached_shows_a_helpful_message_not_a_crash(app, monkeypatch):
    _mock_managed_dems(monkeypatch, [])
    app.run()
    assert not app.exception
    assert any("No DEMs cached yet" in getattr(i, "value", "") for i in app.info)


def test_direct_analysis_tabs_are_present(app):
    app.run()
    assert not app.exception
    assert len(app.tabs) == 6


# ---------------------------------------------------------------------------
# 2/9/11. Coordinate input + rover safety, with structured PASS/REVIEW_REQUIRED/FAIL badges
# ---------------------------------------------------------------------------


def _rover_result(status: str) -> dict[str, Any]:
    return {
        "status": "ok",
        "overall_status": status,
        "risk_score": 42.0,
        "analysis": {
            "max_slope_deg": 12.3,
            "coverage_fraction": 1.0,
            "configured_thresholds": {
                "statement": "Configured analysis threshold: 15.0°. This is NOT a certified safety limit."
            },
            "violated_segments": [0] if status == "FAIL" else [],
            "incomplete_segments": [1] if status == "REVIEW_REQUIRED" else [],
            "segments": [
                {
                    "segment_index": 0,
                    "status": status,
                    "max_slope_deg": 12.3,
                    "mean_tri_m": 0.4,
                    "coverage_fraction": 1.0,
                }
            ],
            "disclaimer": DISCLAIMER,
        },
        "data_source": "flat.tif",
        "disclaimer": DISCLAIMER,
    }


@pytest.mark.parametrize(
    "status, badge_html",
    [("PASS", 'badge-pass">✅ PASS'), ("REVIEW_REQUIRED", 'badge-review">⚠️ REVIEW_REQUIRED'), ("FAIL", 'badge-fail">⛔ FAIL')],
)
def test_rover_status_renders_the_matching_structured_badge(app, monkeypatch, status, badge_html):
    _mock_managed_dems(monkeypatch, ["flat.tif"])
    _mock_dispatch(monkeypatch, {"evaluate_traverse_route": _rover_result(status)})

    app.run()
    tab = app.tabs[3]  # Rover Route
    submit = next(b for b in tab.button if "Evaluate route safety" in b.label)
    submit.click().run()

    assert not app.exception
    assert any(badge_html in getattr(m, "value", "") for m in app.markdown)
    # The configured-threshold statement (AGENTS.md §4) must be visible, verbatim.
    assert any("NOT a certified safety limit" in getattr(c, "value", "") for c in app.caption)


def test_rover_map_and_metrics_render_without_exception(app, monkeypatch):
    _mock_managed_dems(monkeypatch, ["flat.tif"])
    _mock_dispatch(monkeypatch, {"evaluate_traverse_route": _rover_result("PASS")})

    app.run()
    tab = app.tabs[3]
    submit = next(b for b in tab.button if "Evaluate route safety" in b.label)
    submit.click().run()

    assert not app.exception
    assert len(app.get("plotly_chart")) >= 1


# ---------------------------------------------------------------------------
# Landing sites: badges per ranked site
# ---------------------------------------------------------------------------


def _landing_result() -> dict[str, Any]:
    ranked_site = {
        "site_id": "site-a", "lat": -89.9, "lon": 0.0, "status": "PASS",
        "max_slope_deg": 5.0, "flat_radius_m": 150.0, "coverage_fraction": 1.0,
    }
    unranked_site_analysis = {
        "site_id": "site-b", "lat": -89.85, "lon": 45.0, "status": "REVIEW_REQUIRED",
        "max_slope_deg": None, "flat_radius_m": None, "coverage_fraction": 0.4,
    }
    return {
        "status": "ok",
        "sites_evaluated": 2, "sites_ranked": 1, "sites_unranked": 1,
        "analysis": {
            "ranked": [{"rank": 1, "analysis": ranked_site}],
            "unranked": [{"site_id": "site-b", "status": "REVIEW_REQUIRED", "reasons": ["incomplete coverage"]}],
            "sites": [ranked_site, unranked_site_analysis],
            "disclaimer": DISCLAIMER,
        },
        "disclaimer": DISCLAIMER,
    }


def test_landing_sites_render_ranked_badges_and_unranked_reasons(app, monkeypatch):
    _mock_managed_dems(monkeypatch, ["flat.tif"])
    _mock_dispatch(monkeypatch, {"evaluate_landing_sites": _landing_result()})

    app.run()
    tab = app.tabs[4]  # Landing Sites
    submit = next(b for b in tab.button if "Evaluate landing sites" in b.label)
    submit.click().run()

    assert not app.exception
    assert any('badge-pass">✅ PASS' in getattr(m, "value", "") for m in app.markdown)
    assert any("incomplete coverage" in getattr(m, "value", "") or "incomplete coverage" in getattr(w, "value", "") for m in app.markdown for w in [m])


# ---------------------------------------------------------------------------
# Safe regions
# ---------------------------------------------------------------------------


def test_safe_regions_outcome_and_table_render(app, monkeypatch):
    _mock_managed_dems(monkeypatch, ["flat.tif"])
    result = {
        "status": "ok",
        "outcome": "regions_found",
        "analysis": {
            "outcome": "regions_found",
            "regions": [
                {
                    "region_id": "r0", "cells": 120, "centroid_lat": -89.9, "centroid_lon": 0.0,
                    "largest_inscribed_circle_radius_m": 500.0, "max_slope_deg": 8.0, "mean_elevation_m": 1500.0,
                }
            ],
            "threshold_statement": "Configured analysis threshold: 12.0°. This is NOT a certified safety limit.",
            "disclaimer": DISCLAIMER,
        },
        "disclaimer": DISCLAIMER,
    }
    _mock_dispatch(monkeypatch, {"find_safe_regions": result})

    app.run()
    tab = app.tabs[2]  # Safe Regions
    submit = next(b for b in tab.button if "Find safe regions" in b.label)
    submit.click().run()

    assert not app.exception
    assert any("regions_found" in getattr(m, "value", "") for m in app.markdown)
    assert len(app.dataframe) >= 1


def test_no_terrain_data_outcome_is_shown_distinctly_not_as_pass(app, monkeypatch):
    _mock_managed_dems(monkeypatch, ["flat.tif"])
    result = {
        "status": "no_terrain_data",
        "outcome": "no_terrain_data",
        "analysis": {"outcome": "no_terrain_data", "regions": [], "threshold_statement": "", "disclaimer": DISCLAIMER},
        "disclaimer": DISCLAIMER,
    }
    _mock_dispatch(monkeypatch, {"find_safe_regions": result})

    app.run()
    tab = app.tabs[2]
    submit = next(b for b in tab.button if "Find safe regions" in b.label)
    submit.click().run()

    assert not app.exception
    assert any("no_terrain_data" in getattr(m, "value", "") for m in app.markdown)
    # 'badge-pass">' is how an actually-rendered PASS badge looks (see status_badge_html);
    # the bare class name also appears in this page's own <style> block, so check for the
    # rendered form specifically, not just the substring "badge-pass".
    assert not any('badge-pass">' in getattr(m, "value", "") for m in app.markdown)


# ---------------------------------------------------------------------------
# Terrain statistics / dataset info / provenance
# ---------------------------------------------------------------------------


def test_terrain_statistics_panel_renders_metrics(app, monkeypatch):
    _mock_managed_dems(monkeypatch, ["ramp.tif"])

    def fake_dispatch(tool_name, args, dem_cache_dir=None):
        base = {
            "status": "ok",
            "analysis": {
                "terrain_available": True,
                "coverage_fraction": 1.0,
                "resolution_m": 30.3,
                "elevation": {"mean_m": 1500.0, "min_m": 1490.0, "max_m": 1510.0},
                "slope": None,
                "roughness": None,
            },
            "resolution_m": 30.3,
            "disclaimer": DISCLAIMER,
        }
        if tool_name == "get_slope_stats":
            base["analysis"]["slope"] = {"mean_slope_deg": 10.0, "median_slope_deg": 9.5, "max_slope_deg": 11.0}
        if tool_name == "get_roughness_stats":
            base["analysis"]["roughness"] = {"mean_tri_m": 0.3, "max_tri_m": 0.9, "std_tri_m": 0.1}
        return base

    _mock_dispatch(monkeypatch, fake_dispatch)

    app.run()
    tab = app.tabs[1]  # Terrain Statistics
    submit = next(b for b in tab.button if "Compute terrain statistics" in b.label)
    submit.click().run()

    assert not app.exception
    assert len(app.metric) >= 6  # elevation(3) + slope(3) + roughness(3)


def test_terrain_statistics_missing_data_shows_a_warning_not_a_crash(app, monkeypatch):
    _mock_managed_dems(monkeypatch, ["ramp.tif"])
    _mock_dispatch(
        monkeypatch,
        lambda tool_name, args, dem_cache_dir=None: {
            "status": "no_terrain_data",
            "analysis": {"terrain_available": False},
            "disclaimer": DISCLAIMER,
        },
    )

    app.run()
    tab = app.tabs[1]
    submit = next(b for b in tab.button if "Compute terrain statistics" in b.label)
    submit.click().run()

    assert not app.exception
    assert any("No terrain data is available" in getattr(w, "value", "") for w in app.warning)


def test_dataset_information_panel_shows_provenance(app, monkeypatch):
    _mock_managed_dems(monkeypatch, ["ldem.tif"])
    dataset = {
        "file_name": "ldem.tif", "provenance": "sidecar", "mission": "LRO", "instrument": "LOLA",
        "product_type": "GDRDEM", "product_id": "ldem_75s_240m", "crs_kind": "projected",
        "width": 3812, "height": 3812, "resolution_m": 240.0,
        "elevation_reference": "height above a reference sphere, not a geoid", "warnings": [],
    }
    _mock_dispatch(monkeypatch, {"dataset_information": {"status": "ok", "dataset": dataset, "disclaimer": DISCLAIMER}})

    app.run()
    tab = app.tabs[0]  # Dataset Info
    submit = next(b for b in tab.button if "Describe dataset" in b.label)
    submit.click().run()

    assert not app.exception
    assert any("LRO" in getattr(m, "value", "") for m in app.markdown)
    assert any("not a geoid" in getattr(c, "value", "") for c in app.caption)


# ---------------------------------------------------------------------------
# Error messages understandable to a student/research user; no internal leakage
# ---------------------------------------------------------------------------


def test_tool_error_shows_a_friendly_message_not_raw_internals(app, monkeypatch):
    _mock_managed_dems(monkeypatch, ["flat.tif"])
    _mock_dispatch(
        monkeypatch,
        {
            "evaluate_traverse_route": {
                "status": "error",
                "error_type": "InvalidCoordinateError",
                "error": "Latitude 999.0 is outside the valid range.",
                "disclaimer": DISCLAIMER,
            }
        },
    )

    app.run()
    tab = app.tabs[3]
    submit = next(b for b in tab.button if "Evaluate route safety" in b.label)
    submit.click().run()

    assert not app.exception
    all_errors = " ".join(getattr(e, "value", "") for e in app.error)
    assert "not valid" in all_errors.lower()
    # Nothing that looks like a stack trace or an internal file path is ever shown.
    assert "Traceback" not in all_errors
    assert "\\src\\terrain_agent" not in all_errors and "/src/terrain_agent" not in all_errors
    assert ".py:" not in all_errors


def test_no_page_content_ever_contains_a_traceback_or_stack_frame(app, monkeypatch):
    """Even an unexpected internal failure must reach the user only as the sanitized
    ``InternalError`` message that ``dispatch_tool_call`` already produces, never a raw
    exception repr or traceback."""
    _mock_managed_dems(monkeypatch, ["flat.tif"])
    _mock_dispatch(
        monkeypatch,
        {
            "evaluate_traverse_route": {
                "status": "error",
                "error_type": "InternalError",
                "error": "The tool failed unexpectedly. Details were written to the server log.",
                "disclaimer": DISCLAIMER,
            }
        },
    )

    app.run()
    tab = app.tabs[3]
    submit = next(b for b in tab.button if "Evaluate route safety" in b.label)
    submit.click().run()

    assert not app.exception
    page_text = " ".join(getattr(e, "value", "") for e in list(app.error) + list(app.markdown) + list(app.caption))
    assert "Traceback (most recent call last)" not in page_text
    assert "line " not in page_text or ".py" not in page_text


def test_api_key_value_is_never_rendered_anywhere_on_the_page(app, monkeypatch):
    secret = "sk-super-secret-gemini-key-should-never-appear"
    from terrain_agent.config.settings import settings as global_settings

    monkeypatch.setattr(global_settings.model, "api_key", secret)

    app.run()

    assert not app.exception
    everything = []
    for kind in ("markdown", "caption", "text", "info", "success", "warning", "error"):
        everything.extend(getattr(e, "value", "") for e in getattr(app, kind))
    assert secret not in " ".join(everything)


# ---------------------------------------------------------------------------
# Natural-language chat (Phase 6 agent), with a mocked TALUSAgent
# ---------------------------------------------------------------------------


class _FakeAgent:
    is_live = True

    def __init__(self, text: str, tool_calls: list[dict[str, Any]] | None = None, is_demo: bool = False):
        self._text = text
        self._tool_calls = tool_calls or []
        self._is_demo = is_demo

    def chat(self, user_message, history=None, max_slope_deg=15.0, on_event=None):
        return {"status": "ok", "text": self._text, "tool_calls": self._tool_calls, "is_demo": self._is_demo}


def test_chat_round_trip_with_a_mocked_agent(app, monkeypatch):
    import terrain_agent.agent as agent_pkg

    fake = _FakeAgent(
        "The mean slope is 8 degrees. This is a research and demonstration system; not certified.",
        tool_calls=[{"tool": "get_slope_stats", "result_status": "ok", "summary": "Calculated maximum and mean slope for the requested area."}],
    )
    monkeypatch.setattr(agent_pkg, "TALUSAgent", lambda *a, **k: fake)

    _connect(app, monkeypatch)
    app.chat_input[0].set_value("what is the slope near the south pole?").run()

    assert not app.exception
    assert any("mean slope is 8 degrees" in getattr(m, "value", "") for m in app.markdown)
    assert any("Calculated maximum and mean slope" in getattr(c, "value", "") for c in app.caption)


def test_chat_message_content_is_html_escaped_not_rendered_as_markup(app, monkeypatch):
    """Phase 8 hardening: the user's own message and the assistant's text must never be
    rendered as live HTML; both must stay inert text, since the
    assistant's text can in turn echo external data (e.g. a NASA product description) the
    model saw -- neither is trusted HTML."""
    import terrain_agent.agent as agent_pkg

    payload = "<img src=x onerror=alert(1)>"
    fake = _FakeAgent(f"Product description says: {payload}")
    monkeypatch.setattr(agent_pkg, "TALUSAgent", lambda *a, **k: fake)

    _connect(app, monkeypatch)
    app.chat_input[0].set_value(payload).run()

    assert not app.exception
    # Chat content is rendered with st.chat_message + st.text / st.markdown, never with
    # unsafe_allow_html. The payload must still be shown (as inert text), and no element that
    # is allowed to render raw HTML may contain the live tag.
    assert all(payload not in m.value for m in app.markdown if m.proto.allow_html)
    shown = [t.value for t in app.text] + [m.value for m in app.markdown if not m.proto.allow_html]
    assert any(payload in s for s in shown)


def test_chat_demo_mode_banner_shown_when_no_live_model(app, monkeypatch):
    import terrain_agent.agent as agent_pkg

    fake = _FakeAgent("TALUS Demo Mode response.", is_demo=True)
    fake.is_live = False
    monkeypatch.setattr(agent_pkg, "TALUSAgent", lambda *a, **k: fake)

    _connect(app, monkeypatch)
    app.chat_input[0].set_value("hello").run()

    assert not app.exception
    assert any("Demo Mode" in getattr(i, "value", "") for i in app.info)


def test_chat_rejects_are_shown_as_a_warning_not_a_crash(app, monkeypatch):
    """A request-validation or rate-limit response from the agent (Phase 6) must be shown
    plainly, not treated as a normal analysis result."""
    import terrain_agent.agent as agent_pkg

    class _RejectingAgent:
        is_live = True

        def chat(self, user_message, history=None, max_slope_deg=15.0, on_event=None):
            return {"status": "rate_limited", "text": "Too many requests. Please wait a moment and try again.", "tool_calls": [], "is_demo": False}

    monkeypatch.setattr(agent_pkg, "TALUSAgent", lambda *a, **k: _RejectingAgent())

    _connect(app, monkeypatch)
    app.chat_input[0].set_value("go go go").run()

    assert not app.exception
    assert any("Too many requests" in getattr(w, "value", "") for w in app.warning)


# ---------------------------------------------------------------------------
# NASA DEM search & fetch tab
# ---------------------------------------------------------------------------


def test_dem_search_tab_shows_results_table(app, monkeypatch):
    products_result = {
        "status": "ok",
        "count": 1,
        "products": [
            {"product_id": "ldem_75s_240m", "mission": "LRO", "instrument": "LOLA", "dataset": "LOLA Gridded DEM"}
        ],
        "disclaimer": DISCLAIMER,
    }
    _mock_dispatch(monkeypatch, {"search_dem_products": products_result})

    app.run()
    tab = app.tabs[5]  # NASA DEM Search
    submit = next(b for b in tab.button if "Search DEM Products" in b.label)
    submit.click().run()

    assert not app.exception
    assert any("Found 1 DEM product" in getattr(s, "value", "") for s in app.success)
    assert len(app.dataframe) >= 1


def test_access_gate_blocks_and_admits_correctly(app, monkeypatch):
    """Phase 8 hardening: with TALUS_ACCESS_TOKEN set, the app must not render its content
    until the correct token is entered -- an unguessable URL is not a security control."""
    monkeypatch.setenv("TALUS_ACCESS_TOKEN", "correct-horse-battery-staple")

    app.run()
    assert not app.exception
    assert len(app.tabs) == 0
    page_text = " ".join(getattr(e, "value", "") for e in list(app.markdown) + list(app.caption)).lower()
    assert "access token" in page_text

    token_input = next(w for w in app.text_input if w.label == "Access token")
    token_input.set_value("wrong-guess")
    enter = next(b for b in app.button if b.label == "Enter")
    enter.click().run()
    assert not app.exception
    assert len(app.tabs) == 0
    assert any("Incorrect access token" in getattr(e, "value", "") for e in app.error)

    token_input = next(w for w in app.text_input if w.label == "Access token")
    token_input.set_value("correct-horse-battery-staple")
    enter = next(b for b in app.button if b.label == "Enter")
    enter.click().run()
    assert not app.exception
    assert len(app.tabs) == 6


def test_no_access_token_shows_an_operator_warning_not_a_silent_open_deployment(app, monkeypatch):
    monkeypatch.delenv("TALUS_ACCESS_TOKEN", raising=False)
    app.run()
    assert not app.exception
    assert any("No access token is configured" in getattr(m, "value", "") for m in app.markdown)
    assert len(app.tabs) == 6  # still usable locally/dev by default


def test_dem_fetch_tab_reports_disabled_downloads_plainly(app, monkeypatch):
    _mock_dispatch(monkeypatch, {"fetch_nasa_dem": {"status": "disabled", "error": "downloads disabled", "disclaimer": DISCLAIMER}})

    app.run()
    tab = app.tabs[5]
    submit = next(b for b in tab.button if "Fetch DEM" in b.label)
    submit.click().run()

    assert not app.exception
    assert any("disabled" in getattr(w, "value", "").lower() for w in app.warning)


def test_dem_fetch_tab_reports_success(app, monkeypatch):
    fetch_result = {
        "status": "ok",
        "dem_path": "nasa/ldem_75s_240m-6d162ce24928.tif",
        "from_cache": False,
        "provenance": {
            "mission": "LRO", "instrument": "LOLA",
            "product_type": "GDRDEM", "product_id": "ldem_75s_240m",
            "pixel_size_m": [240.0, 240.0], "width": 3812, "height": 3812,
            "elevation_reference": "height above a reference sphere, not a geoid",
        },
        "disclaimer": DISCLAIMER,
    }
    _mock_dispatch(monkeypatch, {"fetch_nasa_dem": fetch_result})

    app.run()
    tab = app.tabs[5]  # NASA DEM Search
    submit = next(b for b in tab.button if "Fetch DEM" in b.label)
    submit.click().run()

    assert not app.exception
    assert any("DEM ready" in getattr(s, "value", "") for s in app.success)
    assert any("ldem_75s_240m" in getattr(s, "value", "") for s in app.success)


def test_dem_fetch_tab_reports_no_product_found_not_a_crash(app, monkeypatch):
    """Regression test for the "ODE Products section has an unexpected structure" bug: a
    query with zero matching NASA ODE products (e.g. SLDEM has no coverage at a given point)
    must surface as a plain, clear result -- never an unhandled parser exception reaching the
    Streamlit UI."""
    _mock_dispatch(
        monkeypatch,
        {
            "fetch_nasa_dem": {
                "status": "no_product",
                "error": "NASA ODE lists no supported DEM product for this area.",
                "excluded_products": [],
                "disclaimer": DISCLAIMER,
            }
        },
    )

    app.run()
    tab = app.tabs[5]
    submit = next(b for b in tab.button if "Fetch DEM" in b.label)
    submit.click().run()

    assert not app.exception
    assert any("no nasa dem product" in getattr(w, "value", "").lower() for w in app.warning)


# ---------------------------------------------------------------------------
# Release audit: Shackleton result cards, NASA status, history roles, no duplicate calls
# ---------------------------------------------------------------------------


def _shackleton_tool_calls() -> list[dict[str, Any]]:
    fetch = {
        "status": "ok", "tool": "fetch_nasa_dem", "dem_path": "nasa/ldem_75s_240m-6d162ce24928.tif", "from_cache": False,
        "provenance": {
            "provider": "nasa_ode", "mission": "LRO", "instrument": "LOLA", "product_type": "GDRDEM",
            "product_id": "ldem_75s_240m", "pixel_size_m": [240.0, 240.0], "width": 3812, "height": 3812,
            "acquired_at": "2026-09-30T00:08:19Z",
            "elevation_reference": "Height above a sphere of radius 1737400 m, not a geoid.",
        },
    }
    elevation = {
        "status": "ok", "tool": "get_elevation_stats", "resolution_m": 239.999,
        "analysis": {
            "terrain_available": True, "coverage_fraction": 1.0, "resolution_m": 239.999,
            "elevation": {"mean_m": -185.54, "min_m": -2855.5, "max_m": 1954.5},
        },
    }
    feature = {
        "status": "ok", "tool": "resolve_lunar_feature", "source": "TALUS curated lunar feature gazetteer",
        "feature": {"name": "Shackleton Crater", "center_lat": -89.9, "center_lon": 0.0},
        "analysis_area": {"min_lat": -90.0, "max_lat": -89.55, "min_lon": -180.0, "max_lon": 180.0, "radius_km": 10.5},
    }
    return [
        {"tool": "resolve_lunar_feature", "result_status": "ok", "summary": "Resolved Shackleton Crater from the TALUS gazetteer.", "args": {"name": "Shackleton Crater"}, "result": feature},
        {"tool": "fetch_nasa_dem", "result_status": "ok", "summary": "Selected a NASA DEM covering the requested region.", "args": {"lat": -89.9, "lon": 0.0}, "result": fetch},
        {"tool": "get_elevation_stats", "result_status": "ok", "summary": "Calculated elevation statistics for the requested area.", "args": {"dem_path": "x", "min_lat": -90.0, "max_lat": -89.55, "min_lon": -180.0, "max_lon": 180.0}, "result": elevation},
    ]


class _RecordingAgent(_FakeAgent):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls: list[dict[str, Any]] = []

    def chat(self, user_message, history=None, max_slope_deg=15.0, on_event=None):
        self.calls.append({"message": user_message, "history": history})
        if on_event:
            on_event("tool_start", {"tool": "fetch_nasa_dem"})
            on_event("tool_end", {"tool": "fetch_nasa_dem", "status": "ok", "summary": "done"})
        return super().chat(user_message, history, max_slope_deg)


def test_shackleton_answer_renders_result_cards_evidence_and_nasa_status(app, monkeypatch):
    import terrain_agent.agent as agent_pkg

    fake = _RecordingAgent("Based on the retrieved LOLA DEM, the mean elevation is -185.54 m. Research/demo, not certified.", tool_calls=_shackleton_tool_calls())
    monkeypatch.setattr(agent_pkg, "TALUSAgent", lambda *a, **k: fake)

    _connect(app, monkeypatch)
    # Offline suite: the network probe is off, so NASA is not claimed healthy before any contact.
    assert any("NASA: 🟡" in m.value for m in app.markdown)
    app.chat_input[0].set_value("What is the average elevation around Shackleton Crater?").run()

    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Elevation (mean)"] == "-185.54 m"
    assert metrics["Dataset"] == "LRO / LOLA"
    assert metrics["Safety score"] == "Not evaluated"  # no safety tool ran: nothing invented
    captions = " ".join(c.value for c in app.caption)
    assert "Analysis status: Completed" in captions
    assert "ldem_75s_240m" in captions and "NASA ODE" in captions
    assert "Resolving lunar coordinates" in captions and "Calculating elevation" in captions
    assert any("ldem_75s_240m" in m.value for m in app.markdown)
    assert any("NASA: 🟢 Connected" in m.value for m in app.markdown)


def test_follow_up_history_uses_gemini_roles_and_reruns_do_not_repeat_calls(app, monkeypatch):
    import terrain_agent.agent as agent_pkg

    fake = _RecordingAgent("Answer. Research/demo, not certified.")
    monkeypatch.setattr(agent_pkg, "TALUSAgent", lambda *a, **k: fake)

    _connect(app, monkeypatch)
    app.chat_input[0].set_value("first question").run()
    app.run()  # a plain rerun (e.g. a widget change) must not call the agent again
    assert len(fake.calls) == 1
    app.chat_input[0].set_value("second question").run()

    assert not app.exception
    assert len(fake.calls) == 2
    history = fake.calls[1]["history"]
    assert [h["role"] for h in history] == ["user", "model"]
    assert history[0]["parts"][0]["text"] == "first question"


def test_suggested_prompt_button_runs_the_query_once(app, monkeypatch):
    import terrain_agent.agent as agent_pkg

    fake = _RecordingAgent("Answer. Research/demo, not certified.")
    monkeypatch.setattr(agent_pkg, "TALUSAgent", lambda *a, **k: fake)

    _connect(app, monkeypatch)
    next(b for b in app.button if "Shackleton" in b.label).click().run()

    assert not app.exception
    assert [c["message"] for c in fake.calls] == ["What is the average elevation around Shackleton Crater?"]


def test_model_error_keeps_deterministic_results_visible(app, monkeypatch):
    import terrain_agent.agent as agent_pkg

    class _Failing(_FakeAgent):
        def chat(self, user_message, history=None, max_slope_deg=15.0, on_event=None):
            return {"status": "model_error", "text": "Gemini service is currently unavailable. Please retry the analysis.", "tool_calls": _shackleton_tool_calls()[:2], "is_demo": False}

    monkeypatch.setattr(agent_pkg, "TALUSAgent", lambda *a, **k: _Failing(""))

    _connect(app, monkeypatch)
    app.chat_input[0].set_value("elevation around Shackleton?").run()

    assert not app.exception
    assert any("Gemini service is currently unavailable" in w.value for w in app.warning)
    assert any("Analysis status: Not completed" in c.value for c in app.caption)
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Dataset"] == "LRO / LOLA"
    assert metrics["Elevation (mean)"] == "—"  # the elevation step never ran: shown as missing

# ---------------------------------------------------------------------------
# Deployment hardening: Gemini fallback display, auto DEM download, secrets mirroring
# ---------------------------------------------------------------------------


def test_quota_fallback_answer_shows_the_notice_and_deterministic_cards(app, monkeypatch):
    import terrain_agent.agent as agent_pkg

    class _FallbackAgent(_FakeAgent):
        def chat(self, user_message, history=None, max_slope_deg=15.0, on_event=None):
            return {
                "status": "fallback", "notice": "Gemini daily quota reached. Terrain tools still available.",
                "text": "Deterministic analysis of **Shackleton Crater**: mean elevation **-185.54 m**.",
                "tool_calls": _shackleton_tool_calls(), "is_demo": False, "error_category": "quota_daily",
            }

    monkeypatch.setattr(agent_pkg, "TALUSAgent", lambda *a, **k: _FallbackAgent(""))

    _connect(app, monkeypatch)
    app.chat_input[0].set_value("What is the average elevation around Shackleton Crater?").run()

    assert not app.exception
    assert any("Gemini daily quota reached. Terrain tools still available." in w.value for w in app.warning)
    assert {m.label: m.value for m in app.metric}["Elevation (mean)"] == "-185.54 m"
    assert any("Analysis status: Completed" in c.value for c in app.caption)


def test_statistics_tab_auto_downloads_a_dem_when_none_is_cached(app, monkeypatch):
    _mock_managed_dems(monkeypatch, [])
    calls = []

    def fake_dispatch(tool_name, args, dem_cache_dir=None):
        calls.append((tool_name, dict(args)))
        if tool_name == "fetch_nasa_dem":
            return {"status": "ok", "dem_path": "nasa/ldem_75s_240m-x.tif", "from_cache": False,
                    "provenance": {"mission": "LRO", "instrument": "LOLA", "product_id": "ldem_75s_240m"}}
        return {"status": "ok", "resolution_m": 240.0, "analysis": {
            "terrain_available": True, "coverage_fraction": 1.0,
            "elevation": {"mean_m": -185.5, "min_m": -2855.5, "max_m": 1954.5}, "slope": None, "roughness": None}}

    _mock_dispatch(monkeypatch, fake_dispatch)

    app.run()
    tab = app.tabs[1]  # Statistics
    assert tab.selectbox[0].value.startswith("⬇️ Auto")
    next(b for b in tab.button if "Compute terrain statistics" in b.label).click().run()

    assert not app.exception
    assert calls[0][0] == "fetch_nasa_dem"
    assert {c[0] for c in calls[1:]} == {"get_elevation_stats", "get_slope_stats", "get_roughness_stats"}
    assert all(c[1]["dem_path"] == "nasa/ldem_75s_240m-x.tif" for c in calls[1:])
    assert any("ldem_75s_240m" in c.value for c in app.caption)


def test_dataset_tab_never_offers_auto_download(app, monkeypatch):
    _mock_managed_dems(monkeypatch, ["flat.tif"])
    app.run()
    assert all(not o.startswith("⬇️") for o in app.tabs[0].selectbox[0].options)


def test_toml_boolean_secrets_are_mirrored_into_the_environment(app, monkeypatch):
    import os

    monkeypatch.delenv("TALUS_TEST_BOOL_SECRET", raising=False)
    monkeypatch.delenv("TALUS_TEST_STR_SECRET", raising=False)
    app.secrets["TALUS_TEST_BOOL_SECRET"] = True
    app.secrets["TALUS_TEST_STR_SECRET"] = "value"
    app.run()
    try:
        assert os.environ.get("TALUS_TEST_BOOL_SECRET") == "true"
        assert os.environ.get("TALUS_TEST_STR_SECRET") == "value"
    finally:
        os.environ.pop("TALUS_TEST_BOOL_SECRET", None)
        os.environ.pop("TALUS_TEST_STR_SECRET", None)


def test_header_and_status_bar_render_without_network_access(app):
    app.run()
    assert not app.exception
    page = " ".join(m.value for m in app.markdown)
    assert "TALUS AI" in page and "Lunar Terrain Intelligence Agent" in page
    for label in ("NVIDIA NIM:", "NASA:", "DEM Cache:"):
        assert label in page