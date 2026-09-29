"""
TALUS Streamlit Frontend.

Terrain Analysis for Landing and Uncrewed Systems — research/demo interface.

This module renders results and wires up user input; it performs no terrain calculation of
its own. Every number shown here comes from ``terrain_agent.agent.agent.dispatch_tool_call``
(the same deterministic bridge the conversational agent uses) or from
``terrain_agent.agent.TALUSAgent.chat()``, whose ``tool_calls`` already carry the full
structured backend result for each tool it ran. Status badges are drawn only from the
structured ``status`` / ``overall_status`` enum fields on those results -- never parsed out of
response text (see ``ui_helpers.status_badge_html``).

Layout (chat first, usable on small screens):
- Header and a status bar (Gemini / NASA ODE / DEM engine)
- Agent conversation: each answer carries an observable activity trace, result cards built
  from structured tool output, and an evidence panel (dataset provenance, coordinates, method)
- Direct Structured Analysis: forms that call the deterministic tools directly, no model
- Sidebar: mission parameters, Gemini health check, conversation controls

No external service is contacted at startup: Gemini and NASA are only called when the user
asks for something, so the UI loads even if either is unavailable.

Research/Demo Disclaimer
------------------------
TALUS is a research and demonstration system. It must NEVER claim to provide
certified flight safety, operational landing approval, autonomous spacecraft
control, or guaranteed rover safety.
"""

from __future__ import annotations

import hmac
import html
import logging
import os
from typing import Any

import streamlit as st

from ui_helpers import (
    TOOL_LABELS,
    analysis_outcome,
    build_terrain_map_figure,
    build_trace,
    fmt,
    fmt_pct,
    friendly_error_message,
    list_managed_dems,
    nasa_status_from_tool_calls,
    sanitize_model_markdown,
    status_badge_html,
)

# ---------------------------------------------------------------------------
# Streamlit Community Cloud secrets -> environment
#
# Secrets pasted into Advanced settings are loaded into st.secrets; mirror root-level string
# values into os.environ so the plain os.getenv() calls in terrain_agent.config (and the
# TALUS_ACCESS_TOKEN check just below) see them exactly as they would a local .env file,
# regardless of exactly when st.secrets itself is populated.
# ---------------------------------------------------------------------------

try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str) and _key not in os.environ:
            os.environ[_key] = _value
except Exception:
    pass

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TALUS — Lunar Terrain Analysis",
    page_icon="🌕",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "About": (
            "**TALUS** — Terrain Analysis for Landing and Uncrewed Systems\n\n"
            "Research/Demo System only. NOT certified for flight operations."
        )
    },
)

# ---------------------------------------------------------------------------
# Minimal CSS: status pills, badges, disclaimer. No background images or animations.
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
.talus-sub { color:#94a3b8; font-size:0.9rem; margin-top:-0.6rem; }
.status-bar { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 10px 0; }
.pill { border:1px solid #334155; border-radius:999px; padding:2px 10px; font-size:0.78rem; color:#cbd5e1; white-space:nowrap; }
.dot-ok { color:#22c55e; } .dot-warn { color:#f59e0b; } .dot-err { color:#ef4444; } .dot-idle { color:#64748b; }
.badge-pass { background: #166534; color: #86efac; padding: 3px 10px; border-radius: 20px; font-weight: 700; font-size: 0.8rem; }
.badge-review { background: #78350f; color: #fde68a; padding: 3px 10px; border-radius: 20px; font-weight: 700; font-size: 0.8rem; }
.badge-fail { background: #7f1d1d; color: #fca5a5; padding: 3px 10px; border-radius: 20px; font-weight: 700; font-size: 0.8rem; }
.section-divider { border-top: 1px solid #1e3a5f; margin: 16px 0; }
.disclaimer {
    background: #1c1917; border: 1px solid #78350f; border-radius: 8px;
    padding: 8px 12px; color: #d97706; font-size: 0.78rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Logging (server-side only — never shown to the user)
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#: Conversation turns kept in session state. Older turns are dropped so a long session cannot
#: grow memory without bound; each stored turn holds only small structured tool results.
MAX_STORED_MESSAGES = 40


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def _init_session() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "nasa_status": "unknown",
        "gemini_health": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ---------------------------------------------------------------------------
# Optional access gate
#
# Streamlit has no built-in authentication. A public deployment must not assume that an
# unguessable URL is a security control (Phase 8 hardening, requirement #4): if
# TALUS_ACCESS_TOKEN is set, a matching token is required before anything else renders. Off
# by default for local/dev use, where the operator controls network access another way.
# ---------------------------------------------------------------------------


def _check_access() -> bool:
    required_token = os.environ.get("TALUS_ACCESS_TOKEN")
    if not required_token:
        return True
    if st.session_state.get("_authenticated"):
        return True

    st.markdown("## 🔒 TALUS Access")
    st.caption("This deployment requires an access token. Contact the operator for one.")
    with st.form("access_form"):
        entered = st.text_input("Access token", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        # Constant-time comparison so the check itself does not leak the token via timing.
        if hmac.compare_digest(entered, required_token):
            st.session_state["_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect access token.")
    return False


if not _check_access():
    st.stop()


# ---------------------------------------------------------------------------
# Agent singleton and cached tool calls
# ---------------------------------------------------------------------------


@st.cache_resource
def _build_agent(api_key: str | None, model_name: str) -> Any:
    """Build and cache a TALUSAgent instance. Never logs or displays the API key itself."""
    try:
        from terrain_agent.agent import TALUSAgent
        return TALUSAgent(api_key=api_key, model_name=model_name)
    except Exception:
        log.exception("Failed to initialise TALUSAgent")
        return None


@st.cache_data(ttl=30, show_spinner=False)
def _cached_managed_dems() -> list[str]:
    return list_managed_dems()


@st.cache_data(ttl=120, show_spinner=False)
def _cached_dispatch(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Cache read-only deterministic tool calls for a short window, so re-rendering the same
    query (a common Streamlit rerun pattern) doesn't repeat a raster read unnecessarily."""
    from terrain_agent.agent.agent import dispatch_tool_call
    return dispatch_tool_call(tool_name, args, dem_cache_dir=None)


def _dispatch(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    from terrain_agent.agent.agent import dispatch_tool_call
    return dispatch_tool_call(tool_name, args, dem_cache_dir=None)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

try:
    from terrain_agent.config import settings
    _APP_VERSION = settings.app_version
    _DEFAULT_SLOPE = settings.safety.default_max_slope_deg
    _GEMINI_KEY = settings.model.api_key
    _MODEL_NAME = settings.model.model_name
    _NASA_DOWNLOADS = settings.nasa.downloads_enabled
except Exception:
    log.exception("Failed to load configuration; using safe defaults")
    _APP_VERSION = "0.1.0"
    _DEFAULT_SLOPE = 15.0
    _GEMINI_KEY = None
    _MODEL_NAME = "gemini-3.6-flash"
    _NASA_DOWNLOADS = False

agent = _build_agent(_GEMINI_KEY, _MODEL_NAME)
available_dems = _cached_managed_dems()


# ---------------------------------------------------------------------------
# Sidebar: mission parameters and operator controls (collapsed by default on small screens)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🌕 TALUS")
    st.caption(f"v{_APP_VERSION} · Research/Demo System")

    st.markdown("### ⚙️ Mission Parameters")
    max_slope = st.slider(
        "Max Slope Threshold (°)",
        min_value=5.0, max_value=35.0, value=_DEFAULT_SLOPE, step=0.5,
        help="Configured mission slope limit. NOT a universal safety standard.",
    )
    st.markdown(
        f'<div class="disclaimer">Configured analysis threshold: <strong>{max_slope:.1f}°</strong>. '
        "This is NOT a certified safety limit.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### 🗂️ Default DEM")
    if available_dems:
        default_dem = st.selectbox(
            "DEM for the Direct Analysis tools",
            options=available_dems,
            help="Only DEMs already present in the managed cache/sample directories can be "
            "selected — never an arbitrary filesystem path.",
        )
    else:
        default_dem = None
        st.info("No DEMs cached yet. Ask the agent about a place, or use the NASA DEM Search "
                "tab, and a DEM will be fetched from NASA.")

    st.markdown("### 🤖 AI Status")
    if agent is not None and getattr(agent, "is_live", False):
        st.success(f"Gemini Active ({_MODEL_NAME})")
    else:
        st.info("Demo Mode — set GEMINI_API_KEY for live AI analysis. "
                "The Direct Analysis tools work without one.")
    if st.button("Run Gemini health check", width="stretch"):
        from terrain_agent.agent import check_gemini_health
        with st.spinner("Contacting Gemini…"):
            st.session_state.gemini_health = check_gemini_health(_GEMINI_KEY, _MODEL_NAME)
    health = st.session_state.gemini_health
    if health:
        st.caption(
            f"Configured: {'YES' if health['configured'] else 'NO'} · "
            f"Request: {health['request']} · "
            f"Latency: {fmt(health['latency_s'], ' s')}"
            + (f" · Error: {health['error_category']}" if health.get("error_category") else "")
        )

    if st.button("🗑️ Clear Conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    if not os.environ.get("TALUS_ACCESS_TOKEN"):
        st.markdown(
            '<div class="disclaimer">🔓 No access token is configured (TALUS_ACCESS_TOKEN unset). '
            "This deployment has no authentication. Do not expose it on a public network without "
            "one — an unguessable URL is not a security control.</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Header and status bar
# ---------------------------------------------------------------------------

st.markdown("# 🌕 TALUS")
st.markdown(
    '<div class="talus-sub"><strong>Lunar Terrain Analysis Agent</strong> · '
    "AI-assisted terrain intelligence for lunar landing and rover operations</div>",
    unsafe_allow_html=True,
)


def _pill(label: str, state: str, dot: str) -> str:
    return f'<span class="pill">{html.escape(label)}: <span class="{dot}">●</span> {html.escape(state)}</span>'


_gemini_live = agent is not None and getattr(agent, "is_live", False)
_health = st.session_state.gemini_health
if _gemini_live and _health and _health.get("request") == "FAIL":
    gemini_pill = _pill("Gemini", "Error", "dot-err")
elif _gemini_live:
    gemini_pill = _pill("Gemini", "Active", "dot-ok")
else:
    gemini_pill = _pill("Gemini", "Demo mode", "dot-warn")

_nasa_states = {
    "unknown": ("Not yet contacted", "dot-idle"),
    "connected": ("Connected", "dot-ok"),
    "error": ("Error", "dot-err"),
    "disabled": ("Downloads disabled", "dot-warn"),
}
if not _NASA_DOWNLOADS and st.session_state.nasa_status == "unknown":
    nasa_state, nasa_dot = _nasa_states["disabled"]
else:
    nasa_state, nasa_dot = _nasa_states.get(st.session_state.nasa_status, _nasa_states["unknown"])

dem_state = f"Ready · {len(available_dems)} cached" if available_dems else "Ready · none cached"
st.markdown(
    '<div class="status-bar">'
    + gemini_pill
    + _pill("NASA ODE", nasa_state, nasa_dot)
    + _pill("DEM Engine", dem_state, "dot-ok")
    + "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer">⚠️ <strong>RESEARCH &amp; DEMO SYSTEM ONLY</strong>: '
    "TALUS must NEVER be used for certified flight safety, operational landing approval, "
    "autonomous spacecraft control, or guaranteed rover safety.</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Structured result panels
#
# Each panel is built only from typed fields of a dispatch_tool_call result -- never from
# response prose. Free-text values that originate outside this process (site ids the user
# typed, NASA product descriptions) are shown with st.write/st.dataframe, never interpolated
# into raw HTML. ``nested=True`` renders without inner expanders (Streamlit forbids nesting),
# for use inside a chat message's evidence expander.
# ---------------------------------------------------------------------------


def _render_rover_panel(result: dict[str, Any], nested: bool = False) -> None:
    analysis = result.get("analysis") or {}
    st.markdown(status_badge_html(result.get("overall_status")), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk score (0-100, uncalibrated)", fmt(result.get("risk_score"), decimals=0))
    c2.metric("Max slope", fmt(analysis.get("max_slope_deg"), "°"))
    c3.metric("Coverage", fmt_pct(analysis.get("coverage_fraction")))
    thresholds = analysis.get("configured_thresholds") or {}
    if thresholds.get("statement"):
        st.caption(thresholds["statement"])

    violated = analysis.get("violated_segments") or []
    incomplete = analysis.get("incomplete_segments") or []
    if violated:
        st.warning(f"Segment(s) that failed the configured thresholds: {violated}")
    if incomplete:
        st.info(f"Segment(s) with incomplete terrain coverage: {incomplete}")

    segments = analysis.get("segments") or []
    if segments:
        rows = [
            {
                "segment": s.get("segment_index"),
                "status": s.get("status"),
                "max_slope_deg": s.get("max_slope_deg"),
                "mean_tri_m": s.get("mean_tri_m"),
                "coverage": s.get("coverage_fraction"),
            }
            for s in segments
        ]
        if nested:
            st.caption(f"Per-segment detail ({len(segments)} segments)")
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            with st.expander(f"Per-segment detail ({len(segments)} segments)"):
                st.dataframe(rows, width="stretch", hide_index=True)
    if result.get("data_source"):
        st.caption(f"Data source: {result['data_source']}")
    st.caption(analysis.get("disclaimer") or result.get("disclaimer") or "")


def _render_landing_panel(result: dict[str, Any], nested: bool = False) -> None:
    analysis = result.get("analysis") or {}
    ranked = analysis.get("ranked") or []
    unranked = analysis.get("unranked") or []
    st.write(f"{len(ranked)} site(s) ranked, {len(unranked)} unranked.")
    for entry in ranked:
        site = entry.get("analysis") or {}
        cols = st.columns([2, 1, 3])
        cols[0].markdown(f"**#{entry.get('rank')} — {site.get('site_id')}**")
        cols[1].markdown(status_badge_html(site.get("status")), unsafe_allow_html=True)
        cols[2].caption(
            f"max slope {fmt(site.get('max_slope_deg'), '°')} · "
            f"flat radius {fmt(site.get('flat_radius_m'), ' m')} · "
            f"coverage {fmt_pct(site.get('coverage_fraction'))}"
        )

    def _unranked_body() -> None:
        for u in unranked:
            st.markdown(status_badge_html(u.get("status")), unsafe_allow_html=True)
            st.write(f"**{u.get('site_id')}**: {'; '.join(u.get('reasons') or [])}")

    if unranked:
        if nested:
            st.caption(f"Unranked sites ({len(unranked)})")
            _unranked_body()
        else:
            with st.expander(f"Unranked sites ({len(unranked)})"):
                _unranked_body()
    st.caption(analysis.get("disclaimer") or result.get("disclaimer") or "")


def _render_safe_regions_panel(result: dict[str, Any]) -> None:
    analysis = result.get("analysis") or {}
    outcome = analysis.get("outcome")
    icon = {
        "regions_found": "✅",
        "no_safe_region_in_assessed_area": "⚠️",
        "no_terrain_data": "⛔",
    }.get(outcome, "❔")
    st.markdown(f"{icon} **{outcome or 'unknown'}**")
    regions = analysis.get("regions") or []
    if regions:
        st.dataframe(
            [
                {
                    "region": r.get("region_id"),
                    "cells": r.get("cells"),
                    "max_slope_deg": r.get("max_slope_deg"),
                    "radius_m": r.get("largest_inscribed_circle_radius_m"),
                    "mean_elevation_m": r.get("mean_elevation_m"),
                }
                for r in regions
            ],
            width="stretch",
            hide_index=True,
        )
    if analysis.get("threshold_statement"):
        st.caption(analysis["threshold_statement"])
    st.caption(analysis.get("disclaimer") or result.get("disclaimer") or "")


_STAT_FIELDS: dict[str, list[tuple[str, str, str]]] = {
    "elevation": [("mean_m", "Mean", " m"), ("min_m", "Min", " m"), ("max_m", "Max", " m")],
    "slope": [("mean_slope_deg", "Mean", "°"), ("median_slope_deg", "Median", "°"), ("max_slope_deg", "Max", "°")],
    "roughness": [("mean_tri_m", "Mean TRI", " m"), ("max_tri_m", "Max TRI", " m"), ("std_tri_m", "Std TRI", " m")],
}


def _render_terrain_stats_panel(analysis: dict[str, Any], resolution_m: float | None = None) -> None:
    if not analysis.get("terrain_available"):
        st.warning("No terrain data is available for the requested area.")
        return
    for key, fields in _STAT_FIELDS.items():
        block = analysis.get(key)
        if not block:
            continue
        st.markdown(f"**{key.capitalize()}**")
        cols = st.columns(len(fields))
        for col, (field, label, unit) in zip(cols, fields):
            col.metric(label, fmt(block.get(field), unit))
    st.caption(
        f"Coverage: {fmt_pct(analysis.get('coverage_fraction'))} · "
        f"Resolution: {fmt(resolution_m if resolution_m is not None else analysis.get('resolution_m'), ' m')}"
    )


def _render_dataset_panel(dataset: dict[str, Any] | None) -> None:
    if not dataset:
        st.info("No dataset information available.")
        return
    if dataset.get("provenance") == "none":
        st.warning("No provenance sidecar found for this DEM — its source is unknown; do not "
                    "present it as a specific NASA product.")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**File:** {dataset.get('file_name')}")
        st.write(f"**Mission / instrument:** {dataset.get('mission') or '—'} / {dataset.get('instrument') or '—'}")
        st.write(f"**Product:** {dataset.get('product_type') or '—'} ({dataset.get('product_id') or '—'})")
    with c2:
        st.write(f"**CRS:** {dataset.get('crs_kind')}")
        st.write(f"**Size:** {dataset.get('width')} × {dataset.get('height')} cells")
        st.write(f"**Resolution:** {fmt(dataset.get('resolution_m'), ' m')}")
    if dataset.get("elevation_reference"):
        st.caption(dataset["elevation_reference"])
    for w in dataset.get("warnings") or []:
        st.caption(f"⚠️ {w}")


def _pixel_size(prov: dict[str, Any]) -> Any:
    size = prov.get("pixel_size_m")
    return size[0] if isinstance(size, list) and size else None


def _render_fetch_panel(result: dict[str, Any]) -> None:
    if result.get("status") != "ok":
        if result.get("status") == "no_product":
            st.warning("No NASA DEM product covers that location at the requested resolution.")
        elif result.get("status") == "disabled":
            st.warning("NASA DEM downloads are disabled in this deployment.")
        else:
            st.error(friendly_error_message(result))
        return
    origin = "from cache" if result.get("from_cache") else "downloaded from NASA"
    prov = result.get("provenance") or {}
    st.success(f"DEM ready ({origin}): {prov.get('product_id') or '—'} · `{result.get('dem_path')}`")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Mission / instrument:** {prov.get('mission')} / {prov.get('instrument')}")
        st.write(f"**Product:** {prov.get('product_type')} ({prov.get('product_id')})")
    with c2:
        st.write(f"**Resolution:** {fmt(_pixel_size(prov), ' m')}")
        st.write(f"**Size:** {prov.get('width')} × {prov.get('height')} cells")
    if prov.get("elevation_reference"):
        st.caption(prov["elevation_reference"])


def _render_search_panel(result: dict[str, Any]) -> None:
    products = result.get("products") or []
    if not products:
        st.info(result.get("message") or "No DEM products found for that region.")
        return
    st.success(f"Found {len(products)} DEM product(s)")
    st.dataframe(
        [
            {
                "product_id": p.get("product_id"),
                "mission": p.get("mission"),
                "instrument": p.get("instrument"),
                "dataset": p.get("dataset"),
            }
            for p in products
        ],
        width="stretch",
        hide_index=True,
    )


def _render_tool_call_panel(tc: dict[str, Any], nested: bool = False) -> None:
    """Dispatch a chat tool-call entry to its structured panel, by tool name."""
    tool = tc.get("tool")
    result = tc.get("result")
    if not isinstance(result, dict):
        st.markdown(f"- {tc.get('summary', tool)}")
        return
    if result.get("status") in ("error", "rate_limited"):
        st.error(friendly_error_message(result))
        return
    if tool == "evaluate_traverse_route":
        _render_rover_panel(result, nested=nested)
    elif tool == "evaluate_landing_sites":
        _render_landing_panel(result, nested=nested)
    elif tool == "find_safe_regions":
        _render_safe_regions_panel(result)
    elif tool in ("get_elevation_stats", "get_slope_stats", "get_roughness_stats"):
        _render_terrain_stats_panel(result.get("analysis") or {}, result.get("resolution_m"))
    elif tool == "dataset_information":
        _render_dataset_panel(result.get("dataset"))
    elif tool == "fetch_nasa_dem":
        _render_fetch_panel(result)
    elif tool == "search_dem_products":
        _render_search_panel(result)
    elif tool == "resolve_lunar_feature" and result.get("status") != "ok":
        st.warning(result.get("error") or "Feature not found.")
        known = result.get("known_features") or []
        if known:
            st.caption("Known features: " + ", ".join(known))
    else:
        st.markdown(f"- {tc.get('summary', tool)}")


# ---------------------------------------------------------------------------
# Chat message rendering: answer text, activity trace, result cards, evidence
# ---------------------------------------------------------------------------


def _latest(tool_calls: list[dict[str, Any]], *tools: str) -> dict[str, Any] | None:
    for tc in reversed(tool_calls):
        if tc.get("tool") in tools and isinstance(tc.get("result"), dict) and tc["result"].get("status") == "ok":
            return tc["result"]
    return None


def _render_result_cards(tool_calls: list[dict[str, Any]]) -> None:
    """Compact metric cards built only from structured tool results."""
    stats: dict[str, Any] = {}
    resolution = None
    for tc in tool_calls:
        result = tc.get("result")
        if tc.get("tool") in ("get_elevation_stats", "get_slope_stats", "get_roughness_stats") and isinstance(result, dict):
            analysis = result.get("analysis") or {}
            if analysis.get("terrain_available"):
                for key in ("elevation", "slope", "roughness"):
                    if analysis.get(key):
                        stats[key] = analysis[key]
                resolution = result.get("resolution_m") or resolution
    cards: list[tuple[str, str]] = []
    if stats.get("elevation"):
        e = stats["elevation"]
        cards += [("Average elevation", fmt(e.get("mean_m"), " m")), ("Min / max elevation", f"{fmt(e.get('min_m'), ' m', 0)} / {fmt(e.get('max_m'), ' m', 0)}")]
    if stats.get("slope"):
        s = stats["slope"]
        cards += [("Mean slope", fmt(s.get("mean_slope_deg"), "°")), ("Max slope", fmt(s.get("max_slope_deg"), "°"))]
    if stats.get("roughness"):
        cards.append(("Mean roughness (TRI)", fmt(stats["roughness"].get("mean_tri_m"), " m")))
    rover = _latest(tool_calls, "evaluate_traverse_route")
    if rover:
        cards.append(("Rover route status", str(rover.get("overall_status") or "—")))
    regions = _latest(tool_calls, "find_safe_regions")
    if regions:
        n = len(((regions.get("analysis") or {}).get("regions")) or [])
        cards.append(("Safe regions found", str(n)))
    landing = _latest(tool_calls, "evaluate_landing_sites")
    if landing:
        cards.append(("Landing sites ranked", f"{landing.get('sites_ranked', 0)} of {landing.get('sites_evaluated', 0)}"))

    fetch = _latest(tool_calls, "fetch_nasa_dem")
    if fetch:
        prov = fetch.get("provenance") or {}
        cards.append(("Dataset", f"{prov.get('mission') or '—'} / {prov.get('instrument') or '—'}"))
        cards.append(("Source", "NASA ODE / PDS"))
    if resolution is not None:
        cards.append(("Resolution", fmt(resolution, " m", 1)))

    for start in range(0, len(cards), 3):
        row = cards[start : start + 3]
        cols = st.columns(3)
        for col, (label, value) in zip(cols, row):
            col.metric(label, value)


def _render_evidence(tool_calls: list[dict[str, Any]]) -> None:
    feature = _latest(tool_calls, "resolve_lunar_feature")
    fetch = _latest(tool_calls, "fetch_nasa_dem")
    stats_call = next(
        (tc for tc in reversed(tool_calls) if tc.get("tool") in ("get_elevation_stats", "get_slope_stats", "get_roughness_stats")),
        None,
    )
    if feature:
        f = feature.get("feature") or {}
        area = feature.get("analysis_area") or {}
        st.markdown("**Coordinate region**")
        st.write(
            f"{f.get('name')} — centre {fmt(f.get('center_lat'), '°', 3)}, {fmt(f.get('center_lon'), '°', 3)}; "
            f"analysis radius {fmt(area.get('radius_km'), ' km', 1)}; "
            f"box lat {fmt(area.get('min_lat'), '°', 3)} to {fmt(area.get('max_lat'), '°', 3)}, "
            f"lon {fmt(area.get('min_lon'), '°', 1)} to {fmt(area.get('max_lon'), '°', 1)}"
        )
        st.caption(f"Coordinates from: {feature.get('source')}")
    elif stats_call and isinstance(stats_call.get("args"), dict):
        a = stats_call["args"]
        st.markdown("**Coordinate region**")
        st.write(
            f"lat {fmt(a.get('min_lat'), '°', 3)} to {fmt(a.get('max_lat'), '°', 3)}, "
            f"lon {fmt(a.get('min_lon'), '°', 2)} to {fmt(a.get('max_lon'), '°', 2)}"
        )
    if fetch:
        prov = fetch.get("provenance") or {}
        st.markdown("**Dataset provenance**")
        st.write(
            f"Mission: {prov.get('mission')} · Instrument: {prov.get('instrument')} · "
            f"Product: {prov.get('product_type')} {prov.get('product_id')}"
        )
        st.write(
            f"Source: NASA ODE ({prov.get('provider') or 'nasa_ode'}) · "
            f"Cell size: {fmt(_pixel_size(prov), ' m', 1)} · "
            f"Raster: {prov.get('width')} × {prov.get('height')} · "
            f"{'Served from local cache' if fetch.get('from_cache') else 'Downloaded from NASA'}"
        )
        if prov.get("acquired_at"):
            st.caption(f"Acquired: {prov['acquired_at']}")
        if prov.get("elevation_reference"):
            st.caption(prov["elevation_reference"])
    if stats_call:
        st.markdown("**Processing method**")
        st.caption(
            "Deterministic Python (terrain_agent): windowed raster read (≤ 2048 × 2048 cells), "
            "elevation statistics over valid cells, Horn's-method slope, TRI roughness. "
            "The language model performed no terrain calculation."
        )
    for tc in tool_calls:
        if tc.get("tool") in ("resolve_lunar_feature",) and (tc.get("result") or {}).get("status") == "ok":
            continue
        st.markdown(f"**{TOOL_LABELS.get(tc.get('tool'), tc.get('tool'))}**")
        _render_tool_call_panel(tc, nested=True)


def _render_assistant_message(msg: dict[str, Any]) -> None:
    status = msg.get("status")
    tool_calls = msg.get("tool_calls") or []
    if msg.get("is_demo"):
        st.info("📡 Demo Mode — set GEMINI_API_KEY for live AI analysis, or use the Direct "
                "Analysis tools below, which work without one.")
    if status in ("invalid_request", "rate_limited", "model_error"):
        st.warning(msg.get("content", ""))
    else:
        st.markdown(sanitize_model_markdown(msg.get("content", "")))
    if tool_calls:
        _render_result_cards(tool_calls)
        outcome = analysis_outcome(status, tool_calls)
        st.caption(f"Analysis status: {outcome}")
        with st.expander("Analysis trace"):
            for step in build_trace(tool_calls):
                st.caption(f"{'✓' if step['ok'] else '⚠'} {step['label']} — {step['detail']}")
        with st.expander("Evidence & details"):
            _render_evidence(tool_calls)


def _history_for_agent() -> list[dict[str, Any]]:
    return [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in st.session_state.messages
        if m.get("content") and m.get("status") not in ("invalid_request", "rate_limited", "model_error")
    ]


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

SUGGESTED = [
    "What is the average elevation around Shackleton Crater?",
    "Check rover safety for a terrain region near Shackleton Crater.",
    "What is the terrain roughness around Haworth Crater?",
    "What DEM data is available for the lunar south pole?",
]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🚀" if msg["role"] == "user" else "🌕"):
        if msg["role"] == "user":
            st.text(msg["content"])
        else:
            _render_assistant_message(msg)

if not st.session_state.messages:
    st.caption("Try one of these, or ask your own terrain question:")
    pill_cols = st.columns(2)
    for idx, q in enumerate(SUGGESTED):
        if pill_cols[idx % 2].button(q, key=f"pill_{idx}", width="stretch"):
            st.session_state["_pending_query"] = q

# Inside a container the chat input renders inline, right under the conversation, instead of
# being pinned over the Direct Analysis section below it.
with st.container():
    typed = st.chat_input("Ask TALUS about lunar terrain, e.g. elevation around Shackleton Crater")
query = (typed or st.session_state.pop("_pending_query", "") or "").strip()

if query:
    history = _history_for_agent()
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍🚀"):
        st.text(query)
    with st.chat_message("assistant", avatar="🌕"):
        with st.status("Analyzing your request…", expanded=True) as activity:
            def _on_event(kind: str, info: dict[str, Any]) -> None:
                if kind == "model" and info.get("phase") == "interpreting":
                    activity.write("⏳ Interpreting terrain request")
                elif kind == "tool_start":
                    activity.write(f"⏳ {TOOL_LABELS.get(info.get('tool'), 'Running terrain tool')}…")
                elif kind == "tool_end":
                    mark = "✓" if info.get("status") in ("ok",) else "⚠"
                    activity.write(f"{mark} {info.get('summary', '')}")

            if agent is not None:
                result = agent.chat(query, history=history, max_slope_deg=max_slope, on_event=_on_event)
            else:
                result = {
                    "status": "model_error",
                    "text": "The TALUS agent could not be initialised. Please check the server logs.",
                    "tool_calls": [], "is_demo": True,
                }
            failed = result.get("status") in ("invalid_request", "rate_limited", "model_error")
            activity.update(
                label="Analysis could not be completed" if failed else "Analysis complete",
                state="error" if failed else "complete",
                expanded=False,
            )
    tool_calls = result.get("tool_calls") or []
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("text", ""),
            "status": result.get("status"),
            "is_demo": result.get("is_demo", False),
            "tool_calls": tool_calls,
        }
    )
    st.session_state.messages = st.session_state.messages[-MAX_STORED_MESSAGES:]
    new_nasa = nasa_status_from_tool_calls(tool_calls)
    if new_nasa:
        st.session_state.nasa_status = new_nasa
    if _latest(tool_calls, "fetch_nasa_dem"):
        _cached_managed_dems.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Direct Structured Analysis — call the deterministic tools directly, no model required.
# Every result here is the exact structured dict dispatch_tool_call returns; nothing is
# recomputed or re-derived in this file.
# ---------------------------------------------------------------------------

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("### 🧪 Direct Structured Analysis")
st.caption(
    "Run a deterministic terrain tool directly with structured coordinates and thresholds — "
    "no natural-language interpretation, no model required."
)

tab_dataset, tab_stats, tab_regions, tab_rover, tab_landing, tab_search = st.tabs(
    ["📄 Dataset", "📈 Statistics", "🟢 Safe Regions", "🚗 Rover Route", "🛬 Landing", "🔎 NASA DEM"]
)

_DEM_HELP = "Only DEMs already present in the managed cache/sample directories are offered."


def _dem_selector(key: str) -> str | None:
    if not available_dems:
        st.info("No DEMs cached yet — fetch one from the NASA DEM tab first.")
        return None
    index = available_dems.index(default_dem) if default_dem in available_dems else 0
    return st.selectbox("DEM file", options=available_dems, index=index, help=_DEM_HELP, key=key)


with tab_dataset:
    dem = _dem_selector("dataset_dem")
    with st.form("dataset_form"):
        c1, c2 = st.columns(2)
        lat = c1.number_input("Reference latitude (° , optional)", value=0.0, min_value=-90.0, max_value=90.0, key="ds_lat")
        lon = c2.number_input("Reference longitude (° , optional)", value=0.0, min_value=-180.0, max_value=360.0, key="ds_lon")
        use_ref = st.checkbox("Use the reference point above (otherwise the DEM centre is used)")
        go = st.form_submit_button("Describe dataset", disabled=dem is None)
    if go and dem:
        with st.spinner("Reading dataset metadata…"):
            args: dict[str, Any] = {"dem_path": dem}
            if use_ref:
                args["lat"], args["lon"] = float(lat), float(lon)
            result = _cached_dispatch("dataset_information", args)
        if result.get("status") in ("error", "rate_limited"):
            st.error(friendly_error_message(result))
        else:
            _render_dataset_panel(result.get("dataset"))


with tab_stats:
    dem = _dem_selector("stats_dem")
    with st.form("stats_form"):
        c1, c2 = st.columns(2)
        min_lat = c1.number_input("Min latitude (°)", value=-89.95, min_value=-90.0, max_value=90.0)
        max_lat = c1.number_input("Max latitude (°)", value=-89.85, min_value=-90.0, max_value=90.0)
        min_lon = c2.number_input("Min longitude (°)", value=-10.0, min_value=-180.0, max_value=360.0)
        max_lon = c2.number_input("Max longitude (°)", value=10.0, min_value=-180.0, max_value=360.0)
        st.caption("Requests are limited to a 2048×2048 cell read window; a box that is too "
                   "large for the DEM's resolution is rejected rather than silently truncated.")
        go = st.form_submit_button("Compute terrain statistics", disabled=dem is None)
    if go and dem:
        box = {"dem_path": dem, "min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon}
        with st.spinner("Computing elevation, slope and roughness…"):
            elev = _cached_dispatch("get_elevation_stats", box)
            slope = _cached_dispatch("get_slope_stats", box)
            rough = _cached_dispatch("get_roughness_stats", box)
        errored = next((r for r in (elev, slope, rough) if r.get("status") in ("error", "rate_limited")), None)
        if errored is not None:
            st.error(friendly_error_message(errored))
        else:
            merged = dict(elev.get("analysis") or {})
            merged["slope"] = (slope.get("analysis") or {}).get("slope")
            merged["roughness"] = (rough.get("analysis") or {}).get("roughness")
            _render_terrain_stats_panel(merged, elev.get("resolution_m"))
            st.plotly_chart(
                build_terrain_map_figure(requested_bbox=(min_lat, max_lat, min_lon, max_lon)),
                width="stretch",
            )


with tab_regions:
    dem = _dem_selector("regions_dem")
    with st.form("regions_form"):
        c1, c2, c3 = st.columns(3)
        center_lat = c1.number_input("Center latitude (°)", value=-89.9, min_value=-90.0, max_value=90.0)
        center_lon = c2.number_input("Center longitude (°)", value=0.0, min_value=-180.0, max_value=360.0)
        radius_m = c3.number_input("Search radius (m)", value=3000.0, min_value=100.0, max_value=50_000.0, step=100.0)
        c4, c5 = st.columns(2)
        region_slope = c4.number_input("Max slope (°)", value=float(max_slope), min_value=0.1, max_value=90.0)
        min_area = c5.number_input("Min region area (m²)", value=50_000.0, min_value=100.0, step=1000.0)
        go = st.form_submit_button("Find safe regions", disabled=dem is None)
    if go and dem:
        with st.spinner("Searching for safe regions…"):
            result = _cached_dispatch(
                "find_safe_regions",
                {
                    "center_lat": center_lat, "center_lon": center_lon, "radius_m": radius_m,
                    "dem_path": dem, "max_slope_deg": region_slope, "min_area_m2": min_area,
                },
            )
        if result.get("status") in ("error", "rate_limited"):
            st.error(friendly_error_message(result))
        else:
            _render_safe_regions_panel(result)
            deg_offset = (radius_m / 1_737_400.0) * (180.0 / 3.14159265)
            regions = ((result.get("analysis") or {}).get("regions")) or []
            st.plotly_chart(
                build_terrain_map_figure(
                    requested_bbox=(center_lat - deg_offset, center_lat + deg_offset, center_lon - deg_offset, center_lon + deg_offset),
                    safe_regions=regions,
                ),
                width="stretch",
            )


with tab_rover:
    dem = _dem_selector("rover_dem")
    st.caption("Enter route waypoints as latitude/longitude pairs, in order.")
    waypoints_df = st.data_editor(
        [{"lat": -89.9, "lon": 0.0}, {"lat": -89.85, "lon": 30.0}, {"lat": -89.8, "lon": 60.0}],
        num_rows="dynamic", key="rover_waypoints", width="stretch",
    )
    with st.form("rover_form"):
        c1, c2 = st.columns(2)
        rover_slope = c1.number_input("Max slope (°)", value=float(max_slope), min_value=0.1, max_value=90.0)
        rover_roughness = c2.number_input("Max roughness TRI (m, 0 = not evaluated)", value=0.0, min_value=0.0)
        go = st.form_submit_button("Evaluate route safety", disabled=dem is None)
    if go and dem:
        waypoints = [[float(r["lat"]), float(r["lon"])] for r in waypoints_df if r.get("lat") is not None and r.get("lon") is not None]
        args: dict[str, Any] = {"waypoints": waypoints, "dem_path": dem, "max_slope_deg": rover_slope}
        if rover_roughness > 0:
            args["max_roughness_tri"] = rover_roughness
        with st.spinner("Evaluating route…"):
            result = _dispatch("evaluate_traverse_route", args)
        if result.get("status") in ("error", "rate_limited"):
            st.error(friendly_error_message(result))
        else:
            _render_rover_panel(result)
            segments = ((result.get("analysis") or {}).get("segments")) or []
            seg_status = [s.get("status") for s in segments]
            st.plotly_chart(
                build_terrain_map_figure(route=[(w[0], w[1]) for w in waypoints], route_segment_status=seg_status),
                width="stretch",
            )


with tab_landing:
    dem = _dem_selector("landing_dem")
    st.caption("Enter candidate landing sites: a short id, latitude and longitude.")
    sites_df = st.data_editor(
        [{"id": "site-a", "lat": -89.9, "lon": 0.0}, {"id": "site-b", "lat": -89.85, "lon": 45.0}],
        num_rows="dynamic", key="landing_sites", width="stretch",
    )
    with st.form("landing_form"):
        c1, c2 = st.columns(2)
        landing_slope = c1.number_input("Max slope (°)", value=float(max_slope), min_value=0.1, max_value=90.0, key="landing_slope")
        landing_flat_radius = c2.number_input("Min flat radius (m)", value=100.0, min_value=0.0)
        go = st.form_submit_button("Evaluate landing sites", disabled=dem is None)
    if go and dem:
        sites = [{"id": str(r["id"]), "lat": float(r["lat"]), "lon": float(r["lon"])} for r in sites_df if r.get("id")]
        with st.spinner("Evaluating candidate sites…"):
            result = _dispatch(
                "evaluate_landing_sites",
                {"sites": sites, "dem_path": dem, "max_slope_deg": landing_slope, "min_flat_radius_m": landing_flat_radius},
            )
        if result.get("status") in ("error", "rate_limited"):
            st.error(friendly_error_message(result))
        else:
            _render_landing_panel(result)
            all_sites = ((result.get("analysis") or {}).get("sites")) or []
            st.plotly_chart(build_terrain_map_figure(landing_sites=all_sites), width="stretch")


with tab_search:
    st.markdown("#### Search NASA ODE for DEM products")
    with st.form("search_form"):
        c1, c2 = st.columns(2)
        s_min_lat = c1.number_input("Min latitude (°)", value=-90.0, min_value=-90.0, max_value=90.0, key="s_min_lat")
        s_max_lat = c1.number_input("Max latitude (°)", value=-85.0, min_value=-90.0, max_value=90.0, key="s_max_lat")
        s_min_lon = c2.number_input("Min longitude (°)", value=0.0, min_value=-180.0, max_value=360.0, key="s_min_lon")
        s_max_lon = c2.number_input("Max longitude (°)", value=10.0, min_value=-180.0, max_value=360.0, key="s_max_lon")
        search_go = st.form_submit_button("🔍 Search DEM Products")
    if search_go:
        with st.spinner("Searching NASA ODE…"):
            result = _cached_dispatch(
                "search_dem_products",
                {"min_lat": s_min_lat, "max_lat": s_max_lat, "min_lon": s_min_lon, "max_lon": s_max_lon},
            )
        if result.get("status") in ("error", "rate_limited"):
            st.error(friendly_error_message(result))
        else:
            _render_search_panel(result)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Fetch a NASA DEM covering a point")
    with st.form("fetch_form"):
        c1, c2, c3 = st.columns(3)
        f_lat = c1.number_input("Latitude (°)", value=-89.9, min_value=-90.0, max_value=90.0, key="f_lat")
        f_lon = c2.number_input("Longitude (°)", value=0.0, min_value=-180.0, max_value=360.0, key="f_lon")
        f_radius = c3.number_input("Radius (km, max 50)", value=5.0, min_value=0.1, max_value=50.0, key="f_radius")
        fetch_go = st.form_submit_button("⬇️ Fetch DEM")
    if fetch_go:
        with st.spinner("Contacting NASA ODE / PDS (only official NASA hosts are ever contacted)…"):
            result = _dispatch("fetch_nasa_dem", {"lat": f_lat, "lon": f_lon, "radius_km": f_radius})
        _render_fetch_panel(result)
        new_nasa = nasa_status_from_tool_calls([{"tool": "fetch_nasa_dem", "result": result}])
        if new_nasa:
            st.session_state.nasa_status = new_nasa
        if result.get("status") == "ok":
            _cached_managed_dems.clear()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(
    f"TALUS v{_APP_VERSION} · Research & Demo System · All terrain values originate from "
    "deterministic NASA DEM processing · NOT certified for flight operations"
)
