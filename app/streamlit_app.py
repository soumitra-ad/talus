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

Panels:
- Sidebar: mission parameters & status
- Chat: natural language terrain queries (Phase 6 agent)
- Results: structured panels for whatever tools the last chat turn ran
- Direct Structured Analysis: coordinate/region forms that call the deterministic tools
  directly, without needing a live model
- Schematic map: requested region / DEM coverage / safe regions / rover route / landing sites

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
    build_terrain_map_figure,
    fmt,
    fmt_pct,
    friendly_error_message,
    list_managed_dems,
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
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**TALUS** — Terrain Analysis for Landing and Uncrewed Systems\n\n"
            "Research/Demo System only. NOT certified for flight operations."
        )
    },
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0d1a 0%, #111827 50%, #0a0f1e 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0d1626 100%);
    border-right: 1px solid #1e3a5f;
}
.user-bubble {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px; margin: 8px 0; max-width: 85%; margin-left: auto;
    color: #fff; font-size: 0.95rem; box-shadow: 0 2px 12px rgba(30, 64, 175, 0.4);
}
.assistant-bubble {
    background: linear-gradient(135deg, #1a2744, #1e3a5f);
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px; margin: 8px 0; max-width: 95%;
    color: #cbd5e1; font-size: 0.95rem; border-left: 3px solid #3b82f6;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
}
.badge-pass { background: #166534; color: #86efac; padding: 3px 10px; border-radius: 20px; font-weight: 700; font-size: 0.8rem; }
.badge-review { background: #78350f; color: #fde68a; padding: 3px 10px; border-radius: 20px; font-weight: 700; font-size: 0.8rem; }
.badge-fail { background: #7f1d1d; color: #fca5a5; padding: 3px 10px; border-radius: 20px; font-weight: 700; font-size: 0.8rem; }
.section-divider { border-top: 1px solid #1e3a5f; margin: 16px 0; }
.disclaimer {
    background: #1c1917; border: 1px solid #78350f; border-radius: 8px;
    padding: 10px 14px; color: #d97706; font-size: 0.8rem;
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


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def _init_session() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "tool_calls": [],
        "last_result": None,
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
# Agent singleton
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
except Exception:
    log.exception("Failed to load configuration; using safe defaults")
    _APP_VERSION = "0.1.0"
    _DEFAULT_SLOPE = 15.0
    _GEMINI_KEY = None
    _MODEL_NAME = "gemini-3.6-flash"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🌕 TALUS")
    st.caption(f"v{_APP_VERSION} · Research/Demo System")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🗂️ Default DEM")
    available_dems = _cached_managed_dems()
    if available_dems:
        default_dem = st.selectbox(
            "DEM for the Direct Analysis tools below",
            options=available_dems,
            help="Only DEMs already present in the managed cache/sample directories can be "
            "selected — never an arbitrary filesystem path.",
        )
    else:
        default_dem = None
        st.info("No DEMs cached yet. Use the NASA DEM Search & Fetch tool below, or ask the "
                "chat assistant to fetch one.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🤖 AI Status")
    agent = _build_agent(_GEMINI_KEY, _MODEL_NAME)
    if agent is not None and getattr(agent, "is_live", False):
        st.success(f"Gemini Active ({_MODEL_NAME})")
    else:
        st.info("Demo Mode — set GEMINI_API_KEY for live AI analysis. "
                "The Direct Analysis tools below work without one.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Conversation", width="stretch"):
        st.session_state.messages = []
        st.session_state.tool_calls = []
        st.session_state.last_result = None
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
<h1 style='font-size:1.8rem; font-weight:700; margin-bottom:0;'>
🌕 Terrain Analysis for Landing and Uncrewed Systems
</h1>
<p style='color:#94a3b8; font-size:0.85rem; margin-top:4px;'>
Research & Demo System · NASA Lunar Terrain Analysis
</p>
""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer">⚠️ <strong>RESEARCH &amp; DEMO SYSTEM ONLY</strong>: '
    "TALUS must NEVER be used for certified flight safety, operational landing approval, "
    "autonomous spacecraft control, or guaranteed rover safety.</div>",
    unsafe_allow_html=True,
)
if not os.environ.get("TALUS_ACCESS_TOKEN"):
    st.markdown(
        '<div class="disclaimer">🔓 No access token is configured (TALUS_ACCESS_TOKEN unset). '
        "This deployment has no authentication. Do not expose it on a public network without "
        "one — an unguessable URL is not a security control.</div>",
        unsafe_allow_html=True,
    )
st.markdown("")


# ---------------------------------------------------------------------------
# Structured result panels
#
# Each panel is built only from typed fields of a dispatch_tool_call result -- never from
# response prose. Free-text values that originate outside this process (site ids the user
# typed, NASA product descriptions) are shown with st.write/st.dataframe, never interpolated
# into raw HTML.
# ---------------------------------------------------------------------------


def _render_rover_panel(result: dict[str, Any]) -> None:
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
        with st.expander(f"Per-segment detail ({len(segments)} segments)"):
            st.dataframe(
                [
                    {
                        "segment": s.get("segment_index"),
                        "status": s.get("status"),
                        "max_slope_deg": s.get("max_slope_deg"),
                        "mean_tri_m": s.get("mean_tri_m"),
                        "coverage": s.get("coverage_fraction"),
                    }
                    for s in segments
                ],
                width="stretch",
                hide_index=True,
            )
    if result.get("data_source"):
        st.caption(f"Data source: {result['data_source']}")
    st.caption(analysis.get("disclaimer") or result.get("disclaimer") or "")


def _render_landing_panel(result: dict[str, Any]) -> None:
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
    if unranked:
        with st.expander(f"Unranked sites ({len(unranked)})"):
            for u in unranked:
                st.markdown(status_badge_html(u.get("status")), unsafe_allow_html=True)
                st.write(f"**{u.get('site_id')}**: {'; '.join(u.get('reasons') or [])}")
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
    st.success(f"DEM ready ({origin}): `{result.get('dem_path')}`")
    prov = result.get("provenance") or {}
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Mission / instrument:** {prov.get('mission')} / {prov.get('instrument')}")
        st.write(f"**Product:** {prov.get('product_type')} ({prov.get('product_id')})")
    with c2:
        st.write(f"**Resolution:** {fmt(prov.get('pixel_size_m', [None])[0] if isinstance(prov.get('pixel_size_m'), list) else None, ' m')}")
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


def _render_tool_call_panel(tc: dict[str, Any]) -> None:
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
        _render_rover_panel(result)
    elif tool == "evaluate_landing_sites":
        _render_landing_panel(result)
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
    else:
        st.markdown(f"- {tc.get('summary', tool)}")


# ---------------------------------------------------------------------------
# Chat + Results layout
# ---------------------------------------------------------------------------

col_chat, col_results = st.columns([3, 2], gap="large")

SUGGESTED = [
    "What is the average elevation around Shackleton Crater?",
    "Find regions near Shackleton with slope below 12°",
    "What DEM data is available for the lunar south pole?",
    "Is a traverse from -89.9°, 0° to -89.5°, 0° safe?",
    "Compare three candidate landing sites near Shackleton",
    "What is the terrain roughness around Haworth Crater?",
]

with col_chat:
    st.markdown("### 💬 Ask a Terrain Question")

    pill_cols = st.columns(3)
    for idx, q in enumerate(SUGGESTED):
        if pill_cols[idx % 3].button(
            q[:40] + ("…" if len(q) > 40 else ""), key=f"pill_{idx}", width="stretch", help=q,
        ):
            st.session_state["_pending_query"] = q

    chat_container = st.container(height=400, border=False)
    with chat_container:
        for msg in st.session_state.messages:
            css = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
            icon = "👤" if msg["role"] == "user" else "🌕"
            # Escaped: this is the user's own message, or the assistant's text, which can in
            # turn echo external data (e.g. a NASA product description) the model saw. Neither
            # is trusted HTML -- rendering it unescaped would let injected markup/script run
            # in the browser (Phase 8 hardening).
            safe_content = html.escape(str(msg["content"]))
            st.markdown(f'<div class="{css}">{icon} {safe_content}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        pending = st.session_state.pop("_pending_query", "")
        user_input = st.text_input(
            "Your question", value=pending,
            placeholder="e.g. What is the slope near Shackleton Crater?",
            label_visibility="collapsed",
        )
        submit = st.form_submit_button("Send →", width="stretch")

    if submit and user_input.strip():
        query = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": query})
        history = [
            {"role": m["role"], "parts": [{"text": m["content"]}]}
            for m in st.session_state.messages[:-1]
        ]
        with st.spinner("Analysing terrain…"):
            if agent is not None:
                result = agent.chat(query, history=history, max_slope_deg=max_slope)
            else:
                result = {
                    "text": "TALUS agent could not be initialised. Please check the server logs.",
                    "tool_calls": [], "is_demo": True,
                }
        st.session_state.messages.append({"role": "assistant", "content": result["text"]})
        st.session_state.tool_calls = result.get("tool_calls", [])
        st.session_state.last_result = result
        st.rerun()


with col_results:
    st.markdown("### 📊 Analysis Results")
    last = st.session_state.last_result

    if last is None:
        st.markdown(
            "<div style='text-align:center; color:#475569; padding:60px 20px;'>"
            "<div style='font-size:3rem;'>🌑</div>"
            "<div style='margin-top:12px; font-size:0.95rem;'>Ask a terrain question to see results here.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        if last.get("is_demo"):
            st.info("📡 Demo Mode — set GEMINI_API_KEY for live AI analysis, or use the Direct "
                     "Analysis tools below, which work without one.")
        if last.get("status") in ("invalid_request", "rate_limited", "model_error"):
            st.warning(last.get("text", ""))
        else:
            tool_calls = st.session_state.tool_calls
            if tool_calls:
                st.markdown("#### Tools executed")
                for tc in tool_calls:
                    icon = "✅" if tc.get("result_status") in ("ok",) else "⚠️"
                    st.caption(f"{icon} `{tc.get('tool')}` — {tc.get('summary', '')}")
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                for tc in tool_calls:
                    with st.expander(f"🔧 {tc.get('tool')}", expanded=True):
                        _render_tool_call_panel(tc)
            st.markdown("#### Response")
            st.markdown(last.get("text", ""))


# ---------------------------------------------------------------------------
# Direct Structured Analysis — call the deterministic tools directly, no model required.
# Every result here is the exact structured dict dispatch_tool_call returns; nothing is
# recomputed or re-derived in this file.
# ---------------------------------------------------------------------------

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🧪 Direct Structured Analysis")
st.caption(
    "Run a deterministic terrain tool directly with structured coordinates and thresholds — "
    "no natural-language interpretation, no model required."
)

tab_dataset, tab_stats, tab_regions, tab_rover, tab_landing, tab_search = st.tabs(
    ["📄 Dataset Info", "📈 Terrain Statistics", "🟢 Safe Regions", "🚗 Rover Route", "🛬 Landing Sites", "🔎 NASA DEM Search"]
)

_DEM_HELP = "Only DEMs already present in the managed cache/sample directories are offered."


def _dem_selector(key: str) -> str | None:
    if not available_dems:
        st.info("No DEMs cached yet — fetch one from the NASA DEM Search tab first.")
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
        if result.get("status") == "ok":
            _cached_managed_dems.clear()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center; color:#475569; font-size:0.75rem; padding:8px 0;'>"
    f"TALUS v{_APP_VERSION} · Research &amp; Demo System · "
    "All terrain values originate from deterministic NASA DEM data only · "
    "NOT certified for flight operations</div>",
    unsafe_allow_html=True,
)
