"""Pure, Streamlit-independent presentation helpers for the TALUS UI.

Kept separate from ``streamlit_app.py`` so status-badge mapping, error-message formatting,
DEM listing, and the schematic map figure can be unit-tested directly, without a running
Streamlit script. Nothing here performs a terrain calculation, parses a safety status out of
prose, or makes a network call -- it only formats and lays out structured results already
produced by the deterministic ``terrain_agent`` tools (see ``terrain_agent.agent.agent``).
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Status badges
#
# Every status shown to the user comes straight from a structured backend field
# (SafetyStatus.value: "PASS" / "REVIEW_REQUIRED" / "FAIL"). Nothing here inspects or parses
# response text -- a status can only ever be one of these three values or missing.
# ---------------------------------------------------------------------------

_STATUS_BADGE_CLASS = {"PASS": "badge-pass", "REVIEW_REQUIRED": "badge-review", "FAIL": "badge-fail"}
_STATUS_ICON = {"PASS": "✅", "REVIEW_REQUIRED": "⚠️", "FAIL": "⛔"}
_STATUS_COLOR = {"PASS": "#22c55e", "REVIEW_REQUIRED": "#f59e0b", "FAIL": "#ef4444"}
_STATUS_COLOR_DEFAULT = "#64748b"


def status_badge_html(status: str | None) -> str:
    """Render a PASS / REVIEW_REQUIRED / FAIL badge from a structured status value."""
    css_class = _STATUS_BADGE_CLASS.get(status or "", "badge-review")
    icon = _STATUS_ICON.get(status or "", "❔")
    text = status or "UNKNOWN"
    return f'<span class="{css_class}">{icon} {text}</span>'


def status_color(status: str | None) -> str:
    """A fixed colour for a structured status value, for map markers and route segments."""
    return _STATUS_COLOR.get(status or "", _STATUS_COLOR_DEFAULT)


# ---------------------------------------------------------------------------
# Error messages
#
# The deterministic tool layer (``dispatch_tool_call``) already sanitizes error text -- no
# stack traces, no file paths (see tests/unit/test_agent_tools_phase5.py). This adds a short,
# plain-language prefix on top, aimed at a student/research user rather than a developer.
# ---------------------------------------------------------------------------

_ERROR_PREFIXES: dict[str, str] = {
    "InvalidCoordinateError": "One of the coordinates you entered is not valid.",
    "InvalidThresholdError": "One of the configured thresholds is not valid.",
    "InvalidWaypointError": "The route waypoints are not in a valid format.",
    "OversizedRequestError": "That request covers too large an area to analyse in one step.",
    "TerrainAnalysisError": "The requested DEM file could not be used.",
    "MissingArgument": "A required field is missing.",
    "InvalidArgument": "One of the values you entered is not valid.",
    "InternalError": "The terrain tool could not complete this operation.",
    "UnknownTool": "The terrain tool could not complete this operation.",
    # NASA acquisition failures (terrain_agent.acquisition.errors)
    "ProviderUnavailableError": (
        "NASA ODE could not be reached. The terrain analysis cannot continue without terrain data."
    ),
    "ProviderResponseError": "NASA ODE returned a response TALUS could not interpret.",
    "NoCoverageError": "No suitable lunar DEM was found for this location.",
    "NoSuitableProductError": "No suitable lunar DEM was found for this location.",
    "DownloadError": "The NASA terrain product could not be downloaded.",
    "HostPolicyError": "The NASA terrain product could not be downloaded.",
    "DownloadTooLargeError": "The NASA terrain product is larger than this deployment's download limit.",
    "DownloadIncompleteError": "The NASA terrain product could not be downloaded completely.",
    "DownloadTimeoutError": "The NASA terrain product download timed out.",
    "DownloadChecksumError": "The downloaded terrain product failed its integrity check.",
    "DemValidationError": "The downloaded terrain product could not be opened as a valid raster.",
    "CacheError": "The downloaded terrain product could not be stored in the local cache.",
}
_DEFAULT_ERROR_PREFIX = "The analysis could not be completed."


def friendly_error_message(result: dict[str, Any]) -> str:
    """A plain-language error message for a student/research user.

    Built only from structured fields of a ``dispatch_tool_call`` error or rate-limited
    result -- never a raw exception, stack trace, or filesystem path.
    """
    detail = str(result.get("error") or "").strip()
    if result.get("status") == "rate_limited":
        return detail or "Too many requests in a short period. Please wait a moment and try again."
    prefix = _ERROR_PREFIXES.get(result.get("error_type") or "", _DEFAULT_ERROR_PREFIX)
    return f"{prefix} {detail}".strip() if detail else prefix


# ---------------------------------------------------------------------------
# Agent activity trace
#
# Observable actions only: each line is derived from a tool name and its structured result
# status / deterministic summary -- never from model reasoning or model prose.
# ---------------------------------------------------------------------------

TOOL_LABELS: dict[str, str] = {
    "resolve_lunar_feature": "Resolving lunar coordinates",
    "search_dem_products": "Searching NASA ODE",
    "fetch_nasa_dem": "Searching NASA ODE and loading the LOLA DEM (a first download can take several minutes)",
    "get_elevation_stats": "Calculating elevation",
    "get_slope_stats": "Calculating slope",
    "get_roughness_stats": "Calculating roughness",
    "evaluate_traverse_route": "Evaluating rover route safety",
    "evaluate_landing_sites": "Evaluating landing sites",
    "find_safe_regions": "Finding safe regions",
    "dataset_information": "Reading dataset provenance",
}

_OK_STATUSES = frozenset({"ok"})


def build_trace(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ordered ``{"label", "detail", "ok"}`` steps for a finished agent turn."""
    steps = [{"label": "Interpreting terrain request", "detail": "Request understood.", "ok": True}]
    for tc in tool_calls:
        tool = tc.get("tool")
        status = tc.get("result_status") or (tc.get("result") or {}).get("status")
        steps.append(
            {
                "label": TOOL_LABELS.get(tool, str(tool)),
                "detail": str(tc.get("summary") or ""),
                "ok": status in _OK_STATUSES,
            }
        )
    steps.append({"label": "Preparing evidence", "detail": "Results and provenance attached.", "ok": True})
    return steps


def analysis_outcome(status: str | None, tool_calls: list[dict[str, Any]]) -> str:
    """Completed / Partially completed / Not completed, from structured statuses only."""
    if status not in ("ok", None) or not tool_calls:
        return "Not completed"
    statuses = [tc.get("result_status") or (tc.get("result") or {}).get("status") for tc in tool_calls]
    if all(s in _OK_STATUSES for s in statuses):
        return "Completed"
    if any(s in _OK_STATUSES for s in statuses):
        return "Partially completed"
    return "Not completed"


def nasa_status_from_tool_calls(tool_calls: list[dict[str, Any]]) -> str | None:
    """Latest observed NASA ODE state from network-tool results, or ``None`` if NASA was not
    contacted (a cache hit says nothing about whether NASA is reachable right now)."""
    state: str | None = None
    for tc in tool_calls:
        if tc.get("tool") not in ("search_dem_products", "fetch_nasa_dem"):
            continue
        result = tc.get("result") or {}
        status = result.get("status")
        if status == "disabled":
            state = "disabled"
        elif status in ("ok", "no_product", "no_products_found"):
            if not result.get("from_cache"):
                state = "connected"
        elif status == "error" and result.get("error_type") in ("ProviderUnavailableError", "ProviderResponseError"):
            state = "error"
    return state


_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")


def sanitize_model_markdown(text: str) -> str:
    """Render model text as Markdown without letting it load remote images.

    Streamlit's Markdown never renders raw HTML (it is escaped), but ``![alt](url)`` would make
    the viewer's browser fetch an arbitrary URL -- model text can echo external data, so image
    syntax is reduced to its alt text.
    """
    return _MD_IMAGE_RE.sub(r"\1", str(text or ""))


# ---------------------------------------------------------------------------
# Managed DEM listing
#
# The UI offers a picklist of DEM files that already exist in the managed cache/sample
# directories -- it never accepts an arbitrary filesystem path from the user, matching the
# same "no arbitrary paths" contract ``dispatch_tool_call`` enforces for the model.
# ---------------------------------------------------------------------------


def list_managed_dems() -> list[str]:
    """DEM file names available under the configured cache and sample directories.

    Names are relative to their managed root (e.g. ``"nasa/ldem_75s_240m-....tif"``), never an
    absolute filesystem path -- safe to display and to pass straight back as ``dem_path``.
    """
    from terrain_agent.config import settings

    names: set[str] = set()
    for root in (settings.paths.cache_dir, settings.paths.sample_dir):
        if not root.is_dir():
            continue
        for pattern in ("*.tif", "*.tiff"):
            for path in root.rglob(pattern):
                try:
                    names.add(str(path.relative_to(root)).replace("\\", "/"))
                except ValueError:
                    continue
    return sorted(names)


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def fmt(value: Any, unit: str = "", decimals: int = 2) -> str:
    """Format a possibly-missing numeric value for display, with a unit suffix."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}{unit}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(fraction: Any) -> str:
    if fraction is None:
        return "—"
    try:
        return f"{float(fraction) * 100:.0f}%"
    except (TypeError, ValueError):
        return str(fraction)


# ---------------------------------------------------------------------------
# Schematic map / terrain visualisation
#
# The Moon has no public tile service inside the AGENTS.md network allowlist (NASA ODE, PDS
# Geosciences, LROC only), so this is a to-scale scientific scatter/shape plot in lat/lon
# space -- not a georeferenced basemap -- built entirely from structured coordinates already
# present in backend results. It makes no network request of its own.
# ---------------------------------------------------------------------------


def build_terrain_map_figure(
    *,
    requested_bbox: tuple[float, float, float, float] | None = None,
    dem_bounds: tuple[float, float, float, float] | None = None,
    safe_regions: list[dict[str, Any]] | None = None,
    route: list[tuple[float, float]] | None = None,
    route_segment_status: list[str | None] | None = None,
    landing_sites: list[dict[str, Any]] | None = None,
):
    """Build a Plotly figure of the analysed area.

    All bbox tuples are ``(min_lat, max_lat, min_lon, max_lon)``. ``safe_regions`` are dicts
    with ``centroid_lat``, ``centroid_lon``, ``largest_inscribed_circle_radius_m`` (as returned
    by ``find_safe_regions``). ``landing_sites`` are dicts with ``lat``, ``lon``, ``site_id``,
    ``status``. Returns a ``plotly.graph_objects.Figure`` with at least one trace, so it always
    renders even when the caller supplies nothing but a center point.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    has_content = False

    def _rect(bbox: tuple[float, float, float, float], color: str, name: str, dash: str = "dot") -> None:
        nonlocal has_content
        min_lat, max_lat, min_lon, max_lon = bbox
        fig.add_trace(
            go.Scattergl(
                x=[min_lon, max_lon, max_lon, min_lon, min_lon],
                y=[min_lat, min_lat, max_lat, max_lat, min_lat],
                mode="lines",
                line=dict(color=color, dash=dash, width=2),
                name=name,
                hoverinfo="name",
            )
        )
        has_content = True

    if requested_bbox is not None:
        _rect(requested_bbox, "#60a5fa", "Requested region")
    if dem_bounds is not None:
        _rect(dem_bounds, "#a78bfa", "DEM coverage", dash="dash")

    if safe_regions:
        for region in safe_regions:
            lat = region.get("centroid_lat")
            lon = region.get("centroid_lon")
            radius_m = region.get("largest_inscribed_circle_radius_m") or 0.0
            if lat is None or lon is None:
                continue
            # Schematic only: a fixed small marker scaled by radius, not a geodetically
            # accurate circle (degrees of latitude and longitude are not equal-area).
            fig.add_trace(
                go.Scattergl(
                    x=[lon],
                    y=[lat],
                    mode="markers",
                    marker=dict(
                        size=max(8, min(40, radius_m / 20.0)),
                        color="rgba(34,197,94,0.35)",
                        line=dict(color="#22c55e", width=1),
                    ),
                    name=region.get("region_id", "safe region"),
                    hovertemplate=(
                        f"Safe region {region.get('region_id', '')}<br>"
                        f"radius ~{radius_m:.0f} m<extra></extra>"
                    ),
                )
            )
            has_content = True

    if route:
        lats = [p[0] for p in route]
        lons = [p[1] for p in route]
        statuses = route_segment_status or []
        for i in range(len(route) - 1):
            seg_status = statuses[i] if i < len(statuses) else None
            fig.add_trace(
                go.Scattergl(
                    x=[lons[i], lons[i + 1]],
                    y=[lats[i], lats[i + 1]],
                    mode="lines+markers",
                    line=dict(color=status_color(seg_status), width=3),
                    marker=dict(size=6, color=status_color(seg_status)),
                    name=f"Route segment {i} ({seg_status or 'unknown'})",
                    hoverinfo="name",
                )
            )
        has_content = True

    if landing_sites:
        fig.add_trace(
            go.Scattergl(
                x=[s.get("lon") for s in landing_sites],
                y=[s.get("lat") for s in landing_sites],
                mode="markers+text",
                marker=dict(
                    size=14,
                    color=[status_color(s.get("status")) for s in landing_sites],
                    symbol="diamond",
                    line=dict(color="#0f172a", width=1),
                ),
                text=[s.get("site_id", "") for s in landing_sites],
                textposition="top center",
                name="Landing sites",
                hovertemplate="%{text}<extra></extra>",
            )
        )
        has_content = True

    if not has_content:
        fig.add_trace(go.Scattergl(x=[], y=[], mode="markers", name="No data yet"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.6)",
        xaxis_title="Longitude (°)",
        yaxis_title="Latitude (°)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )
    return fig
