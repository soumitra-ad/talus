"""Unit tests for the pure, Streamlit-independent UI presentation helpers (Phase 7).

These run without any Streamlit runtime -- ``ui_helpers`` performs no rendering itself, only
formatting and data shaping, so it is tested directly like any other pure module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parents[2] / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from ui_helpers import (  # noqa: E402
    build_terrain_map_figure,
    fmt,
    fmt_pct,
    friendly_error_message,
    list_managed_dems,
    status_badge_html,
    status_color,
)


# ---------------------------------------------------------------------------
# Status badges — must reflect only the structured status value, never prose
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status, expected_class, expected_icon",
    [("PASS", "badge-pass", "✅"), ("REVIEW_REQUIRED", "badge-review", "⚠️"), ("FAIL", "badge-fail", "⛔")],
)
def test_status_badge_html_maps_each_structured_status(status, expected_class, expected_icon):
    html = status_badge_html(status)
    assert expected_class in html
    assert expected_icon in html
    assert status in html


def test_status_badge_html_handles_unknown_or_missing_status_without_crashing():
    assert "UNKNOWN" in status_badge_html(None)
    assert "badge-review" in status_badge_html("something-unexpected")


def test_status_color_is_distinct_per_status():
    colors = {status_color("PASS"), status_color("REVIEW_REQUIRED"), status_color("FAIL"), status_color(None)}
    assert len(colors) == 4


# ---------------------------------------------------------------------------
# Friendly error messages
# ---------------------------------------------------------------------------


def test_friendly_error_message_prefixes_known_error_types():
    msg = friendly_error_message({"error_type": "InvalidCoordinateError", "error": "Latitude 999 is out of range."})
    assert "not valid" in msg
    assert "Latitude 999 is out of range." in msg


def test_friendly_error_message_falls_back_for_unknown_error_types():
    msg = friendly_error_message({"error_type": "SomeNewException", "error": "detail"})
    assert msg  # never empty
    assert "detail" in msg


def test_friendly_error_message_never_crashes_on_missing_fields():
    assert friendly_error_message({}) == "The analysis could not be completed."


# ---------------------------------------------------------------------------
# Managed DEM listing — never an arbitrary filesystem path
# ---------------------------------------------------------------------------


def test_list_managed_dems_only_lists_files_under_the_managed_roots(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    sample_dir = tmp_path / "sample"
    (cache_dir / "nasa").mkdir(parents=True)
    sample_dir.mkdir()
    (cache_dir / "nasa" / "a.tif").write_bytes(b"x")
    (sample_dir / "b.tiff").write_bytes(b"x")
    (tmp_path / "outside.tif").write_bytes(b"x")  # not under either managed root

    from terrain_agent.config.settings import settings as global_settings

    monkeypatch.setattr(global_settings.paths, "cache_dir", cache_dir)
    monkeypatch.setattr(global_settings.paths, "sample_dir", sample_dir)

    names = list_managed_dems()

    assert set(names) == {"nasa/a.tif", "b.tiff"}
    assert all("outside" not in n for n in names)
    assert all(not n.startswith("/") and ":" not in n for n in names)  # never an absolute path


def test_list_managed_dems_handles_missing_directories(tmp_path, monkeypatch):
    from terrain_agent.config.settings import settings as global_settings

    monkeypatch.setattr(global_settings.paths, "cache_dir", tmp_path / "does-not-exist")
    monkeypatch.setattr(global_settings.paths, "sample_dir", tmp_path / "also-missing")

    assert list_managed_dems() == []


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def test_fmt_handles_none_and_numbers():
    assert fmt(None) == "—"
    assert fmt(12.345, " m", 2) == "12.35 m"
    assert fmt(3, "°", 0) == "3°"


def test_fmt_pct_handles_none_and_fractions():
    assert fmt_pct(None) == "—"
    assert fmt_pct(0.5) == "50%"
    assert fmt_pct(1.0) == "100%"


# ---------------------------------------------------------------------------
# Schematic map figure — pure data-to-figure, no network, always renders
# ---------------------------------------------------------------------------


def test_build_terrain_map_figure_with_no_data_still_returns_a_figure():
    fig = build_terrain_map_figure()
    assert fig is not None
    assert len(fig.data) >= 1


def test_build_terrain_map_figure_includes_requested_and_dem_bounds():
    fig = build_terrain_map_figure(
        requested_bbox=(-90.0, -89.0, 0.0, 10.0),
        dem_bounds=(-90.0, -85.0, -5.0, 15.0),
    )
    names = [trace.name for trace in fig.data]
    assert "Requested region" in names
    assert "DEM coverage" in names


def test_build_terrain_map_figure_colors_route_segments_by_status():
    fig = build_terrain_map_figure(
        route=[(-89.9, 0.0), (-89.85, 30.0), (-89.8, 60.0)],
        route_segment_status=["PASS", "FAIL"],
    )
    colors = {trace.line.color for trace in fig.data if trace.name and trace.name.startswith("Route segment")}
    assert status_color("PASS") in colors
    assert status_color("FAIL") in colors


def test_build_terrain_map_figure_includes_landing_site_markers_with_status_colors():
    sites = [
        {"lat": -89.9, "lon": 0.0, "site_id": "a", "status": "PASS"},
        {"lat": -89.8, "lon": 10.0, "site_id": "b", "status": "FAIL"},
    ]
    fig = build_terrain_map_figure(landing_sites=sites)
    marker_trace = next(t for t in fig.data if t.name == "Landing sites")
    assert list(marker_trace.marker.color) == [status_color("PASS"), status_color("FAIL")]


def test_build_terrain_map_figure_includes_safe_region_markers():
    regions = [{"region_id": "r0", "centroid_lat": -89.9, "centroid_lon": 0.0, "largest_inscribed_circle_radius_m": 400.0}]
    fig = build_terrain_map_figure(safe_regions=regions)
    assert any(t.name == "r0" for t in fig.data)


# ---------------------------------------------------------------------------
# Release audit: activity trace, outcome, NASA status, markdown sanitising, NASA errors
# ---------------------------------------------------------------------------

import ui_helpers as _uh  # noqa: E402


def _tc(tool, status, **result):
    return {"tool": tool, "result_status": status, "summary": f"{tool} {status}", "result": {"status": status, **result}}


def test_trace_lists_only_observable_steps_in_order():
    steps = _uh.build_trace([_tc("resolve_lunar_feature", "ok"), _tc("fetch_nasa_dem", "error")])
    assert [s["label"] for s in steps] == [
        "Interpreting terrain request",
        "Resolving lunar coordinates",
        _uh.TOOL_LABELS["fetch_nasa_dem"],
        "Preparing evidence",
    ]
    assert [s["ok"] for s in steps] == [True, True, False, True]


@pytest.mark.parametrize(
    "status, calls, expected",
    [
        ("ok", [_tc("get_elevation_stats", "ok")], "Completed"),
        ("ok", [_tc("fetch_nasa_dem", "ok"), _tc("get_elevation_stats", "error")], "Partially completed"),
        ("ok", [_tc("fetch_nasa_dem", "no_product")], "Not completed"),
        ("model_error", [_tc("fetch_nasa_dem", "ok")], "Not completed"),
        ("ok", [], "Not completed"),
    ],
)
def test_analysis_outcome(status, calls, expected):
    assert _uh.analysis_outcome(status, calls) == expected


def test_nasa_status_tracks_real_contact_only():
    assert _uh.nasa_status_from_tool_calls([]) is None
    assert _uh.nasa_status_from_tool_calls([_tc("fetch_nasa_dem", "ok", from_cache=True)]) is None
    assert _uh.nasa_status_from_tool_calls([_tc("fetch_nasa_dem", "ok", from_cache=False)]) == "connected"
    assert _uh.nasa_status_from_tool_calls([_tc("search_dem_products", "no_products_found")]) == "connected"
    assert _uh.nasa_status_from_tool_calls([_tc("fetch_nasa_dem", "disabled")]) == "disabled"
    err = _tc("fetch_nasa_dem", "error", error_type="ProviderUnavailableError")
    assert _uh.nasa_status_from_tool_calls([err]) == "error"
    assert _uh.nasa_status_from_tool_calls([_tc("get_elevation_stats", "error", error_type="ProviderUnavailableError")]) is None


def test_model_markdown_cannot_load_remote_images():
    text = "See ![tracker](https://evil.example.com/p.png) and **bold**."
    assert _uh.sanitize_model_markdown(text) == "See tracker and **bold**."
    assert _uh.sanitize_model_markdown(None) == ""


@pytest.mark.parametrize(
    "error_type, expected",
    [
        ("ProviderUnavailableError", "NASA ODE could not be reached. The terrain analysis cannot continue without terrain data."),
        ("NoCoverageError", "No suitable lunar DEM was found for this location."),
        ("DemValidationError", "The downloaded terrain product could not be opened as a valid raster."),
        ("InternalError", "The terrain tool could not complete this operation."),
    ],
)
def test_nasa_and_tool_failures_have_plain_messages(error_type, expected):
    assert friendly_error_message({"status": "error", "error_type": error_type}) == expected

def test_result_cards_show_only_tool_values():
    fetch = _tc("fetch_nasa_dem", "ok", provenance={"mission": "LRO", "instrument": "LOLA", "product_id": "ldem_75s_240m", "pixel_size_m": [240.0, 240.0]})
    elev = _tc("get_elevation_stats", "ok", resolution_m=240.0, analysis={"terrain_available": True, "elevation": {"mean_m": -185.54, "min_m": -2855.5, "max_m": 1954.5}})
    cards = {c["label"]: c for c in _uh.result_cards([fetch, elev])}
    assert cards["Elevation (mean)"]["value"] == "-185.54 m"
    assert cards["Slope (mean)"]["value"] == "—"
    assert cards["Dataset"]["value"] == "LRO / LOLA" and "ldem_75s_240m" in cards["Dataset"]["detail"]
    assert cards["Safety score"]["value"] == "Not evaluated"
    assert _uh.result_cards([]) is None


def test_safety_score_comes_from_the_safety_tools():
    rover = _tc("evaluate_traverse_route", "ok", overall_status="FAIL", risk_score=72.0)
    assert {c["label"]: c for c in _uh.result_cards([rover])}["Safety score"]["value"] == "FAIL"
    regions = _tc("find_safe_regions", "ok", configured_threshold_deg=15.0, analysis={"safe_fraction_of_assessed": 0.5427, "regions": [{}, {}]})
    card = {c["label"]: c for c in _uh.result_cards([regions])}["Safety score"]
    assert card["value"] == "54%" and "15.0°" in card["detail"] and "2 safe region" in card["detail"]


def test_status_bar_states():
    base = dict(gemini_configured=True, gemini_block=None, gemini_health=None, nasa_observed="unknown",
                nasa_probe={"state": "ok"}, downloads_enabled=True, dem_count=1, cache_writable=True)
    assert {k: v[0] for k, v in _uh.status_bar_states(**base).items()} == {"Gemini": "ok", "NASA": "ok", "DEM Cache": "ok"}
    s = _uh.status_bar_states(**{**base, "gemini_block": {"category": "quota_daily"}})
    assert s["Gemini"] == ("fail", "Daily quota reached")
    assert _uh.status_bar_states(**{**base, "gemini_configured": False})["Gemini"][0] == "warn"
    assert _uh.status_bar_states(**{**base, "nasa_observed": "error"})["NASA"][0] == "fail"
    assert _uh.status_bar_states(**{**base, "nasa_probe": None})["NASA"] == ("warn", "Not checked")
    assert _uh.status_bar_states(**{**base, "downloads_enabled": False})["NASA"] == ("warn", "Downloads disabled")
    assert _uh.status_bar_states(**{**base, "dem_count": 0})["DEM Cache"][0] == "warn"
    assert _uh.status_bar_states(**{**base, "cache_writable": False})["DEM Cache"][0] == "fail"


def test_fallback_counts_as_a_completed_analysis():
    assert _uh.analysis_outcome("fallback", [_tc("get_elevation_stats", "ok")]) == "Completed"