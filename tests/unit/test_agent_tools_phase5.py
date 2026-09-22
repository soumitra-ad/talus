"""Tests for the agent tool layer and terrain statistics tool (Phase 5).

These call ``dispatch_tool_call`` exactly as the model would, on synthetic DEMs. Before
Phase 5 nothing tested this layer, and four of its tools failed on import.
"""

from __future__ import annotations

import json
import os

import pytest

from terrain_agent.agent.agent import TALUS_TOOL_DECLARATIONS, dispatch_tool_call
from terrain_agent.terrain import InvalidCoordinateError, OversizedRequestError
from terrain_agent.tools.terrain_stats import analyze_terrain_bbox


def call(tool, args, cache):
    return dispatch_tool_call(tool, args, dem_cache_dir=str(cache))


def wp(b, *cells):
    return [list(p) for p in b.route(*cells)]


# ---------------------------------------------------------------------------
# Tool declarations
# ---------------------------------------------------------------------------


def test_every_declared_tool_has_a_handler(tmp_path):
    names = [t["name"] for t in TALUS_TOOL_DECLARATIONS]
    assert "find_safe_regions" in names
    for name in names:
        result = call(name, {}, tmp_path)
        assert "Unknown tool" not in result.get("error", ""), name


def test_declarations_ask_for_managed_file_names_not_arbitrary_paths():
    text = json.dumps(TALUS_TOOL_DECLARATIONS)
    assert "Absolute path" not in text
    assert "managed cache or sample directory" in text


# ---------------------------------------------------------------------------
# Rover, landing and safe-region tools
# ---------------------------------------------------------------------------


def test_traverse_tool_returns_the_deterministic_analysis(dem_builder, tmp_path):
    dem_builder.geographic("flat.tif", dem_builder.flat())
    args = {"waypoints": wp(dem_builder, (200, 50), (200, 150), (250, 300)), "dem_path": "flat.tif"}
    result = call("evaluate_traverse_route", args, tmp_path)

    assert result["status"] == "ok"
    assert result["overall_status"] == "PASS"
    assert result["risk_score"] == 0.0
    analysis = result["analysis"]
    assert analysis["total_segments"] == 2 and analysis["coverage_fraction"] == 1.0
    assert "NOT certified" in result["disclaimer"]
    json.dumps(result)  # must be serialisable for the model


def test_traverse_tool_passes_thresholds_and_no_go_zones_through(dem_builder, tmp_path):
    dem_builder.geographic("flat.tif", dem_builder.flat())
    lat, lon = dem_builder.latlon(225, 225)
    args = {
        "waypoints": wp(dem_builder, (200, 50), (200, 150), (250, 300)),
        "dem_path": "flat.tif",
        "max_slope_deg": 8.0,
        "no_go_zones": [{"name": "Z", "lat": lat, "lon": lon, "radius_m": 100.0}],
    }
    result = call("evaluate_traverse_route", args, tmp_path)
    assert result["overall_status"] == "FAIL"
    assert result["configured_threshold_deg"] == 8.0
    assert result["analysis"]["violated_segments"] == [1]


def test_traverse_tool_never_reports_pass_without_data(dem_builder, tmp_path):
    dem_builder.geographic("flat.tif", dem_builder.flat())
    args = {"waypoints": [[0.2, 20.0], [0.2, 20.1]], "dem_path": "flat.tif"}
    result = call("evaluate_traverse_route", args, tmp_path)
    assert result["status"] == "ok"
    assert result["overall_status"] == "REVIEW_REQUIRED"


def test_landing_tool_ranks_and_lists_unranked_sites(dem_builder, tmp_path):
    dem_builder.geographic("zones.tif", dem_builder.bands([(150, 300, 2.0), (300, 400, 10.0)]))
    lat_a, lon_a = dem_builder.latlon(200, 60)
    lat_c, lon_c = dem_builder.latlon(200, 350)
    args = {
        "sites": [
            {"id": "flat", "lat": lat_a, "lon": lon_a},
            {"id": "steep", "lat": lat_c, "lon": lon_c},
            {"id": "far", "lat": 0.2, "lon": 20.0},
        ],
        "dem_path": "zones.tif",
        "radius_m": 200.0,
        "min_flat_radius_m": 60.0,
    }
    result = call("evaluate_landing_sites", args, tmp_path)
    assert result["status"] == "ok"
    assert (result["sites_ranked"], result["sites_unranked"]) == (2, 1)
    ranked = [r["analysis"]["site_id"] for r in result["analysis"]["ranked"]]
    assert ranked == ["flat", "steep"]
    assert result["analysis"]["unranked"][0]["site_id"] == "far"


def test_safe_region_tool(dem_builder, tmp_path):
    dem_builder.geographic("half.tif", dem_builder.bands([(200, 400, 20.0)]))
    lat, lon = dem_builder.latlon(200, 200)
    args = {"center_lat": lat, "center_lon": lon, "radius_m": 1500.0, "dem_path": "half.tif", "max_slope_deg": 10.0}
    result = call("find_safe_regions", args, tmp_path)
    assert result["status"] == "ok" and result["outcome"] == "regions_found"
    assert len(result["analysis"]["regions"]) == 1


# ---------------------------------------------------------------------------
# DEM path handling: the model cannot choose arbitrary files
# ---------------------------------------------------------------------------


def test_paths_outside_the_managed_directory_are_rejected(dem_builder, tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "ok.tif").write_bytes(b"x")
    secret = dem_builder.__class__(outside).geographic("secret.tif", dem_builder.flat())
    waypoints = wp(dem_builder, (200, 50), (200, 150))

    for bad in (str(secret), "../outside/secret.tif", "..\\outside\\secret.tif", "/etc/passwd", "", "   ", 123, None, ["a"]):
        result = dispatch_tool_call(
            "evaluate_traverse_route", {"waypoints": waypoints, "dem_path": bad}, dem_cache_dir=str(allowed)
        )
        assert result["status"] == "error", bad
        assert result["error_type"] == "TerrainAnalysisError"
        assert "outside" not in result["error"] and "secret" not in result["error"]


def test_symbolic_link_out_of_the_managed_directory_is_rejected(dem_builder, tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    secret = dem_builder.__class__(outside).geographic("secret.tif", dem_builder.flat())
    try:
        os.symlink(secret, allowed / "link.tif")
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available on this system")
    result = dispatch_tool_call(
        "get_slope_stats",
        {"dem_path": "link.tif", "min_lat": 0.1, "max_lat": 0.2, "min_lon": 10.1, "max_lon": 10.2},
        dem_cache_dir=str(allowed),
    )
    assert result["status"] == "error"


def test_missing_file_inside_the_directory_is_an_error_without_paths(dem_builder, tmp_path):
    result = call("evaluate_traverse_route", {"waypoints": wp(dem_builder, (200, 50), (200, 150)), "dem_path": "nope.tif"}, tmp_path)
    assert result["status"] == "error"
    assert tmp_path.name not in result["error"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_tool_and_missing_arguments(tmp_path):
    assert "Unknown tool" in call("delete_everything", {}, tmp_path)["error"]
    missing = call("evaluate_traverse_route", {}, tmp_path)
    assert missing["status"] == "error" and "Missing required argument" in missing["error"]


def test_invalid_coordinates_and_waypoints_surface_as_typed_errors(dem_builder, tmp_path):
    dem_builder.geographic("flat.tif", dem_builder.flat())
    coords = call("evaluate_traverse_route", {"waypoints": [[95.0, 10.0], [0.2, 10.1]], "dem_path": "flat.tif"}, tmp_path)
    assert coords["status"] == "error" and coords["error_type"] == "InvalidCoordinateError"
    shape = call("evaluate_traverse_route", {"waypoints": "abc", "dem_path": "flat.tif"}, tmp_path)
    assert shape["error_type"] == "InvalidWaypointError"
    threshold = call(
        "evaluate_traverse_route",
        {"waypoints": wp(dem_builder, (200, 50), (200, 150)), "dem_path": "flat.tif", "max_slope_deg": -3},
        tmp_path,
    )
    assert threshold["error_type"] == "InvalidThresholdError"


def test_unexpected_failures_do_not_leak_internal_details(dem_builder, tmp_path, monkeypatch):
    dem_builder.geographic("flat.tif", dem_builder.flat())

    def explode(*args, **kwargs):
        raise RuntimeError("boom at C:/secret/location/file.tif")

    monkeypatch.setattr("terrain_agent.safety.rover.check_rover_safety", explode)
    result = call(
        "evaluate_traverse_route", {"waypoints": wp(dem_builder, (200, 50), (200, 150)), "dem_path": "flat.tif"}, tmp_path
    )
    assert result["status"] == "error" and result["error_type"] == "InternalError"
    assert "secret" not in json.dumps(result) and "boom" not in json.dumps(result)


# ---------------------------------------------------------------------------
# Terrain statistics tools
# ---------------------------------------------------------------------------


BOX = {"min_lat": 0.15, "max_lat": 0.25, "min_lon": 10.1, "max_lon": 10.2}


@pytest.mark.parametrize(
    "tool, section",
    [("get_elevation_stats", "elevation"), ("get_slope_stats", "slope"), ("get_roughness_stats", "roughness")],
)
def test_statistics_tools_return_only_their_section(dem_builder, tmp_path, tool, section):
    dem_builder.geographic("ramp.tif", dem_builder.ramp(10.0))
    result = call(tool, {"dem_path": "ramp.tif", **BOX}, tmp_path)
    assert result["status"] == "ok"
    analysis = result["analysis"]
    assert analysis[section] is not None
    assert all(other not in analysis for other in {"elevation", "slope", "roughness"} - {section})
    assert analysis["coverage_fraction"] == 1.0
    assert result["resolution_m"] == pytest.approx(30.3, abs=0.2)


def test_statistics_values_come_from_the_dem(dem_builder):
    dem = dem_builder.geographic("ramp.tif", dem_builder.ramp(10.0))
    stats = analyze_terrain_bbox(dem, **BOX)
    assert stats.slope.mean_slope_deg == pytest.approx(10.0, abs=0.1)
    assert stats.elevation.min_m < stats.elevation.mean_m < stats.elevation.max_m
    assert stats.dataset.file_name == "ramp.tif"


def test_statistics_report_partial_coverage_and_no_data(dem_builder, tmp_path):
    dem = dem_builder.geographic("flat.tif", dem_builder.flat())
    partly = analyze_terrain_bbox(dem, 0.15, 0.25, 10.3, 10.5)
    assert partly.terrain_available and 0.0 < partly.coverage_fraction < 1.0
    assert any("requested area" in w for w in partly.warnings)

    outside = analyze_terrain_bbox(dem, 0.15, 0.25, 20.0, 20.1)
    assert not outside.terrain_available and outside.coverage_fraction == 0.0
    assert outside.elevation is None

    missing = analyze_terrain_bbox(tmp_path / "nope.tif", **BOX)
    assert not missing.terrain_available and missing.elevation is None
    assert tmp_path.name not in missing.model_dump_json()
    tool = call(
        "get_elevation_stats",
        {"dem_path": "flat.tif", "min_lat": 0.15, "max_lat": 0.25, "min_lon": 20.0, "max_lon": 20.1},
        tmp_path,
    )
    assert tool["status"] == "no_terrain_data"


def test_statistics_reject_invalid_boxes(dem_builder):
    dem = dem_builder.geographic("flat.tif", dem_builder.flat())
    with pytest.raises(InvalidCoordinateError):
        analyze_terrain_bbox(dem, 0.25, 0.15, 10.1, 10.2)
    with pytest.raises(InvalidCoordinateError):
        analyze_terrain_bbox(dem, 95.0, 96.0, 10.1, 10.2)
    with pytest.raises(InvalidCoordinateError):
        analyze_terrain_bbox(dem, 0.1, 0.2, 170.0, 190.0)  # crosses the antimeridian
    with pytest.raises(OversizedRequestError):
        analyze_terrain_bbox(dem, 0.0, 0.4, 10.0, 13.0)  # needs a window far above the read limit
