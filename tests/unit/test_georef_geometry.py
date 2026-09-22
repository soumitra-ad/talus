"""Tests for georeferencing, route densification and no-go geometry (Phase 5)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from terrain_agent.safety import SafetyStatus, check_rover_safety
from terrain_agent.terrain import (
    InvalidCoordinateError,
    InvalidNoGoZoneError,
    InvalidWaypointError,
    MalformedRasterError,
    OversizedRequestError,
    UnsupportedCRSError,
    densify_route,
    find_no_go_hits,
    open_dem_context,
    parse_no_go_zones,
    validate_route,
)

R = 1_737_400.0


# ---------------------------------------------------------------------------
# Georeferencing
# ---------------------------------------------------------------------------


def stereographic_ctx(b, name="stereo.tif"):
    dem = b.stereographic(name, np.zeros((50, 50)), west=-250.0, north=250.0)
    return open_dem_context(dem)


def test_lat_lon_maps_to_the_analytic_polar_stereographic_position(dem_builder):
    ctx = stereographic_ctx(dem_builder)
    x, y = ctx.to_raster_xy([-89.9, -89.9], [0.0, 90.0])
    expected = 2.0 * R * math.tan(math.radians(0.1) / 2.0)  # about 3032.336 m from the pole
    assert (x[0], y[0]) == pytest.approx((0.0, expected), abs=0.01)
    assert (x[1], y[1]) == pytest.approx((expected, 0.0), abs=0.01)


def test_lat_lon_round_trip_through_the_dem_projection(dem_builder):
    ctx = stereographic_ctx(dem_builder)
    lats, lons = [-89.95, -88.0, -85.5], [10.0, -120.0, 179.0]
    x, y = ctx.to_raster_xy(lats, lons)
    back_lat, back_lon = ctx.to_latlon(x, y)
    assert back_lat == pytest.approx(lats, abs=1e-9)
    assert back_lon == pytest.approx(lons, abs=1e-9)


@pytest.mark.parametrize("lat", [-89.9, -85.0, -80.0, -70.0])
def test_scale_factor_matches_the_polar_stereographic_formula(dem_builder, lat):
    ctx = stereographic_ctx(dem_builder)
    expected = 2.0 / (1.0 + math.sin(math.radians(abs(lat))))
    assert ctx.scale_factor(lat, 25.0) == pytest.approx(expected, abs=1e-5)
    dx, dy = ctx.pixel_size_m(lat, 25.0)
    assert dx == pytest.approx(10.0 / expected, rel=1e-5) and dy == pytest.approx(dx)


def test_slope_is_correct_at_80_degrees_south_where_the_map_scale_is_not_one(dem_builder):
    """A plane of true slope 20 degrees. Ignoring the 0.77% map scale would read 19.86 degrees."""
    x0, y0 = dem_builder.lunar_to_stereographic(-80.0, 0.0)
    k = 2.0 / (1.0 + math.sin(math.radians(80.0)))
    columns = (np.arange(100) + 0.5) * 10.0 - 500.0
    elevation = np.tile(1500.0 + math.tan(math.radians(20.0)) * columns / k, (100, 1))
    dem = dem_builder.stereographic("tilt80.tif", elevation, west=x0 - 500.0, north=y0 + 500.0)
    ctx = open_dem_context(dem)
    lats, lons = ctx.to_latlon([x0 - 300.0, x0 + 300.0], [y0, y0])
    route = list(zip(lats.tolist(), lons.tolist()))

    result = check_rover_safety(route, 25.0, dem_path=dem)
    assert result.coverage_fraction == 1.0
    assert result.max_slope_deg == pytest.approx(20.0, abs=0.05)
    assert result.dataset.resolution_m == pytest.approx(10.0 / k, rel=1e-4)


def test_geographic_cell_size_uses_the_cosine_of_latitude(dem_builder):
    dem = dem_builder.geographic("mid_lat.tif", dem_builder.flat(), lat_top=60.4)
    ctx = open_dem_context(dem)
    dx, dy = ctx.pixel_size_m(60.2, 10.2)
    metres_per_degree = math.radians(1.0) * R
    assert dy == pytest.approx(0.001 * metres_per_degree, rel=1e-9)
    assert dx == pytest.approx(0.001 * metres_per_degree * math.cos(math.radians(60.2)), rel=1e-9)


def test_geographic_dem_near_a_pole_is_refused_and_never_passes(dem_builder):
    dem = dem_builder.geographic("polar_geo.tif", dem_builder.flat(), lat_top=89.9)
    ctx = open_dem_context(dem)
    with pytest.raises(UnsupportedCRSError):
        ctx.pixel_size_m(89.8, 10.2)
    result = check_rover_safety([(89.75, 10.1), (89.74, 10.2)], 15.0, dem_path=dem)
    assert result.status is SafetyStatus.REVIEW_REQUIRED
    assert not result.terrain_available
    assert any("pole" in w for w in result.warnings)


def test_zero_to_360_longitude_convention_is_handled(dem_builder):
    dem = dem_builder.geographic("lon360.tif", dem_builder.flat(), lon0=350.0)
    ctx = open_dem_context(dem)
    assert ctx.uses_lon_360
    x, _ = ctx.to_raster_xy([0.2], [-9.9])
    assert x[0] == pytest.approx(350.1)
    result = check_rover_safety([(0.2, -9.95), (0.2, -9.85)], 15.0, dem_path=dem)
    assert result.status is SafetyStatus.PASS
    assert result.coverage_fraction == 1.0


def test_a_dem_declared_on_earth_radius_is_flagged(dem_builder, tmp_path):
    path = tmp_path / "earth_crs.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=50,
        width=50,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=from_origin(10.0, 0.4, 0.001, 0.001),
    ) as dst:
        dst.write(np.full((50, 50), 1500.0, dtype="float32"), 1)
    ctx = open_dem_context(path)
    assert any("differs from the lunar reference radius" in w for w in ctx.warnings)
    result = check_rover_safety([(0.37, 10.01), (0.37, 10.04)], 15.0, dem_path=path)
    assert any("differs from the lunar reference radius" in w for w in result.warnings)


def test_open_errors_are_typed(dem_builder, tmp_path):
    with pytest.raises(FileNotFoundError):
        open_dem_context(tmp_path / "missing.tif")
    bad = tmp_path / "bad.tif"
    bad.write_bytes(b"not a raster")
    with pytest.raises(MalformedRasterError):
        open_dem_context(bad)


# ---------------------------------------------------------------------------
# Route densification
# ---------------------------------------------------------------------------


def test_densified_route_follows_great_circles_and_respects_spacing():
    waypoints = [(0.0, 0.0), (0.0, 90.0)]
    samples = densify_route(waypoints, 100_000.0)
    quarter = math.pi * R / 2.0
    assert samples.segment_lengths_m[0] == pytest.approx(quarter, rel=1e-9)
    assert samples.lat[0] == pytest.approx(0.0, abs=1e-9) and samples.lon[0] == pytest.approx(0.0, abs=1e-9)
    assert samples.lat[-1] == pytest.approx(0.0, abs=1e-9) and samples.lon[-1] == pytest.approx(90.0, abs=1e-9)
    assert np.all(np.abs(samples.lat) < 1e-9)  # the equator is a great circle
    steps = np.diff(samples.along_m)
    assert np.all(steps > 0) and np.all(steps <= 100_000.0 + 1e-6)
    assert samples.along_m[-1] == pytest.approx(quarter, rel=1e-9)


def test_densified_samples_are_tagged_by_segment_and_include_every_waypoint():
    waypoints = [(0.0, 10.0), (0.1, 10.0), (0.1, 10.1)]
    samples = densify_route(waypoints, 500.0)
    assert set(samples.segment.tolist()) == {0, 1}
    for index, (lat, lon) in enumerate(waypoints[:-1]):
        first = np.flatnonzero(samples.segment == index)[0]
        assert (samples.lat[first], samples.lon[first]) == pytest.approx((lat, lon), abs=1e-9)
    end_of_first = np.flatnonzero(samples.segment == 0)[-1]
    assert (samples.lat[end_of_first], samples.lon[end_of_first]) == pytest.approx((0.1, 10.0), abs=1e-9)


def test_route_across_the_pole_is_sampled_through_the_pole():
    samples = densify_route([(-89.9, 0.0), (-89.9, 180.0)], 200.0)
    # The two ends are about 6 km apart across the pole, so some sample must lie within
    # half a spacing (100 m) of the pole.
    metres_from_pole = (90.0 - np.abs(samples.lat).max()) * math.radians(1.0) * R
    assert metres_from_pole <= 101.0


def test_densify_rejects_bad_spacing_and_oversized_requests():
    with pytest.raises(ValueError):
        densify_route([(0.0, 0.0), (0.0, 1.0)], 0.0)
    with pytest.raises(ValueError):
        densify_route([(0.0, 0.0), (0.0, 1.0)], float("nan"))
    with pytest.raises(OversizedRequestError):
        densify_route([(0.0, 0.0), (0.0, 1.0)], 1.0, max_samples=1000)


# ---------------------------------------------------------------------------
# Route validation
# ---------------------------------------------------------------------------


def test_route_validation_normalises_longitude_and_accepts_common_containers():
    normalised = validate_route([(0.0, 350.0), [0.0, 350.1]])
    assert normalised == [(0.0, pytest.approx(-10.0)), (0.0, pytest.approx(-9.9))]
    assert validate_route(np.array([[0.0, 1.0], [0.0, 2.0]])) == [(0.0, 1.0), (0.0, 2.0)]
    assert validate_route(((0, 1), (0, 2))) == [(0.0, 1.0), (0.0, 2.0)]


def test_route_validation_rejects_bad_input():
    with pytest.raises(InvalidWaypointError):
        validate_route([(0.0, 1.0)])
    with pytest.raises(InvalidCoordinateError):
        validate_route([(0.0, 1.0), (91.0, 1.0)])
    with pytest.raises(InvalidWaypointError):
        validate_route([(0.0, 1.0), (0.0, 1.0000000001)])  # closer than one metre


# ---------------------------------------------------------------------------
# No-go zone parsing and geometry
# ---------------------------------------------------------------------------


def test_zone_definitions_accept_alternative_keys():
    zones = parse_no_go_zones(
        [
            {"center_lat": 0.1, "center_lon": 10.0, "radius_km": 2.5, "name": "km zone"},
            {"vertices": [[0, 10], [0, 10.1], [0.1, 10.1], [0.1, 10.0], [0, 10]]},
        ]
    )
    assert zones[0].radius_m == 2500.0 and zones[0].name == "km zone"
    assert zones[1].kind == "polygon" and len(zones[1].vertices) == 4  # closing vertex dropped
    assert zones[1].name == "no_go_zone_2"


def test_concave_polygon_hits_only_where_the_route_enters_the_shape():
    # An L shape. The notch (latitude above 0.03 and longitude above 10.03) is outside it.
    l_shape = [[0, 10], [0, 10.1], [0.03, 10.1], [0.03, 10.03], [0.1, 10.03], [0.1, 10]]
    zones = parse_no_go_zones([{"name": "L", "polygon": l_shape}])

    through_notch = densify_route([(0.06, 10.06), (0.09, 10.09)], 250.0)
    assert find_no_go_hits(through_notch, zones) == {}

    through_arm = densify_route([(0.01, 10.05), (0.02, 10.08)], 250.0)
    assert find_no_go_hits(through_arm, zones) == {0: ["L"]}

    grazing_from_outside = densify_route([(-0.05, 10.05), (0.01, 10.05)], 250.0)
    assert find_no_go_hits(grazing_from_outside, zones) == {0: ["L"]}


def test_zone_hits_are_reported_per_segment_with_every_zone_name():
    waypoints = [(0.0, 10.0), (0.0, 10.1), (0.0, 10.2)]
    zones = parse_no_go_zones(
        [
            {"name": "A", "lat": 0.0, "lon": 10.05, "radius_m": 500.0},
            {"name": "B", "lat": 0.0, "lon": 10.05, "radius_m": 300.0},
            {"name": "C", "lat": 0.0, "lon": 10.15, "radius_m": 300.0},
        ]
    )
    hits = find_no_go_hits(densify_route(waypoints, 250.0), zones)
    assert hits == {0: ["A", "B"], 1: ["C"]}


def test_zone_validation_limits():
    with pytest.raises(InvalidNoGoZoneError):
        parse_no_go_zones([{"lat": 0.0, "lon": 10.0, "radius_m": 1000.0, "polygon": [[0, 0], [1, 1], [0, 1]]}])
    with pytest.raises(InvalidNoGoZoneError):
        parse_no_go_zones([{"polygon": [[0, 0], [0, 1], [1, 1], [True, 0]]}])
    huge = [[-60, 0], [-60, 120], [60, 120], [60, 0]]
    with pytest.raises(InvalidNoGoZoneError):
        parse_no_go_zones([{"polygon": huge}])
