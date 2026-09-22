"""Real-NASA-data validation of the Phase 5 terrain and safety pipeline.

This module does NOT rebuild or replace Phase 5. It exercises the existing, deterministic
Phase 5 functions -- ``analyze_terrain_bbox``, ``check_rover_safety``, ``analyze_landing_site``,
``find_safe_regions`` -- against a real NASA lunar DEM acquired through the existing Phase 4
acquisition service, to confirm the pipeline behaves correctly on real terrain rather than only
on the synthetic rasters used by ``tests/unit``.

Real NASA product used
-----------------------
LRO/LOLA GDR polar DEM ``ldem_75s_240m`` (product type ``GDRDEM``), downloaded from the PDS
Geosciences Node through the NASA ODE REST API -- the same path the running application uses
(``terrain_agent.acquisition.service.build_default_service``). It is a south-polar stereographic
raster, 240 m/pixel, 3812 x 3812 cells, covering the south pole down to about -75 degrees
latitude at every longitude. Elevations are height above a 1,737,400 m reference sphere, not a
geoid, per the project's elevation-reference convention.

Fixture, not a mock
--------------------
The file is downloaded once, offline of pytest, by ``tests/fixtures/_download_real_dem.py``
into the gitignored ``tests/fixtures/nasa_real_dem_cache`` directory. These tests then read that
cached, validated GeoTIFF directly -- no network call happens inside the test run, so the suite
stays fast, offline and deterministic, while still exercising real measured lunar terrain (real
elevation noise, real slopes, real coverage gaps at the tile edge) that a synthetic DEM could not
produce. If the fixture has not been created yet, the whole module is skipped with instructions,
rather than failing the offline test suite.

Assertion style
----------------
Per the project's own live-data test (``tests/live/test_nasa_live.py``), assertions here check
facts that follow from the data itself -- physical plausibility, internal consistency, coverage
and status/violation relationships, and safety invariants -- not specific numeric terrain values,
which would make the suite brittle to (legitimate) future re-downloads of the same product.
The specific status values asserted in the edge-case tests (partial coverage, roughness-only
violation, coarse-resolution flat-radius evidence) were confirmed by first running the real
scenario and observing the actual, code-determined outcome -- they are not guesses.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from terrain_agent.acquisition.models import CoverageRequest
from terrain_agent.acquisition.service import build_default_service
from terrain_agent.safety import (
    SafetyStatus,
    analyze_landing_site,
    check_rover_safety,
    find_safe_regions,
)
from terrain_agent.terrain import open_dem_context
from terrain_agent.terrain.resource_safety import InvalidCoordinateError, InvalidThresholdError
from terrain_agent.tools.terrain_stats import analyze_terrain_bbox

FIXTURE_CACHE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "nasa_real_dem_cache"

# Matches tests/fixtures/_download_real_dem.py and tests/live/test_nasa_live.py: Shackleton
# crater vicinity, lunar south pole. The acquired product covers far more than this point (see
# module docstring), which is what lets the edge/out-of-coverage tests below work.
FIXTURE_LAT, FIXTURE_LON, FIXTURE_RADIUS_M = -89.9, 0.0, 5000.0

# The product's projected extent is a 914,880 m square centred on the pole (measured from the
# downloaded raster: +/-457,440 m). Along the lon=0 meridian that boundary falls at about -75.0
# degrees latitude. These two points straddle it by design, one just inside and one just outside,
# so a route or footprint between/around them is genuinely, reproducibly partially covered.
EDGE_LAT_INSIDE = -75.8
EDGE_LAT_OUTSIDE = -74.8
EDGE_LON = 0.0

# Well beyond the product's south-polar coverage altogether, but still a valid lunar coordinate.
FAR_OUTSIDE_LAT, FAR_OUTSIDE_LON = -60.0, 0.0

pytestmark = pytest.mark.real_dem


def _find_fixture_dem():
    service = build_default_service(cache_dir=FIXTURE_CACHE_DIR, enabled=False)
    request = CoverageRequest.from_point(FIXTURE_LAT, FIXTURE_LON, FIXTURE_RADIUS_M)
    return service.find_cached(request, ["GDRDEM"])


@pytest.fixture(scope="module")
def acquired_dem():
    cached = _find_fixture_dem()
    if cached is None:
        pytest.skip(
            "No local real-NASA-DEM fixture found in tests/fixtures/nasa_real_dem_cache. Run "
            "`python tests/fixtures/_download_real_dem.py` once (needs network access to NASA "
            "ODE/PDS) to create it; these tests then run fully offline."
        )
    return cached


@pytest.fixture(scope="module")
def dem_path(acquired_dem) -> Path:
    return acquired_dem.dem_path


@pytest.fixture(scope="module")
def provenance(acquired_dem) -> dict:
    return acquired_dem.provenance


# ---------------------------------------------------------------------------
# 1. Real DEM loading
# ---------------------------------------------------------------------------


def test_real_dem_loads(dem_path):
    assert dem_path.is_file()
    ctx = open_dem_context(dem_path)
    assert ctx.width > 0 and ctx.height > 0
    assert ctx.metadata.driver == "GTiff"
    assert ctx.metadata.count == 1
    assert ctx.metadata.res_x > 0 and ctx.metadata.res_y > 0


# ---------------------------------------------------------------------------
# 2. Real CRS handling
# ---------------------------------------------------------------------------


def test_real_crs_handling(dem_path):
    ctx = open_dem_context(dem_path)
    assert not ctx.is_geographic
    assert ctx.crs.is_projected
    assert ctx.projection == "stere"
    assert ctx.metadata.crs_wkt is not None and "Moon" in ctx.metadata.crs_wkt
    # The DEM's own reference sphere matches the lunar radius TALUS assumes, so georef.py
    # should not have had to fall back to a mismatch warning.
    assert not any("reference radius" in w for w in ctx.warnings)


# ---------------------------------------------------------------------------
# 3. Real geographic bounds
# ---------------------------------------------------------------------------


def test_real_geographic_bounds(dem_path, provenance):
    ctx = open_dem_context(dem_path)

    lat_bounds = provenance["raster"]["product_bounds_lat"]
    assert lat_bounds[0] == pytest.approx(-90.0, abs=0.5)
    assert -80.0 < lat_bounds[1] < -70.0

    # The south pole itself must land inside the raster's cell grid, at its projected origin.
    row, col, finite = ctx.latlon_to_rowcol_float([-90.0], [0.0])
    assert bool(finite[0])
    assert 0 <= row[0] < ctx.height and 0 <= col[0] < ctx.width

    # The fixture location used throughout this module is comfortably inside the raster.
    row_f, col_f, finite_f = ctx.latlon_to_rowcol_float([FIXTURE_LAT], [FIXTURE_LON])
    assert bool(finite_f[0])
    assert 0 <= row_f[0] < ctx.height and 0 <= col_f[0] < ctx.width


# ---------------------------------------------------------------------------
# 4. Elevation queries
# ---------------------------------------------------------------------------


def test_elevation_queries(dem_path):
    stats = analyze_terrain_bbox(dem_path, -89.95, -89.85, -10.0, 10.0)
    assert stats.terrain_available
    assert stats.coverage_fraction > 0.9
    assert stats.elevation is not None
    assert stats.elevation.valid_cell_count > 0
    # Physical plausibility: the Moon spans roughly -9 km to +11 km about the reference sphere.
    assert -12_000.0 < stats.elevation.min_m <= stats.elevation.mean_m <= stats.elevation.max_m < 12_000.0
    assert stats.dataset is not None
    assert stats.dataset.provenance == "sidecar"
    assert stats.dataset.product_type == "GDRDEM"
    assert stats.dataset.elevation_reference is not None
    assert "not a geoid" in stats.dataset.elevation_reference


# ---------------------------------------------------------------------------
# 5. Slope calculations
# ---------------------------------------------------------------------------


def test_slope_calculations(dem_path):
    stats = analyze_terrain_bbox(dem_path, -89.95, -89.85, -10.0, 10.0)
    slope = stats.slope
    assert slope is not None
    assert slope.valid_cells > 0
    assert 0.0 <= slope.mean_slope_deg <= slope.max_slope_deg < 90.0
    assert slope.median_slope_deg <= slope.max_slope_deg
    assert sum(slope.histogram_counts) == slope.valid_cells
    assert slope.percentiles["p10"] <= slope.percentiles["p50"] <= slope.percentiles["p90"]


# ---------------------------------------------------------------------------
# 6. Roughness calculations
# ---------------------------------------------------------------------------


def test_roughness_calculations(dem_path):
    stats = analyze_terrain_bbox(dem_path, -89.95, -89.85, -10.0, 10.0)
    roughness = stats.roughness
    assert roughness is not None
    assert roughness.valid_cells > 0
    assert 0.0 <= roughness.mean_tri_m <= roughness.max_tri_m
    assert roughness.std_tri_m >= 0.0
    assert "NOT a certified or universal" in roughness.disclaimer


# ---------------------------------------------------------------------------
# 7. Safe-region detection
# ---------------------------------------------------------------------------


def test_safe_region_detection(dem_path):
    generous = find_safe_regions(
        dem_path, FIXTURE_LAT, FIXTURE_LON, 3000.0, maximum_slope_deg=25.0, min_area_m2=50_000.0
    )
    assert generous.terrain_available
    assert generous.assessed_fraction == pytest.approx(1.0, abs=0.02)
    assert generous.outcome in ("regions_found", "no_safe_region_in_assessed_area")
    for region in generous.regions:
        assert region.max_slope_deg <= 25.0 + 1e-6
        assert region.area_m2 >= 50_000.0
        assert region.largest_inscribed_circle_radius_m >= 0.0

    # An unrealistically strict slope limit must never find MORE safe area than a generous one.
    strict = find_safe_regions(
        dem_path, FIXTURE_LAT, FIXTURE_LON, 3000.0, maximum_slope_deg=0.05, min_area_m2=50_000.0
    )
    assert strict.terrain_available
    assert strict.regions_total_found <= generous.regions_total_found
    if generous.safe_fraction_of_assessed is not None and strict.safe_fraction_of_assessed is not None:
        assert strict.safe_fraction_of_assessed <= generous.safe_fraction_of_assessed


# ---------------------------------------------------------------------------
# 8. Rover route safety
# ---------------------------------------------------------------------------


def test_rover_route_safety(dem_path):
    route = [(-89.9, 0.0), (-89.85, 30.0), (-89.8, 60.0)]
    result = check_rover_safety(route, 15.0, dem_path=dem_path)

    assert result.terrain_available
    assert result.status in (SafetyStatus.PASS, SafetyStatus.FAIL, SafetyStatus.REVIEW_REQUIRED)
    assert result.max_slope_deg is not None
    assert result.risk_score is not None

    if result.status is SafetyStatus.PASS:
        assert not result.violated_segments
        assert result.coverage_fraction == 1.0
    if result.violated_segments:
        assert result.status in (SafetyStatus.FAIL, SafetyStatus.REVIEW_REQUIRED)

    assert "uncalibrated" in result.risk_score_method.lower()
    assert "not mission approval" in result.disclaimer
    assert "NOT certified safety limits" in result.disclaimer
    assert "NOT a certified safety limit" in result.configured_thresholds.statement


def test_rover_roughness_only_violation_is_review_required(dem_path):
    """A route with clearly rugged real terrain but a lax slope limit: measured roughness
    violation, no slope violation -> REVIEW_REQUIRED, per the module's own status rules, never
    silently PASS and never escalated to FAIL for a non-slope violation."""
    route = [(-89.9, 0.0), (-89.85, 30.0), (-89.8, 60.0)]
    result = check_rover_safety(route, 89.0, maximum_roughness=5.0, dem_path=dem_path)

    assert result.terrain_available
    assert result.coverage_fraction == 1.0
    assert result.max_slope_deg is not None and result.max_slope_deg <= 89.0
    assert result.mean_tri_m is not None and result.mean_tri_m > 5.0
    assert result.status is SafetyStatus.REVIEW_REQUIRED
    assert all(v.kind == "roughness" for s in result.segments for v in s.violations)


# ---------------------------------------------------------------------------
# 9. Landing-site evaluation
# ---------------------------------------------------------------------------


def test_landing_site_evaluation(dem_path):
    site = analyze_landing_site(
        dem_path, FIXTURE_LAT, FIXTURE_LON, radius_m=1000.0, maximum_slope_deg=10.0, min_flat_radius_m=200.0
    )
    assert site.terrain_available
    assert site.status in (SafetyStatus.PASS, SafetyStatus.FAIL, SafetyStatus.REVIEW_REQUIRED)
    if site.status is SafetyStatus.PASS:
        assert not site.violations
        assert site.coverage_fraction == 1.0
    assert "NOT a certified safety limit" in site.thresholds.statement
    assert "not mission approval" in site.disclaimer


def test_landing_site_review_required_for_insufficient_flat_radius_evidence(dem_path):
    """A 240 m/cell DEM cannot verify a 50 m flat radius (fewer than 4 cells fall inside it).
    With a lax slope limit so slope itself doesn't drive the status, this must be
    REVIEW_REQUIRED -- insufficient evidence, not an invented PASS or FAIL."""
    site = analyze_landing_site(
        dem_path, FIXTURE_LAT, FIXTURE_LON, radius_m=1000.0, maximum_slope_deg=90.0, min_flat_radius_m=50.0
    )
    assert site.terrain_available
    assert site.coverage_fraction == 1.0
    assert site.status is SafetyStatus.REVIEW_REQUIRED
    assert any("too coarse" in issue for issue in site.data_issues)


# ---------------------------------------------------------------------------
# 10. Missing-data behaviour
# ---------------------------------------------------------------------------


def test_missing_dem_never_produces_pass(dem_path):
    route = [(-89.9, 0.0), (-89.85, 30.0)]

    rover = check_rover_safety(route, 15.0, dem_path=None)
    assert not rover.terrain_available
    assert rover.status is not SafetyStatus.PASS

    landing = analyze_landing_site(None, FIXTURE_LAT, FIXTURE_LON)
    assert landing.status is not SafetyStatus.PASS
    assert not landing.rankable

    regions = find_safe_regions(None, FIXTURE_LAT, FIXTURE_LON, 3000.0, maximum_slope_deg=15.0)
    assert regions.outcome == "no_terrain_data"
    assert not regions.terrain_available

    stats = analyze_terrain_bbox(None, -90.0, -89.0, -1.0, 1.0)
    assert not stats.terrain_available

    nonexistent = Path(str(dem_path) + ".does-not-exist")
    rover2 = check_rover_safety(route, 15.0, dem_path=nonexistent)
    assert not rover2.terrain_available
    assert rover2.status is not SafetyStatus.PASS


# ---------------------------------------------------------------------------
# 11. Partially covered route / footprint behaviour
# ---------------------------------------------------------------------------


def test_partial_coverage_never_produces_pass(dem_path):
    # This route crosses the product's real projected-extent boundary (see module docstring):
    # genuinely, reproducibly partial coverage on real data, not a synthetic gap.
    route = [(EDGE_LAT_INSIDE, EDGE_LON), (EDGE_LAT_OUTSIDE, EDGE_LON)]
    rover = check_rover_safety(route, 15.0, dem_path=dem_path)
    assert rover.terrain_available
    assert 0.0 < rover.coverage_fraction < 1.0
    assert rover.status is not SafetyStatus.PASS
    assert rover.incomplete_segments

    mid_lat = 0.5 * (EDGE_LAT_INSIDE + EDGE_LAT_OUTSIDE)
    site = analyze_landing_site(
        dem_path, mid_lat, EDGE_LON, radius_m=30_000.0, maximum_slope_deg=25.0, min_flat_radius_m=100.0
    )
    assert site.terrain_available
    assert 0.0 < site.coverage_fraction < 1.0
    assert site.status is not SafetyStatus.PASS
    assert not site.rankable
    assert any("was measured" in issue for issue in site.data_issues)

    regions = find_safe_regions(
        dem_path, mid_lat, EDGE_LON, 30_000.0, maximum_slope_deg=25.0, min_area_m2=50_000.0
    )
    assert regions.terrain_available
    assert 0.0 < regions.assessed_fraction < 1.0


def test_far_outside_coverage_is_never_pass_and_not_no_terrain_data_confused(dem_path):
    """A valid lunar coordinate that the product does not cover at all (far from the pole).
    Distinguishes "no data at this location" (this test) from "no DEM at all" (test 10)."""
    site = analyze_landing_site(dem_path, FAR_OUTSIDE_LAT, FAR_OUTSIDE_LON, radius_m=1000.0)
    assert site.coverage_fraction == 0.0
    assert site.status is not SafetyStatus.PASS
    assert not site.rankable

    regions = find_safe_regions(dem_path, FAR_OUTSIDE_LAT, FAR_OUTSIDE_LON, 1000.0, maximum_slope_deg=15.0)
    assert regions.outcome == "no_terrain_data"
    assert not regions.terrain_available


# ---------------------------------------------------------------------------
# 12. Invalid-coordinate behaviour
# ---------------------------------------------------------------------------


def test_invalid_coordinates_are_rejected(dem_path):
    with pytest.raises(InvalidCoordinateError):
        check_rover_safety([(999.0, 0.0), (-89.9, 10.0)], 15.0, dem_path=dem_path)
    with pytest.raises(InvalidCoordinateError):
        analyze_landing_site(dem_path, 999.0, 0.0)
    with pytest.raises(InvalidCoordinateError):
        find_safe_regions(dem_path, -89.9, 999.0, 1000.0, maximum_slope_deg=15.0)
    with pytest.raises(InvalidCoordinateError):
        analyze_terrain_bbox(dem_path, math.nan, -89.0, -1.0, 1.0)


# ---------------------------------------------------------------------------
# 13. Threshold validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_threshold", [-5.0, 0.0, 91.0, float("nan"), float("inf")])
def test_invalid_thresholds_are_rejected(dem_path, bad_threshold):
    route = [(-89.9, 0.0), (-89.85, 30.0)]
    with pytest.raises(InvalidThresholdError):
        check_rover_safety(route, bad_threshold, dem_path=dem_path)
    with pytest.raises(InvalidThresholdError):
        analyze_landing_site(dem_path, FIXTURE_LAT, FIXTURE_LON, maximum_slope_deg=bad_threshold)
    with pytest.raises(InvalidThresholdError):
        find_safe_regions(
            dem_path, FIXTURE_LAT, FIXTURE_LON, 3000.0, maximum_slope_deg=bad_threshold
        )


def test_configured_threshold_is_echoed_back_unaltered(dem_path):
    """The implementation must report the caller's configured threshold exactly, never a
    threshold silently adjusted to make real terrain pass."""
    route = [(-89.9, 0.0), (-89.85, 30.0)]
    result = check_rover_safety(route, 12.34, dem_path=dem_path)
    assert result.configured_thresholds.maximum_slope_deg == 12.34

    site = analyze_landing_site(dem_path, FIXTURE_LAT, FIXTURE_LON, maximum_slope_deg=7.65)
    assert site.thresholds.maximum_slope_deg == 7.65

    regions = find_safe_regions(dem_path, FIXTURE_LAT, FIXTURE_LON, 3000.0, maximum_slope_deg=8.76)
    assert regions.maximum_slope_deg == 8.76


# ---------------------------------------------------------------------------
# 14. Repeated-run determinism
# ---------------------------------------------------------------------------


def test_repeated_run_determinism(dem_path):
    route = [(-89.9, 0.0), (-89.85, 30.0), (-89.8, 60.0)]

    r1 = check_rover_safety(route, 15.0, dem_path=dem_path)
    r2 = check_rover_safety(route, 15.0, dem_path=dem_path)
    assert r1.model_dump() == r2.model_dump()

    s1 = analyze_landing_site(dem_path, FIXTURE_LAT, FIXTURE_LON, radius_m=1000.0, maximum_slope_deg=10.0)
    s2 = analyze_landing_site(dem_path, FIXTURE_LAT, FIXTURE_LON, radius_m=1000.0, maximum_slope_deg=10.0)
    assert s1.model_dump() == s2.model_dump()

    g1 = find_safe_regions(
        dem_path, FIXTURE_LAT, FIXTURE_LON, 3000.0, maximum_slope_deg=25.0, min_area_m2=50_000.0
    )
    g2 = find_safe_regions(
        dem_path, FIXTURE_LAT, FIXTURE_LON, 3000.0, maximum_slope_deg=25.0, min_area_m2=50_000.0
    )
    assert g1.model_dump() == g2.model_dump()

    t1 = analyze_terrain_bbox(dem_path, -89.95, -89.85, -10.0, 10.0)
    t2 = analyze_terrain_bbox(dem_path, -89.95, -89.85, -10.0, 10.0)
    assert t1.model_dump() == t2.model_dump()
