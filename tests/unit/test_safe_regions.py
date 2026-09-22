"""Tests for deterministic safe-region detection (Phase 5)."""

from __future__ import annotations

import json

import pytest

from terrain_agent.safety import find_safe_regions
from terrain_agent.terrain import InvalidCoordinateError, InvalidThresholdError, OversizedRequestError

BOUNDARY_LON = 10.0 + 200 * 0.001  # longitude of column 200 in the default DEM


def centre(b, row=200, col=200):
    return b.latlon(row, col)


def test_only_the_flat_half_is_reported_as_safe(dem_builder):
    dem = dem_builder.geographic("half.tif", dem_builder.bands([(200, 400, 20.0)]))
    lat, lon = centre(dem_builder)
    result = find_safe_regions(dem, lat, lon, 2000.0, maximum_slope_deg=10.0)

    assert result.outcome == "regions_found"
    assert result.terrain_available
    assert result.assessed_fraction == 1.0
    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.region_id == "R1"
    assert region.centroid_lon < BOUNDARY_LON and region.best_point_lon < BOUNDARY_LON
    assert region.max_slope_deg <= 10.0
    # A half disc of radius 2000 m: largest inscribed circle is about half the radius.
    assert 850.0 < region.largest_inscribed_circle_radius_m < 1050.0
    assert region.area_m2 > 0.4 * 0.5 * 3.14159 * 2000.0**2
    assert 0.45 < result.safe_fraction_of_assessed < 0.55
    assert "NOT a certified safety limit" in result.threshold_statement
    assert result.limitations


def test_all_steep_terrain_reports_no_safe_region_among_measured_cells(dem_builder):
    dem = dem_builder.geographic("steep.tif", dem_builder.ramp(20.0))
    lat, lon = centre(dem_builder)
    result = find_safe_regions(dem, lat, lon, 1000.0, maximum_slope_deg=10.0)
    assert result.outcome == "no_safe_region_in_assessed_area"
    assert result.terrain_available
    assert result.regions == []
    assert result.safe_fraction_of_assessed == 0.0


def test_missing_terrain_is_not_reported_as_unsafe_terrain(dem_builder, tmp_path):
    lat, lon = centre(dem_builder)
    for missing in (tmp_path / "nope.tif", None):
        result = find_safe_regions(missing, lat, lon, 1000.0, maximum_slope_deg=10.0)
        assert result.outcome == "no_terrain_data"
        assert not result.terrain_available
        assert result.regions == []
        assert result.safe_fraction_of_assessed is None
        assert tmp_path.name not in result.model_dump_json()


def test_area_entirely_outside_the_dem_has_no_terrain_data(dem_builder):
    dem = dem_builder.geographic("flat.tif", dem_builder.flat())
    result = find_safe_regions(dem, 0.2, 20.0, 500.0, maximum_slope_deg=10.0)
    assert result.outcome == "no_terrain_data"
    assert result.cells_measured == 0


def test_unmeasured_cells_are_never_treated_as_safe(dem_builder):
    elevation = dem_builder.flat()
    elevation[190:211, 190:211] = -9999.0  # nodata block of 21 x 21 cells
    dem = dem_builder.geographic("hole.tif", elevation)
    lat, lon = centre(dem_builder)
    result = find_safe_regions(dem, lat, lon, 1000.0, maximum_slope_deg=10.0)

    assert result.outcome == "regions_found"
    assert result.assessed_fraction < 1.0
    assert result.footprint_cells - result.cells_measured >= 21 * 21
    assert result.safe_fraction_of_assessed == 1.0  # every measured cell is flat
    assert result.regions[0].cells == result.cells_measured  # and none of the hole is included
    assert any("not treated as safe" in w for w in result.warnings)


def test_minimum_area_filters_small_regions(dem_builder):
    # A flat strip about 9 cells wide inside otherwise steep terrain.
    dem = dem_builder.geographic("strip.tif", dem_builder.bands([(0, 190, 20.0), (202, 400, 20.0)]))
    lat, lon = centre(dem_builder, 200, 196)
    kept = find_safe_regions(dem, lat, lon, 1000.0, maximum_slope_deg=10.0, min_area_m2=10_000.0)
    dropped = find_safe_regions(dem, lat, lon, 1000.0, maximum_slope_deg=10.0, min_area_m2=1.0e7)
    assert kept.outcome == "regions_found" and len(kept.regions) == 1
    assert dropped.outcome == "no_safe_region_in_assessed_area" and dropped.regions == []


def test_roughness_limit_removes_rough_cells(dem_builder):
    dem = dem_builder.geographic("rough.tif", dem_builder.rough(0.5))
    lat, lon = centre(dem_builder)
    # About 7% of cells are individually smooth enough, scattered in small clusters. The
    # minimum area is raised above the largest chance cluster (12 cells, about 11,000 m2).
    area = 100_000.0
    unlimited = find_safe_regions(dem, lat, lon, 1000.0, maximum_slope_deg=45.0, min_area_m2=area)
    limited = find_safe_regions(
        dem, lat, lon, 1000.0, maximum_slope_deg=45.0, maximum_roughness=0.3, min_area_m2=area
    )
    assert unlimited.outcome == "regions_found"
    assert limited.safe_fraction_of_assessed < 0.15
    assert limited.outcome == "no_safe_region_in_assessed_area"


def test_regions_are_ranked_and_truncated_deterministically(dem_builder):
    dem = dem_builder.geographic("three.tif", dem_builder.bands([(120, 140, 20.0), (260, 280, 20.0)]))
    lat, lon = centre(dem_builder)
    full = find_safe_regions(dem, lat, lon, 3000.0, maximum_slope_deg=10.0)
    one = find_safe_regions(dem, lat, lon, 3000.0, maximum_slope_deg=10.0, max_regions=1)

    assert full.regions_total_found >= 2
    radii = [r.largest_inscribed_circle_radius_m for r in full.regions]
    assert radii == sorted(radii, reverse=True)
    assert len(one.regions) == 1 and one.regions_truncated
    assert one.regions[0] == full.regions[0]
    assert one.regions_total_found == full.regions_total_found


def test_results_are_identical_across_runs(dem_builder):
    dem = dem_builder.geographic("three.tif", dem_builder.bands([(120, 140, 20.0), (260, 280, 20.0)]))
    lat, lon = centre(dem_builder)
    dumps = [
        json.dumps(
            find_safe_regions(dem, lat, lon, 3000.0, maximum_slope_deg=10.0).model_dump(mode="json"),
            sort_keys=True,
        )
        for _ in range(3)
    ]
    assert dumps[0] == dumps[1] == dumps[2]


def test_invalid_inputs_are_rejected(dem_builder):
    dem = dem_builder.geographic("flat.tif", dem_builder.flat())
    lat, lon = centre(dem_builder)
    with pytest.raises(InvalidCoordinateError):
        find_safe_regions(dem, 95.0, lon, 500.0, maximum_slope_deg=10.0)
    with pytest.raises(OversizedRequestError):
        find_safe_regions(dem, lat, lon, 60_000.0, maximum_slope_deg=10.0)
    for kwargs in (
        dict(radius_m=0.0),
        dict(radius_m=float("nan")),
        dict(maximum_slope_deg=0.0),
        dict(maximum_slope_deg=91.0),
        dict(maximum_roughness=0.0),
        dict(min_area_m2=-1.0),
        dict(max_regions=0),
        dict(max_regions=101),
        dict(max_regions=True),
    ):
        args = dict(radius_m=500.0, maximum_slope_deg=10.0)
        args.update(kwargs)
        radius = args.pop("radius_m")
        with pytest.raises(InvalidThresholdError):
            find_safe_regions(dem, lat, lon, radius, **args)
