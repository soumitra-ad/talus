"""Tests for deterministic landing-site analysis and comparison (Phase 5)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from terrain_agent.safety import (
    SafetyStatus,
    analyze_landing_site,
    compare_landing_candidates,
    rank_landing_sites,
)
from terrain_agent.terrain import (
    InvalidCoordinateError,
    InvalidSiteError,
    InvalidThresholdError,
    OversizedRequestError,
)

# Footprint and flat-radius settings that suit the 30 m cells of the synthetic DEM.
CFG = dict(radius_m=200.0, maximum_slope_deg=5.0, min_flat_radius_m=60.0)


def site(b, row, col):
    return b.latlon(row, col)


# ---------------------------------------------------------------------------
# Single-site measurements
# ---------------------------------------------------------------------------


def test_flat_site_passes_and_reports_all_measurable_properties(dem_builder):
    dem = dem_builder.geographic("flat.tif", dem_builder.flat(1234.0))
    lat, lon = site(dem_builder, 200, 200)
    result = analyze_landing_site(dem, lat, lon, site_id="flat", **CFG)

    assert result.status is SafetyStatus.PASS
    assert result.terrain_available
    assert result.coverage_fraction == 1.0
    assert result.cells_measured == result.footprint_cells > 100
    assert result.elevation_mean_m == 1234.0
    assert result.elevation_min_m == result.elevation_max_m == 1234.0
    assert result.mean_slope_deg == 0.0 and result.max_slope_deg == 0.0
    assert result.mean_tri_m == 0.0
    assert result.resolution_m == pytest.approx(30.3, abs=0.2)
    assert result.flat_radius_m == 200.0 and result.flat_radius_is_lower_bound
    assert result.flat_radius_requirement_met is True
    assert result.rankable and result.unrankable_reasons == []
    assert result.dataset is not None
    assert "NOT a certified safety limit" in result.thresholds.statement
    assert result.limitations


def test_steep_site_fails(dem_builder):
    dem = dem_builder.geographic("ramp.tif", dem_builder.ramp(10.0))
    lat, lon = site(dem_builder, 200, 200)
    result = analyze_landing_site(dem, lat, lon, **CFG)
    assert result.status is SafetyStatus.FAIL
    assert result.max_slope_deg == pytest.approx(10.0, abs=0.1)
    assert result.violations[0].kind == "slope"
    assert result.violations[0].threshold == 5.0


def test_rough_site_requires_review_when_a_roughness_limit_is_set(dem_builder):
    dem = dem_builder.geographic("rough.tif", dem_builder.rough(0.5))
    lat, lon = site(dem_builder, 200, 200)
    limited = analyze_landing_site(dem, lat, lon, maximum_roughness=0.3, **CFG)
    assert limited.status is SafetyStatus.REVIEW_REQUIRED
    assert limited.mean_tri_m > 0.3
    assert {v.kind for v in limited.violations} == {"roughness"}
    unlimited = analyze_landing_site(dem, lat, lon, **CFG)
    assert unlimited.status is SafetyStatus.PASS


def test_flat_radius_is_measured_to_the_nearest_steep_cell(dem_builder):
    elevation = dem_builder.flat()
    elevation[190:210, 200:260] += 200.0  # a raised block with steep faces
    dem = dem_builder.geographic("block.tif", elevation)
    lat, lon = site(dem_builder, 200, 150)
    result = analyze_landing_site(
        dem, lat, lon, radius_m=2000.0, maximum_slope_deg=5.0, min_flat_radius_m=1000.0
    )
    # The first steep cell is 49 cells (about 1486 m) east of the site.
    assert result.flat_radius_m == pytest.approx(1470.0, abs=50.0)
    assert not result.flat_radius_is_lower_bound
    assert result.flat_radius_requirement_met is True
    assert result.status is SafetyStatus.FAIL  # steep block faces lie inside the footprint


def test_site_near_the_dem_edge_has_incomplete_coverage(dem_builder):
    dem = dem_builder.geographic("flat.tif", dem_builder.flat())
    lat, lon = site(dem_builder, 200, 396)
    result = analyze_landing_site(dem, lat, lon, **CFG)
    assert result.status is SafetyStatus.REVIEW_REQUIRED
    assert 0.0 < result.coverage_fraction < 1.0
    assert result.cells_measured < result.footprint_cells
    assert not result.rankable
    assert any("Incomplete DEM coverage" in r for r in result.unrankable_reasons)


def test_site_outside_the_dem_is_unmeasured(dem_builder):
    dem = dem_builder.geographic("flat.tif", dem_builder.flat())
    result = analyze_landing_site(dem, 0.2, 20.0, **CFG)
    assert result.status is SafetyStatus.REVIEW_REQUIRED
    assert result.cells_measured == 0
    assert result.max_slope_deg is None and result.elevation_mean_m is None
    assert not result.rankable


def test_missing_dem_never_passes(dem_builder, tmp_path):
    lat, lon = site(dem_builder, 200, 200)
    for missing in (tmp_path / "nope.tif", None):
        result = analyze_landing_site(missing, lat, lon, **CFG)
        assert result.status is SafetyStatus.REVIEW_REQUIRED
        assert not result.terrain_available
        assert result.max_slope_deg is None and result.mean_tri_m is None
        assert result.flat_radius_m is None
        assert not result.rankable and result.unrankable_reasons
        assert tmp_path.name not in result.model_dump_json()


def test_coarse_resolution_cannot_verify_a_small_flat_radius(dem_builder):
    dem = dem_builder.geographic("flat.tif", dem_builder.flat())
    lat, lon = site(dem_builder, 200, 200)
    result = analyze_landing_site(dem, lat, lon, radius_m=200.0, min_flat_radius_m=10.0)
    assert result.status is SafetyStatus.REVIEW_REQUIRED
    assert any("too coarse" in issue for issue in result.data_issues)


def test_invalid_inputs_are_rejected(dem_builder):
    dem = dem_builder.geographic("flat.tif", dem_builder.flat())
    lat, lon = site(dem_builder, 200, 200)
    with pytest.raises(InvalidCoordinateError):
        analyze_landing_site(dem, 95.0, lon, **CFG)
    with pytest.raises(InvalidCoordinateError):
        analyze_landing_site(dem, lat, float("nan"), **CFG)
    with pytest.raises(InvalidCoordinateError):
        analyze_landing_site(dem, 120.0, 15.0, **CFG)  # probable swap
    with pytest.raises(OversizedRequestError):
        analyze_landing_site(dem, lat, lon, radius_m=60_000.0)
    for kwargs in (
        dict(radius_m=0.0),
        dict(radius_m=float("nan")),
        dict(maximum_slope_deg=0.0),
        dict(maximum_slope_deg=91.0),
        dict(maximum_roughness=-1.0),
        dict(radius_m=100.0, min_flat_radius_m=150.0),
    ):
        with pytest.raises(InvalidThresholdError):
            analyze_landing_site(dem, lat, lon, **{**CFG, **kwargs})


# ---------------------------------------------------------------------------
# Comparison and ranking
# ---------------------------------------------------------------------------


def three_zone_dem(b):
    """Flat to column 150, 2 degrees to column 300, then 10 degrees."""
    return b.geographic("zones.tif", b.bands([(150, 300, 2.0), (300, 400, 10.0)]))


def candidate_sites(b):
    def s(name, row, col):
        lat, lon = b.latlon(row, col)
        return {"id": name, "lat": lat, "lon": lon}

    return [
        s("C_steep", 200, 350),
        {"id": "D_outside", "lat": 0.2, "lon": 20.0},
        s("B_gentle", 200, 225),
        s("A_flat_2", 200, 80),
        s("A_flat", 200, 60),
    ]


def test_comparison_ranks_only_sites_with_complete_data(dem_builder):
    dem = three_zone_dem(dem_builder)
    result = compare_landing_candidates(dem, candidate_sites(dem_builder), **CFG)

    ranked = [(r.rank, r.analysis.site_id, r.analysis.status.value) for r in result.ranked]
    assert ranked == [
        (1, "A_flat", "PASS"),  # identical to A_flat_2, so the id breaks the tie
        (2, "A_flat_2", "PASS"),
        (3, "B_gentle", "PASS"),
        (4, "C_steep", "FAIL"),
    ]
    assert [u.site_id for u in result.unranked] == ["D_outside"]
    assert result.unranked[0].reasons
    assert result.unranked[0].status is SafetyStatus.REVIEW_REQUIRED
    assert any("not ranked" in w for w in result.warnings)
    assert "Lexicographic" in result.ranking_method
    assert len(result.sites) == 5


def test_no_ranking_is_invented_when_no_site_has_data(dem_builder, tmp_path):
    result = compare_landing_candidates(tmp_path / "missing.tif", candidate_sites(dem_builder), **CFG)
    assert result.ranked == []
    assert len(result.unranked) == 5
    assert all(u.status is SafetyStatus.REVIEW_REQUIRED for u in result.unranked)


def test_comparison_is_deterministic_and_order_independent(dem_builder):
    dem = three_zone_dem(dem_builder)
    sites = candidate_sites(dem_builder)
    first = compare_landing_candidates(dem, sites, **CFG)
    again = compare_landing_candidates(dem, sites, **CFG)
    shuffled = compare_landing_candidates(dem, list(reversed(sites)), **CFG)

    assert json.dumps(first.model_dump(mode="json"), sort_keys=True) == json.dumps(
        again.model_dump(mode="json"), sort_keys=True
    )
    assert [r.analysis.site_id for r in first.ranked] == [r.analysis.site_id for r in shuffled.ranked]


def test_different_dem_resolutions_are_flagged(dem_builder):
    fine = dem_builder.geographic("fine.tif", dem_builder.flat())
    coarse = dem_builder.geographic("coarse.tif", np.full((200, 200), 1500.0), res_deg=0.002)
    lat, lon = site(dem_builder, 200, 200)
    a = analyze_landing_site(fine, lat, lon, site_id="fine", radius_m=400.0, maximum_slope_deg=5.0, min_flat_radius_m=120.0)
    b = analyze_landing_site(coarse, lat, lon, site_id="coarse", radius_m=400.0, maximum_slope_deg=5.0, min_flat_radius_m=120.0)
    assert a.rankable and b.rankable
    comparison = rank_landing_sites([a, b])
    assert len(comparison.ranked) == 2
    assert any("different DEM" in w for w in comparison.warnings)
    assert any("resolution" in w for w in comparison.warnings)


def test_candidates_are_validated_before_any_analysis(dem_builder):
    dem = three_zone_dem(dem_builder)
    good = {"id": "ok", "lat": 0.2, "lon": 10.1}
    with pytest.raises(InvalidCoordinateError):
        compare_landing_candidates(dem, [good, {"id": "bad", "lat": 95.0, "lon": 10.0}], **CFG)
    with pytest.raises(InvalidSiteError):
        compare_landing_candidates(dem, [good, {"id": "ok", "lat": 0.2, "lon": 10.2}], **CFG)
    with pytest.raises(InvalidSiteError):
        compare_landing_candidates(dem, [], **CFG)
    with pytest.raises(InvalidSiteError):
        compare_landing_candidates(dem, "sites", **CFG)
    with pytest.raises(InvalidSiteError):
        compare_landing_candidates(dem, [{"id": "no_lon", "lat": 0.2}], **CFG)
    with pytest.raises(InvalidSiteError):
        compare_landing_candidates(dem, [(0.2, 10.1)] * 51, **CFG)
    with pytest.raises(InvalidCoordinateError):
        compare_landing_candidates(dem, [{"id": "s", "lat": "north", "lon": 10.1}], **CFG)


def test_coordinate_pairs_are_accepted_with_generated_ids(dem_builder):
    dem = three_zone_dem(dem_builder)
    result = compare_landing_candidates(dem, [dem_builder.latlon(200, 60), dem_builder.latlon(200, 80)], **CFG)
    assert [r.analysis.site_id for r in result.ranked] == ["site_1", "site_2"]
