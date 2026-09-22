"""
Unit tests for terrain_agent.tools.ode_search and
terrain_agent.safety.evaluator.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# ODE search — offline unit tests (no network)
# ---------------------------------------------------------------------------


class TestODESearchValidation:
    """Validate bounding-box input checks in search_lunar_dem."""

    from terrain_agent.tools.ode_search import search_lunar_dem

    def test_infinite_lat_raises(self):
        from terrain_agent.tools.ode_search import search_lunar_dem
        import math
        with pytest.raises(ValueError, match="must be finite"):
            search_lunar_dem(
                min_lat=math.inf,
                max_lat=0.0,
                min_lon=0.0,
                max_lon=10.0,
            )

    def test_min_lat_greater_than_max_lat_raises(self):
        from terrain_agent.tools.ode_search import search_lunar_dem
        with pytest.raises(ValueError, match="must be"):
            search_lunar_dem(
                min_lat=10.0,
                max_lat=5.0,
                min_lon=0.0,
                max_lon=10.0,
            )

    def test_lat_out_of_range_raises(self):
        from terrain_agent.tools.ode_search import search_lunar_dem
        with pytest.raises(ValueError, match="Latitude out of range"):
            search_lunar_dem(
                min_lat=-95.0,
                max_lat=0.0,
                min_lon=0.0,
                max_lon=10.0,
            )

    def test_negative_longitude_normalised(self, monkeypatch):
        """Negative longitudes should be converted to 0-360 before ODE query."""
        from terrain_agent.tools.ode_search import search_lunar_dem

        captured_params: list[dict] = []

        def _mock_query_ode(client, ihid, iid, pt, dataset_label,
                            min_lat, max_lat, min_lon, max_lon, limit):
            captured_params.append({"min_lon": min_lon, "max_lon": max_lon})
            return []

        monkeypatch.setattr(
            "terrain_agent.tools.ode_search._query_ode", _mock_query_ode
        )
        import httpx
        monkeypatch.setattr(
            "terrain_agent.tools.ode_search.httpx.Client",
            lambda **kw: _FakeClientContextManager(kw),
        )

        search_lunar_dem(
            min_lat=-90.0,
            max_lat=-85.0,
            min_lon=-10.0,   # should become 350
            max_lon=10.0,    # should stay 10
        )
        assert captured_params[0]["min_lon"] == pytest.approx(350.0)
        assert captured_params[0]["max_lon"] == pytest.approx(10.0)


class _FakeClientContextManager:
    """Minimal httpx.Client context manager mock."""
    def __init__(self, kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


class TestODESearchStrategyOrder:
    """Verify preferred dataset appears first in strategy ordering."""

    def test_lola_preferred_first(self, monkeypatch):
        from terrain_agent.tools.ode_search import search_lunar_dem, DATASET_STRATEGIES

        call_order: list[str] = []

        def _mock_query_ode(client, ihid, iid, pt, dataset_label,
                            min_lat, max_lat, min_lon, max_lon, limit):
            call_order.append(pt)
            return []

        monkeypatch.setattr(
            "terrain_agent.tools.ode_search._query_ode", _mock_query_ode
        )
        monkeypatch.setattr(
            "terrain_agent.tools.ode_search.httpx.Client",
            lambda **kw: _FakeClientContextManager(kw),
        )

        search_lunar_dem(
            min_lat=-90.0, max_lat=-85.0,
            min_lon=0.0, max_lon=10.0,
            preferred_dataset="lola",
        )
        # First call must be GDRDEM (LOLA preference)
        assert call_order[0] == "GDRDEM"

    def test_sldem_preferred_first(self, monkeypatch):
        from terrain_agent.tools.ode_search import search_lunar_dem

        call_order: list[str] = []

        def _mock_query_ode(client, ihid, iid, pt, dataset_label,
                            min_lat, max_lat, min_lon, max_lon, limit):
            call_order.append(pt)
            return []

        monkeypatch.setattr(
            "terrain_agent.tools.ode_search._query_ode", _mock_query_ode
        )
        monkeypatch.setattr(
            "terrain_agent.tools.ode_search.httpx.Client",
            lambda **kw: _FakeClientContextManager(kw),
        )

        search_lunar_dem(
            min_lat=-90.0, max_lat=-85.0,
            min_lon=0.0, max_lon=10.0,
            preferred_dataset="sldem",
        )
        assert call_order[0] == "SLDEM"


class TestODESearchMaxResults:
    """Verify max_results cap is respected."""

    def test_max_results_respected(self, monkeypatch):
        from terrain_agent.tools.ode_search import search_lunar_dem, DEMProductRecord

        def _mock_query_ode(client, ihid, iid, pt, dataset_label,
                            min_lat, max_lat, min_lon, max_lon, limit):
            # Return 'limit' dummy records
            return [
                DEMProductRecord(
                    product_id=f"p{i}",
                    mission="LRO", instrument="LOLA",
                    dataset="test", product_type=pt,
                    file_url="", files_page_url="",
                    min_lat=-90.0, max_lat=-85.0,
                    min_lon=0.0, max_lon=10.0,
                    center_lat=-87.5, center_lon=5.0,
                )
                for i in range(limit)
            ]

        monkeypatch.setattr(
            "terrain_agent.tools.ode_search._query_ode", _mock_query_ode
        )
        monkeypatch.setattr(
            "terrain_agent.tools.ode_search.httpx.Client",
            lambda **kw: _FakeClientContextManager(kw),
        )

        results = search_lunar_dem(
            min_lat=-90.0, max_lat=-85.0,
            min_lon=0.0, max_lon=10.0,
            max_results=3,
        )
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Safety evaluator — unit tests
# ---------------------------------------------------------------------------


class TestSafetyStatus:
    def test_pass_value(self):
        from terrain_agent.safety.evaluator import SafetyStatus
        assert SafetyStatus.PASS.value == "PASS"

    def test_review_required_value(self):
        from terrain_agent.safety.evaluator import SafetyStatus
        assert SafetyStatus.REVIEW_REQUIRED.value == "REVIEW_REQUIRED"

    def test_fail_value(self):
        from terrain_agent.safety.evaluator import SafetyStatus
        assert SafetyStatus.FAIL.value == "FAIL"


class TestEvaluateTraverseSegment:
    def test_pass_when_all_within_threshold(self):
        from terrain_agent.safety.evaluator import (
            evaluate_traverse_segment, TraverseSafetyConfig, SafetyStatus
        )
        config = TraverseSafetyConfig(max_slope_deg=15.0, max_roughness_tri=0.5)
        result = evaluate_traverse_segment(
            segment_index=0,
            start_lat=-89.9, start_lon=0.0,
            end_lat=-89.5, end_lon=0.0,
            mean_slope_deg=8.0,
            max_slope_deg=12.0,
            mean_roughness_tri=0.2,
            config=config,
        )
        assert result.status == SafetyStatus.PASS
        assert result.violations == []

    def test_fail_when_slope_exceeds_threshold(self):
        from terrain_agent.safety.evaluator import (
            evaluate_traverse_segment, TraverseSafetyConfig, SafetyStatus
        )
        config = TraverseSafetyConfig(max_slope_deg=15.0)
        result = evaluate_traverse_segment(
            segment_index=0,
            start_lat=-89.9, start_lon=0.0,
            end_lat=-89.5, end_lon=0.0,
            max_slope_deg=20.0,
            config=config,
        )
        assert result.status == SafetyStatus.FAIL
        assert any(v.metric == "slope_deg" for v in result.violations)

    def test_review_required_when_only_roughness_violated(self):
        from terrain_agent.safety.evaluator import (
            evaluate_traverse_segment, TraverseSafetyConfig, SafetyStatus
        )
        config = TraverseSafetyConfig(max_slope_deg=15.0, max_roughness_tri=0.3)
        result = evaluate_traverse_segment(
            segment_index=0,
            start_lat=-89.9, start_lon=0.0,
            end_lat=-89.5, end_lon=0.0,
            max_slope_deg=10.0,   # under limit
            mean_roughness_tri=0.6,  # over limit
            config=config,
        )
        assert result.status == SafetyStatus.REVIEW_REQUIRED

    def test_violation_describes_correctly(self):
        from terrain_agent.safety.evaluator import (
            evaluate_traverse_segment, TraverseSafetyConfig
        )
        config = TraverseSafetyConfig(max_slope_deg=15.0)
        result = evaluate_traverse_segment(
            segment_index=2,
            start_lat=-89.9, start_lon=0.0,
            end_lat=-89.5, end_lon=0.0,
            max_slope_deg=25.0,
            config=config,
        )
        desc = result.violations[0].describe()
        assert "slope_deg" in desc
        assert "25" in desc
        assert "15" in desc
        assert "segment 2" in desc


class TestEvaluateTraverse:
    def test_requires_at_least_two_waypoints(self):
        from terrain_agent.safety.evaluator import evaluate_traverse
        with pytest.raises(ValueError, match="at least 2 waypoints"):
            evaluate_traverse(
                waypoints=[(-89.9, 0.0)],
                segment_metrics=[],
            )

    def test_segment_count_mismatch_raises(self):
        from terrain_agent.safety.evaluator import evaluate_traverse
        with pytest.raises(ValueError, match="Expected 1 segment"):
            evaluate_traverse(
                waypoints=[(-89.9, 0.0), (-89.5, 0.0)],
                segment_metrics=[],
            )

    def test_all_pass_gives_overall_pass(self):
        from terrain_agent.safety.evaluator import (
            evaluate_traverse, TraverseSafetyConfig, SafetyStatus
        )
        config = TraverseSafetyConfig(max_slope_deg=15.0)
        report = evaluate_traverse(
            waypoints=[(-89.9, 0.0), (-89.5, 0.0), (-89.0, 0.0)],
            segment_metrics=[
                {"mean_slope_deg": 5.0, "max_slope_deg": 8.0, "data_source": "test"},
                {"mean_slope_deg": 6.0, "max_slope_deg": 9.0, "data_source": "test"},
            ],
            config=config,
        )
        assert report.overall_status == SafetyStatus.PASS
        assert report.passing_segments == 2
        assert report.failing_segments == 0

    def test_one_fail_gives_overall_fail(self):
        from terrain_agent.safety.evaluator import (
            evaluate_traverse, TraverseSafetyConfig, SafetyStatus
        )
        config = TraverseSafetyConfig(max_slope_deg=15.0)
        report = evaluate_traverse(
            waypoints=[(-89.9, 0.0), (-89.5, 0.0), (-89.0, 0.0)],
            segment_metrics=[
                {"mean_slope_deg": 5.0, "max_slope_deg": 8.0},
                {"mean_slope_deg": 20.0, "max_slope_deg": 25.0},  # FAIL
            ],
            config=config,
        )
        assert report.overall_status == SafetyStatus.FAIL
        assert report.failing_segments == 1

    def test_report_summary_contains_disclaimer(self):
        from terrain_agent.safety.evaluator import evaluate_traverse, TraverseSafetyConfig
        config = TraverseSafetyConfig(max_slope_deg=15.0)
        report = evaluate_traverse(
            waypoints=[(-89.9, 0.0), (-89.5, 0.0)],
            segment_metrics=[{"mean_slope_deg": 5.0}],
            config=config,
        )
        summary = report.summary_text()
        assert "15.0" in summary
        assert "NOT" in summary


class TestEvaluateLandingSite:
    def test_pass_below_thresholds(self):
        from terrain_agent.safety.evaluator import (
            evaluate_landing_site, LandingSiteConfig, SafetyStatus
        )
        config = LandingSiteConfig(max_slope_deg=5.0, max_roughness_tri=0.3)
        report = evaluate_landing_site(
            "site_A", lat=-89.9, lon=0.0,
            mean_slope_deg=2.0, max_slope_deg=4.0,
            mean_roughness_tri=0.1,
            config=config,
        )
        assert report.status == SafetyStatus.PASS

    def test_fail_above_slope(self):
        from terrain_agent.safety.evaluator import (
            evaluate_landing_site, LandingSiteConfig, SafetyStatus
        )
        config = LandingSiteConfig(max_slope_deg=5.0)
        report = evaluate_landing_site(
            "site_B", lat=-89.9, lon=0.0,
            max_slope_deg=8.0,
            config=config,
        )
        assert report.status == SafetyStatus.FAIL

    def test_disclaimer_in_summary(self):
        from terrain_agent.safety.evaluator import evaluate_landing_site, LandingSiteConfig
        config = LandingSiteConfig(max_slope_deg=5.0)
        report = evaluate_landing_site(
            "site_C", lat=-89.9, lon=0.0, config=config
        )
        assert "NOT" in report.disclaimer


class TestCompareLandingSites:
    def test_pass_sites_ranked_first(self):
        from terrain_agent.safety.evaluator import (
            evaluate_landing_site, LandingSiteConfig,
            compare_landing_sites, SafetyStatus
        )
        config = LandingSiteConfig(max_slope_deg=5.0)
        site_pass = evaluate_landing_site("pass", -89.9, 0.0, max_slope_deg=3.0, config=config)
        site_fail = evaluate_landing_site("fail", -89.9, 0.0, max_slope_deg=8.0, config=config)
        ranked = compare_landing_sites([site_fail, site_pass])
        assert ranked[0].status == SafetyStatus.PASS
        assert ranked[1].status == SafetyStatus.FAIL
