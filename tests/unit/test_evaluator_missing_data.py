"""Regression tests: missing measurements must never produce PASS in the Phase 3 evaluator.

Before Phase 5 a traverse segment or landing site with no measurements at all was reported
as PASS. Callers that supply a subset of measurements are unaffected.
"""

from __future__ import annotations

from terrain_agent.safety.evaluator import (
    SafetyStatus,
    evaluate_landing_site,
    evaluate_traverse,
    evaluate_traverse_segment,
)


def test_segment_without_any_measurement_requires_review():
    result = evaluate_traverse_segment(0, 0.0, 0.0, 0.0, 1.0)
    assert result.status is SafetyStatus.REVIEW_REQUIRED
    assert result.data_missing


def test_traverse_of_empty_metrics_requires_review():
    metrics = [{"mean_slope_deg": None, "max_slope_deg": None, "mean_roughness_tri": None}] * 2
    report = evaluate_traverse([(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)], metrics)
    assert report.overall_status is SafetyStatus.REVIEW_REQUIRED
    assert [s.status for s in report.segments] == [SafetyStatus.REVIEW_REQUIRED] * 2


def test_landing_site_without_any_measurement_requires_review():
    report = evaluate_landing_site("X", 0.0, 0.0, elevation_m=1000.0)
    assert report.status is SafetyStatus.REVIEW_REQUIRED
    assert report.data_missing
    assert "No terrain measurements" in report.summary_text()


def test_measured_data_still_passes_and_fails_as_before():
    passing = evaluate_traverse_segment(0, 0.0, 0.0, 0.0, 1.0, max_slope_deg=3.0)
    assert passing.status is SafetyStatus.PASS and not passing.data_missing
    failing = evaluate_traverse_segment(0, 0.0, 0.0, 0.0, 1.0, max_slope_deg=30.0)
    assert failing.status is SafetyStatus.FAIL
    site = evaluate_landing_site("Y", 0.0, 0.0, max_slope_deg=2.0)
    assert site.status is SafetyStatus.PASS and not site.data_missing
