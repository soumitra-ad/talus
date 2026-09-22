"""
Deterministic safety evaluation for traverse routes and landing-site candidates.

All numerical thresholds are CONFIGURED mission parameters, not universal
scientific limits. Output statuses are PASS, REVIEW_REQUIRED, or FAIL only.

Research/Demo Disclaimer
------------------------
TALUS is a research and demonstration system. These evaluations must NOT
be used for certified flight safety, operational landing approval,
autonomous spacecraft control, or guaranteed rover safety.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status enumeration
# ---------------------------------------------------------------------------


class SafetyStatus(str, Enum):
    """Allowed safety assessment output values.

    Per TALUS operational directives: only these three statuses may appear
    in any safety report. They are determined solely by configured thresholds.
    """

    PASS = "PASS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Shared structures
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    """A single threshold violation with location context."""

    metric: str
    """Name of the violated metric (e.g. 'slope_deg', 'roughness_tri')."""

    value: float
    """Measured value that caused the violation."""

    threshold: float
    """The configured threshold that was exceeded."""

    segment_index: int | None = None
    """Route segment index (for traverse evaluations), or None."""

    lat: float | None = None
    """Approximate latitude of the violation, if available."""

    lon: float | None = None
    """Approximate longitude of the violation, if available."""

    def describe(self) -> str:
        """Human-readable violation description."""
        loc = ""
        if self.segment_index is not None:
            loc = f" at segment {self.segment_index}"
        if self.lat is not None and self.lon is not None:
            loc += f" ({self.lat:.4f}°, {self.lon:.4f}°)"
        return (
            f"{self.metric}={self.value:.4g} exceeds configured threshold "
            f"{self.threshold:.4g}{loc}"
        )


# ---------------------------------------------------------------------------
# Traverse safety evaluation
# ---------------------------------------------------------------------------


@dataclass
class TraverseSafetyConfig:
    """Configured safety thresholds for a rover traverse mission."""

    max_slope_deg: float = 15.0
    """Configured maximum allowable traverse slope in degrees."""

    max_roughness_tri: float = 0.5
    """Configured maximum allowable Terrain Roughness Index (TRI)."""

    max_elevation_change_m: float | None = None
    """Optional configured maximum elevation change per segment in metres."""

    disclaimer: str = (
        "Configured analysis threshold: {max_slope_deg}° slope, "
        "{max_roughness_tri} TRI. "
        "NOT a certified safety limit."
    )

    def format_disclaimer(self) -> str:
        return self.disclaimer.format(
            max_slope_deg=self.max_slope_deg,
            max_roughness_tri=self.max_roughness_tri,
        )


@dataclass
class SegmentResult:
    """Safety assessment for a single route segment."""

    segment_index: int
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    mean_slope_deg: float | None
    max_slope_deg: float | None
    mean_roughness_tri: float | None
    elevation_change_m: float | None
    status: SafetyStatus
    violations: list[Violation] = field(default_factory=list)
    data_source: str = "unknown"
    resolution_m: float | None = None
    data_missing: bool = False
    """True when no slope or roughness measurement was supplied. Status is then never PASS."""


@dataclass
class TraverseSafetyReport:
    """Complete traverse safety evaluation report."""

    overall_status: SafetyStatus
    """Aggregate status across all segments."""

    segments: list[SegmentResult]
    """Per-segment results."""

    total_segments: int
    """Total number of evaluated segments."""

    passing_segments: int
    """Segments with PASS status."""

    review_segments: int
    """Segments with REVIEW_REQUIRED status."""

    failing_segments: int
    """Segments with FAIL status."""

    config: TraverseSafetyConfig
    """The configured thresholds used for this evaluation."""

    disclaimer: str = ""
    """Mandatory disclaimer text."""

    all_violations: list[Violation] = field(default_factory=list)
    """Flattened list of all violations across all segments."""

    data_sources: list[str] = field(default_factory=list)
    """DEM product IDs or dataset names used in this evaluation."""

    def summary_text(self) -> str:
        """Generate a human-readable report summary."""
        lines = [
            "=== TALUS Traverse Safety Report ===",
            f"Overall Status: {self.overall_status.value}",
            f"Segments evaluated: {self.total_segments}",
            f"  PASS: {self.passing_segments}",
            f"  REVIEW_REQUIRED: {self.review_segments}",
            f"  FAIL: {self.failing_segments}",
            "",
            f"Configured analysis threshold: {self.config.max_slope_deg}°",
            "",
        ]
        if self.all_violations:
            lines.append("Violations:")
            for v in self.all_violations:
                lines.append(f"  - {v.describe()}")
        else:
            lines.append("No threshold violations detected.")
        lines += ["", self.disclaimer]
        return "\n".join(lines)


def evaluate_traverse_segment(
    segment_index: int,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    *,
    mean_slope_deg: float | None = None,
    max_slope_deg: float | None = None,
    mean_roughness_tri: float | None = None,
    elevation_change_m: float | None = None,
    config: TraverseSafetyConfig | None = None,
    data_source: str = "unknown",
    resolution_m: float | None = None,
) -> SegmentResult:
    """
    Evaluate a single traverse segment against configured safety thresholds.

    Parameters
    ----------
    segment_index:
        Zero-based index of this segment within the overall route.
    start_lat, start_lon:
        Start waypoint coordinates (decimal degrees).
    end_lat, end_lon:
        End waypoint coordinates (decimal degrees).
    mean_slope_deg:
        Mean slope over the segment in degrees (from deterministic tool).
    max_slope_deg:
        Maximum slope value within the segment (from deterministic tool).
    mean_roughness_tri:
        Mean Terrain Roughness Index over the segment (from deterministic tool).
    elevation_change_m:
        Total elevation change over the segment in metres.
    config:
        Configured thresholds. Uses defaults if None.
    data_source:
        DEM product ID or dataset label for provenance reporting.
    resolution_m:
        DEM resolution in metres at this location.

    Returns
    -------
    SegmentResult
        Assessment with status PASS, REVIEW_REQUIRED, or FAIL.
    """
    if config is None:
        config = TraverseSafetyConfig()

    violations: list[Violation] = []

    # Check slope (use max_slope if available, otherwise mean_slope)
    slope_check_value = max_slope_deg if max_slope_deg is not None else mean_slope_deg
    if slope_check_value is not None:
        if slope_check_value > config.max_slope_deg:
            violations.append(
                Violation(
                    metric="slope_deg",
                    value=slope_check_value,
                    threshold=config.max_slope_deg,
                    segment_index=segment_index,
                    lat=(start_lat + end_lat) / 2,
                    lon=(start_lon + end_lon) / 2,
                )
            )

    # Check roughness
    if mean_roughness_tri is not None:
        if mean_roughness_tri > config.max_roughness_tri:
            violations.append(
                Violation(
                    metric="roughness_tri",
                    value=mean_roughness_tri,
                    threshold=config.max_roughness_tri,
                    segment_index=segment_index,
                    lat=(start_lat + end_lat) / 2,
                    lon=(start_lon + end_lon) / 2,
                )
            )

    # Check optional elevation change
    if elevation_change_m is not None and config.max_elevation_change_m is not None:
        if abs(elevation_change_m) > config.max_elevation_change_m:
            violations.append(
                Violation(
                    metric="elevation_change_m",
                    value=abs(elevation_change_m),
                    threshold=config.max_elevation_change_m,
                    segment_index=segment_index,
                    lat=(start_lat + end_lat) / 2,
                    lon=(start_lon + end_lon) / 2,
                )
            )

    # Missing measurements must never produce PASS.
    no_measurements = slope_check_value is None and mean_roughness_tri is None

    # Determine segment status
    if not violations:
        status = SafetyStatus.REVIEW_REQUIRED if no_measurements else SafetyStatus.PASS
    else:
        # Any slope violation is a FAIL; roughness-only violation is REVIEW_REQUIRED
        slope_violated = any(v.metric == "slope_deg" for v in violations)
        status = SafetyStatus.FAIL if slope_violated else SafetyStatus.REVIEW_REQUIRED

    return SegmentResult(
        segment_index=segment_index,
        start_lat=start_lat,
        start_lon=start_lon,
        end_lat=end_lat,
        end_lon=end_lon,
        mean_slope_deg=mean_slope_deg,
        max_slope_deg=max_slope_deg,
        mean_roughness_tri=mean_roughness_tri,
        elevation_change_m=elevation_change_m,
        status=status,
        violations=violations,
        data_source=data_source,
        resolution_m=resolution_m,
        data_missing=no_measurements,
    )


def evaluate_traverse(
    waypoints: list[tuple[float, float]],
    segment_metrics: list[dict[str, Any]],
    config: TraverseSafetyConfig | None = None,
) -> TraverseSafetyReport:
    """
    Evaluate a complete rover traverse route for safety compliance.

    Parameters
    ----------
    waypoints:
        Ordered list of (lat, lon) tuples defining the route.
        Must have at least 2 waypoints.
    segment_metrics:
        List of metric dictionaries, one per segment (len = len(waypoints) - 1).
        Each dict may contain keys: mean_slope_deg, max_slope_deg,
        mean_roughness_tri, elevation_change_m, data_source, resolution_m.
    config:
        Configured safety thresholds. Uses defaults if None.

    Returns
    -------
    TraverseSafetyReport

    Raises
    ------
    ValueError
        If waypoints or segment_metrics are malformed.
    """
    if config is None:
        config = TraverseSafetyConfig()

    n_waypoints = len(waypoints)
    if n_waypoints < 2:
        raise ValueError(
            f"A traverse requires at least 2 waypoints, got {n_waypoints}."
        )

    expected_segments = n_waypoints - 1
    if len(segment_metrics) != expected_segments:
        raise ValueError(
            f"Expected {expected_segments} segment metric dicts for "
            f"{n_waypoints} waypoints, got {len(segment_metrics)}."
        )

    segments: list[SegmentResult] = []
    all_violations: list[Violation] = []
    data_sources: list[str] = []

    for i in range(expected_segments):
        start_lat, start_lon = waypoints[i]
        end_lat, end_lon = waypoints[i + 1]
        m = segment_metrics[i]

        result = evaluate_traverse_segment(
            segment_index=i,
            start_lat=start_lat,
            start_lon=start_lon,
            end_lat=end_lat,
            end_lon=end_lon,
            mean_slope_deg=m.get("mean_slope_deg"),
            max_slope_deg=m.get("max_slope_deg"),
            mean_roughness_tri=m.get("mean_roughness_tri"),
            elevation_change_m=m.get("elevation_change_m"),
            config=config,
            data_source=m.get("data_source", "unknown"),
            resolution_m=m.get("resolution_m"),
        )
        segments.append(result)
        all_violations.extend(result.violations)

        src = m.get("data_source", "unknown")
        if src not in data_sources:
            data_sources.append(src)

    # Aggregate status
    statuses = {s.status for s in segments}
    if SafetyStatus.FAIL in statuses:
        overall = SafetyStatus.FAIL
    elif SafetyStatus.REVIEW_REQUIRED in statuses:
        overall = SafetyStatus.REVIEW_REQUIRED
    else:
        overall = SafetyStatus.PASS

    return TraverseSafetyReport(
        overall_status=overall,
        segments=segments,
        total_segments=expected_segments,
        passing_segments=sum(1 for s in segments if s.status == SafetyStatus.PASS),
        review_segments=sum(
            1 for s in segments if s.status == SafetyStatus.REVIEW_REQUIRED
        ),
        failing_segments=sum(1 for s in segments if s.status == SafetyStatus.FAIL),
        config=config,
        disclaimer=config.format_disclaimer(),
        all_violations=all_violations,
        data_sources=data_sources,
    )


# ---------------------------------------------------------------------------
# Landing-site safety evaluation
# ---------------------------------------------------------------------------


@dataclass
class LandingSiteConfig:
    """Configured safety thresholds for a landing site candidate evaluation."""

    max_slope_deg: float = 5.0
    """Configured maximum allowable landing zone slope in degrees."""

    max_roughness_tri: float = 0.3
    """Configured maximum allowable TRI for a landing zone."""

    min_flat_radius_m: float = 10.0
    """Configured minimum flat-area radius in metres around the landing point."""

    disclaimer: str = (
        "Configured analysis threshold: {max_slope_deg}° slope, "
        "{max_roughness_tri} TRI, {min_flat_radius_m}m flat radius. "
        "NOT a certified safety limit."
    )

    def format_disclaimer(self) -> str:
        return self.disclaimer.format(
            max_slope_deg=self.max_slope_deg,
            max_roughness_tri=self.max_roughness_tri,
            min_flat_radius_m=self.min_flat_radius_m,
        )


@dataclass
class LandingSiteReport:
    """Safety evaluation for a single landing site candidate."""

    site_id: str
    lat: float
    lon: float
    elevation_m: float | None
    mean_slope_deg: float | None
    max_slope_deg: float | None
    mean_roughness_tri: float | None
    status: SafetyStatus
    violations: list[Violation] = field(default_factory=list)
    config: LandingSiteConfig = field(default_factory=LandingSiteConfig)
    data_source: str = "unknown"
    resolution_m: float | None = None
    disclaimer: str = ""
    data_missing: bool = False

    def summary_text(self) -> str:
        lines = [
            f"=== Landing Site {self.site_id} ===",
            f"Location: {self.lat:.4f}°, {self.lon:.4f}°",
        ]
        if self.elevation_m is not None:
            lines.append(f"Elevation: {self.elevation_m:.2f} m")
        if self.mean_slope_deg is not None:
            lines.append(f"Mean slope: {self.mean_slope_deg:.2f}°")
        if self.max_slope_deg is not None:
            lines.append(f"Max slope: {self.max_slope_deg:.2f}°")
        if self.mean_roughness_tri is not None:
            lines.append(f"Mean roughness (TRI): {self.mean_roughness_tri:.4f}")
        if self.data_missing:
            lines.append("No terrain measurements were available for this site.")
        lines += [
            f"Status: {self.status.value}",
            f"Data source: {self.data_source}",
        ]
        if self.resolution_m is not None:
            lines.append(f"Resolution: {self.resolution_m:.1f} m/px")
        if self.violations:
            lines.append("Violations:")
            for v in self.violations:
                lines.append(f"  - {v.describe()}")
        lines += ["", self.disclaimer]
        return "\n".join(lines)


def evaluate_landing_site(
    site_id: str,
    lat: float,
    lon: float,
    *,
    elevation_m: float | None = None,
    mean_slope_deg: float | None = None,
    max_slope_deg: float | None = None,
    mean_roughness_tri: float | None = None,
    config: LandingSiteConfig | None = None,
    data_source: str = "unknown",
    resolution_m: float | None = None,
) -> LandingSiteReport:
    """
    Evaluate a landing site candidate against configured safety thresholds.

    All numerical inputs must originate from deterministic terrain tools —
    never from LLM reasoning or approximation.

    Parameters
    ----------
    site_id:
        Unique identifier for this candidate site.
    lat, lon:
        Candidate site coordinates (decimal degrees).
    elevation_m:
        Elevation in metres above the lunar datum.
    mean_slope_deg:
        Mean slope within the landing zone radius.
    max_slope_deg:
        Maximum slope within the landing zone radius.
    mean_roughness_tri:
        Mean TRI within the landing zone radius.
    config:
        Configured thresholds. Uses defaults if None.
    data_source:
        DEM product ID or dataset label.
    resolution_m:
        DEM resolution in metres.

    Returns
    -------
    LandingSiteReport
    """
    if config is None:
        config = LandingSiteConfig()

    violations: list[Violation] = []

    slope_check = max_slope_deg if max_slope_deg is not None else mean_slope_deg
    if slope_check is not None and slope_check > config.max_slope_deg:
        violations.append(
            Violation(
                metric="slope_deg",
                value=slope_check,
                threshold=config.max_slope_deg,
                lat=lat,
                lon=lon,
            )
        )

    if mean_roughness_tri is not None and mean_roughness_tri > config.max_roughness_tri:
        violations.append(
            Violation(
                metric="roughness_tri",
                value=mean_roughness_tri,
                threshold=config.max_roughness_tri,
                lat=lat,
                lon=lon,
            )
        )

    # Missing measurements must never produce PASS.
    no_measurements = slope_check is None and mean_roughness_tri is None

    if not violations:
        status = SafetyStatus.REVIEW_REQUIRED if no_measurements else SafetyStatus.PASS
    else:
        slope_violated = any(v.metric == "slope_deg" for v in violations)
        status = SafetyStatus.FAIL if slope_violated else SafetyStatus.REVIEW_REQUIRED

    return LandingSiteReport(
        site_id=site_id,
        lat=lat,
        lon=lon,
        elevation_m=elevation_m,
        mean_slope_deg=mean_slope_deg,
        max_slope_deg=max_slope_deg,
        mean_roughness_tri=mean_roughness_tri,
        status=status,
        violations=violations,
        config=config,
        data_source=data_source,
        resolution_m=resolution_m,
        disclaimer=config.format_disclaimer(),
        data_missing=no_measurements,
    )


def compare_landing_sites(
    sites: list[LandingSiteReport],
) -> list[LandingSiteReport]:
    """
    Sort a list of landing site reports by safety preference.

    Order: PASS first, then REVIEW_REQUIRED, then FAIL.
    Within the same status, sort by ascending max slope.

    Parameters
    ----------
    sites:
        List of evaluated LandingSiteReport objects.

    Returns
    -------
    list[LandingSiteReport]
        Sorted copy of the input list.
    """
    _status_rank = {
        SafetyStatus.PASS: 0,
        SafetyStatus.REVIEW_REQUIRED: 1,
        SafetyStatus.FAIL: 2,
    }

    def _sort_key(site: LandingSiteReport) -> tuple[int, float]:
        slope = site.max_slope_deg if site.max_slope_deg is not None else 999.0
        return (_status_rank[site.status], slope)

    return sorted(sites, key=_sort_key)
