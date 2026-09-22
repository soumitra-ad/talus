"""Deterministic rover route safety analysis (research/demo).

``check_rover_safety`` measures slope and terrain ruggedness along a route on a DEM and
compares the measurements with thresholds supplied by the caller. Every number in the
result comes from deterministic Python code. Nothing here is certified for mission use.

Status rules
------------
Statuses are the three allowed values only. Per segment, the worst applicable rule wins.

* FAIL: the measured maximum slope on the segment exceeds ``maximum_slope_deg``, or the
  segment touches a no-go zone.
* REVIEW_REQUIRED: mean terrain ruggedness (TRI) exceeds ``maximum_roughness`` (only if a
  limit was given), or any part of the segment is unmeasured, or no terrain data exists.
* PASS: every cell on the segment was measured and no rule above applies.

Missing terrain data can therefore never produce PASS. The route status is the worst
segment status. A no-go zone crossing is geometric and is reported even without a DEM.

Risk score
----------
The risk score is a heuristic between 0 and 100. It is not a probability, it is not
calibrated against real rover performance, and it must not be read as a safety margin.

Per segment, with the segment measurements and configured limits:

    slope_component     = min(max_slope / maximum_slope_deg, 2) / 2
    roughness_component = min(mean_tri / maximum_roughness, 2) / 2   (only if a limit is given)
    segment_risk        = 100 * max(slope_component, roughness_component)
    segment_risk        = 100 if the segment touches a no-go zone

So a segment exactly at its limit scores 50, and a segment at twice its limit or worse
scores 100. For the route:

    route_risk = 0.5 * length_weighted_mean(segment_risk) + 0.5 * max(segment_risk)

The maximum term stops one dangerous segment being averaged away. Only segments with
measurements (or a no-go crossing) contribute. Unmeasured parts are reported through
coverage and status, not scored. If no segment has any basis, the score is ``None``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel

from terrain_agent.data.dataset_info import DatasetInfo, build_dataset_info
from terrain_agent.safety.evaluator import SafetyStatus
from terrain_agent.terrain.geometry import (
    MAX_ROUTE_SAMPLES,
    NO_GO_SAMPLE_SPACING_M,
    NoGoZone,
    densify_route,
    find_no_go_hits,
    parse_no_go_zones,
    segment_lengths_m,
    validate_route,
)
from terrain_agent.terrain.georef import DemGeoContext, open_dem_context
from terrain_agent.terrain.resource_safety import (
    InvalidThresholdError,
    TerrainAnalysisError,
)
from terrain_agent.terrain.route_analysis import (
    RouteTerrainMeasurement,
    SegmentTerrainStats,
    measure_route_terrain,
)

DISCLAIMER = (
    "TALUS research/demo analysis. Thresholds are configured analysis parameters, NOT "
    "certified safety limits. This result is not mission approval or guaranteed rover safety."
)

LIMITATIONS: tuple[str, ...] = (
    "Terrain is measured along the route centreline only. Rover width, wheel-soil interaction "
    "and cross-track terrain are not assessed.",
    "Slope is the gradient magnitude from Horn's method at the DEM native cell size. Direction "
    "of travel is not considered, and coarse or smoothed DEMs understate steep local slopes.",
    "Boulders, small craters and other hazards smaller than one DEM cell are not resolved.",
    "Terrain ruggedness (TRI) depends on DEM resolution. Values from different DEMs are not "
    "directly comparable, and a limit chosen for one resolution may not suit another.",
    "Elevation quality can be lower in permanently shadowed or poorly illuminated areas. "
    "Vertical accuracy of the DEM is not evaluated here.",
    "The risk score is an uncalibrated heuristic, not a probability of failure.",
    "Power, thermal, communication, localisation and mobility-system constraints are not modelled.",
)

RISK_SCORE_METHOD = (
    "segment_risk = 100 * max(min(max_slope/limit, 2)/2, min(mean_tri/roughness_limit, 2)/2); "
    "100 for a no-go crossing. route_risk = 0.5 * length-weighted mean + 0.5 * maximum. "
    "Heuristic, uncalibrated."
)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ConfiguredThresholds(BaseModel):
    """Thresholds the analysis was configured with."""

    maximum_slope_deg: float
    maximum_roughness_tri_m: Optional[float]
    roughness_evaluated: bool
    no_go_zone_count: int
    statement: str


class ViolationRecord(BaseModel):
    """One threshold or no-go violation on a segment."""

    kind: Literal["slope", "roughness", "no_go_zone"]
    description: str
    measured: Optional[float] = None
    threshold: Optional[float] = None


class SegmentAnalysis(BaseModel):
    """Assessment of one route segment."""

    segment_index: int
    start: tuple[float, float]
    end: tuple[float, float]
    length_m: float
    status: SafetyStatus
    cells_total: int
    cells_measured: int
    coverage_fraction: float
    max_slope_deg: Optional[float] = None
    mean_slope_deg: Optional[float] = None
    mean_tri_m: Optional[float] = None
    max_tri_m: Optional[float] = None
    min_elevation_m: Optional[float] = None
    max_elevation_m: Optional[float] = None
    elevation_change_m: Optional[float] = None
    risk_score: Optional[float] = None
    violations: list[ViolationRecord] = []
    no_go_zones: list[str] = []
    data_issues: list[str] = []


class RoverSafetyResult(BaseModel):
    """Complete route analysis result."""

    status: SafetyStatus
    terrain_available: bool
    risk_score: Optional[float]
    risk_score_basis: str
    risk_score_method: str = RISK_SCORE_METHOD
    total_segments: int
    segments_analysed: int
    segments_with_incomplete_coverage: int
    route_length_m: float
    coverage_fraction: float
    max_slope_deg: Optional[float]
    mean_slope_deg: Optional[float]
    mean_tri_m: Optional[float]
    max_tri_m: Optional[float]
    violated_segments: list[int]
    incomplete_segments: list[int]
    segments: list[SegmentAnalysis]
    configured_thresholds: ConfiguredThresholds
    dataset: Optional[DatasetInfo]
    sample_spacing_m: Optional[float]
    warnings: list[str]
    limitations: list[str]
    disclaimer: str = DISCLAIMER


# ---------------------------------------------------------------------------
# Validation and small helpers
# ---------------------------------------------------------------------------


def _validate_max_slope(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidThresholdError("maximum_slope_deg must be a number.")
    number = float(value)
    if not math.isfinite(number) or not (0.0 < number <= 90.0):
        raise InvalidThresholdError(
            f"maximum_slope_deg must be greater than 0 and at most 90, got {value!r}."
        )
    return number


def _validate_max_roughness(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidThresholdError("maximum_roughness must be a number or None.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise InvalidThresholdError(f"maximum_roughness must be positive and finite, got {value!r}.")
    return number


def _validate_spacing(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidThresholdError("sample_spacing_m must be a number or None.")
    number = float(value)
    if not math.isfinite(number) or not (0.1 <= number <= 10_000.0):
        raise InvalidThresholdError("sample_spacing_m must be between 0.1 and 10000 metres.")
    return number


def safe_error_text(exc: BaseException, path: Any) -> str:
    """Error text with any file system path reduced to the file name."""
    text = str(exc)
    try:
        candidate = Path(path)
        for variant in {str(candidate), str(candidate.resolve()), candidate.as_posix()}:
            text = text.replace(variant, candidate.name)
    except (OSError, ValueError):
        pass
    return text


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def segment_risk_score(
    max_slope_deg: Optional[float],
    mean_tri_m: Optional[float],
    maximum_slope_deg: float,
    maximum_roughness: Optional[float],
    touches_no_go: bool,
) -> Optional[float]:
    """Risk score for one segment. See the module docstring for the definition."""
    if touches_no_go:
        return 100.0
    components: list[float] = []
    if max_slope_deg is not None:
        components.append(min(max_slope_deg / maximum_slope_deg, 2.0) / 2.0)
    if maximum_roughness is not None and mean_tri_m is not None:
        components.append(min(mean_tri_m / maximum_roughness, 2.0) / 2.0)
    if not components:
        return None
    return round(100.0 * max(components), 2)


def route_risk_score(
    segment_scores: Sequence[Optional[float]], segment_lengths: Sequence[float]
) -> Optional[float]:
    """Route risk score: half length-weighted mean, half worst segment."""
    scored = [(s, l) for s, l in zip(segment_scores, segment_lengths) if s is not None]
    if not scored:
        return None
    total_length = sum(l for _, l in scored)
    mean = sum(s * l for s, l in scored) / total_length if total_length > 0 else 0.0
    worst = max(s for s, _ in scored)
    return round(0.5 * mean + 0.5 * worst, 2)


def _threshold_statement(max_slope: float, max_roughness: Optional[float]) -> str:
    text = f"Configured analysis threshold: {max_slope:g}° slope"
    if max_roughness is not None:
        text += f", {max_roughness:g} m TRI"
    return text + ". This is NOT a certified safety limit."


# ---------------------------------------------------------------------------
# Segment assessment
# ---------------------------------------------------------------------------


def _assess_segment(
    index: int,
    start: tuple[float, float],
    end: tuple[float, float],
    length_m: float,
    stats: Optional[SegmentTerrainStats],
    terrain_available: bool,
    no_go_names: list[str],
    maximum_slope_deg: float,
    maximum_roughness: Optional[float],
) -> SegmentAnalysis:
    violations: list[ViolationRecord] = []
    issues: list[str] = []
    status = SafetyStatus.PASS

    if no_go_names:
        violations.append(
            ViolationRecord(
                kind="no_go_zone",
                description="Segment touches no-go zone(s): " + ", ".join(no_go_names),
            )
        )
        status = SafetyStatus.FAIL

    cells_total = stats.cells_total if stats else 0
    cells_valid = stats.cells_valid if stats else 0

    if not terrain_available:
        issues.append("No terrain data was available for this segment.")
    elif cells_valid == 0:
        issues.append("No cell along this segment could be measured.")
    elif cells_valid < cells_total:
        issues.append(
            f"Only {100.0 * cells_valid / cells_total:.1f}% of cells along this segment were "
            "measured (outside the DEM, nodata, or DEM edge)."
        )

    if stats is not None and cells_valid > 0:
        assert stats.max_slope_deg is not None and stats.mean_tri_m is not None
        if stats.max_slope_deg > maximum_slope_deg:
            violations.append(
                ViolationRecord(
                    kind="slope",
                    description=(
                        f"Maximum slope {stats.max_slope_deg:.2f}° exceeds configured "
                        f"threshold {maximum_slope_deg:g}°"
                    ),
                    measured=round(stats.max_slope_deg, 2),
                    threshold=maximum_slope_deg,
                )
            )
            status = SafetyStatus.FAIL
        if maximum_roughness is not None and stats.mean_tri_m > maximum_roughness:
            violations.append(
                ViolationRecord(
                    kind="roughness",
                    description=(
                        f"Mean TRI {stats.mean_tri_m:.3f} m exceeds configured "
                        f"threshold {maximum_roughness:g} m"
                    ),
                    measured=round(stats.mean_tri_m, 3),
                    threshold=maximum_roughness,
                )
            )
            if status is not SafetyStatus.FAIL:
                status = SafetyStatus.REVIEW_REQUIRED

    if issues and status is SafetyStatus.PASS:
        status = SafetyStatus.REVIEW_REQUIRED

    risk = segment_risk_score(
        stats.max_slope_deg if stats else None,
        stats.mean_tri_m if stats else None,
        maximum_slope_deg,
        maximum_roughness,
        bool(no_go_names),
    )

    return SegmentAnalysis(
        segment_index=index,
        start=start,
        end=end,
        length_m=round(length_m, 2),
        status=status,
        cells_total=cells_total,
        cells_measured=cells_valid,
        coverage_fraction=round(cells_valid / cells_total, 4) if cells_total else 0.0,
        max_slope_deg=_round(stats.max_slope_deg) if stats else None,
        mean_slope_deg=_round(stats.mean_slope_deg) if stats else None,
        mean_tri_m=_round(stats.mean_tri_m, 3) if stats else None,
        max_tri_m=_round(stats.max_tri_m, 3) if stats else None,
        min_elevation_m=_round(stats.min_elevation_m) if stats else None,
        max_elevation_m=_round(stats.max_elevation_m) if stats else None,
        elevation_change_m=_round(stats.elevation_change_m) if stats else None,
        risk_score=risk,
        violations=violations,
        no_go_zones=list(no_go_names),
        data_issues=issues,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_rover_safety(
    waypoints: Any,
    maximum_slope_deg: float,
    maximum_roughness: Optional[float] = None,
    no_go_zones: Optional[Sequence[Any]] = None,
    *,
    dem_path: Any = None,
    sample_spacing_m: Optional[float] = None,
) -> RoverSafetyResult:
    """Analyse a rover route against configured slope and roughness thresholds.

    Parameters
    ----------
    waypoints:
        Ordered ``(latitude, longitude)`` pairs in decimal degrees, at least two.
    maximum_slope_deg:
        Configured maximum slope in degrees, greater than 0 and at most 90.
    maximum_roughness:
        Configured maximum mean TRI in metres, or ``None`` to report roughness without a limit.
    no_go_zones:
        Optional circles or polygons the route must not touch. See
        :func:`terrain_agent.terrain.geometry.parse_no_go_zones`.
    dem_path:
        DEM raster to measure. Without a readable DEM the result is never PASS.
    sample_spacing_m:
        Optional sample spacing along the route. Defaults to half the DEM cell size.

    Raises
    ------
    InvalidCoordinateError
        Invalid, non-finite, out-of-range or order-inverted coordinates.
    InvalidWaypointError
        Fewer than two waypoints, malformed entries, or duplicate consecutive waypoints.
    InvalidThresholdError
        A threshold is not a finite number in its allowed range.
    InvalidNoGoZoneError
        A no-go zone definition is malformed.
    OversizedRequestError
        Too many waypoints or zones, or a route longer than the configured maximum.
    """
    max_slope = _validate_max_slope(maximum_slope_deg)
    max_rough = _validate_max_roughness(maximum_roughness)
    requested_spacing = _validate_spacing(sample_spacing_m)
    route = validate_route(waypoints)
    zones: list[NoGoZone] = parse_no_go_zones(no_go_zones)

    n_segments = len(route) - 1
    lengths = segment_lengths_m(route)
    route_length = float(sum(lengths))

    # Geometry only, independent of any DEM.
    no_go_hits = find_no_go_hits(densify_route(route, NO_GO_SAMPLE_SPACING_M), zones)

    warnings: list[str] = []
    ctx: Optional[DemGeoContext] = None
    dataset: Optional[DatasetInfo] = None
    measurement: Optional[RouteTerrainMeasurement] = None
    used_spacing: Optional[float] = None

    if dem_path is None:
        warnings.append("No DEM was provided; terrain could not be assessed.")
    else:
        try:
            ctx = open_dem_context(dem_path)
        except FileNotFoundError:
            warnings.append(f"DEM file not found: {Path(str(dem_path)).name}")
        except TerrainAnalysisError as exc:
            warnings.append(f"DEM could not be used: {safe_error_text(exc, dem_path)}")

    if ctx is not None:
        ref_lat, ref_lon = route[len(route) // 2]
        dataset = build_dataset_info(ctx, ref_lat, ref_lon)
        warnings.extend(dataset.warnings)
        try:
            pixel = ctx.pixel_size_m(ref_lat, ref_lon)
            wanted = requested_spacing if requested_spacing else 0.5 * min(pixel)
            floor = route_length / (0.5 * MAX_ROUTE_SAMPLES)
            used_spacing = max(wanted, floor)
            if used_spacing > wanted:
                warnings.append(
                    f"Sample spacing was increased to {used_spacing:.2f} m to keep the number "
                    "of samples within the limit."
                )
            samples = densify_route(route, used_spacing)
            measurement = measure_route_terrain(ctx, samples, n_segments)
        except TerrainAnalysisError as exc:
            measurement = None
            used_spacing = None
            warnings.append(f"Terrain could not be measured: {safe_error_text(exc, ctx.path)}")

    terrain_available = measurement is not None

    segments: list[SegmentAnalysis] = []
    for index in range(n_segments):
        segments.append(
            _assess_segment(
                index,
                route[index],
                route[index + 1],
                lengths[index],
                measurement.segments[index] if measurement else None,
                terrain_available,
                no_go_hits.get(index, []),
                max_slope,
                max_rough,
            )
        )

    statuses = {s.status for s in segments}
    if SafetyStatus.FAIL in statuses:
        overall = SafetyStatus.FAIL
    elif SafetyStatus.REVIEW_REQUIRED in statuses:
        overall = SafetyStatus.REVIEW_REQUIRED
    else:
        overall = SafetyStatus.PASS

    violated = [s.segment_index for s in segments if s.violations]
    incomplete = [s.segment_index for s in segments if s.data_issues]
    analysed = [s for s in segments if s.cells_measured > 0]

    total_cells = sum(s.cells_total for s in segments)
    measured_cells = sum(s.cells_measured for s in segments)
    coverage = round(measured_cells / total_cells, 4) if total_cells else 0.0

    max_slope_route = max((s.max_slope_deg for s in analysed if s.max_slope_deg is not None), default=None)
    max_tri_route = max((s.max_tri_m for s in analysed if s.max_tri_m is not None), default=None)
    weight = sum(s.cells_measured for s in analysed)
    mean_slope_route = (
        sum(s.mean_slope_deg * s.cells_measured for s in analysed if s.mean_slope_deg is not None) / weight
        if weight
        else None
    )
    mean_tri_route = (
        sum(s.mean_tri_m * s.cells_measured for s in analysed if s.mean_tri_m is not None) / weight
        if weight
        else None
    )

    scores = [s.risk_score for s in segments]
    risk = route_risk_score(scores, lengths)
    scored = sum(1 for s in scores if s is not None)
    if risk is None:
        basis = "No score: no segment has terrain measurements or a no-go crossing."
    elif scored == n_segments and all(not s.data_issues for s in segments):
        basis = f"All {n_segments} segments fully measured."
    else:
        basis = (
            f"Partial: {scored} of {n_segments} segments contribute, and measured segments may "
            "have incomplete coverage. Unmeasured terrain is not scored."
        )

    for s in segments:
        for issue in s.data_issues:
            warnings.append(f"Segment {s.segment_index}: {issue}")
    if any(s.no_go_zones for s in segments):
        warnings.append(
            "Segments touching a no-go zone: "
            + ", ".join(str(s.segment_index) for s in segments if s.no_go_zones)
        )

    return RoverSafetyResult(
        status=overall,
        terrain_available=terrain_available,
        risk_score=risk,
        risk_score_basis=basis,
        total_segments=n_segments,
        segments_analysed=len(analysed),
        segments_with_incomplete_coverage=len(incomplete),
        route_length_m=round(route_length, 2),
        coverage_fraction=coverage,
        max_slope_deg=_round(max_slope_route),
        mean_slope_deg=_round(mean_slope_route),
        mean_tri_m=_round(mean_tri_route, 3),
        max_tri_m=_round(max_tri_route, 3),
        violated_segments=violated,
        incomplete_segments=incomplete,
        segments=segments,
        configured_thresholds=ConfiguredThresholds(
            maximum_slope_deg=max_slope,
            maximum_roughness_tri_m=max_rough,
            roughness_evaluated=max_rough is not None,
            no_go_zone_count=len(zones),
            statement=_threshold_statement(max_slope, max_rough),
        ),
        dataset=dataset,
        sample_spacing_m=_round(used_spacing, 3),
        warnings=warnings,
        limitations=list(LIMITATIONS),
    )
