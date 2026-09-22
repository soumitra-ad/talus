"""Deterministic landing-site analysis and comparison (research/demo).

Each candidate is measured inside a circular footprint on a DEM. The measurements are
elevation, slope, terrain ruggedness (TRI), DEM resolution, data coverage, and the radius
of the largest flat circle around the site. Comparison ranks only candidates whose
required measurements are complete. A candidate with missing data is listed as unranked
with the reason. No rank, score or status is ever invented for missing data.

Status rules (per site)
-----------------------
* FAIL: measured maximum slope inside the footprint exceeds ``maximum_slope_deg``.
* REVIEW_REQUIRED: mean TRI exceeds ``maximum_roughness`` (if given), or any footprint cell
  is unmeasured, or the DEM is too coarse to verify ``min_flat_radius_m``, or nothing could
  be measured.
* PASS: every footprint cell is measured, resolution is adequate, and no rule above applies.

Ranking order
-------------
Lexicographic and fully transparent, in this order: status (PASS, REVIEW_REQUIRED, FAIL),
maximum slope ascending, mean TRI ascending, mean slope ascending, flat radius descending,
site id. There is no weighted score.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from pydantic import BaseModel

from terrain_agent.data.dataset_info import DatasetInfo, build_dataset_info
from terrain_agent.safety.evaluator import SafetyStatus
from terrain_agent.safety.rover import DISCLAIMER, ViolationRecord, safe_error_text
from terrain_agent.terrain.coordinates import validate_analysis_radius, validate_lunar_coordinate
from terrain_agent.terrain.footprint import load_footprint
from terrain_agent.terrain.georef import DemGeoContext, open_dem_context
from terrain_agent.terrain.resource_safety import (
    InvalidCoordinateError,
    InvalidSiteError,
    InvalidThresholdError,
    OversizedRequestError,
    TerrainAnalysisError,
)

DEFAULT_FOOTPRINT_RADIUS_M = 500.0
DEFAULT_MAX_SLOPE_DEG = 5.0
DEFAULT_MIN_FLAT_RADIUS_M = 10.0
MAX_SITES = 50

#: Fewer than this many cells inside the required flat radius means the DEM is too coarse.
MIN_CELLS_FOR_FLAT_RADIUS = 4

LIMITATIONS: tuple[str, ...] = (
    "Measurements are from a DEM at its native cell size. Boulders, small craters and other "
    "hazards smaller than one cell are not resolved.",
    "Slope is the gradient magnitude from Horn's method. Coarse or smoothed DEMs understate "
    "steep local slopes.",
    "Terrain ruggedness (TRI) depends on DEM resolution. Values from different DEMs are not "
    "directly comparable.",
    "Illumination, communication, thermal, hazard-avoidance and landing-system constraints are "
    "not modelled. Only terrain measured on the supplied DEM is considered.",
    "Elevation quality can be lower in permanently shadowed or poorly illuminated areas. "
    "Vertical accuracy of the DEM is not evaluated here.",
)

RANKING_METHOD = (
    "Lexicographic order: status (PASS, REVIEW_REQUIRED, FAIL), then maximum slope ascending, "
    "then mean TRI ascending, then mean slope ascending, then flat radius descending, then "
    "site id. Only sites with complete measurements are ranked."
)


class LandingThresholds(BaseModel):
    """Thresholds the site analysis was configured with."""

    maximum_slope_deg: float
    maximum_roughness_tri_m: Optional[float]
    min_flat_radius_m: float
    footprint_radius_m: float
    statement: str


class LandingSiteAnalysis(BaseModel):
    """Measurements and assessment for one candidate site."""

    site_id: str
    lat: float
    lon: float
    status: SafetyStatus
    terrain_available: bool
    footprint_radius_m: float
    footprint_cells: int
    cells_measured: int
    coverage_fraction: float
    elevation_mean_m: Optional[float] = None
    elevation_min_m: Optional[float] = None
    elevation_max_m: Optional[float] = None
    mean_slope_deg: Optional[float] = None
    max_slope_deg: Optional[float] = None
    mean_tri_m: Optional[float] = None
    max_tri_m: Optional[float] = None
    flat_radius_m: Optional[float] = None
    flat_radius_is_lower_bound: bool = False
    flat_radius_requirement_met: Optional[bool] = None
    resolution_m: Optional[float] = None
    violations: list[ViolationRecord] = []
    data_issues: list[str] = []
    warnings: list[str] = []
    dataset: Optional[DatasetInfo] = None
    thresholds: LandingThresholds
    rankable: bool = False
    unrankable_reasons: list[str] = []
    limitations: list[str] = list(LIMITATIONS)
    disclaimer: str = DISCLAIMER


class RankedSite(BaseModel):
    rank: int
    analysis: LandingSiteAnalysis


class UnrankedSite(BaseModel):
    site_id: str
    status: SafetyStatus
    reasons: list[str]


class LandingComparison(BaseModel):
    """Comparison of candidate sites. Sites with incomplete data are never ranked."""

    ranking_method: str = RANKING_METHOD
    ranked: list[RankedSite]
    unranked: list[UnrankedSite]
    sites: list[LandingSiteAnalysis]
    warnings: list[str]
    limitations: list[str] = list(LIMITATIONS)
    disclaimer: str = DISCLAIMER


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _clean_site_id(raw: Any, fallback: str) -> str:
    if raw is None:
        return fallback
    text = "".join(ch for ch in str(raw) if ch.isprintable()).strip()[:64]
    return text or fallback


def _check_number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidThresholdError(f"{name} must be a number.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or (number == 0.0 and not allow_zero):
        raise InvalidThresholdError(f"{name} must be a positive finite number, got {value!r}.")
    return number


def _validate_config(
    radius_m: Any,
    maximum_slope_deg: Any,
    maximum_roughness: Any,
    min_flat_radius_m: Any,
) -> tuple[float, float, Optional[float], float]:
    radius = _check_number(radius_m, "radius_m")
    validate_analysis_radius(radius / 1000.0)
    slope = _check_number(maximum_slope_deg, "maximum_slope_deg")
    if slope > 90.0:
        raise InvalidThresholdError("maximum_slope_deg must be at most 90.")
    rough = None if maximum_roughness is None else _check_number(maximum_roughness, "maximum_roughness")
    flat = _check_number(min_flat_radius_m, "min_flat_radius_m", allow_zero=True)
    if flat > radius:
        raise InvalidThresholdError("min_flat_radius_m cannot exceed the footprint radius.")
    return radius, slope, rough, flat


def _thresholds(radius: float, slope: float, rough: Optional[float], flat: float) -> LandingThresholds:
    text = f"Configured analysis threshold: {slope:g}° slope"
    if rough is not None:
        text += f", {rough:g} m TRI"
    text += f", {flat:g} m flat radius within a {radius:g} m footprint. This is NOT a certified safety limit."
    return LandingThresholds(
        maximum_slope_deg=slope,
        maximum_roughness_tri_m=rough,
        min_flat_radius_m=flat,
        footprint_radius_m=radius,
        statement=text,
    )


def _rank_blockers(a: LandingSiteAnalysis) -> list[str]:
    reasons: list[str] = []
    if not a.terrain_available or a.cells_measured == 0:
        reasons.append("No terrain measurements are available for this site.")
    else:
        if a.coverage_fraction < 1.0:
            reasons.append(
                f"Incomplete DEM coverage: {100.0 * a.coverage_fraction:.1f}% of the footprint "
                "was measured."
            )
        for name, value in (
            ("maximum slope", a.max_slope_deg),
            ("mean slope", a.mean_slope_deg),
            ("mean TRI", a.mean_tri_m),
        ):
            if value is None:
                reasons.append(f"Required measurement missing: {name}.")
    return reasons


# ---------------------------------------------------------------------------
# Single-site analysis
# ---------------------------------------------------------------------------


def analyze_landing_site(
    dem_path: Any,
    lat: float,
    lon: float,
    *,
    site_id: Optional[str] = None,
    radius_m: float = DEFAULT_FOOTPRINT_RADIUS_M,
    maximum_slope_deg: float = DEFAULT_MAX_SLOPE_DEG,
    maximum_roughness: Optional[float] = None,
    min_flat_radius_m: float = DEFAULT_MIN_FLAT_RADIUS_M,
) -> LandingSiteAnalysis:
    """Measure and assess one candidate landing site.

    Raises
    ------
    InvalidCoordinateError
        Invalid coordinates or a footprint radius outside the allowed range.
    InvalidThresholdError
        A threshold is not a finite number in its allowed range.
    OversizedRequestError
        The footprint needs a window larger than the raster read limit.

    A missing, unreadable or non-covering DEM does not raise. The result reports the data
    problem, has status REVIEW_REQUIRED, and is not rankable.
    """
    site_lat, site_lon = validate_lunar_coordinate(lat, lon)
    radius, max_slope, max_rough, min_flat = _validate_config(
        radius_m, maximum_slope_deg, maximum_roughness, min_flat_radius_m
    )
    sid = _clean_site_id(site_id, f"site_{site_lat:.4f}_{site_lon:.4f}")
    thresholds = _thresholds(radius, max_slope, max_rough, min_flat)

    def unavailable(reason: str, dataset: Optional[DatasetInfo] = None) -> LandingSiteAnalysis:
        analysis = LandingSiteAnalysis(
            site_id=sid,
            lat=site_lat,
            lon=site_lon,
            status=SafetyStatus.REVIEW_REQUIRED,
            terrain_available=False,
            footprint_radius_m=radius,
            footprint_cells=0,
            cells_measured=0,
            coverage_fraction=0.0,
            data_issues=[reason],
            warnings=[reason],
            dataset=dataset,
            thresholds=thresholds,
        )
        analysis.unrankable_reasons = _rank_blockers(analysis)
        return analysis

    if dem_path is None:
        return unavailable("No DEM was provided; terrain could not be assessed.")

    ctx: DemGeoContext
    try:
        ctx = open_dem_context(dem_path)
    except FileNotFoundError:
        return unavailable(f"DEM file not found: {Path(str(dem_path)).name}")
    except TerrainAnalysisError as exc:
        return unavailable(f"DEM could not be used: {safe_error_text(exc, dem_path)}")

    dataset = build_dataset_info(ctx, site_lat, site_lon)
    try:
        footprint = load_footprint(ctx, site_lat, site_lon, radius)
    except OversizedRequestError:
        raise
    except TerrainAnalysisError as exc:
        return unavailable(f"Terrain could not be measured: {safe_error_text(exc, ctx.path)}", dataset)

    grids, dist, mask = footprint.grids, footprint.distance_m, footprint.mask
    dx_m, dy_m = grids.pixel_size_m
    footprint_cells = int(mask.sum())
    measured = mask & grids.slope_valid
    cells_measured = int(measured.sum())
    coverage = cells_measured / footprint_cells if footprint_cells else 0.0

    violations: list[ViolationRecord] = []
    issues: list[str] = []
    status = SafetyStatus.PASS

    stats: dict[str, Optional[float]] = dict.fromkeys(
        ("mean_slope", "max_slope", "mean_tri", "max_tri", "e_mean", "e_min", "e_max"), None
    )
    if cells_measured:
        slope_vals = grids.slope_deg[measured]
        tri_vals = grids.tri[measured]
        stats.update(
            mean_slope=float(np.mean(slope_vals)),
            max_slope=float(np.max(slope_vals)),
            mean_tri=float(np.mean(tri_vals)),
            max_tri=float(np.max(tri_vals)),
        )
    elev_cells = mask & np.isfinite(grids.elevation)
    if elev_cells.any():
        ev = grids.elevation[elev_cells]
        stats.update(e_mean=float(np.mean(ev)), e_min=float(np.min(ev)), e_max=float(np.max(ev)))

    # Largest flat circle: distance to the nearest cell that is steeper than the limit or unmeasured.
    nonflat = mask & ~(grids.slope_valid & (grids.slope_deg <= max_slope))
    flat_radius: Optional[float]
    lower_bound = False
    if not cells_measured:
        flat_radius = None
    elif nonflat.any():
        flat_radius = max(0.0, float(dist[nonflat].min()) - 0.5 * max(dx_m, dy_m))
    else:
        flat_radius = radius
        lower_bound = True
    flat_met = None if flat_radius is None else flat_radius >= min_flat

    if not cells_measured:
        issues.append("No cell in the footprint could be measured.")
    else:
        assert stats["max_slope"] is not None and stats["mean_tri"] is not None
        if stats["max_slope"] > max_slope:
            violations.append(
                ViolationRecord(
                    kind="slope",
                    description=(
                        f"Maximum slope {stats['max_slope']:.2f}° exceeds configured "
                        f"threshold {max_slope:g}°"
                    ),
                    measured=round(stats["max_slope"], 2),
                    threshold=max_slope,
                )
            )
            status = SafetyStatus.FAIL
        if max_rough is not None and stats["mean_tri"] > max_rough:
            violations.append(
                ViolationRecord(
                    kind="roughness",
                    description=(
                        f"Mean TRI {stats['mean_tri']:.3f} m exceeds configured "
                        f"threshold {max_rough:g} m"
                    ),
                    measured=round(stats["mean_tri"], 3),
                    threshold=max_rough,
                )
            )
            if status is not SafetyStatus.FAIL:
                status = SafetyStatus.REVIEW_REQUIRED
        if coverage < 1.0:
            issues.append(
                f"Only {100.0 * coverage:.1f}% of the footprint was measured "
                "(outside the DEM, nodata, or DEM edge)."
            )
        cells_in_flat_disc = int((dist <= min_flat).sum()) if min_flat > 0 else MIN_CELLS_FOR_FLAT_RADIUS
        if cells_in_flat_disc < MIN_CELLS_FOR_FLAT_RADIUS:
            issues.append(
                f"DEM resolution (about {0.5 * (dx_m + dy_m):.1f} m per cell) is too coarse to "
                f"verify a flat radius of {min_flat:g} m."
            )

    if issues and status is SafetyStatus.PASS:
        status = SafetyStatus.REVIEW_REQUIRED

    def r(value: Optional[float], digits: int = 2) -> Optional[float]:
        return None if value is None else round(float(value), digits)

    analysis = LandingSiteAnalysis(
        site_id=sid,
        lat=site_lat,
        lon=site_lon,
        status=status,
        terrain_available=True,
        footprint_radius_m=radius,
        footprint_cells=footprint_cells,
        cells_measured=cells_measured,
        coverage_fraction=round(coverage, 4),
        elevation_mean_m=r(stats["e_mean"]),
        elevation_min_m=r(stats["e_min"]),
        elevation_max_m=r(stats["e_max"]),
        mean_slope_deg=r(stats["mean_slope"]),
        max_slope_deg=r(stats["max_slope"]),
        mean_tri_m=r(stats["mean_tri"], 3),
        max_tri_m=r(stats["max_tri"], 3),
        flat_radius_m=r(flat_radius),
        flat_radius_is_lower_bound=lower_bound,
        flat_radius_requirement_met=flat_met,
        resolution_m=r(0.5 * (dx_m + dy_m)),
        violations=violations,
        data_issues=issues,
        warnings=list(dataset.warnings) + issues,
        dataset=dataset,
        thresholds=thresholds,
    )
    analysis.unrankable_reasons = _rank_blockers(analysis)
    analysis.rankable = not analysis.unrankable_reasons
    return analysis


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

_STATUS_RANK = {SafetyStatus.PASS: 0, SafetyStatus.REVIEW_REQUIRED: 1, SafetyStatus.FAIL: 2}


def rank_landing_sites(analyses: Sequence[LandingSiteAnalysis]) -> LandingComparison:
    """Rank analysed sites. Sites that are not rankable are listed with reasons, not ranked."""
    ids = [a.site_id for a in analyses]
    if len(set(ids)) != len(ids):
        raise InvalidSiteError("Site ids must be unique.")

    rankable = [a for a in analyses if a.rankable]
    ordered = sorted(
        rankable,
        key=lambda a: (
            _STATUS_RANK[a.status],
            a.max_slope_deg,
            a.mean_tri_m,
            a.mean_slope_deg,
            -(a.flat_radius_m if a.flat_radius_m is not None else 0.0),
            a.site_id,
        ),
    )
    ranked = [RankedSite(rank=i + 1, analysis=a) for i, a in enumerate(ordered)]
    unranked = [
        UnrankedSite(site_id=a.site_id, status=a.status, reasons=list(a.unrankable_reasons))
        for a in analyses
        if not a.rankable
    ]

    warnings: list[str] = []
    if len(ordered) > 1:
        sources = {
            (a.dataset.file_name, a.dataset.sha256_recorded) if a.dataset else None for a in ordered
        }
        if len(sources) > 1:
            warnings.append("Ranked sites were measured on different DEM files; values may not be comparable.")
        resolutions = [a.resolution_m for a in ordered if a.resolution_m]
        if resolutions and max(resolutions) > 1.1 * min(resolutions):
            warnings.append("Ranked sites were measured at different DEM resolutions.")
        if len({a.thresholds.statement for a in ordered}) > 1:
            warnings.append("Ranked sites were analysed with different thresholds.")
    if unranked:
        warnings.append(
            f"{len(unranked)} site(s) were not ranked because required data is missing or incomplete."
        )

    return LandingComparison(ranked=ranked, unranked=unranked, sites=list(analyses), warnings=warnings)


def compare_landing_candidates(
    dem_path: Any,
    sites: Sequence[Any],
    *,
    radius_m: float = DEFAULT_FOOTPRINT_RADIUS_M,
    maximum_slope_deg: float = DEFAULT_MAX_SLOPE_DEG,
    maximum_roughness: Optional[float] = None,
    min_flat_radius_m: float = DEFAULT_MIN_FLAT_RADIUS_M,
) -> LandingComparison:
    """Analyse every candidate on one DEM, then rank those with complete data.

    Each site is a mapping with ``lat``, ``lon`` and optional ``id``, or a ``(lat, lon)`` pair.
    All sites are validated before any analysis, so one invalid coordinate rejects the request.

    Raises
    ------
    InvalidSiteError
        No sites, too many sites, malformed entries, or duplicate ids.
    InvalidCoordinateError
        Any site has an invalid coordinate.
    """
    if isinstance(sites, (str, bytes)) or not isinstance(sites, Sequence) or not sites:
        raise InvalidSiteError("sites must be a non-empty list.")
    if len(sites) > MAX_SITES:
        raise InvalidSiteError(f"At most {MAX_SITES} sites can be compared, got {len(sites)}.")
    _validate_config(radius_m, maximum_slope_deg, maximum_roughness, min_flat_radius_m)

    parsed: list[tuple[str, float, float]] = []
    for index, raw in enumerate(sites):
        if isinstance(raw, dict):
            if "lat" not in raw or "lon" not in raw:
                raise InvalidSiteError(f"Site {index + 1} needs lat and lon.")
            lat_raw, lon_raw, id_raw = raw["lat"], raw["lon"], raw.get("id")
        elif isinstance(raw, (tuple, list)) and len(raw) == 2:
            lat_raw, lon_raw, id_raw = raw[0], raw[1], None
        else:
            raise InvalidSiteError(f"Site {index + 1} must be a mapping or a (lat, lon) pair.")
        for value in (lat_raw, lon_raw):
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise InvalidCoordinateError(f"Site {index + 1} coordinates must be numeric.")
        lat_c, lon_c = validate_lunar_coordinate(float(lat_raw), float(lon_raw))
        parsed.append((_clean_site_id(id_raw, f"site_{index + 1}"), lat_c, lon_c))

    ids = [p[0] for p in parsed]
    if len(set(ids)) != len(ids):
        raise InvalidSiteError("Site ids must be unique.")

    analyses = [
        analyze_landing_site(
            dem_path,
            lat_c,
            lon_c,
            site_id=sid,
            radius_m=radius_m,
            maximum_slope_deg=maximum_slope_deg,
            maximum_roughness=maximum_roughness,
            min_flat_radius_m=min_flat_radius_m,
        )
        for sid, lat_c, lon_c in parsed
    ]
    return rank_landing_sites(analyses)
