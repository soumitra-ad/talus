"""Deterministic safe-region detection on a DEM (research/demo).

Method
------
1. Load terrain grids for a circular footprint around a centre point.
2. A cell is safe only if it was measured (valid Horn slope) and its slope does not exceed
   ``maximum_slope_deg`` and, if a roughness limit is given, its own TRI does not exceed
   ``maximum_roughness``. Unmeasured cells are never safe.
3. Safe cells are grouped into 4-connected regions. Diagonal-only contact does not join
   regions, which is the conservative choice.
4. Regions smaller than ``min_area_m2`` are dropped. Area is cell count times metric cell area.
5. Each region reports its centroid, its largest inscribed circle (distance to the nearest
   cell that is not safe, reduced by half a cell), and its slope, TRI and elevation statistics.

The outcome distinguishes "no safe region among the cells that were measured" from "no
terrain data". An empty result is never reported as unsafe terrain when the terrain was
not measured.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel
from scipy import ndimage

from terrain_agent.data.dataset_info import DatasetInfo, build_dataset_info
from terrain_agent.safety.rover import DISCLAIMER, safe_error_text
from terrain_agent.terrain.coordinates import validate_analysis_radius, validate_lunar_coordinate
from terrain_agent.terrain.footprint import load_footprint
from terrain_agent.terrain.georef import DemGeoContext, open_dem_context
from terrain_agent.terrain.resource_safety import (
    InvalidThresholdError,
    OversizedRequestError,
    TerrainAnalysisError,
)

DEFAULT_MIN_REGION_AREA_M2 = 10_000.0
DEFAULT_MAX_REGIONS = 20
MAX_REGIONS_LIMIT = 100

LIMITATIONS: tuple[str, ...] = (
    "Safe means only that measured cells satisfy the configured slope and roughness limits on "
    "this DEM. It is not a landing or traverse approval.",
    "Hazards smaller than one DEM cell are not resolved, and coarse DEMs understate local slopes.",
    "Per-cell TRI depends on DEM resolution. A roughness limit chosen for one DEM may not suit another.",
    "Illumination, communication, thermal and mobility constraints are not modelled.",
    "The largest inscribed circle is accurate to about one cell.",
)


class SafeRegion(BaseModel):
    """One connected area of measured cells that satisfy the configured limits."""

    region_id: str
    area_m2: float
    cells: int
    centroid_lat: float
    centroid_lon: float
    best_point_lat: float
    best_point_lon: float
    largest_inscribed_circle_radius_m: float
    mean_slope_deg: float
    max_slope_deg: float
    mean_tri_m: float
    mean_elevation_m: float


class SafeRegionResult(BaseModel):
    """Result of safe-region detection."""

    outcome: Literal["regions_found", "no_safe_region_in_assessed_area", "no_terrain_data"]
    terrain_available: bool
    center_lat: float
    center_lon: float
    footprint_radius_m: float
    footprint_cells: int
    cells_measured: int
    assessed_fraction: float
    safe_fraction_of_assessed: Optional[float]
    maximum_slope_deg: float
    maximum_roughness_tri_m: Optional[float]
    min_area_m2: float
    regions: list[SafeRegion]
    regions_total_found: int
    regions_truncated: bool
    dataset: Optional[DatasetInfo]
    threshold_statement: str
    warnings: list[str]
    limitations: list[str] = list(LIMITATIONS)
    disclaimer: str = DISCLAIMER


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidThresholdError(f"{name} must be a number.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or (number == 0.0 and not allow_zero):
        raise InvalidThresholdError(f"{name} must be a positive finite number, got {value!r}.")
    return number


def _cell_xy(transform, row: float, col: float) -> tuple[float, float]:
    return (
        transform.a * col + transform.b * row + transform.c,
        transform.d * col + transform.e * row + transform.f,
    )


def find_safe_regions(
    dem_path: Any,
    center_lat: float,
    center_lon: float,
    radius_m: float,
    *,
    maximum_slope_deg: float,
    maximum_roughness: Optional[float] = None,
    min_area_m2: float = DEFAULT_MIN_REGION_AREA_M2,
    max_regions: int = DEFAULT_MAX_REGIONS,
) -> SafeRegionResult:
    """Find connected areas around a point whose measured cells meet the configured limits.

    Raises
    ------
    InvalidCoordinateError
        Invalid coordinates or a radius outside the allowed range.
    InvalidThresholdError
        A threshold or count is not a finite number in its allowed range.
    OversizedRequestError
        The footprint needs a window larger than the raster read limit.

    A missing or unreadable DEM does not raise. The outcome is ``no_terrain_data``.
    """
    lat, lon = validate_lunar_coordinate(center_lat, center_lon)
    radius = _positive(radius_m, "radius_m")
    validate_analysis_radius(radius / 1000.0)
    max_slope = _positive(maximum_slope_deg, "maximum_slope_deg")
    if max_slope > 90.0:
        raise InvalidThresholdError("maximum_slope_deg must be at most 90.")
    max_rough = None if maximum_roughness is None else _positive(maximum_roughness, "maximum_roughness")
    min_area = _positive(min_area_m2, "min_area_m2", allow_zero=True)
    if isinstance(max_regions, bool) or not isinstance(max_regions, int) or not (1 <= max_regions <= MAX_REGIONS_LIMIT):
        raise InvalidThresholdError(f"max_regions must be an integer from 1 to {MAX_REGIONS_LIMIT}.")

    statement = f"Configured analysis threshold: {max_slope:g}° slope"
    if max_rough is not None:
        statement += f", {max_rough:g} m TRI per cell"
    statement += ". This is NOT a certified safety limit."

    def empty(reason: str, dataset: Optional[DatasetInfo] = None) -> SafeRegionResult:
        return SafeRegionResult(
            outcome="no_terrain_data",
            terrain_available=False,
            center_lat=lat,
            center_lon=lon,
            footprint_radius_m=radius,
            footprint_cells=0,
            cells_measured=0,
            assessed_fraction=0.0,
            safe_fraction_of_assessed=None,
            maximum_slope_deg=max_slope,
            maximum_roughness_tri_m=max_rough,
            min_area_m2=min_area,
            regions=[],
            regions_total_found=0,
            regions_truncated=False,
            dataset=dataset,
            threshold_statement=statement,
            warnings=[reason],
        )

    if dem_path is None:
        return empty("No DEM was provided; terrain could not be assessed.")
    try:
        ctx: DemGeoContext = open_dem_context(dem_path)
    except FileNotFoundError:
        from pathlib import Path

        return empty(f"DEM file not found: {Path(str(dem_path)).name}")
    except TerrainAnalysisError as exc:
        return empty(f"DEM could not be used: {safe_error_text(exc, dem_path)}")

    dataset = build_dataset_info(ctx, lat, lon)
    try:
        footprint = load_footprint(ctx, lat, lon, radius)
    except OversizedRequestError:
        raise
    except TerrainAnalysisError as exc:
        return empty(f"Terrain could not be measured: {safe_error_text(exc, ctx.path)}", dataset)

    grids, dist, mask = footprint.grids, footprint.distance_m, footprint.mask
    dx_m, dy_m = grids.pixel_size_m
    cell_area = dx_m * dy_m
    footprint_cells = int(mask.sum())
    assessed = mask & grids.slope_valid
    cells_measured = int(assessed.sum())
    warnings: list[str] = list(dataset.warnings)

    if cells_measured == 0:
        result = empty("No cell in the footprint could be measured.", dataset)
        result.footprint_cells = footprint_cells
        result.warnings = warnings + result.warnings
        return result

    safe = assessed & (grids.slope_deg <= max_slope)
    if max_rough is not None:
        safe &= grids.tri_valid & (grids.tri <= max_rough)

    assessed_fraction = cells_measured / footprint_cells
    if assessed_fraction < 1.0:
        warnings.append(
            f"Only {100.0 * assessed_fraction:.1f}% of the footprint was measured. Unmeasured "
            "cells are not treated as safe, and the unmeasured area may contain safe terrain."
        )

    labels, n_labels = ndimage.label(safe)
    counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
    keep = [k for k in range(1, n_labels + 1) if counts[k] * cell_area >= min_area]

    regions: list[SafeRegion] = []
    if keep:
        padded = np.pad(safe, 1, constant_values=False)
        edt = ndimage.distance_transform_edt(padded, sampling=(dy_m, dx_m))[1:-1, 1:-1]
        half_cell = 0.5 * max(dx_m, dy_m)
        radii = ndimage.maximum(edt, labels, index=keep)
        best_pos = ndimage.maximum_position(edt, labels, index=keep)
        centroids = ndimage.center_of_mass(safe, labels, index=keep)
        mean_slope = ndimage.mean(grids.slope_deg, labels, index=keep)
        max_slope_v = ndimage.maximum(grids.slope_deg, labels, index=keep)
        mean_tri = ndimage.mean(grids.tri, labels, index=keep)
        mean_elev = ndimage.mean(grids.elevation, labels, index=keep)

        order = sorted(
            range(len(keep)), key=lambda i: (-float(radii[i]), -int(counts[keep[i]]), keep[i])
        )
        for position, i in enumerate(order[:max_regions]):
            c_row, c_col = centroids[i]
            b_row, b_col = best_pos[i]
            cx, cy = _cell_xy(grids.transform, c_row + 0.5, c_col + 0.5)
            bx, by = _cell_xy(grids.transform, b_row + 0.5, b_col + 0.5)
            lats, lons = ctx.to_latlon([cx, bx], [cy, by])
            regions.append(
                SafeRegion(
                    region_id=f"R{position + 1}",
                    area_m2=round(float(counts[keep[i]] * cell_area), 1),
                    cells=int(counts[keep[i]]),
                    centroid_lat=round(float(lats[0]), 6),
                    centroid_lon=round(float(lons[0]), 6),
                    best_point_lat=round(float(lats[1]), 6),
                    best_point_lon=round(float(lons[1]), 6),
                    largest_inscribed_circle_radius_m=round(max(0.0, float(radii[i]) - half_cell), 1),
                    mean_slope_deg=round(float(mean_slope[i]), 2),
                    max_slope_deg=round(float(max_slope_v[i]), 2),
                    mean_tri_m=round(float(mean_tri[i]), 3),
                    mean_elevation_m=round(float(mean_elev[i]), 2),
                )
            )

    safe_fraction = float(safe.sum()) / cells_measured
    return SafeRegionResult(
        outcome="regions_found" if regions else "no_safe_region_in_assessed_area",
        terrain_available=True,
        center_lat=lat,
        center_lon=lon,
        footprint_radius_m=radius,
        footprint_cells=footprint_cells,
        cells_measured=cells_measured,
        assessed_fraction=round(assessed_fraction, 4),
        safe_fraction_of_assessed=round(safe_fraction, 4),
        maximum_slope_deg=max_slope,
        maximum_roughness_tri_m=max_rough,
        min_area_m2=min_area,
        regions=regions,
        regions_total_found=len(keep),
        regions_truncated=len(keep) > len(regions),
        dataset=dataset,
        threshold_statement=statement,
        warnings=warnings,
    )
