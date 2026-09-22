"""Terrain statistics (elevation, slope, roughness) for a latitude/longitude bounding box.

The box is converted to the smallest enclosing window in the DEM own coordinates by
sampling its edges and interior. Statistics cover that window. For polar projections the
enclosing window can extend beyond the exact latitude/longitude box. Cells outside the raster
are reported through coverage and are never invented.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel

from terrain_agent.data.dataset_info import DatasetInfo, build_dataset_info
from terrain_agent.safety.rover import DISCLAIMER, safe_error_text
from terrain_agent.terrain.coordinates import validate_lunar_coordinate
from terrain_agent.terrain.elevation import ElevationStats, get_elevation_stats
from terrain_agent.terrain.georef import open_dem_context
from terrain_agent.terrain.resource_safety import (
    InvalidCoordinateError,
    OversizedRequestError,
    TerrainAnalysisError,
)
from terrain_agent.terrain.roughness import RoughnessStats, calculate_roughness_stats
from terrain_agent.terrain.slope import SlopeStats, calculate_slope_stats
from terrain_agent.terrain.route_analysis import PAD_CELLS
from terrain_agent.terrain.terrain_grids import load_terrain_grids

_EDGE_SAMPLES = 9

LIMITATIONS: tuple[str, ...] = (
    "Statistics are from a DEM at its native cell size and describe the enclosing window of the "
    "requested box.",
    "Slope is the gradient magnitude from Horn's method. TRI depends on DEM resolution.",
    "Hazards smaller than one cell are not resolved.",
)


class TerrainRegionStats(BaseModel):
    """Elevation, slope and roughness statistics for a bounding box."""

    terrain_available: bool
    window_cells: int
    cells_measured: int
    coverage_fraction: float
    resolution_m: Optional[float]
    elevation: Optional[ElevationStats]
    slope: Optional[SlopeStats]
    roughness: Optional[RoughnessStats]
    dataset: Optional[DatasetInfo]
    warnings: list[str]
    limitations: list[str] = list(LIMITATIONS)
    disclaimer: str = DISCLAIMER


def analyze_terrain_bbox(
    dem_path: Any,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> TerrainRegionStats:
    """Compute terrain statistics for a latitude/longitude box.

    Raises
    ------
    InvalidCoordinateError
        Invalid coordinates, a box with min greater than or equal to max, or a box that
        crosses the antimeridian.
    OversizedRequestError
        The box needs a window larger than the raster read limit.

    A missing or unreadable DEM does not raise. The result has ``terrain_available`` false.
    """
    lat_lo, lon_lo = validate_lunar_coordinate(min_lat, min_lon)
    lat_hi, lon_hi = validate_lunar_coordinate(max_lat, max_lon)
    if not lat_lo < lat_hi:
        raise InvalidCoordinateError("min_lat must be less than max_lat.")
    if not lon_lo < lon_hi:
        raise InvalidCoordinateError(
            "min_lon must be less than max_lon after normalising to -180..180. Boxes that "
            "cross the antimeridian are not supported."
        )

    def unavailable(reason: str, dataset: Optional[DatasetInfo] = None) -> TerrainRegionStats:
        return TerrainRegionStats(
            terrain_available=False,
            window_cells=0,
            cells_measured=0,
            coverage_fraction=0.0,
            resolution_m=None,
            elevation=None,
            slope=None,
            roughness=None,
            dataset=dataset,
            warnings=[reason],
        )

    if dem_path is None:
        return unavailable("No DEM was provided; terrain could not be assessed.")
    try:
        ctx = open_dem_context(dem_path)
    except FileNotFoundError:
        return unavailable(f"DEM file not found: {Path(str(dem_path)).name}")
    except TerrainAnalysisError as exc:
        return unavailable(f"DEM could not be used: {safe_error_text(exc, dem_path)}")

    ref_lat, ref_lon = 0.5 * (lat_lo + lat_hi), 0.5 * (lon_lo + lon_hi)
    dataset = build_dataset_info(ctx, ref_lat, ref_lon)

    lat_grid, lon_grid = np.meshgrid(
        np.linspace(lat_lo, lat_hi, _EDGE_SAMPLES), np.linspace(lon_lo, lon_hi, _EDGE_SAMPLES)
    )
    row_f, col_f, finite = ctx.latlon_to_rowcol_float(lat_grid.ravel(), lon_grid.ravel())
    if not finite.any():
        return unavailable("The box cannot be placed in the DEM coordinate system.", dataset)
    row_min = int(math.floor(float(np.clip(row_f[finite].min(), -1e9, 1e9)))) - PAD_CELLS
    row_max = int(math.ceil(float(np.clip(row_f[finite].max(), -1e9, 1e9)))) + PAD_CELLS
    col_min = int(math.floor(float(np.clip(col_f[finite].min(), -1e9, 1e9)))) - PAD_CELLS
    col_max = int(math.ceil(float(np.clip(col_f[finite].max(), -1e9, 1e9)))) + PAD_CELLS

    try:
        grids = load_terrain_grids(
            ctx,
            row_min,
            col_min,
            row_max - row_min + 1,
            col_max - col_min + 1,
            ref_lat=ref_lat,
            ref_lon=ref_lon,
        )
    except OversizedRequestError:
        raise
    except TerrainAnalysisError as exc:
        return unavailable(f"Terrain could not be measured: {safe_error_text(exc, ctx.path)}", dataset)

    dx_m, dy_m = grids.pixel_size_m
    # Coverage is judged on the requested box, without the padding ring added for the
    # slope and roughness neighbourhoods.
    core = (slice(PAD_CELLS, -PAD_CELLS), slice(PAD_CELLS, -PAD_CELLS))
    window_cells = (grids.height - 2 * PAD_CELLS) * (grids.width - 2 * PAD_CELLS)
    measured = int(grids.slope_valid[core].sum())
    warnings = list(dataset.warnings)
    if measured < window_cells:
        warnings.append(
            f"Only {100.0 * measured / window_cells:.1f}% of the requested area has measurable "
            "slope (outside the DEM, nodata, or DEM edge)."
        )

    elevation = slope = roughness = None
    try:
        elevation = get_elevation_stats(grids.elevation)
    except TerrainAnalysisError:
        warnings.append("No valid elevation cells in the window.")
    try:
        slope = calculate_slope_stats(grids.elevation, dx_m, dy_m)
    except TerrainAnalysisError:
        warnings.append("Slope could not be computed for the window.")
    try:
        roughness = calculate_roughness_stats(grids.elevation)
    except TerrainAnalysisError:
        warnings.append("Roughness could not be computed for the window.")

    return TerrainRegionStats(
        terrain_available=elevation is not None,
        window_cells=window_cells,
        cells_measured=measured,
        coverage_fraction=round(measured / window_cells, 4),
        resolution_m=round(0.5 * (dx_m + dy_m), 3),
        elevation=elevation,
        slope=slope,
        roughness=roughness,
        dataset=dataset,
        warnings=warnings,
    )
