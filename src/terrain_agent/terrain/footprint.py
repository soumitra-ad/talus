"""Circular analysis footprint around a point, with metric distances per cell.

Used by landing-site analysis and safe-region detection. The footprint window may extend
past the raster edge. Cells outside the raster are invalid, so coverage is measured, not
assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from terrain_agent.terrain.georef import DemGeoContext
from terrain_agent.terrain.resource_safety import UnsupportedCRSError
from terrain_agent.terrain.route_analysis import PAD_CELLS
from terrain_agent.terrain.terrain_grids import TerrainGrids, load_terrain_grids


@dataclass
class Footprint:
    """Terrain grids around a point plus each cell distance from that point."""

    grids: TerrainGrids
    distance_m: np.ndarray
    mask: np.ndarray
    center_row_col: tuple[float, float]


def load_footprint(ctx: DemGeoContext, lat: float, lon: float, radius_m: float) -> Footprint:
    """Load terrain grids for a circle of *radius_m* around a location.

    The cell containing the location is always inside the footprint mask, even when the
    radius is smaller than one cell.

    Raises
    ------
    UnsupportedCRSError
        If the location cannot be placed in the DEM coordinate system, or metric cell size
        cannot be determined.
    OversizedRequestError
        If the footprint needs a window larger than the raster read limit.
    """
    row_f, col_f, finite = ctx.latlon_to_rowcol_float([lat], [lon])
    if not bool(finite[0]):
        raise UnsupportedCRSError(
            f"Location ({lat:.4f}, {lon:.4f}) cannot be placed in the coordinate system of "
            f"{ctx.path.name}."
        )
    row_pos = float(np.clip(row_f[0], -1e9, 1e9))
    col_pos = float(np.clip(col_f[0], -1e9, 1e9))

    dx_m, dy_m = ctx.pixel_size_m(lat, lon)
    half_rows = math.ceil(radius_m / dy_m) + PAD_CELLS
    half_cols = math.ceil(radius_m / dx_m) + PAD_CELLS
    row_c = math.floor(row_pos)
    col_c = math.floor(col_pos)
    row0, col0 = row_c - half_rows, col_c - half_cols
    height, width = 2 * half_rows + 1, 2 * half_cols + 1

    grids = load_terrain_grids(ctx, row0, col0, height, width, ref_lat=lat, ref_lon=lon)

    rows_idx, cols_idx = np.mgrid[0:height, 0:width]
    d_row = (rows_idx + 0.5 + row0 - row_pos) * dy_m
    d_col = (cols_idx + 0.5 + col0 - col_pos) * dx_m
    distance = np.hypot(d_col, d_row)
    mask = distance <= radius_m
    mask[row_c - row0, col_c - col0] = True

    return Footprint(
        grids=grids, distance_m=distance, mask=mask, center_row_col=(row_pos, col_pos)
    )
