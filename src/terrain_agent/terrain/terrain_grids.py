"""Windowed terrain grids (elevation, slope, roughness) with explicit validity masks.

The requested window may extend past the raster edge. Cells outside the raster, nodata
cells, and cells whose 3x3 neighbourhood is incomplete are marked invalid. Nothing outside
the raster is ever invented, so callers can measure data coverage honestly.

Window size is bounded by the same 2048 x 2048 cell limit as every other raster read.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio.transform import Affine
from rasterio.windows import Window

from terrain_agent.terrain.dem_reader import read_dem_window
from terrain_agent.terrain.georef import DemGeoContext
from terrain_agent.terrain.resource_safety import validate_raster_dimensions
from terrain_agent.terrain.roughness import calculate_tri_grid
from terrain_agent.terrain.slope import calculate_slope_grid

#: Elevations at or beyond this magnitude are treated as nodata sentinels (project convention).
ELEVATION_SENTINEL_LIMIT_M = 30000.0


@dataclass
class TerrainGrids:
    """Terrain rasters for one window, in the DEM raster grid orientation."""

    row0: int
    col0: int
    height: int
    width: int
    transform: Affine
    elevation: np.ndarray
    slope_deg: np.ndarray
    slope_valid: np.ndarray
    tri: np.ndarray
    tri_valid: np.ndarray
    pixel_size_m: tuple[float, float]


def load_terrain_grids(
    ctx: DemGeoContext,
    row0: int,
    col0: int,
    height: int,
    width: int,
    *,
    ref_lat: float,
    ref_lon: float,
) -> TerrainGrids:
    """Read a window and compute slope and terrain ruggedness on it.

    Parameters
    ----------
    ctx:
        Georeferencing context from :func:`open_dem_context`.
    row0, col0:
        Window origin in raster cell coordinates. May be negative.
    height, width:
        Window size in cells. Must not exceed the raster read limit.
    ref_lat, ref_lon:
        Location used to convert cell size to metres.
    """
    validate_raster_dimensions(width, height)

    dx_m, dy_m = ctx.pixel_size_m(ref_lat, ref_lon)

    r_lo = max(row0, 0)
    c_lo = max(col0, 0)
    r_hi = min(row0 + height, ctx.height)
    c_hi = min(col0 + width, ctx.width)

    elevation = np.full((height, width), np.nan, dtype=np.float32)
    if r_hi > r_lo and c_hi > c_lo:
        data, _transform, metadata = read_dem_window(
            ctx.path, window=Window(c_lo, r_lo, c_hi - c_lo, r_hi - r_lo)
        )
        data = data.astype(np.float32, copy=True)
        if metadata.nodata_value is not None and np.isfinite(metadata.nodata_value):
            data[data == np.float32(metadata.nodata_value)] = np.nan
        data[~np.isfinite(data)] = np.nan
        data[np.abs(data) >= ELEVATION_SENTINEL_LIMIT_M] = np.nan
        elevation[r_lo - row0 : r_hi - row0, c_lo - col0 : c_hi - col0] = data

    slope_deg, slope_valid = calculate_slope_grid(elevation, dx_m, dy_m)
    tri, tri_valid = calculate_tri_grid(elevation)

    window_transform = ctx.transform * Affine.translation(col0, row0)
    return TerrainGrids(
        row0=row0,
        col0=col0,
        height=height,
        width=width,
        transform=window_transform,
        elevation=elevation,
        slope_deg=slope_deg,
        slope_valid=slope_valid,
        tri=tri,
        tri_valid=tri_valid,
        pixel_size_m=(dx_m, dy_m),
    )
