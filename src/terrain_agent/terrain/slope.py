"""Deterministic slope calculation from DEM gradients using Horn's finite difference method."""

import math
from typing import Optional, Tuple, Dict, List
import numpy as np
from scipy.ndimage import convolve
from pydantic import BaseModel, Field

from terrain_agent.terrain.resource_safety import TerrainAnalysisError


class SlopeStats(BaseModel):
    """Deterministic slope characterization metrics in degrees."""
    mean_slope_deg: float = Field(description="Mean surface slope in degrees")
    median_slope_deg: float = Field(description="Median surface slope in degrees")
    max_slope_deg: float = Field(description="Maximum observed slope in degrees")
    percentiles: Dict[str, float] = Field(description="Slope distribution percentiles (p10, p25, p50, p75, p90, p95, p99)")
    histogram_counts: List[int] = Field(description="Histogram bin counts (e.g. 10 bins across 0-90°)")
    histogram_bin_edges: List[float] = Field(description="Histogram bin edge boundaries in degrees")
    valid_cells: int = Field(description="Count of valid slope calculation cells")


def calculate_slope_grid(
    elevation_array: np.ndarray,
    res_x_m: float,
    res_y_m: float,
    nodata_val: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate 2D slope grid in degrees using Horn's (1981) 3x3 finite-difference convolution.
    
    Returns:
        Tuple of (slope_degrees_grid, valid_mask)
    """
    if elevation_array is None or not isinstance(elevation_array, np.ndarray):
        raise TerrainAnalysisError("elevation_array must be a valid numpy array.")
    
    if elevation_array.ndim != 2:
        raise TerrainAnalysisError(f"elevation_array must be 2-dimensional, got {elevation_array.ndim}D.")

    if elevation_array.shape[0] < 3 or elevation_array.shape[1] < 3:
        raise TerrainAnalysisError(
            f"Raster dimensions {elevation_array.shape} are too small for 3x3 gradient computation (minimum 3x3 required)."
        )

    res_x = abs(float(res_x_m))
    res_y = abs(float(res_y_m))
    if res_x <= 0 or res_y <= 0:
        raise TerrainAnalysisError(f"Grid resolution must be strictly positive, got ({res_x}, {res_y}).")

    # Mask invalid cells
    valid_elevation = np.isfinite(elevation_array)
    if nodata_val is not None and not np.isnan(nodata_val):
        valid_elevation = valid_elevation & (elevation_array != nodata_val)
    valid_elevation = valid_elevation & (elevation_array > -30000.0) & (elevation_array < 30000.0)

    # Horn's 3x3 convolution kernels
    # Horizontal kernel (dz/dx)
    kx = np.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0]
    ], dtype=np.float32) / (8.0 * res_x)

    # Vertical kernel (dz/dy)
    ky = np.array([
        [-1.0, -2.0, -1.0],
        [ 0.0,  0.0,  0.0],
        [ 1.0,  2.0,  1.0]
    ], dtype=np.float32) / (8.0 * res_y)

    # Convolve with constant reflection
    dz_dx = convolve(elevation_array, kx, mode="nearest")
    dz_dy = convolve(elevation_array, ky, mode="nearest")

    # Slope gradient magnitude
    gradient_mag = np.sqrt(dz_dx**2 + dz_dy**2)
    slope_rad = np.arctan(gradient_mag)
    slope_deg = np.degrees(slope_rad)

    # To avoid edge artifacts, invalidate borders and cells touching nodata
    valid_mask = valid_elevation.copy()
    valid_mask[0, :] = False
    valid_mask[-1, :] = False
    valid_mask[:, 0] = False
    valid_mask[:, -1] = False

    # Mask cells whose 3x3 neighbors contain nodata
    invalid_mask = ~valid_elevation
    if np.any(invalid_mask):
        neighbor_invalid = convolve(invalid_mask.astype(np.float32), np.ones((3, 3)), mode="constant", cval=1.0) > 0
        valid_mask = valid_mask & (~neighbor_invalid)

    slope_deg[~valid_mask] = np.nan
    return slope_deg, valid_mask


def calculate_slope_stats(
    elevation_array: np.ndarray,
    res_x_m: float,
    res_y_m: float,
    nodata_val: Optional[float] = None,
) -> SlopeStats:
    """Calculate slope metrics, percentiles, and histogram distribution across a DEM window."""
    slope_grid, valid_mask = calculate_slope_grid(
        elevation_array, res_x_m, res_y_m, nodata_val=nodata_val
    )
    
    valid_slopes = slope_grid[valid_mask]
    if valid_slopes.size == 0:
        raise TerrainAnalysisError("No valid slope cells could be computed in the provided DEM window.")

    mean_val = float(np.mean(valid_slopes))
    median_val = float(np.median(valid_slopes))
    max_val = float(np.max(valid_slopes))

    percentiles = {
        "p10": round(float(np.percentile(valid_slopes, 10)), 2),
        "p25": round(float(np.percentile(valid_slopes, 25)), 2),
        "p50": round(float(np.percentile(valid_slopes, 50)), 2),
        "p75": round(float(np.percentile(valid_slopes, 75)), 2),
        "p90": round(float(np.percentile(valid_slopes, 90)), 2),
        "p95": round(float(np.percentile(valid_slopes, 95)), 2),
        "p99": round(float(np.percentile(valid_slopes, 99)), 2),
    }

    # Generate 10-bin histogram up to 60° (or max slope + 5°)
    max_hist_edge = max(30.0, math.ceil(max_val / 5.0) * 5.0)
    counts, bin_edges = np.histogram(valid_slopes, bins=10, range=(0.0, max_hist_edge))

    return SlopeStats(
        mean_slope_deg=round(mean_val, 2),
        median_slope_deg=round(median_val, 2),
        max_slope_deg=round(max_val, 2),
        percentiles=percentiles,
        histogram_counts=[int(c) for c in counts],
        histogram_bin_edges=[round(float(b), 2) for b in bin_edges],
        valid_cells=int(valid_slopes.size),
    )
