"""Deterministic terrain roughness computation using the Terrain Ruggedness Index (TRI)."""

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import generic_filter
from pydantic import BaseModel, Field

from terrain_agent.terrain.resource_safety import TerrainAnalysisError


class RoughnessStats(BaseModel):
    """Deterministic terrain roughness metrics container."""
    mean_tri_m: float = Field(description="Mean Terrain Ruggedness Index in meters")
    median_tri_m: float = Field(description="Median Terrain Ruggedness Index in meters")
    max_tri_m: float = Field(description="Maximum observed Terrain Ruggedness Index in meters")
    std_tri_m: float = Field(description="Standard deviation of Terrain Ruggedness Index in meters")
    formula: str = Field(description="Mathematical definition of the computed metric")
    disclaimer: str = Field(description="Scientific limitation and non-standard disclaimer")
    valid_cells: int = Field(description="Count of valid cells evaluated")


def _tri_kernel_3x3(neighbors: np.ndarray) -> float:
    """Calculate mean absolute elevation difference between focal center cell (index 4) and 8 neighbors."""
    center = neighbors[4]
    if not np.isfinite(center):
        return np.nan
    # Indices 0..3 and 5..8 are the 8 neighbors
    diffs = np.abs(neighbors - center)
    diffs[4] = 0.0  # exclude self
    valid = np.isfinite(diffs)
    valid_count = np.sum(valid) - 1  # exclude center
    if valid_count < 4:
        return np.nan
    return np.sum(diffs[valid]) / valid_count


def calculate_tri_grid(
    elevation_array: np.ndarray,
    nodata_val: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the 2D Terrain Ruggedness Index (TRI; Riley et al. 1999) grid in meters.
    
    Formula:
        TRI(x, y) = (1 / 8) * sum(|z(x+i, y+j) - z(x, y)|) for (i, j) in {-1, 0, 1}^2 \\ {(0, 0)}
    
    Returns:
        Tuple of (tri_grid_meters, valid_mask)
    """
    if elevation_array is None or not isinstance(elevation_array, np.ndarray):
        raise TerrainAnalysisError("elevation_array must be a valid numpy array.")
    
    if elevation_array.ndim != 2:
        raise TerrainAnalysisError(f"elevation_array must be 2D, got {elevation_array.ndim}D.")

    if elevation_array.shape[0] < 3 or elevation_array.shape[1] < 3:
        raise TerrainAnalysisError(
            f"Raster shape {elevation_array.shape} too small for 3x3 roughness computation."
        )

    grid = elevation_array.astype(np.float64).copy()
    
    # Mask nodata
    if nodata_val is not None and not np.isnan(nodata_val):
        grid[grid == nodata_val] = np.nan
    grid[(grid <= -30000.0) | (grid >= 30000.0)] = np.nan

    # Vectorized computation of the 8 neighbor differences for fast execution
    padded = np.pad(grid, pad_width=1, mode="edge")
    h, w = grid.shape
    
    sum_diff = np.zeros((h, w), dtype=np.float64)
    valid_neighbor_count = np.zeros((h, w), dtype=np.int32)
    
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            neighbor = padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
            valid = np.isfinite(neighbor) & np.isfinite(grid)
            diff = np.where(valid, np.abs(neighbor - grid), 0.0)
            sum_diff += diff
            valid_neighbor_count += valid.astype(np.int32)
            
    # Mask cells with fewer than 4 valid neighbors or border cells
    valid_mask = (valid_neighbor_count >= 4) & np.isfinite(grid)
    valid_mask[0, :] = False
    valid_mask[-1, :] = False
    valid_mask[:, 0] = False
    valid_mask[:, -1] = False

    tri_grid = np.full((h, w), np.nan, dtype=np.float32)
    tri_grid[valid_mask] = (sum_diff[valid_mask] / valid_neighbor_count[valid_mask]).astype(np.float32)
    
    return tri_grid, valid_mask


def calculate_roughness_stats(
    elevation_array: np.ndarray,
    nodata_val: Optional[float] = None,
) -> RoughnessStats:
    """Calculate deterministic roughness metrics across a DEM window."""
    tri_grid, valid_mask = calculate_tri_grid(elevation_array, nodata_val=nodata_val)
    
    valid_tri = tri_grid[valid_mask]
    if valid_tri.size == 0:
        raise TerrainAnalysisError("No valid roughness cells could be computed in the provided DEM window.")

    mean_val = float(np.mean(valid_tri))
    median_val = float(np.median(valid_tri))
    max_val = float(np.max(valid_tri))
    std_val = float(np.std(valid_tri))

    formula_text = "TRI = (1 / N) * sum(|z_neighbor - z_center|) for 8-connected local neighborhood (Riley et al. 1999)"
    disclaimer_text = (
        "Scientific Notice: Terrain Ruggedness Index (TRI) is a localized elevation variability metric. "
        "It is NOT a certified or universal lunar rover safety standard. True surface traversability depends "
        "on sub-pixel boulder hazards, geotechnical regolith properties, and vehicle-specific mobility parameters."
    )

    return RoughnessStats(
        mean_tri_m=round(mean_val, 2),
        median_tri_m=round(median_val, 2),
        max_tri_m=round(max_val, 2),
        std_tri_m=round(std_val, 2),
        formula=formula_text,
        disclaimer=disclaimer_text,
        valid_cells=int(valid_tri.size),
    )
