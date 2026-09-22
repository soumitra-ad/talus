"""Deterministic elevation statistics computation from raster arrays."""

from typing import Optional, Union
import numpy as np
from pydantic import BaseModel, Field

from terrain_agent.terrain.resource_safety import TerrainAnalysisError


class ElevationStats(BaseModel):
    """Deterministic elevation metrics container."""
    min_m: float = Field(description="Minimum elevation in meters")
    max_m: float = Field(description="Maximum elevation in meters")
    mean_m: float = Field(description="Mean elevation in meters")
    median_m: float = Field(description="Median elevation in meters")
    std_m: float = Field(description="Standard deviation of elevation in meters")
    valid_cell_count: int = Field(description="Count of valid, non-nodata cells")
    nodata_count: int = Field(description="Count of nodata or NaN cells")


def get_elevation_stats(
    elevation_array: np.ndarray,
    nodata_val: Optional[float] = None,
) -> ElevationStats:
    """Calculate deterministic elevation metrics on a 2D numpy array, handling nodata and non-finite cells."""
    if elevation_array is None or not isinstance(elevation_array, np.ndarray):
        raise TerrainAnalysisError("elevation_array must be a valid numpy array.")
    
    if elevation_array.size == 0:
        raise TerrainAnalysisError("elevation_array is empty.")

    # Build boolean mask of valid cells
    valid_mask = np.isfinite(elevation_array)
    
    if nodata_val is not None and not np.isnan(nodata_val):
        valid_mask = valid_mask & (elevation_array != nodata_val)
        
    # Also mask common planetary nodata sentinels (< -30000 m or > 30000 m relative to lunar datum)
    valid_mask = valid_mask & (elevation_array > -30000.0) & (elevation_array < 30000.0)
    
    valid_cells = elevation_array[valid_mask]
    total_cells = elevation_array.size
    valid_count = int(valid_cells.size)
    nodata_count = total_cells - valid_count
    
    if valid_count == 0:
        raise TerrainAnalysisError("No valid elevation cells found in the requested DEM window (all cells are nodata).")
        
    min_val = float(np.min(valid_cells))
    max_val = float(np.max(valid_cells))
    mean_val = float(np.mean(valid_cells))
    median_val = float(np.median(valid_cells))
    std_val = float(np.std(valid_cells))
    
    return ElevationStats(
        min_m=round(min_val, 2),
        max_m=round(max_val, 2),
        mean_m=round(mean_val, 2),
        median_m=round(median_val, 2),
        std_m=round(std_val, 2),
        valid_cell_count=valid_count,
        nodata_count=nodata_count,
    )
