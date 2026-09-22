"""Resource safety constraints, exceptions, and execution guardrails for terrain analysis."""

from typing import Tuple, List
import math


# Strict operational resource limits
MAX_RASTER_DIM = 2048
MAX_CELLS_PER_OP = MAX_RASTER_DIM * MAX_RASTER_DIM  # 4,194,304 cells
MAX_ANALYSIS_RADIUS_KM = 50.0
MAX_PATH_LENGTH_KM = 250.0
MAX_WAYPOINTS = 100
MAX_ESTIMATED_MEMORY_MB = 128.0


class TerrainAnalysisError(Exception):
    """Base exception for all terrain analysis errors."""
    pass


class InvalidCoordinateError(TerrainAnalysisError):
    """Raised when coordinates violate lunar geographical bounds or are non-finite."""
    pass


class OversizedRequestError(TerrainAnalysisError):
    """Raised when an operation exceeds allowed memory, dimension, or radius limits."""
    pass


class MalformedRasterError(TerrainAnalysisError):
    """Raised when a DEM file has invalid dimensions, unreadable headers, or corrupt data."""
    pass


class UnsupportedCRSError(TerrainAnalysisError):
    """Raised when a DEM uses an unrecognized or unprojectable coordinate reference system."""
    pass


def validate_raster_dimensions(width: int, height: int) -> None:
    """Ensure raster dimensions are positive and do not exceed safety limits."""
    if width <= 0 or height <= 0:
        raise MalformedRasterError(
            f"Invalid raster dimensions: width={width}, height={height}. Dimensions must be positive."
        )
    total_cells = width * height
    if width > MAX_RASTER_DIM or height > MAX_RASTER_DIM or total_cells > MAX_CELLS_PER_OP:
        raise OversizedRequestError(
            f"Requested raster window ({width}x{height} = {total_cells:,} cells) exceeds "
            f"maximum safety limit of {MAX_RASTER_DIM}x{MAX_RASTER_DIM} ({MAX_CELLS_PER_OP:,} cells)."
        )


def estimate_memory_mb(width: int, height: int, bytes_per_cell: int = 4) -> float:
    """Estimate memory consumption in megabytes for a given grid dimension."""
    total_bytes = width * height * bytes_per_cell
    # Accounting for working buffers (elevation, slope, masks ~3x)
    return (total_bytes * 3.0) / (1024.0 * 1024.0)


def split_bounding_box(
    min_x: float, min_y: float, max_x: float, max_y: float, res_x: float, res_y: float, max_cells: int = MAX_RASTER_DIM
) -> List[Tuple[float, float, float, float]]:
    """Split an oversized geographic bounding box into safe, manageable sub-windows."""
    width_m = abs(max_x - min_x)
    height_m = abs(max_y - min_y)
    
    total_cols = max(1, int(math.ceil(width_m / abs(res_x))))
    total_rows = max(1, int(math.ceil(height_m / abs(res_y))))
    
    num_cols_split = math.ceil(total_cols / max_cells)
    num_rows_split = math.ceil(total_rows / max_cells)
    
    col_step = width_m / num_cols_split
    row_step = height_m / num_rows_split
    
    windows = []
    for r in range(num_rows_split):
        sub_min_y = min_y + r * row_step
        sub_max_y = min_y + (r + 1) * row_step
        for c in range(num_cols_split):
            sub_min_x = min_x + c * col_step
            sub_max_x = min_x + (c + 1) * col_step
            windows.append((sub_min_x, sub_min_y, sub_max_x, sub_max_y))
            
    return windows


# ---------------------------------------------------------------------------
# Phase 5 request-validation errors
# ---------------------------------------------------------------------------


class InvalidWaypointError(TerrainAnalysisError):
    """Raised when a traverse route is structurally invalid (count, shape, duplicates)."""
    pass


class InvalidThresholdError(TerrainAnalysisError):
    """Raised when a configured safety threshold is non-finite, non-positive, or out of range."""
    pass


class InvalidNoGoZoneError(TerrainAnalysisError):
    """Raised when a no-go zone definition is malformed, degenerate, or too large."""
    pass


class InvalidSiteError(TerrainAnalysisError):
    """Raised when a landing-site candidate definition is malformed or duplicated."""
    pass
