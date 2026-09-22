"""Deterministic DEM reader supporting GeoTIFF and GDAL-compatible planetary rasters with windowed reads."""

from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any
import math

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds
from pydantic import BaseModel, Field

from terrain_agent.terrain.resource_safety import (
    MalformedRasterError,
    OversizedRequestError,
    UnsupportedCRSError,
    InvalidCoordinateError,
    MAX_RASTER_DIM,
    validate_raster_dimensions,
)
from terrain_agent.terrain.coordinates import (
    validate_lunar_coordinate,
    validate_analysis_radius,
    LUNAR_RADIUS_METERS,
)


class RasterMetadata(BaseModel):
    """Metadata inspection container for a lunar raster."""
    path: str
    driver: str
    width: int
    height: int
    count: int
    crs_wkt: Optional[str]
    crs_epsg: Optional[int]
    bounds: Tuple[float, float, float, float]  # (left, bottom, right, top)
    res_x: float
    res_y: float
    nodata_value: Optional[float]
    is_polar: bool = False
    is_projected: bool = False


def inspect_dem(raster_path: Union[str, Path]) -> RasterMetadata:
    """Inspect DEM headers and metadata without loading raster cell data into memory."""
    path_obj = Path(raster_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"DEM raster file not found: {path_obj}")

    try:
        with rasterio.open(path_obj) as src:
            if src.width <= 0 or src.height <= 0:
                raise MalformedRasterError(
                    f"Malformed raster dimensions: {src.width}x{src.height} in {path_obj.name}"
                )
            if src.count < 1:
                raise MalformedRasterError(f"Raster contains no data bands: {path_obj.name}")

            crs_obj = src.crs
            if crs_obj is None:
                raise UnsupportedCRSError(
                    f"Raster {path_obj.name} does not contain valid CRS/projection metadata."
                )

            bounds_tuple = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            res_x, res_y = src.res

            if res_x <= 0 or res_y <= 0:
                raise MalformedRasterError(
                    f"Invalid raster resolution: ({res_x}, {res_y}) in {path_obj.name}"
                )

            # Determine if projection is polar stereographic
            crs_str = crs_obj.to_string().lower()
            is_polar = "stere" in crs_str or "polar" in crs_str
            is_projected = crs_obj.is_projected

            return RasterMetadata(
                path=str(path_obj.resolve()),
                driver=src.driver,
                width=src.width,
                height=src.height,
                count=src.count,
                crs_wkt=crs_obj.to_wkt(),
                crs_epsg=crs_obj.to_epsg(),
                bounds=bounds_tuple,
                res_x=float(res_x),
                res_y=float(res_y),
                nodata_value=float(src.nodata) if src.nodata is not None else None,
                is_polar=is_polar,
                is_projected=is_projected,
            )
    except (MalformedRasterError, UnsupportedCRSError):
        raise
    except (rasterio.errors.RasterioError, rasterio.errors.RasterioIOError, OSError) as e:
        raise MalformedRasterError(f"Failed to inspect raster {path_obj.name}: {e}") from e


def read_dem_window(
    raster_path: Union[str, Path],
    bounds: Optional[Tuple[float, float, float, float]] = None,
    window: Optional[Window] = None,
    allow_full_read: bool = False,
) -> Tuple[np.ndarray, Affine, RasterMetadata]:
    """Read a bounded window of elevation cells from a DEM raster.
    
    Guarantees:
    - Never reads full global datasets into memory unless allow_full_read is True.
    - Window dimension is validated to be <= MAX_RASTER_DIM (2048 cells).
    - Returns elevation array (float32), affine transform, and raster metadata.
    """
    path_obj = Path(raster_path)
    metadata = inspect_dem(path_obj)

    try:
        with rasterio.open(path_obj) as src:
            target_window: Window

            if window is not None:
                target_window = window
            elif bounds is not None:
                min_x, min_y, max_x, max_y = bounds
                # Verify bounds intersect raster
                r_left, r_bottom, r_right, r_top = metadata.bounds
                if max_x < r_left or min_x > r_right or max_y < r_bottom or min_y > r_top:
                    raise InvalidCoordinateError(
                        f"Requested bounds ({min_x:.1f}, {min_y:.1f}, {max_x:.1f}, {max_y:.1f}) "
                        f"do not intersect raster bounds ({r_left:.1f}, {r_bottom:.1f}, {r_right:.1f}, {r_top:.1f})."
                    )
                target_window = from_bounds(min_x, min_y, max_x, max_y, transform=src.transform)
            elif allow_full_read:
                target_window = Window(0, 0, src.width, src.height)
            else:
                # Disallow unwindowed global read
                if src.width > MAX_RASTER_DIM or src.height > MAX_RASTER_DIM:
                    raise OversizedRequestError(
                        f"Attempted unwindowed read of large raster ({src.width}x{src.height}). "
                        f"Provide explicit window bounds or set allow_full_read=True."
                    )
                target_window = Window(0, 0, src.width, src.height)

            # Clamp window to integer pixel offsets and bounds
            col_off = max(0, int(math.floor(target_window.col_off)))
            row_off = max(0, int(math.floor(target_window.row_off)))
            
            # Width and height calculation respecting raster bounds
            raw_w = int(math.ceil(target_window.width))
            raw_h = int(math.ceil(target_window.height))
            
            clamped_w = min(raw_w, src.width - col_off)
            clamped_h = min(raw_h, src.height - row_off)

            validate_raster_dimensions(clamped_w, clamped_h)

            final_window = Window(col_off=col_off, row_off=row_off, width=clamped_w, height=clamped_h)
            
            # Read single band into memory as float32
            data = src.read(1, window=final_window).astype(np.float32)
            win_transform = rasterio.windows.transform(final_window, src.transform)

            return data, win_transform, metadata
    except (InvalidCoordinateError, OversizedRequestError, MalformedRasterError, UnsupportedCRSError):
        raise
    except (rasterio.errors.RasterioError, rasterio.errors.RasterioIOError, OSError) as e:
        raise MalformedRasterError(f"Raster read error on {path_obj.name}: {e}") from e


def read_dem_around_point(
    raster_path: Union[str, Path],
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> Tuple[np.ndarray, Affine, RasterMetadata]:
    """Read a DEM sub-window centered around a lunar geographic coordinate."""
    clean_lat, clean_lon = validate_lunar_coordinate(center_lat, center_lon)
    clean_radius = validate_analysis_radius(radius_km)
    
    metadata = inspect_dem(raster_path)
    
    # Calculate radius in projected meter coordinates
    radius_m = clean_radius * 1000.0

    if metadata.is_projected:
        # Reproject or convert center_lat, center_lon to projection coordinates
        with rasterio.open(raster_path) as src:
            from rasterio.warp import transform as transform_coords
            xs, ys = transform_coords("EPSG:4326", src.crs, [clean_lon], [clean_lat])
            center_x, center_y = xs[0], ys[0]
            
            min_x = center_x - radius_m
            max_x = center_x + radius_m
            min_y = center_y - radius_m
            max_y = center_y + radius_m
            
            return read_dem_window(raster_path, bounds=(min_x, min_y, max_x, max_y))
    else:
        # Equirectangular / Angular degrees
        # Convert meter radius to degree radius on Moon
        deg_lat = (radius_m / LUNAR_RADIUS_METERS) * (180.0 / math.pi)
        cos_lat = max(math.cos(math.radians(clean_lat)), 0.01)
        deg_lon = deg_lat / cos_lat
        
        min_x = clean_lon - deg_lon
        max_x = clean_lon + deg_lon
        min_y = clean_lat - deg_lat
        max_y = clean_lat + deg_lat
        
        return read_dem_window(raster_path, bounds=(min_x, min_y, max_x, max_y))
