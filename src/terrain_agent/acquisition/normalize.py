"""Convert a validated PDS4 product into a GeoTIFF of elevation in metres.

Why: the terrain engine reads elevation values directly and does not apply raster scale and
offset metadata. NASA products store raw integers or kilometres. Converting once, with the rule
stated in the product label, gives the engine values it can use unchanged.

Rule: elevation (metres) = raw value * scaling_factor * unit factor, relative to the reference
sphere of the label. This is height above a sphere, not above a geoid.

The conversion streams in row blocks, so memory use is bounded regardless of raster size. The
output is a tiled, deflate-compressed float32 GeoTIFF with the source CRS and geotransform.
Invalid values are written as nodata and counted. The same plausibility limit the engine uses
(plus or minus 30 km) marks garbage values invalid.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window

from terrain_agent.acquisition.errors import DemValidationError
from terrain_agent.acquisition.pds_validation import HeightConvention
from terrain_agent.terrain.dem_reader import inspect_dem
from terrain_agent.terrain.resource_safety import TerrainAnalysisError
from terrain_agent.terrain.terrain_grids import ELEVATION_SENTINEL_LIMIT_M

NORMALIZATION_VERSION = "1"
NODATA_M = -9999.0
BLOCK_ROWS = 256


@dataclass(frozen=True)
class NormalizedDem:
    """Result of normalisation."""

    path: Path
    size_bytes: int
    sha256: str
    total_cells: int
    masked_cells: int
    min_m: float
    max_m: float


def normalize_to_geotiff(
    label_path: Path,
    out_path: Path,
    convention: HeightConvention,
    *,
    block_rows: int = BLOCK_ROWS,
    tags: Optional[dict[str, str]] = None,
) -> NormalizedDem:
    """Write ``out_path`` from a validated PDS4 product. Raises on insufficient disk space."""
    out_path = Path(out_path)
    factor = convention.factor_to_metres
    total = 0
    masked = 0
    low, high = float("inf"), float("-inf")

    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        with rasterio.open(label_path) as src:
            needed = src.width * src.height * 4
            free = shutil.disk_usage(out_path.parent).free
            if free < needed * 1.2:
                raise DemValidationError(
                    "disk_space", "There is not enough free disk space to normalise this product."
                )
            profile = {
                "driver": "GTiff",
                "dtype": "float32",
                "count": 1,
                "width": src.width,
                "height": src.height,
                "crs": src.crs,
                "transform": src.transform,
                "nodata": NODATA_M,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "compress": "deflate",
                "predictor": 3,
                "zlevel": 6,
                "BIGTIFF": "IF_SAFER",
            }
            raw_nodata = src.nodata
            with rasterio.open(out_path, "w", **profile) as dst:
                for row in range(0, src.height, block_rows):
                    rows = min(block_rows, src.height - row)
                    window = Window(0, row, src.width, rows)
                    data = src.read(1, window=window)
                    metres = data.astype(np.float64) * factor
                    invalid = ~np.isfinite(metres) | (np.abs(metres) >= ELEVATION_SENTINEL_LIMIT_M)
                    if raw_nodata is not None:
                        invalid |= data == raw_nodata
                    total += metres.size
                    masked += int(invalid.sum())
                    good = metres[~invalid]
                    if good.size:
                        low = min(low, float(good.min()))
                        high = max(high, float(good.max()))
                    out = np.where(invalid, NODATA_M, metres).astype(np.float32)
                    dst.write(out, 1, window=window)
                update = {
                    "TALUS_ELEVATION_UNITS": "metre",
                    "TALUS_HEIGHT_REFERENCE": "sphere",
                    "TALUS_HEIGHT_REFERENCE_RADIUS_M": f"{convention.reference_radius_m:g}",
                    "TALUS_NORMALIZATION_VERSION": NORMALIZATION_VERSION,
                }
                update.update(tags or {})
                dst.update_tags(**update)

    if masked == total or not np.isfinite(low):
        out_path.unlink(missing_ok=True)
        raise DemValidationError("elevation_values", "No valid elevation cell remained after conversion.")
    digest = hashlib.sha256()
    with open(out_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return NormalizedDem(
        path=out_path,
        size_bytes=out_path.stat().st_size,
        sha256=digest.hexdigest(),
        total_cells=total,
        masked_cells=masked,
        min_m=low,
        max_m=high,
    )


def verify_normalized(label_path: Path, out_path: Path, convention: HeightConvention) -> None:
    """Re-open the output with the terrain engine reader and compare it with the source.

    A decimated read of both files uses the same pixel positions, so the values must agree.
    """
    try:
        meta = inspect_dem(out_path)
    except TerrainAnalysisError as exc:
        raise DemValidationError("normalized_output", f"The converted file is not readable: {exc}") from None
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        with rasterio.open(label_path) as src, rasterio.open(out_path) as out:
            if (meta.width, meta.height) != (src.width, src.height):
                raise DemValidationError("normalized_output", "The converted dimensions differ from the source.")
            if src.crs != out.crs or tuple(src.transform)[:6] != tuple(out.transform)[:6]:
                raise DemValidationError("normalized_output", "The converted georeferencing differs from the source.")
            shape = (min(src.height, 256), min(src.width, 256))
            expected = src.read(1, out_shape=shape).astype(np.float64) * convention.factor_to_metres
            actual = out.read(1, out_shape=shape).astype(np.float64)
    valid = (actual != NODATA_M) & np.isfinite(expected)
    if not np.allclose(actual[valid], expected[valid], rtol=1e-6, atol=1e-2):
        raise DemValidationError("normalized_output", "Converted elevations differ from the source values.")
