"""Georeferencing for DEM analysis: latitude/longitude <-> raster cells, and metric pixel size.

Why this module exists
----------------------
Lunar DEMs are usually distributed in a lunar polar stereographic projection or a lunar
geographic (longitude/latitude) grid. Two details matter for correct slope values:

1. Latitude/longitude must be converted using the reference sphere declared by the DEM
   itself. Converting from Earth EPSG:4326 to a lunar projection is not a valid coordinate
   operation, so the geographic CRS is derived here from the DEM ellipsoid definition.
2. Slope needs cell size in metres. Projected rasters need the local map scale factor
   applied, and geographic rasters need degrees converted with the cosine of latitude.
   Conformal projections such as polar stereographic have one scale factor in every
   direction. Equirectangular grids, which NASA cylindrical lunar DEMs use, have different
   east-west and north-south factors, so both are measured. Other projections are refused
   because their metric cell size would have to be guessed.

Nothing here reads elevation data. It only interprets the raster header.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform as _warp_transform

from terrain_agent.terrain.coordinates import LUNAR_RADIUS_METERS
from terrain_agent.terrain.dem_reader import RasterMetadata, inspect_dem
from terrain_agent.terrain.resource_safety import MalformedRasterError, UnsupportedCRSError

#: Geographic rasters closer to a pole than this (cos(latitude) below the value, about 87.1
#: degrees) have cells so narrow in the east-west direction that metric slope is unreliable.
MIN_COS_LATITUDE = 0.05

#: Relative difference between the declared DEM radius and the lunar radius that triggers a warning.
RADIUS_MISMATCH_TOLERANCE = 0.01

#: Projections that preserve angles, so the local scale factor is the same in every direction.
_CONFORMAL_PROJECTIONS = frozenset({"stere", "sterea", "merc", "tmerc", "utm", "lcc", "omerc"})

#: Latitude step in degrees used to measure the local scale factor numerically.
_SCALE_PROBE_DEG = 1.0e-4

_WKT_ELLIPSOID = re.compile(r"(?:SPHEROID|ELLIPSOID)\[\"[^\"]*\",\s*([0-9.]+),\s*([0-9.]+)")


def _ellipsoid_from_crs(crs: CRS) -> tuple[Optional[float], Optional[float]]:
    """Return the (semi-major, semi-minor) axes in metres declared by *crs*, if determinable."""
    try:
        params = crs.to_dict()
    except Exception:  # noqa: BLE001 - rasterio raises several types here
        params = {}
    if "R" in params:
        radius = float(params["R"])
        return radius, radius
    if "a" in params:
        semi_major = float(params["a"])
        if "b" in params:
            return semi_major, float(params["b"])
        if "rf" in params and float(params["rf"]) != 0.0:
            return semi_major, semi_major * (1.0 - 1.0 / float(params["rf"]))
        return semi_major, semi_major
    match = _WKT_ELLIPSOID.search(crs.to_wkt())
    if match:
        semi_major = float(match.group(1))
        inverse_flattening = float(match.group(2))
        if inverse_flattening == 0.0:
            return semi_major, semi_major
        return semi_major, semi_major * (1.0 - 1.0 / inverse_flattening)
    return None, None


@dataclass(frozen=True)
class DemGeoContext:
    """Interpreted georeferencing of one DEM file."""

    path: Path
    metadata: RasterMetadata
    crs: CRS
    transform: Affine
    inv_transform: Affine
    is_geographic: bool
    geographic_crs: Optional[CRS]
    unit_factor: float
    body_radius_m: float
    uses_lon_360: bool
    warnings: tuple[str, ...]
    projection: Optional[str] = None

    @property
    def width(self) -> int:
        return self.metadata.width

    @property
    def height(self) -> int:
        return self.metadata.height

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def to_raster_xy(self, lats, lons) -> tuple[np.ndarray, np.ndarray]:
        """Convert latitude/longitude (degrees, longitude in [-180, 180]) to raster CRS x, y."""
        lat_arr = np.asarray(lats, dtype=np.float64).ravel()
        lon_arr = np.asarray(lons, dtype=np.float64).ravel()
        if self.is_geographic:
            x = np.where(lon_arr < 0.0, lon_arr + 360.0, lon_arr) if self.uses_lon_360 else lon_arr
            return x.copy(), lat_arr.copy()
        xs, ys = _warp_transform(self.geographic_crs, self.crs, lon_arr.tolist(), lat_arr.tolist())
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)

    def to_latlon(self, xs, ys) -> tuple[np.ndarray, np.ndarray]:
        """Convert raster CRS x, y to (latitude, longitude) with longitude in [-180, 180]."""
        x_arr = np.asarray(xs, dtype=np.float64).ravel()
        y_arr = np.asarray(ys, dtype=np.float64).ravel()
        if self.is_geographic:
            lon = x_arr.copy()
            lat = y_arr.copy()
        else:
            lon_list, lat_list = _warp_transform(
                self.crs, self.geographic_crs, x_arr.tolist(), y_arr.tolist()
            )
            lon = np.asarray(lon_list, dtype=np.float64)
            lat = np.asarray(lat_list, dtype=np.float64)
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        return lat, lon

    def latlon_to_rowcol_float(self, lats, lons) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return fractional (row, col) raster positions and a mask of finite results."""
        x, y = self.to_raster_xy(lats, lons)
        finite = np.isfinite(x) & np.isfinite(y)
        x0 = np.where(finite, x, 0.0)
        y0 = np.where(finite, y, 0.0)
        inv = self.inv_transform
        col = inv.a * x0 + inv.b * y0 + inv.c
        row = inv.d * x0 + inv.e * y0 + inv.f
        return row, col, finite

    # ------------------------------------------------------------------
    # Metric cell size
    # ------------------------------------------------------------------

    def scale_factor(self, lat: float, lon: float) -> float:
        """Local map scale factor (grid distance divided by true distance) at a location.

        Measured numerically along a meridian. Exact for conformal projections such as polar
        stereographic, where the factor is the same in every direction. Returns 1.0 for
        geographic rasters, which are handled separately.
        """
        if self.is_geographic:
            return 1.0
        lat2 = lat - _SCALE_PROBE_DEG if lat >= 0.0 else lat + _SCALE_PROBE_DEG
        xs, ys = _warp_transform(self.geographic_crs, self.crs, [lon, lon], [lat, lat2])
        grid_distance = math.hypot(xs[1] - xs[0], ys[1] - ys[0]) * self.unit_factor
        true_distance = self.body_radius_m * math.radians(_SCALE_PROBE_DEG)
        factor = grid_distance / true_distance if true_distance > 0.0 else float("nan")
        if not math.isfinite(factor) or factor <= 0.0:
            raise UnsupportedCRSError(
                f"Cannot determine the local map scale for {self.path.name} at "
                f"({lat:.4f}, {lon:.4f})."
            )
        return factor

    def _parallel_scale_factor(self, lat: float, lon: float) -> float:
        """Local scale factor along a parallel (east-west), measured numerically."""
        cos_lat = math.cos(math.radians(lat))
        if cos_lat < MIN_COS_LATITUDE:
            raise UnsupportedCRSError(
                f"{self.path.name} uses an equirectangular grid and the location is within "
                "about 3 degrees of a pole, where the east-west cell size is not usable. Use a "
                "polar stereographic product for polar analysis."
            )
        xs, ys = _warp_transform(
            self.geographic_crs, self.crs, [lon, lon + _SCALE_PROBE_DEG], [lat, lat]
        )
        grid_distance = math.hypot(xs[1] - xs[0], ys[1] - ys[0]) * self.unit_factor
        true_distance = self.body_radius_m * cos_lat * math.radians(_SCALE_PROBE_DEG)
        factor = grid_distance / true_distance
        if not math.isfinite(factor) or factor <= 0.0:
            raise UnsupportedCRSError(
                f"Cannot determine the east-west map scale for {self.path.name} at "
                f"({lat:.4f}, {lon:.4f})."
            )
        return factor

    def scale_factors(self, lat: float, lon: float) -> tuple[float, float]:
        """Return (east-west, north-south) map scale factors at a location.

        Conformal projections have the same factor in both directions. Equirectangular
        grids have separate factors. Any other projection is refused rather than guessed.
        """
        if self.is_geographic:
            return 1.0, 1.0
        k_north = self.scale_factor(lat, lon)
        if self.projection is None or self.projection in _CONFORMAL_PROJECTIONS:
            return k_north, k_north
        if self.projection == "eqc":
            return self._parallel_scale_factor(lat, lon), k_north
        raise UnsupportedCRSError(
            f"Projection {self.projection!r} in {self.path.name} is not supported for metric "
            "slope. Supported: polar stereographic and other conformal projections, and "
            "equirectangular grids."
        )

    def pixel_size_m(self, lat: float, lon: float) -> tuple[float, float]:
        """Return (east-west, north-south) cell size in metres at a location."""
        res_x = abs(self.metadata.res_x)
        res_y = abs(self.metadata.res_y)
        if self.is_geographic:
            cos_lat = math.cos(math.radians(lat))
            if cos_lat < MIN_COS_LATITUDE:
                raise UnsupportedCRSError(
                    f"{self.path.name} uses geographic coordinates and the location is within "
                    "about 3 degrees of a pole, where metric slope cannot be computed reliably."
                )
            metres_per_degree = math.radians(1.0) * LUNAR_RADIUS_METERS
            return res_x * metres_per_degree * cos_lat, res_y * metres_per_degree
        k_east, k_north = self.scale_factors(lat, lon)
        return res_x * self.unit_factor / k_east, res_y * self.unit_factor / k_north


def open_dem_context(dem_path) -> DemGeoContext:
    """Inspect a DEM header and build its georeferencing context.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    MalformedRasterError
        If the raster cannot be opened or has invalid dimensions.
    UnsupportedCRSError
        If the raster has no CRS or is rotated.
    """
    path = Path(dem_path)
    metadata = inspect_dem(path)
    try:
        with rasterio.open(path) as src:
            crs = src.crs
            transform = src.transform
    except (rasterio.errors.RasterioError, OSError) as exc:
        raise MalformedRasterError(f"Failed to open raster {path.name}: {exc}") from exc

    if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
        raise UnsupportedCRSError(f"Rotated rasters are not supported: {path.name}")

    warnings: list[str] = []
    semi_major, semi_minor = _ellipsoid_from_crs(crs)
    if semi_major is None:
        warnings.append(
            "DEM CRS does not declare a reference ellipsoid; the lunar reference sphere "
            f"({LUNAR_RADIUS_METERS:.0f} m) is assumed."
        )
        semi_major = semi_minor = LUNAR_RADIUS_METERS
    elif abs(semi_major - LUNAR_RADIUS_METERS) / LUNAR_RADIUS_METERS > RADIUS_MISMATCH_TOLERANCE:
        warnings.append(
            f"DEM CRS reference radius ({semi_major:.0f} m) differs from the lunar reference "
            f"radius ({LUNAR_RADIUS_METERS:.0f} m). Verify the DEM georeferencing."
        )
    is_sphere = semi_minor is None or abs(semi_major - semi_minor) <= 1e-6 * semi_major
    if not is_sphere:
        warnings.append(
            "DEM CRS uses a non-spherical ellipsoid; metric cell size assumes a sphere."
        )

    is_geographic = bool(crs.is_geographic)
    geographic_crs: Optional[CRS] = None
    unit_factor = 1.0
    uses_lon_360 = False
    projection: Optional[str] = None

    if is_geographic:
        # The metric conversion uses the lunar sphere regardless of the declared ellipsoid.
        body_radius = LUNAR_RADIUS_METERS
        uses_lon_360 = metadata.bounds[2] > 180.0 + 1e-9
    else:
        body_radius = float(semi_major)
        try:
            unit_factor = float(crs.linear_units_factor[1])
        except Exception:  # noqa: BLE001
            warnings.append("Could not determine the linear unit of the DEM CRS; metres assumed.")
            unit_factor = 1.0
        if is_sphere:
            geographic_crs = CRS.from_string(f"+proj=longlat +R={semi_major!r} +no_defs")
        else:
            geographic_crs = CRS.from_string(
                f"+proj=longlat +a={semi_major!r} +b={semi_minor!r} +no_defs"
            )
        try:
            projection = crs.to_dict().get("proj")
        except Exception:  # noqa: BLE001
            projection = None
        if projection is None:
            warnings.append(
                "The DEM projection could not be identified; the map scale is assumed to be "
                "the same in every direction."
            )

    return DemGeoContext(
        path=path,
        metadata=metadata,
        crs=crs,
        transform=transform,
        inv_transform=~transform,
        is_geographic=is_geographic,
        geographic_crs=geographic_crs,
        unit_factor=unit_factor,
        body_radius_m=body_radius,
        uses_lon_360=uses_lon_360,
        warnings=tuple(warnings),
        projection=projection,
    )
