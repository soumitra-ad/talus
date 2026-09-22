"""Data models shared by the acquisition layer.

``CoverageRequest`` describes the area a caller needs. ``ProductCandidate`` describes one
product a provider found. Provider text is never stored in free form: every text field that
reaches these models has already been checked against a strict pattern by the provider.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict

from terrain_agent.terrain.coordinates import (
    LUNAR_RADIUS_METERS,
    validate_analysis_radius,
    validate_lunar_coordinate,
)
from terrain_agent.terrain.resource_safety import InvalidCoordinateError, InvalidThresholdError

#: Margin added around a route when deriving the area to cover.
DEFAULT_ROUTE_MARGIN_M = 1000.0


class CoverageRequest(BaseModel):
    """The geographic area a DEM must cover.

    Latitudes are planetocentric. Longitudes are degrees east from 0 to 360, the convention
    ODE documents. When ``west_lon`` is greater than ``east_lon`` the area crosses the 0/360
    meridian. ``full_longitude`` means every longitude, which is the case for areas that
    contain a pole.
    """

    model_config = ConfigDict(frozen=True)

    min_lat: float
    max_lat: float
    west_lon: float
    east_lon: float
    full_longitude: bool = False
    center_lat: float
    center_lon: float
    max_pixel_size_m: Optional[float] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_point(
        cls, lat: float, lon: float, radius_m: float, *, max_pixel_size_m: Optional[float] = None
    ) -> "CoverageRequest":
        """Area within *radius_m* of a location."""
        clean_lat, clean_lon = validate_lunar_coordinate(lat, lon)
        if isinstance(radius_m, bool) or not isinstance(radius_m, (int, float)):
            raise InvalidThresholdError("radius_m must be a number.")
        validate_analysis_radius(float(radius_m) / 1000.0)
        return cls._from_center_and_radius(clean_lat, clean_lon, float(radius_m), max_pixel_size_m)

    @classmethod
    def from_bbox(
        cls,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        *,
        max_pixel_size_m: Optional[float] = None,
    ) -> "CoverageRequest":
        """Latitude/longitude box. Longitudes may be -180..360 and may cross a meridian."""
        lat_lo, _ = validate_lunar_coordinate(min_lat, min_lon)
        lat_hi, _ = validate_lunar_coordinate(max_lat, max_lon)
        if not lat_lo < lat_hi:
            raise InvalidCoordinateError("min_lat must be less than max_lat.")
        span = float(max_lon) - float(min_lon)
        width = span % 360.0
        full = span >= 360.0 or (width == 0.0 and span != 0.0)
        if not full and span == 0.0:
            raise InvalidCoordinateError("min_lon and max_lon must differ.")
        west = float(min_lon) % 360.0
        east = float(max_lon) % 360.0
        center_lon = _to_signed(west + (360.0 if full else width) / 2.0)
        return cls._validated(
            lat_lo, lat_hi, west, east, full, 0.5 * (lat_lo + lat_hi), center_lon, max_pixel_size_m
        )

    @classmethod
    def from_waypoints(
        cls,
        waypoints: Sequence[tuple[float, float]],
        *,
        margin_m: float = DEFAULT_ROUTE_MARGIN_M,
        max_pixel_size_m: Optional[float] = None,
    ) -> "CoverageRequest":
        """Smallest area containing every waypoint plus a margin."""
        from terrain_agent.terrain.geometry import validate_route

        route = validate_route(waypoints)
        lats = [p[0] for p in route]
        lons = sorted(p[1] % 360.0 for p in route)
        dlat = math.degrees(margin_m / LUNAR_RADIUS_METERS)
        lat_lo, lat_hi = max(-90.0, min(lats) - dlat), min(90.0, max(lats) + dlat)
        gaps = [(lons[(i + 1) % len(lons)] - lons[i]) % 360.0 for i in range(len(lons))]
        biggest = max(range(len(gaps)), key=lambda i: gaps[i])
        west, east = lons[(biggest + 1) % len(lons)], lons[biggest]
        width = (east - west) % 360.0
        contains_pole = min(lats) - dlat <= -90.0 or max(lats) + dlat >= 90.0
        widest_cos = math.cos(math.radians(min(89.999, max(abs(lat_lo), abs(lat_hi)))))
        dlon = dlat / max(widest_cos, 1e-6)
        if contains_pole or width + 2 * dlon >= 360.0:
            return cls._validated(
                lat_lo, lat_hi, 0.0, 360.0, True, 0.5 * (lat_lo + lat_hi), 0.0, max_pixel_size_m
            )
        west = (west - dlon) % 360.0
        east = (east + dlon) % 360.0
        centre = _to_signed(west + ((east - west) % 360.0) / 2.0)
        return cls._validated(
            lat_lo, lat_hi, west, east, False, 0.5 * (lat_lo + lat_hi), centre, max_pixel_size_m
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _from_center_and_radius(
        cls, lat: float, lon: float, radius_m: float, max_pixel_size_m: Optional[float]
    ) -> "CoverageRequest":
        dlat = math.degrees(radius_m / LUNAR_RADIUS_METERS)
        lat_lo, lat_hi = max(-90.0, lat - dlat), min(90.0, lat + dlat)
        contains_pole = lat - dlat <= -90.0 or lat + dlat >= 90.0
        widest_cos = math.cos(math.radians(min(89.999, max(abs(lat_lo), abs(lat_hi)))))
        dlon = dlat / max(widest_cos, 1e-6)
        if contains_pole or dlon >= 180.0:
            return cls._validated(lat_lo, lat_hi, 0.0, 360.0, True, lat, lon, max_pixel_size_m)
        return cls._validated(
            lat_lo, lat_hi, (lon - dlon) % 360.0, (lon + dlon) % 360.0, False, lat, lon,
            max_pixel_size_m,
        )

    @classmethod
    def _validated(
        cls,
        min_lat: float,
        max_lat: float,
        west: float,
        east: float,
        full: bool,
        center_lat: float,
        center_lon: float,
        max_pixel_size_m: Optional[float],
    ) -> "CoverageRequest":
        if max_pixel_size_m is not None:
            if (
                isinstance(max_pixel_size_m, bool)
                or not isinstance(max_pixel_size_m, (int, float))
                or not math.isfinite(max_pixel_size_m)
                or max_pixel_size_m <= 0.0
            ):
                raise InvalidThresholdError("max_pixel_size_m must be a positive number.")
        return cls(
            min_lat=min_lat,
            max_lat=max_lat,
            west_lon=west,
            east_lon=east,
            full_longitude=full,
            center_lat=center_lat,
            center_lon=center_lon,
            max_pixel_size_m=None if max_pixel_size_m is None else float(max_pixel_size_m),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def lon_width(self) -> float:
        return 360.0 if self.full_longitude else (self.east_lon - self.west_lon) % 360.0

    def sample_points(self) -> list[tuple[float, float]]:
        """Representative (lat, lon) points, longitude in -180..180, used to check coverage."""
        if self.full_longitude:
            lons = [0.0, 90.0, 180.0, -90.0]
        else:
            lons = [
                _to_signed(self.west_lon),
                _to_signed(self.west_lon + self.lon_width / 2.0),
                _to_signed(self.west_lon + self.lon_width),
            ]
        mid_lat = 0.5 * (self.min_lat + self.max_lat)
        points = [(lat, lon) for lat in (self.min_lat, mid_lat, self.max_lat) for lon in lons]
        points.append((self.center_lat, self.center_lon))
        seen: list[tuple[float, float]] = []
        for point in points:
            if point not in seen:
                seen.append(point)
        return seen


def _to_signed(lon_east: float) -> float:
    lon = lon_east % 360.0
    return lon - 360.0 if lon > 180.0 else lon


def arc_contains(
    outer_west: float, outer_width: float, inner_west: float, inner_width: float
) -> bool:
    """True if the longitude arc (inner) lies inside the arc (outer). All values in degrees east."""
    if outer_width >= 360.0 - 1e-9:
        return True
    if inner_width > outer_width + 1e-9:
        return False
    offset = (inner_west - outer_west) % 360.0
    return offset + inner_width <= outer_width + 1e-9


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


class FileRole(str, Enum):
    """What a product file is used for."""

    DATA = "data"
    LABEL_PDS4 = "label_pds4"


class ProductFile(BaseModel):
    """One downloadable file of a product, taken from provider metadata."""

    model_config = ConfigDict(frozen=True)

    role: FileRole
    file_name: str
    url: str
    size_kb: Optional[int] = None

    @property
    def expected_max_bytes(self) -> Optional[int]:
        """Upper bound of the file size implied by the provider size in kilobytes.

        ODE reports sizes in kilobytes of 1000 bytes, rounded up.
        """
        return None if self.size_kb is None else self.size_kb * 1000


class ProductCandidate(BaseModel):
    """A product a provider found for a request."""

    model_config = ConfigDict(frozen=True)

    provider_id: str
    host_id: str
    instrument_id: str
    product_type: str
    product_id: str
    product_lid: Optional[str] = None
    data_set_id: Optional[str] = None
    version: Optional[str] = None
    min_lat: float
    max_lat: float
    west_lon: float
    east_lon: float
    map_scale_m: Optional[float] = None
    map_resolution_ppd: Optional[float] = None
    creation_time: Optional[str] = None
    ode_id: Optional[str] = None
    files: tuple[ProductFile, ...] = ()

    @property
    def lon_width(self) -> float:
        span = self.east_lon - self.west_lon
        return 360.0 if span >= 360.0 - 1e-9 else span % 360.0

    def file(self, role: FileRole) -> Optional[ProductFile]:
        return next((f for f in self.files if f.role is role), None)

    def covers(self, request: CoverageRequest) -> bool:
        """True if the product bounding box contains the whole requested area."""
        eps = 1e-9
        if self.min_lat > request.min_lat + eps or self.max_lat < request.max_lat - eps:
            return False
        return arc_contains(self.west_lon, self.lon_width, request.west_lon, request.lon_width)


class Exclusion(BaseModel):
    """Why a product was not used. Contains only validated, safe values."""

    product_id: str
    reason: str
    detail: str = ""
