"""Route densification and no-go zone geometry on the lunar sphere.

All functions are deterministic and use only NumPy. Distances use the lunar reference
sphere (radius 1,737,400 m), the same model as the haversine helpers in ``coordinates``.

No-go zones are tested in a local azimuthal equidistant plane centred on the zone. Radial
distance from the centre is exact in that plane, and route chords are short (a few hundred
metres), so intersection tests are accurate to well below a metre for zones up to hundreds
of kilometres across.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from terrain_agent.terrain.coordinates import (
    LUNAR_RADIUS_METERS,
    haversine_distance_m,
    validate_lunar_coordinate,
)
from terrain_agent.terrain.resource_safety import (
    InvalidNoGoZoneError,
    InvalidWaypointError,
    OversizedRequestError,
)

#: Consecutive waypoints closer than this are treated as duplicates and rejected.
MIN_SEGMENT_LENGTH_M = 1.0

#: Upper bound on densified samples for one route.
MAX_ROUTE_SAMPLES = 200_000

#: Chord spacing used for no-go zone tests. Independent of any DEM.
NO_GO_SAMPLE_SPACING_M = 250.0

MAX_NO_GO_ZONES = 100
MAX_ZONE_VERTICES = 200
MAX_ZONE_EXTENT_M = 500_000.0
_MAX_ZONE_NAME_LENGTH = 64


# ---------------------------------------------------------------------------
# Unit-vector helpers
# ---------------------------------------------------------------------------


def _unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    cos_lat = np.cos(lat)
    return np.stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], axis=-1)


def _latlon_from_vectors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(vectors[..., 2], -1.0, 1.0)
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(vectors[..., 1], vectors[..., 0]))
    return lat, lon


# ---------------------------------------------------------------------------
# Route densification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteSamples:
    """Points sampled along a great-circle route, tagged with their segment index."""

    lat: np.ndarray
    lon: np.ndarray
    segment: np.ndarray
    along_m: np.ndarray
    segment_lengths_m: tuple[float, ...]
    spacing_m: float

    @property
    def count(self) -> int:
        return int(self.lat.size)


def segment_lengths_m(waypoints: Sequence[tuple[float, float]]) -> list[float]:
    """Great-circle length in metres of each segment between consecutive waypoints."""
    return [
        haversine_distance_m(a[0], a[1], b[0], b[1])
        for a, b in zip(waypoints[:-1], waypoints[1:])
    ]


def densify_route(
    waypoints: Sequence[tuple[float, float]],
    spacing_m: float,
    *,
    max_samples: int = MAX_ROUTE_SAMPLES,
) -> RouteSamples:
    """Sample a route along great circles at no more than *spacing_m* between samples.

    Every segment includes both of its end points, so a waypoint appears once as the end
    of one segment and once as the start of the next.
    """
    if not math.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError(f"spacing_m must be a positive finite number, got {spacing_m!r}")

    lengths = segment_lengths_m(waypoints)
    counts = [max(1, math.ceil(length / spacing_m)) + 1 for length in lengths]
    if sum(counts) > max_samples:
        raise OversizedRequestError(
            f"Route would need {sum(counts):,} samples at {spacing_m:g} m spacing, "
            f"exceeding the limit of {max_samples:,}."
        )

    lat_parts: list[np.ndarray] = []
    lon_parts: list[np.ndarray] = []
    seg_parts: list[np.ndarray] = []
    along_parts: list[np.ndarray] = []
    travelled = 0.0

    for index, ((lat_a, lon_a), (lat_b, lon_b)) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        length = lengths[index]
        n_points = counts[index]
        t = np.linspace(0.0, 1.0, n_points)
        va = _unit_vectors(np.array([lat_a]), np.array([lon_a]))[0]
        vb = _unit_vectors(np.array([lat_b]), np.array([lon_b]))[0]
        omega = length / LUNAR_RADIUS_METERS
        sin_omega = math.sin(omega)
        if sin_omega > 1e-12:
            weight_a = np.sin((1.0 - t) * omega) / sin_omega
            weight_b = np.sin(t * omega) / sin_omega
        else:  # pragma: no cover - guarded by MIN_SEGMENT_LENGTH_M
            weight_a, weight_b = 1.0 - t, t
        points = weight_a[:, None] * va[None, :] + weight_b[:, None] * vb[None, :]
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        lat, lon = _latlon_from_vectors(points)
        lat_parts.append(lat)
        lon_parts.append(lon)
        seg_parts.append(np.full(n_points, index, dtype=np.int64))
        along_parts.append(travelled + t * length)
        travelled += length

    return RouteSamples(
        lat=np.concatenate(lat_parts),
        lon=np.concatenate(lon_parts),
        segment=np.concatenate(seg_parts),
        along_m=np.concatenate(along_parts),
        segment_lengths_m=tuple(lengths),
        spacing_m=float(spacing_m),
    )


# ---------------------------------------------------------------------------
# No-go zones
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoGoZone:
    """A keep-out area: either a circle or a polygon of (lat, lon) vertices."""

    name: str
    kind: str  # "circle" or "polygon"
    center_lat: float
    center_lon: float
    radius_m: Optional[float] = None
    vertices: Optional[tuple[tuple[float, float], ...]] = None


def _clean_name(raw: Any, index: int) -> str:
    default = f"no_go_zone_{index + 1}"
    if raw is None:
        return default
    text = "".join(ch for ch in str(raw) if ch.isprintable()).strip()
    return text[:_MAX_ZONE_NAME_LENGTH] or default


def _finite_number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InvalidNoGoZoneError(f"{what} must be a number, got {type(value).__name__}.")
    number = float(value)
    if not math.isfinite(number):
        raise InvalidNoGoZoneError(f"{what} must be finite.")
    return number


def _segments_cross(p0, p1, a, b) -> bool:
    """Scalar test: do segments p0-p1 and a-b touch or cross?"""

    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])

    if (
        max(min(p0[0], p1[0]), min(a[0], b[0])) > min(max(p0[0], p1[0]), max(a[0], b[0]))
        or max(min(p0[1], p1[1]), min(a[1], b[1])) > min(max(p0[1], p1[1]), max(a[1], b[1]))
    ):
        return False
    return cross(p0, p1, a) * cross(p0, p1, b) <= 0.0 and cross(a, b, p0) * cross(a, b, p1) <= 0.0


class _Plane:
    """Local azimuthal equidistant projection about a reference direction."""

    def __init__(self, ref_vector: np.ndarray) -> None:
        up = ref_vector / np.linalg.norm(ref_vector)
        lon0 = math.atan2(up[1], up[0])
        east = np.array([-math.sin(lon0), math.cos(lon0), 0.0])
        north = np.cross(up, east)
        self.up, self.east, self.north = up, east, north

    def project(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vectors = _unit_vectors(lat_deg, lon_deg)
        cos_c = np.clip(vectors @ self.up, -1.0, 1.0)
        c = np.arccos(cos_c)
        sin_c = np.sin(c)
        scale = np.where(sin_c > 1e-12, c / np.where(sin_c > 1e-12, sin_c, 1.0), 1.0)
        x = LUNAR_RADIUS_METERS * scale * (vectors @ self.east)
        y = LUNAR_RADIUS_METERS * scale * (vectors @ self.north)
        return x, y


def _zone_plane(zone: NoGoZone) -> _Plane:
    if zone.kind == "circle":
        ref = _unit_vectors(np.array([zone.center_lat]), np.array([zone.center_lon]))[0]
    else:
        assert zone.vertices is not None
        verts = np.array(zone.vertices, dtype=np.float64)
        ref = _unit_vectors(verts[:, 0], verts[:, 1]).mean(axis=0)
    return _Plane(ref)


def parse_no_go_zones(zones: Optional[Sequence[Any]]) -> list[NoGoZone]:
    """Validate and normalise no-go zone definitions.

    Each item may be a :class:`NoGoZone` or a mapping. Circles use ``lat``, ``lon`` and
    ``radius_m`` (or ``radius_km``). Polygons use ``polygon`` or ``vertices``: a list of
    ``[lat, lon]`` pairs, at least three, not self-intersecting. Coordinates are validated
    with the standard lunar coordinate checks, so invalid coordinates raise
    :class:`InvalidCoordinateError`. Other problems raise :class:`InvalidNoGoZoneError`.
    """
    if zones is None:
        return []
    if isinstance(zones, (str, bytes, Mapping)) or not isinstance(zones, Sequence):
        raise InvalidNoGoZoneError("no_go_zones must be a list of zone definitions.")
    if len(zones) > MAX_NO_GO_ZONES:
        raise OversizedRequestError(
            f"{len(zones)} no-go zones supplied; the maximum is {MAX_NO_GO_ZONES}."
        )

    parsed: list[NoGoZone] = []
    for index, raw in enumerate(zones):
        if isinstance(raw, NoGoZone):
            parsed.append(_validate_zone(raw))
            continue
        if not isinstance(raw, Mapping):
            raise InvalidNoGoZoneError(f"No-go zone {index + 1} must be a mapping or NoGoZone.")
        name = _clean_name(raw.get("name"), index)
        polygon = raw.get("polygon", raw.get("vertices"))
        has_circle_keys = any(k in raw for k in ("radius_m", "radius_km"))
        if polygon is not None and has_circle_keys:
            raise InvalidNoGoZoneError(
                f"No-go zone {name!r} defines both a polygon and a radius."
            )
        if polygon is not None:
            if isinstance(polygon, (str, bytes)) or not isinstance(polygon, Sequence):
                raise InvalidNoGoZoneError(f"No-go zone {name!r}: polygon must be a list of pairs.")
            vertices: list[tuple[float, float]] = []
            for pair in polygon:
                if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
                    raise InvalidNoGoZoneError(
                        f"No-go zone {name!r}: each polygon vertex must be a [lat, lon] pair."
                    )
                lat, lon = validate_lunar_coordinate(
                    _finite_number(pair[0], "vertex latitude"),
                    _finite_number(pair[1], "vertex longitude"),
                )
                vertices.append((lat, lon))
            if len(vertices) > 1 and vertices[0] == vertices[-1]:
                vertices.pop()
            centre = np.array(vertices, dtype=np.float64).mean(axis=0) if vertices else (0.0, 0.0)
            zone = NoGoZone(
                name=name,
                kind="polygon",
                center_lat=float(centre[0]),
                center_lon=float(centre[1]),
                vertices=tuple(vertices),
            )
        else:
            lat_raw = raw.get("lat", raw.get("center_lat"))
            lon_raw = raw.get("lon", raw.get("center_lon"))
            if lat_raw is None or lon_raw is None:
                raise InvalidNoGoZoneError(
                    f"No-go zone {name!r} needs either a polygon or lat, lon and radius_m."
                )
            if "radius_m" in raw:
                radius = _finite_number(raw["radius_m"], "radius_m")
            elif "radius_km" in raw:
                radius = _finite_number(raw["radius_km"], "radius_km") * 1000.0
            else:
                raise InvalidNoGoZoneError(f"No-go zone {name!r} is missing radius_m.")
            lat, lon = validate_lunar_coordinate(
                _finite_number(lat_raw, "latitude"), _finite_number(lon_raw, "longitude")
            )
            zone = NoGoZone(
                name=name, kind="circle", center_lat=lat, center_lon=lon, radius_m=radius
            )
        parsed.append(_validate_zone(zone))
    return parsed


def _validate_zone(zone: NoGoZone) -> NoGoZone:
    if zone.kind == "circle":
        validate_lunar_coordinate(zone.center_lat, zone.center_lon)
        if zone.radius_m is None or not math.isfinite(zone.radius_m) or zone.radius_m <= 0.0:
            raise InvalidNoGoZoneError(f"No-go zone {zone.name!r}: radius must be positive.")
        if zone.radius_m > MAX_ZONE_EXTENT_M:
            raise InvalidNoGoZoneError(
                f"No-go zone {zone.name!r}: radius exceeds {MAX_ZONE_EXTENT_M / 1000:g} km."
            )
        return zone
    if zone.kind != "polygon" or not zone.vertices:
        raise InvalidNoGoZoneError(f"No-go zone {zone.name!r} has an unsupported definition.")
    if len(zone.vertices) < 3:
        raise InvalidNoGoZoneError(f"No-go zone {zone.name!r}: a polygon needs at least 3 vertices.")
    if len(zone.vertices) > MAX_ZONE_VERTICES:
        raise OversizedRequestError(
            f"No-go zone {zone.name!r}: more than {MAX_ZONE_VERTICES} vertices."
        )
    for lat, lon in zone.vertices:
        validate_lunar_coordinate(lat, lon)

    verts = np.array(zone.vertices, dtype=np.float64)
    mean_vector = _unit_vectors(verts[:, 0], verts[:, 1]).mean(axis=0)
    if np.linalg.norm(mean_vector) < 1e-6:
        raise InvalidNoGoZoneError(f"No-go zone {zone.name!r}: polygon is degenerate or too large.")
    plane = _Plane(mean_vector)
    px, py = plane.project(verts[:, 0], verts[:, 1])
    if float(np.max(np.hypot(px, py))) > MAX_ZONE_EXTENT_M:
        raise InvalidNoGoZoneError(
            f"No-go zone {zone.name!r}: polygon extends more than "
            f"{MAX_ZONE_EXTENT_M / 1000:g} km from its centre."
        )
    area2 = float(np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py))
    if abs(area2) < 1.0:
        raise InvalidNoGoZoneError(f"No-go zone {zone.name!r}: polygon has zero area.")
    n = len(px)
    for i in range(n):
        a0, a1 = (px[i], py[i]), (px[(i + 1) % n], py[(i + 1) % n])
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            b0, b1 = (px[j], py[j]), (px[(j + 1) % n], py[(j + 1) % n])
            if _segments_cross(a0, a1, b0, b1):
                raise InvalidNoGoZoneError(f"No-go zone {zone.name!r}: polygon self-intersects.")
    return zone


def _points_in_polygon(px, py, vx, vy) -> np.ndarray:
    inside = np.zeros(px.shape, dtype=bool)
    n = len(vx)
    j = n - 1
    for i in range(n):
        yi, yj, xi, xj = vy[i], vy[j], vx[i], vx[j]
        crosses = (yi > py) != (yj > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (xj - xi) * (py - yi) / (yj - yi) + xi
        inside ^= crosses & (px < x_at)
        j = i
    return inside


def _chords_cross_edge(p0, p1, a, b) -> np.ndarray:
    """Vectorised: which chords p0[k]-p1[k] touch or cross the edge a-b?"""

    def cross(ox, oy, ux, uy, vx, vy):
        return (ux - ox) * (vy - oy) - (uy - oy) * (vx - ox)

    o1 = cross(p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1], a[0], a[1])
    o2 = cross(p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1], b[0], b[1])
    o3 = cross(a[0], a[1], b[0], b[1], p0[:, 0], p0[:, 1])
    o4 = cross(a[0], a[1], b[0], b[1], p1[:, 0], p1[:, 1])
    overlap = (
        (np.maximum(np.minimum(p0[:, 0], p1[:, 0]), min(a[0], b[0])) <= np.minimum(np.maximum(p0[:, 0], p1[:, 0]), max(a[0], b[0])))
        & (np.maximum(np.minimum(p0[:, 1], p1[:, 1]), min(a[1], b[1])) <= np.minimum(np.maximum(p0[:, 1], p1[:, 1]), max(a[1], b[1])))
    )
    return (o1 * o2 <= 0.0) & (o3 * o4 <= 0.0) & overlap


def find_no_go_hits(samples: RouteSamples, zones: Sequence[NoGoZone]) -> dict[int, list[str]]:
    """Return ``{segment_index: [zone names]}`` for every segment that touches a zone.

    Consecutive samples on the same segment form chords. A chord hits a circle when its
    closest point is within the radius, and hits a polygon when either end is inside or it
    crosses an edge. Zones smaller than the sample spacing are still detected.
    """
    hits: dict[int, list[str]] = {}
    if not zones or samples.count < 2:
        return hits
    same_segment = samples.segment[:-1] == samples.segment[1:]
    chord_segments = samples.segment[:-1][same_segment]

    for zone in zones:
        plane = _zone_plane(zone)
        x, y = plane.project(samples.lat, samples.lon)
        p0 = np.stack([x[:-1], y[:-1]], axis=1)[same_segment]
        p1 = np.stack([x[1:], y[1:]], axis=1)[same_segment]

        if zone.kind == "circle":
            direction = p1 - p0
            length_sq = np.sum(direction**2, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(length_sq > 0.0, -np.sum(p0 * direction, axis=1) / length_sq, 0.0)
            t = np.clip(t, 0.0, 1.0)
            closest = p0 + t[:, None] * direction
            radius = float(zone.radius_m or 0.0)
            hit = np.sum(closest**2, axis=1) <= radius * radius
        else:
            assert zone.vertices is not None
            verts = np.array(zone.vertices, dtype=np.float64)
            vx, vy = plane.project(verts[:, 0], verts[:, 1])
            hit = _points_in_polygon(p0[:, 0], p0[:, 1], vx, vy) | _points_in_polygon(
                p1[:, 0], p1[:, 1], vx, vy
            )
            n = len(vx)
            for i in range(n):
                edge_a = (vx[i], vy[i])
                edge_b = (vx[(i + 1) % n], vy[(i + 1) % n])
                hit |= _chords_cross_edge(p0, p1, edge_a, edge_b)

        for segment in np.unique(chord_segments[hit]):
            names = hits.setdefault(int(segment), [])
            if zone.name not in names:
                names.append(zone.name)
    return hits


# ---------------------------------------------------------------------------
# Route validation shared by rover analysis
# ---------------------------------------------------------------------------


def validate_route(waypoints: Any) -> list[tuple[float, float]]:
    """Validate a traverse route and return normalised ``(lat, lon)`` tuples.

    Raises
    ------
    InvalidWaypointError
        Not a list, fewer than two waypoints, malformed entries, or duplicate consecutive points.
    InvalidCoordinateError
        Non-numeric, non-finite, out-of-range, or order-inverted coordinates.
    OversizedRequestError
        More than the maximum number of waypoints, or a route longer than the maximum length.
    """
    from terrain_agent.terrain.coordinates import validate_waypoints
    from terrain_agent.terrain.resource_safety import InvalidCoordinateError, MAX_WAYPOINTS

    if isinstance(waypoints, (str, bytes, Mapping)) or not isinstance(waypoints, Sequence):
        if isinstance(waypoints, np.ndarray) and waypoints.ndim == 2:
            waypoints = waypoints.tolist()
        else:
            raise InvalidWaypointError("waypoints must be a list of (latitude, longitude) pairs.")
    if len(waypoints) < 2:
        raise InvalidWaypointError(
            f"A route needs at least 2 waypoints, got {len(waypoints)}."
        )
    if len(waypoints) > MAX_WAYPOINTS:
        raise OversizedRequestError(
            f"Route has {len(waypoints)} waypoints; the maximum is {MAX_WAYPOINTS}."
        )

    numeric: list[tuple[float, float]] = []
    for index, point in enumerate(waypoints):
        if isinstance(point, (str, bytes, Mapping)) or not isinstance(point, Sequence):
            raise InvalidWaypointError(f"Waypoint {index} must be a (latitude, longitude) pair.")
        if len(point) != 2:
            raise InvalidWaypointError(
                f"Waypoint {index} must have exactly 2 values, got {len(point)}."
            )
        values = []
        for value in point:
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise InvalidCoordinateError(
                    f"Waypoint {index} coordinates must be numeric, got {type(value).__name__}."
                )
            values.append(float(value))
        numeric.append((values[0], values[1]))

    validated = validate_waypoints(numeric)
    for index in range(len(validated) - 1):
        gap = haversine_distance_m(*validated[index], *validated[index + 1])
        if gap < MIN_SEGMENT_LENGTH_M:
            raise InvalidWaypointError(
                f"Waypoints {index} and {index + 1} are duplicates or closer than "
                f"{MIN_SEGMENT_LENGTH_M:g} m."
            )
    return validated
