"""Deterministic lunar coordinate validation, transformations, and distance metrics."""

from typing import Tuple, List, Union
import math

from terrain_agent.terrain.resource_safety import (
    InvalidCoordinateError,
    OversizedRequestError,
    MAX_ANALYSIS_RADIUS_KM,
    MAX_PATH_LENGTH_KM,
    MAX_WAYPOINTS,
)

# Mean volumetric lunar radius in meters (IAU/IAG Lunar Reference)
LUNAR_RADIUS_METERS = 1_737_400.0
LUNAR_RADIUS_KM = 1_737.4


def validate_latitude(lat: float) -> float:
    """Validate that latitude is a finite number in [-90.0, +90.0]."""
    if lat is None or not isinstance(lat, (int, float)):
        raise InvalidCoordinateError(f"Latitude must be a numeric value, got: {type(lat).__name__}")
    if math.isnan(lat) or math.isinf(lat):
        raise InvalidCoordinateError(f"Latitude must be finite, got: {lat}")
    if not (-90.0 <= lat <= 90.0):
        raise InvalidCoordinateError(
            f"Latitude out of bounds: {lat}°. Lunar latitude must be between -90.0° and +90.0°."
        )
    return float(lat)


def validate_longitude(lon: float) -> float:
    """Validate that longitude is a finite number and normalize to [-180.0, +180.0]."""
    if lon is None or not isinstance(lon, (int, float)):
        raise InvalidCoordinateError(f"Longitude must be a numeric value, got: {type(lon).__name__}")
    if math.isnan(lon) or math.isinf(lon):
        raise InvalidCoordinateError(f"Longitude must be finite, got: {lon}")
    if not (-180.0 <= lon <= 360.0):
        raise InvalidCoordinateError(
            f"Longitude out of bounds: {lon}°. Lunar longitude must be between -180.0° and +360.0°."
        )
    val = float(lon)
    # Normalize [180, 360] to [-180, 0]
    if val > 180.0:
        val -= 360.0
    return val


def validate_lunar_coordinate(lat: float, lon: float) -> Tuple[float, float]:
    """Validate a lunar coordinate pair (latitude, longitude).
    
    Detects probable accidental coordinate order inversions (e.g. lat > 90 but within lon range).
    """
    if isinstance(lat, (int, float)) and abs(lat) > 90.0 and abs(lon) <= 90.0:
        raise InvalidCoordinateError(
            f"Possible inverted coordinate order: latitude={lat}° exceeds 90°, while longitude={lon}° is within latitude range. "
            f"Ensure coordinates are passed in (latitude, longitude) order."
        )
    
    clean_lat = validate_latitude(lat)
    clean_lon = validate_longitude(lon)
    return clean_lat, clean_lon


def validate_analysis_radius(radius_km: float) -> float:
    """Validate that analysis radius is positive and within configured resource limits."""
    if radius_km is None or not isinstance(radius_km, (int, float)):
        raise InvalidCoordinateError(f"Radius must be a numeric value, got: {type(radius_km).__name__}")
    if math.isnan(radius_km) or math.isinf(radius_km):
        raise InvalidCoordinateError(f"Radius must be a finite number, got: {radius_km}")
    if radius_km <= 0.0:
        raise InvalidCoordinateError(f"Radius must be strictly positive, got: {radius_km} km")
    if radius_km > MAX_ANALYSIS_RADIUS_KM:
        raise OversizedRequestError(
            f"Requested analysis radius ({radius_km:.1f} km) exceeds maximum safety limit "
            f"of {MAX_ANALYSIS_RADIUS_KM:.1f} km."
        )
    return float(radius_km)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle geodesic distance between two points on the lunar sphere in meters."""
    c_lat1, c_lon1 = validate_lunar_coordinate(lat1, lon1)
    c_lat2, c_lon2 = validate_lunar_coordinate(lat2, lon2)
    
    phi1 = math.radians(c_lat1)
    phi2 = math.radians(c_lat2)
    delta_phi = math.radians(c_lat2 - c_lat1)
    delta_lambda = math.radians(c_lon2 - c_lon1)
    
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return LUNAR_RADIUS_METERS * c


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on the lunar sphere in kilometers."""
    return haversine_distance_m(lat1, lon1, lat2, lon2) / 1000.0


def validate_waypoints(
    waypoints: List[Union[Tuple[float, float], List[float]]]
) -> List[Tuple[float, float]]:
    """Validate a traverse path consisting of multiple waypoints."""
    if not waypoints or len(waypoints) < 2:
        raise InvalidCoordinateError("A traverse path requires at least 2 waypoints.")
    if len(waypoints) > MAX_WAYPOINTS:
        raise OversizedRequestError(
            f"Traverse contains {len(waypoints)} waypoints, exceeding maximum limit of {MAX_WAYPOINTS}."
        )
    
    validated = []
    total_length_km = 0.0
    
    for idx, pt in enumerate(waypoints):
        if not isinstance(pt, (tuple, list)) or len(pt) != 2:
            raise InvalidCoordinateError(f"Waypoint {idx} must be a (lat, lon) pair, got: {pt}")
        v_lat, v_lon = validate_lunar_coordinate(pt[0], pt[1])
        validated.append((v_lat, v_lon))
        
        if idx > 0:
            prev_lat, prev_lon = validated[idx - 1]
            total_length_km += haversine_distance_km(prev_lat, prev_lon, v_lat, v_lon)
            
    if total_length_km > MAX_PATH_LENGTH_KM:
        raise OversizedRequestError(
            f"Total traverse path length ({total_length_km:.1f} km) exceeds maximum limit "
            f"of {MAX_PATH_LENGTH_KM:.1f} km."
        )
        
    return validated
