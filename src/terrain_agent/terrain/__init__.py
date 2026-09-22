"""Deterministic lunar terrain analysis engine.

Provides modular, mathematically verified computation of elevation metrics,
Horn's slope gradients, Terrain Ruggedness Index, windowed raster IO, and coordinate validation.
"""

from terrain_agent.terrain.resource_safety import (
    TerrainAnalysisError,
    InvalidCoordinateError,
    OversizedRequestError,
    MalformedRasterError,
    UnsupportedCRSError,
    MAX_RASTER_DIM,
    MAX_CELLS_PER_OP,
    MAX_ANALYSIS_RADIUS_KM,
    MAX_PATH_LENGTH_KM,
    MAX_WAYPOINTS,
    validate_raster_dimensions,
    estimate_memory_mb,
    split_bounding_box,
)
from terrain_agent.terrain.coordinates import (
    LUNAR_RADIUS_METERS,
    LUNAR_RADIUS_KM,
    validate_latitude,
    validate_longitude,
    validate_lunar_coordinate,
    validate_analysis_radius,
    validate_waypoints,
    haversine_distance_m,
    haversine_distance_km,
)
from terrain_agent.terrain.dem_reader import (
    RasterMetadata,
    inspect_dem,
    read_dem_window,
    read_dem_around_point,
)
from terrain_agent.terrain.elevation import (
    ElevationStats,
    get_elevation_stats,
)
from terrain_agent.terrain.slope import (
    SlopeStats,
    calculate_slope_grid,
    calculate_slope_stats,
)
from terrain_agent.terrain.roughness import (
    RoughnessStats,
    calculate_tri_grid,
    calculate_roughness_stats,
)

__all__ = [
    # Resource Safety & Errors
    "TerrainAnalysisError",
    "InvalidCoordinateError",
    "OversizedRequestError",
    "MalformedRasterError",
    "UnsupportedCRSError",
    "MAX_RASTER_DIM",
    "MAX_CELLS_PER_OP",
    "MAX_ANALYSIS_RADIUS_KM",
    "MAX_PATH_LENGTH_KM",
    "MAX_WAYPOINTS",
    "validate_raster_dimensions",
    "estimate_memory_mb",
    "split_bounding_box",
    # Coordinates
    "LUNAR_RADIUS_METERS",
    "LUNAR_RADIUS_KM",
    "validate_latitude",
    "validate_longitude",
    "validate_lunar_coordinate",
    "validate_analysis_radius",
    "validate_waypoints",
    "haversine_distance_m",
    "haversine_distance_km",
    # DEM Reader
    "RasterMetadata",
    "inspect_dem",
    "read_dem_window",
    "read_dem_around_point",
    # Elevation
    "ElevationStats",
    "get_elevation_stats",
    # Slope
    "SlopeStats",
    "calculate_slope_grid",
    "calculate_slope_stats",
    # Roughness
    "RoughnessStats",
    "calculate_tri_grid",
    "calculate_roughness_stats",
]


from terrain_agent.terrain.resource_safety import (
    InvalidWaypointError,
    InvalidThresholdError,
    InvalidNoGoZoneError,
    InvalidSiteError,
)
from terrain_agent.terrain.georef import DemGeoContext, open_dem_context
from terrain_agent.terrain.geometry import (
    NoGoZone,
    RouteSamples,
    densify_route,
    find_no_go_hits,
    parse_no_go_zones,
    validate_route,
)
from terrain_agent.terrain.route_analysis import (
    RouteTerrainMeasurement,
    SegmentTerrainStats,
    measure_route_terrain,
)

__all__ += [
    "InvalidWaypointError",
    "InvalidThresholdError",
    "InvalidNoGoZoneError",
    "InvalidSiteError",
    "DemGeoContext",
    "open_dem_context",
    "NoGoZone",
    "RouteSamples",
    "densify_route",
    "find_no_go_hits",
    "parse_no_go_zones",
    "validate_route",
    "RouteTerrainMeasurement",
    "SegmentTerrainStats",
    "measure_route_terrain",
]
