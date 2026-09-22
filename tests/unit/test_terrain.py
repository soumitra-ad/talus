"""Unit tests and mathematical ground-truth verification for the deterministic lunar terrain engine."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from terrain_agent.terrain import (
    TerrainAnalysisError,
    InvalidCoordinateError,
    OversizedRequestError,
    MalformedRasterError,
    UnsupportedCRSError,
    MAX_RASTER_DIM,
    MAX_ANALYSIS_RADIUS_KM,
    validate_latitude,
    validate_longitude,
    validate_lunar_coordinate,
    validate_analysis_radius,
    validate_waypoints,
    haversine_distance_km,
    inspect_dem,
    read_dem_window,
    read_dem_around_point,
    get_elevation_stats,
    calculate_slope_stats,
    calculate_slope_grid,
    calculate_roughness_stats,
    calculate_tri_grid,
)


class TestLunarTerrainEngine(unittest.TestCase):
    """Test suite covering the 8 required deterministic terrain scenarios."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_geotiff(
        self,
        data: np.ndarray,
        res_x: float = 10.0,
        res_y: float = 10.0,
        nodata: float = -9999.0,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        crs_code: str = "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs",
    ) -> Path:
        """Helper to create a temporary synthetic GeoTIFF file with standard lunar CRS."""
        file_path = self.temp_path / f"synthetic_{math.floor(np.random.rand() * 100000)}.tif"
        h, w = data.shape
        transform = from_origin(origin_x, origin_y, res_x, res_y)
        
        with rasterio.open(
            file_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=data.dtype,
            crs=CRS.from_string(crs_code),
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(data, 1)
            
        return file_path

    # =========================================================================
    # Scenario 1: Flat Terrain -> Slope ~ 0.0°
    # =========================================================================
    def test_01_flat_terrain_slope_and_elevation(self):
        """Verify that a perfectly horizontal surface produces slope ~ 0.0° and zero roughness."""
        h, w = 50, 50
        elevation = np.full((h, w), 1500.0, dtype=np.float32)
        
        # Test elevation statistics
        elev_stats = get_elevation_stats(elevation)
        self.assertEqual(elev_stats.min_m, 1500.0)
        self.assertEqual(elev_stats.max_m, 1500.0)
        self.assertEqual(elev_stats.mean_m, 1500.0)
        self.assertEqual(elev_stats.std_m, 0.0)
        self.assertEqual(elev_stats.valid_cell_count, 2500)
        self.assertEqual(elev_stats.nodata_count, 0)
        
        # Test slope statistics (Horn's method)
        slope_stats = calculate_slope_stats(elevation, res_x_m=10.0, res_y_m=10.0)
        self.assertAlmostEqual(slope_stats.mean_slope_deg, 0.0, places=3)
        self.assertAlmostEqual(slope_stats.median_slope_deg, 0.0, places=3)
        self.assertAlmostEqual(slope_stats.max_slope_deg, 0.0, places=3)
        
        # Test roughness (TRI)
        rough_stats = calculate_roughness_stats(elevation)
        self.assertAlmostEqual(rough_stats.mean_tri_m, 0.0, places=3)
        self.assertAlmostEqual(rough_stats.max_tri_m, 0.0, places=3)

    # =========================================================================
    # Scenario 2: Constant Gradient -> Expected Slope (45° and 10°)
    # =========================================================================
    def test_02_constant_gradient_ramps(self):
        """Mathematically verify slope calculation against 45° and 10° synthetic planar ramps."""
        h, w = 60, 60
        res_m = 10.0
        
        # 1. 45° planar ramp along the x-axis: dz/dx = 1.0 (rise of 10m per 10m cell)
        # z(r, c) = c * 10.0
        c_grid = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        ramp_45 = c_grid * res_m
        
        slope_stats_45 = calculate_slope_stats(ramp_45, res_x_m=res_m, res_y_m=res_m)
        self.assertAlmostEqual(slope_stats_45.mean_slope_deg, 45.0, delta=0.05)
        self.assertAlmostEqual(slope_stats_45.median_slope_deg, 45.0, delta=0.05)
        self.assertAlmostEqual(slope_stats_45.max_slope_deg, 45.0, delta=0.05)
        
        # 2. 10° planar ramp along the y-axis: dz/dy = tan(10°)
        tan_10 = math.tan(math.radians(10.0))
        r_grid = np.tile(np.arange(h, dtype=np.float32)[:, np.newaxis], (1, w))
        ramp_10 = r_grid * (res_m * tan_10)
        
        slope_stats_10 = calculate_slope_stats(ramp_10, res_x_m=res_m, res_y_m=res_m)
        self.assertAlmostEqual(slope_stats_10.mean_slope_deg, 10.0, delta=0.05)
        self.assertAlmostEqual(slope_stats_10.median_slope_deg, 10.0, delta=0.05)

    # =========================================================================
    # Scenario 3: Nodata Handling
    # =========================================================================
    def test_03_nodata_masking_and_accounting(self):
        """Verify nodata cells are properly masked and not included in statistics."""
        data = np.full((20, 20), 500.0, dtype=np.float32)
        # Set 50 cells to nodata (-9999.0) and 10 cells to NaN
        data[0:5, 0:10] = -9999.0
        data[10:11, 0:10] = np.nan
        
        elev_stats = get_elevation_stats(data, nodata_val=-9999.0)
        self.assertEqual(elev_stats.valid_cell_count, 340)
        self.assertEqual(elev_stats.nodata_count, 60)
        self.assertEqual(elev_stats.mean_m, 500.0)
        self.assertEqual(elev_stats.min_m, 500.0)
        self.assertEqual(elev_stats.max_m, 500.0)
        
        # Verify slope handles nodata without throwing or outputting NaN
        slope_stats = calculate_slope_stats(data, res_x_m=10.0, res_y_m=10.0, nodata_val=-9999.0)
        self.assertGreater(slope_stats.valid_cells, 0)
        self.assertFalse(np.isnan(slope_stats.mean_slope_deg))

    # =========================================================================
    # Scenario 4: Coordinates Outside Bounds
    # =========================================================================
    def test_04_coordinate_bounds_validation(self):
        """Verify strict rejection of out-of-bounds, inverted, or non-finite coordinates."""
        # Latitude out of bounds
        with self.assertRaises(InvalidCoordinateError):
            validate_latitude(95.0)
        with self.assertRaises(InvalidCoordinateError):
            validate_latitude(-90.1)
        with self.assertRaises(InvalidCoordinateError):
            validate_latitude(float("nan"))
            
        # Longitude normalization and out of bounds
        self.assertAlmostEqual(validate_longitude(270.0), -90.0)
        self.assertAlmostEqual(validate_longitude(-45.0), -45.0)
        with self.assertRaises(InvalidCoordinateError):
            validate_longitude(400.0)
            
        # Inverted coordinate detection (e.g. lat=120, lon=15)
        with self.assertRaises(InvalidCoordinateError) as ctx:
            validate_lunar_coordinate(120.0, 15.0)
        self.assertIn("Possible inverted coordinate order", str(ctx.exception))

    # =========================================================================
    # Scenario 5: Oversized Requests
    # =========================================================================
    def test_05_oversized_requests_and_safety_caps(self):
        """Verify that oversized radii, raster windows, and waypoint paths are capped and rejected."""
        # Excessive radius
        with self.assertRaises(OversizedRequestError):
            validate_analysis_radius(MAX_ANALYSIS_RADIUS_KM + 10.0)
            
        # Excessive raster dimensions
        from terrain_agent.terrain.resource_safety import validate_raster_dimensions
        with self.assertRaises(OversizedRequestError):
            validate_raster_dimensions(MAX_RASTER_DIM + 100, 500)
            
        # Excessive waypoints
        too_many_waypoints = [(0.0, float(i)) for i in range(150)]
        with self.assertRaises(OversizedRequestError):
            validate_waypoints(too_many_waypoints)

    # =========================================================================
    # Scenario 6: Malformed Raster Rejection
    # =========================================================================
    def test_06_malformed_raster_rejection(self):
        """Verify clean errors when encountering corrupted, non-existent, or zero-dimension rasters."""
        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            inspect_dem(self.temp_path / "non_existent.tif")
            
        # Corrupted file
        corrupt_file = self.temp_path / "corrupt.tif"
        corrupt_file.write_bytes(b"NOT A VALID GEOTIFF HEADER DATA")
        with self.assertRaises(MalformedRasterError):
            inspect_dem(corrupt_file)

    # =========================================================================
    # Scenario 7: Roughness Calculation (TRI)
    # =========================================================================
    def test_07_roughness_calculation(self):
        """Verify the Terrain Ruggedness Index (TRI) against known stepped/checkerboard terrain."""
        h, w = 30, 30
        # Checkerboard where alternating cells differ by 8.0 meters
        checkerboard = np.zeros((h, w), dtype=np.float32)
        checkerboard[::2, ::2] = 8.0
        checkerboard[1::2, 1::2] = 8.0
        
        rough_stats = calculate_roughness_stats(checkerboard)
        self.assertGreater(rough_stats.mean_tri_m, 0.0)
        self.assertIn("Riley et al. 1999", rough_stats.formula)
        self.assertIn("Scientific Notice", rough_stats.disclaimer)

    # =========================================================================
    # Scenario 8: Windowed Reading from Disk
    # =========================================================================
    def test_08_windowed_reading(self):
        """Verify that DEM reader reads only requested sub-windows without loading full file."""
        # Create a 200 x 200 synthetic DEM on disk
        data = np.arange(200 * 200, dtype=np.float32).reshape((200, 200))
        dem_file = self._create_synthetic_geotiff(
            data, res_x=10.0, res_y=10.0, origin_x=0.0, origin_y=2000.0
        )
        
        # Read a 50 x 50 cell sub-window: bounds = (500, 1000, 1000, 1500)
        window_data, win_transform, metadata = read_dem_window(
            dem_file, bounds=(500.0, 1000.0, 1000.0, 1500.0)
        )
        
        self.assertEqual(window_data.shape, (50, 50))
        self.assertEqual(metadata.width, 200)
        self.assertEqual(metadata.height, 200)
        self.assertAlmostEqual(win_transform.c, 500.0)  # X origin of window
        self.assertAlmostEqual(win_transform.f, 1500.0)  # Y origin of window
        
        # Reading bounds completely outside raster must raise InvalidCoordinateError
        with self.assertRaises(InvalidCoordinateError):
            read_dem_window(dem_file, bounds=(5000.0, 6000.0, 7000.0, 8000.0))


if __name__ == "__main__":
    unittest.main()
