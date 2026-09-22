"""Georeferencing of equirectangular (cylindrical) lunar grids.

NASA cylindrical lunar DEMs are distributed as equirectangular grids in metres. On such a
grid the east-west cell shrinks with the cosine of latitude while the north-south cell does
not. These tests use synthetic rasters shaped like that projection.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from terrain_agent.safety import SafetyStatus, check_rover_safety
from terrain_agent.terrain import UnsupportedCRSError, open_dem_context

R = 1_737_400.0
EQC = f"+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=180 +x_0=0 +y_0=0 +R={R} +units=m +no_defs"
SINUSOIDAL = f"+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R={R} +units=m +no_defs"
CELL = 500.0  # metres of grid per cell


def write(path, array, crs, west, north):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs=CRS.from_string(crs),
        transform=from_origin(west, north, CELL, CELL),
        nodata=-9999.0,
    ) as dst:
        dst.write(array.astype("float32"), 1)
    return path


def grid_position(lat, lon):
    """Grid coordinates (metres) of a location on the equirectangular grid used here."""
    return R * math.radians(lon - 180.0), R * math.radians(lat)


def test_cell_size_shrinks_east_west_with_the_cosine_of_latitude(tmp_path):
    path = write(tmp_path / "eqc.tif", np.zeros((60, 60)), EQC, -15000.0, 15000.0)
    ctx = open_dem_context(path)
    for lat in (0.0, 30.0, 60.0, 80.0):
        dx, dy = ctx.pixel_size_m(lat, 180.0)
        assert dy == pytest.approx(CELL, rel=1e-6)
        assert dx == pytest.approx(CELL * math.cos(math.radians(lat)), rel=1e-4)


def test_east_west_slope_is_correct_on_an_equirectangular_grid(tmp_path):
    """A plane of true slope 20 degrees rising east at 60 degrees north.

    If the shrinking east-west cell were ignored the measured slope would be about 10 degrees.
    """
    lat = 60.0
    x0, y0 = grid_position(lat, 180.0)
    true_dx = CELL * math.cos(math.radians(lat))
    columns = np.arange(80) * true_dx * math.tan(math.radians(20.0))
    elevation = np.tile(1500.0 + columns, (80, 1))
    west, north = x0 - 40 * CELL, y0 + 40 * CELL
    path = write(tmp_path / "ramp60.tif", elevation, EQC, west, north)

    route = [(lat, 179.9), (lat, 180.1)]
    result = check_rover_safety(route, 30.0, dem_path=path)
    assert result.coverage_fraction == 1.0
    assert result.max_slope_deg == pytest.approx(20.0, abs=0.2)
    assert result.status is SafetyStatus.PASS


def test_near_the_pole_an_equirectangular_grid_is_refused_not_guessed(tmp_path):
    path = write(tmp_path / "eqc.tif", np.zeros((60, 60)), EQC, -15000.0, 15000.0)
    ctx = open_dem_context(path)
    with pytest.raises(UnsupportedCRSError, match="polar"):
        ctx.pixel_size_m(88.0, 180.0)


def test_unsupported_projections_are_refused_not_guessed(tmp_path):
    path = write(tmp_path / "sinu.tif", np.zeros((60, 60)), SINUSOIDAL, -15000.0, 15000.0)
    ctx = open_dem_context(path)
    with pytest.raises(UnsupportedCRSError, match="not supported"):
        ctx.pixel_size_m(10.0, 5.0)
    result = check_rover_safety([(10.0, 5.0), (10.0, 5.05)], 15.0, dem_path=path)
    assert result.status is SafetyStatus.REVIEW_REQUIRED
    assert not result.terrain_available
