"""Pytest configuration and shared fixtures for TALUS."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

# Ensure src/ is on sys.path
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


@pytest.fixture(autouse=True)
def _reset_dispatch_rate_limiters():
    """``dispatch_tool_call`` is bounded by module-level, process-wide rate limiters (Phase 8
    hardening) so every caller -- the chat agent and any UI calling tools directly -- is
    covered, not just one agent instance. That means they persist for the life of the test
    process; without a reset, a test file that calls a network tool (e.g. fetch_nasa_dem)
    many times in a row would trip the limiter and fail later, unrelated tests."""
    from terrain_agent.agent import agent as agent_module

    agent_module._DISPATCH_RATE_LIMITER.reset()
    agent_module._NETWORK_TOOL_RATE_LIMITER.reset()
    yield

LUNAR_RADIUS_M = 1_737_400.0
GEOGRAPHIC_CRS = f"+proj=longlat +R={LUNAR_RADIUS_M} +no_defs"
POLAR_STEREOGRAPHIC_CRS = (
    f"+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R={LUNAR_RADIUS_M} "
    "+units=m +no_defs"
)


class DemBuilder:
    """Builds small synthetic GeoTIFF DEMs for tests.

    The default geographic DEM is 400 x 400 cells covering longitude 10.0 to 10.4 degrees and
    latitude 0.0 to 0.4 degrees at 0.001 degree cell size (about 30.3 m). Every value is
    synthetic and generated in code. No NASA data is involved.
    """

    SIZE = 400
    LON0 = 10.0
    LAT_TOP = 0.4
    RES_DEG = 0.001

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.pixel_m = self.RES_DEG * math.radians(1.0) * LUNAR_RADIUS_M * math.cos(
            math.radians(self.LAT_TOP / 2)
        )

    # ------------------------------------------------------------------
    # Elevation generators, each returns a SIZE x SIZE float array
    # ------------------------------------------------------------------

    def flat(self, value: float = 1500.0) -> np.ndarray:
        return np.full((self.SIZE, self.SIZE), value, dtype=np.float64)

    def ramp(self, slope_deg: float) -> np.ndarray:
        """Uniform slope rising toward the east (increasing column)."""
        columns = np.arange(self.SIZE) * self.pixel_m * math.tan(math.radians(slope_deg))
        return np.tile(columns, (self.SIZE, 1)) + 1500.0

    def bands(self, bands: list[tuple[int, int, float]]) -> np.ndarray:
        """Slope bands along the column axis: (first_col, last_col_exclusive, slope_deg)."""
        slope_per_step = np.zeros(self.SIZE)
        for first, last, slope_deg in bands:
            slope_per_step[first:last] = math.tan(math.radians(slope_deg))
        profile = np.concatenate([[0.0], np.cumsum(slope_per_step[:-1] * self.pixel_m)]) + 1500.0
        return np.tile(profile, (self.SIZE, 1))

    def rough(self, sigma_m: float = 0.5, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(1500.0, sigma_m, (self.SIZE, self.SIZE))

    # ------------------------------------------------------------------
    # File writers
    # ------------------------------------------------------------------

    def geographic(
        self,
        name: str,
        elevation: np.ndarray,
        *,
        res_deg: float | None = None,
        lon0: float | None = None,
        lat_top: float | None = None,
        nodata: float = -9999.0,
        sidecar: dict | None = None,
    ) -> Path:
        res = self.RES_DEG if res_deg is None else res_deg
        west = self.LON0 if lon0 is None else lon0
        north = self.LAT_TOP if lat_top is None else lat_top
        path = self.directory / name
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=elevation.shape[0],
            width=elevation.shape[1],
            count=1,
            dtype="float32",
            crs=CRS.from_string(GEOGRAPHIC_CRS),
            transform=from_origin(west, north, res, res),
            nodata=nodata,
        ) as dst:
            dst.write(elevation.astype("float32"), 1)
        if sidecar is not None:
            path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
        return path

    def stereographic(
        self,
        name: str,
        elevation: np.ndarray,
        *,
        west: float,
        north: float,
        res_m: float = 10.0,
        nodata: float = -9999.0,
    ) -> Path:
        path = self.directory / name
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=elevation.shape[0],
            width=elevation.shape[1],
            count=1,
            dtype="float32",
            crs=CRS.from_string(POLAR_STEREOGRAPHIC_CRS),
            transform=from_origin(west, north, res_m, res_m),
            nodata=nodata,
        ) as dst:
            dst.write(elevation.astype("float32"), 1)
        return path

    # ------------------------------------------------------------------
    # Location helpers for the default geographic DEM
    # ------------------------------------------------------------------

    def latlon(self, row: float, col: float) -> tuple[float, float]:
        """Latitude and longitude of a cell centre in the default geographic grid."""
        return (self.LAT_TOP - (row + 0.5) * self.RES_DEG, self.LON0 + (col + 0.5) * self.RES_DEG)

    def route(self, *cells: tuple[float, float]) -> list[tuple[float, float]]:
        return [self.latlon(r, c) for r, c in cells]

    @staticmethod
    def lunar_to_stereographic(lat: float, lon: float) -> tuple[float, float]:
        xs, ys = warp_transform(
            CRS.from_string(GEOGRAPHIC_CRS), CRS.from_string(POLAR_STEREOGRAPHIC_CRS), [lon], [lat]
        )
        return xs[0], ys[0]


@pytest.fixture
def dem_builder(tmp_path: Path) -> DemBuilder:
    """Factory for synthetic DEM files in a temporary directory."""
    return DemBuilder(tmp_path)


# ---------------------------------------------------------------------------
# NASA acquisition fixtures (offline, synthetic)
# ---------------------------------------------------------------------------

from tests.acquisition_fixtures import MockNasa, public_resolver, write_product  # noqa: E402


@pytest.fixture
def mock_nasa() -> MockNasa:
    """Mock ODE REST service and PDS data server."""
    return MockNasa()


@pytest.fixture
def pds_product(tmp_path: Path):
    """Write a synthetic PDS4 product (label and data file) and return their paths."""

    def make(name: str = "ldem_75s_240m", **kwargs):
        return write_product(tmp_path / "product", name, **kwargs)

    return make


@pytest.fixture
def make_service(tmp_path: Path, mock_nasa: MockNasa):
    """Build a NasaDemService wired to the mock servers, with no sleeping and no DNS."""
    import httpx

    from terrain_agent.acquisition.cache import DemCache
    from terrain_agent.acquisition.download import DownloadManager
    from terrain_agent.acquisition.http import HttpSettings
    from terrain_agent.acquisition.net_policy import NASA_DOWNLOAD_HOSTS, HostPolicy
    from terrain_agent.acquisition.ode_provider import ODE_API_HOSTS, OdeProvider
    from terrain_agent.acquisition.service import NasaDemService

    def build(
        *,
        max_download_bytes: int = 50_000_000,
        max_cache_bytes: int = 500_000_000,
        retries: int = 2,
        enabled: bool = True,
        verify_on_use: bool = True,
        cache_dir: Path | None = None,
        resolver=public_resolver,
    ) -> NasaDemService:
        http = HttpSettings(max_retries=retries, backoff_base_s=1.0)
        api_policy = HostPolicy(ODE_API_HOSTS, resolver)
        file_policy = HostPolicy(NASA_DOWNLOAD_HOSTS, resolver)
        client = httpx.Client(transport=mock_nasa.transport, follow_redirects=False)
        provider = OdeProvider(
            policy=api_policy,
            file_policy=file_policy,
            settings=http,
            client=client,
            sleep=lambda _s: None,
        )
        downloader = DownloadManager(
            policy=file_policy,
            settings=http,
            max_bytes=max_download_bytes,
            client_factory=lambda: httpx.Client(transport=mock_nasa.transport, follow_redirects=False),
            sleep=lambda _s: None,
        )
        cache = DemCache(cache_dir or (tmp_path / "cache"), max_bytes=max_cache_bytes, verify_on_use=verify_on_use)
        return NasaDemService(provider, downloader, cache, enabled=enabled)

    return build
