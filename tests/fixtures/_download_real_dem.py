"""One-time helper: acquire a real NASA lunar DEM into the local test fixture cache.

This is not a pytest test. It is a small script a developer runs once, with network
access to the NASA ODE/PDS domains, to populate ``tests/fixtures/nasa_real_dem_cache``.
Once populated, ``tests/integration/test_real_nasa_dem_pipeline.py`` reads that cache
directly and needs no network at all, so the deterministic integration suite stays
fast and offline on every subsequent run (including in CI, where the tests self-skip
if the fixture has not been created).

It goes through the same Phase 4 acquisition path as the rest of the application
(search -> select -> download -> validate -> normalise -> cache with provenance), just
against a fixed small area near the lunar south pole (Shackleton crater vicinity) so the
download stays small. The chosen NASA product actually covers a much larger area (the
whole south-polar LOLA GDR tile), which is what makes it possible to test edge-of-coverage
and out-of-coverage behaviour deterministically in the integration tests.

Usage
-----
    python tests/fixtures/_download_real_dem.py

The cache directory is gitignored; each developer/CI machine that wants to run the real-data
integration tests populates it locally.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from terrain_agent.acquisition.models import CoverageRequest  # noqa: E402
from terrain_agent.acquisition.service import build_default_service  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / "nasa_real_dem_cache"

# Shackleton crater vicinity, lunar south pole -- matches tests/live/test_nasa_live.py.
LAT, LON, RADIUS_M = -89.9, 0.0, 5000.0


def main() -> None:
    t0 = time.time()
    service = build_default_service(cache_dir=CACHE_DIR, enabled=True)
    request = CoverageRequest.from_point(LAT, LON, RADIUS_M)
    acquired = service.acquire(request, product_types=["GDRDEM"])
    elapsed = time.time() - t0

    p = acquired.provenance
    print(f"Elapsed: {elapsed:.1f}s  from_cache={acquired.from_cache}")
    print(f"dem_path: {acquired.dem_path}")
    print(f"size_bytes: {acquired.dem_path.stat().st_size}")
    print(
        json.dumps(
            {
                "product_id": p["product_id"],
                "product_type": p["product_type"],
                "mission": p["mission"],
                "instrument": p["instrument"],
                "crs_proj": p["raster"]["crs_proj"],
                "native_pixel_size": p["raster"]["native_pixel_size"],
                "width": p["raster"]["width"],
                "height": p["raster"]["height"],
                "elevation_min_m": p["normalization"]["elevation_min_m"],
                "elevation_max_m": p["normalization"]["elevation_max_m"],
                "product_bounds_lat": p["raster"]["product_bounds_lat"],
                "product_bounds_lon_east": p["raster"]["product_bounds_lon_east"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
