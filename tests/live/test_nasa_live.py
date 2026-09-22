"""OPTIONAL live NASA integration test. This is the only test in the project that uses the network.

It is skipped unless the environment variable TALUS_RUN_LIVE_NASA is set to 1, so the normal
suite stays offline and deterministic.

What it does, with no mocks:

1. Asks NASA ODE (the documented REST interface) for LOLA gridded DEM products covering the
   Shackleton crater area at the lunar south pole.
2. Selects a product that fits the configured download size limit and downloads it from the
   PDS Geosciences Node.
3. Validates it, converts it to metres, caches it with provenance.
4. Runs the existing Phase 5 analyses on the real DEM.

The test asserts only facts that follow from the data itself (physical plausibility, internal
consistency, provenance structure). It does not assert specific terrain values.

Run it:

    PowerShell:  $env:TALUS_RUN_LIVE_NASA = "1"; python -m pytest tests/live -m live_nasa -s -p no:cacheprovider
    bash:        TALUS_RUN_LIVE_NASA=1 python -m pytest tests/live -m live_nasa -s -p no:cacheprovider

Optional: set TALUS_LIVE_CACHE_DIR to a directory to keep the downloaded product between runs.
The download can take several minutes because the NASA data server is slow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from terrain_agent.acquisition.models import CoverageRequest
from terrain_agent.acquisition.ode_provider import ODE_LIVE2_URL
from terrain_agent.acquisition.service import build_default_service
from terrain_agent.safety import (
    SafetyStatus,
    analyze_landing_site,
    check_rover_safety,
    find_safe_regions,
)
from terrain_agent.terrain import open_dem_context
from terrain_agent.tools.terrain_stats import analyze_terrain_bbox

pytestmark = [
    pytest.mark.live_nasa,
    pytest.mark.skipif(
        os.getenv("TALUS_RUN_LIVE_NASA") != "1",
        reason="live NASA test: set TALUS_RUN_LIVE_NASA=1 to run it (needs internet access)",
    ),
]

# Shackleton crater region, lunar south pole. These are the exact coordinates reported against
# the deployed Streamlit "Fetch DEM" tool (radius given there in km): a real regression case for
# the "ODE Products section has an unexpected structure" bug, which only reproduced with every
# default product type queried (GDRDEM *and* SLDEM) -- SLDEM has zero coverage at this location,
# and ODE represents that as the literal string "No Products Found", not an absent/null field.
LAT, LON, RADIUS_M = -89.90, 0.16, 5.08 * 1000.0


def test_real_nasa_lola_dem_end_to_end(tmp_path, capsys):
    cache_dir = Path(os.getenv("TALUS_LIVE_CACHE_DIR") or tmp_path / "cache")

    # 0. The documented IIPT query lists the product type we are about to use.
    iipt = httpx.get(
        ODE_LIVE2_URL,
        params={"query": "iipt", "odemetadb": "moon", "output": "JSON"},
        timeout=60.0,
        headers={"User-Agent": "TALUS-terrain-agent/0.1 (research; lunar terrain analysis)"},
    )
    assert iipt.status_code == 200
    sets = json.loads(iipt.content.decode("utf-8-sig"))["ODEResults"]["IIPTSets"]["IIPTSet"]
    assert any((s["IHID"], s["IID"], s["PT"]) == ("LRO", "LOLA", "GDRDEM") for s in sets)

    # 1-3. Search, select, download, validate, normalise, cache.
    service = build_default_service(cache_dir=cache_dir)
    request = CoverageRequest.from_point(LAT, LON, RADIUS_M)
    # product_types=None, exactly as the Streamlit "Fetch DEM" tool calls it with no
    # preferred_dataset: every default type (GDRDEM and SLDEM) is queried.
    acquired = service.acquire(request, product_types=None)
    p = acquired.provenance

    assert acquired.dem_path.is_file() and acquired.dem_path.parent == service.cache.root
    assert (p["mission"], p["instrument"], p["product_type"]) == ("LRO", "LOLA", "GDRDEM")
    assert p["provider_id"] == "nasa_ode" and p["product_id"].startswith("ldem_")
    assert p["product_lid"] and p["product_lid"].startswith("urn:nasa:pds:")
    assert p["source"]["data_url"].startswith("https://pds-geosciences.wustl.edu/")
    assert len(p["source"]["data_sha256"]) == 64 and len(p["normalization"]["output_sha256"]) == 64
    assert p["validation"]["passed"] and "coverage" in p["validation"]["checks"]
    assert p["height"]["reference_radius_m"] == pytest.approx(1_737_400.0)
    assert p["raster"]["crs_kind"] == "projected" and p["raster"]["projection"] in ("stere", "eqc")
    assert p["raster"]["width"] > 0 and p["raster"]["height"] > 0
    assert 1.0 < p["raster"]["native_pixel_size"][0] < 30_000.0

    # Physical plausibility: the Moon spans roughly -9 km to +11 km about the reference sphere.
    assert -12_000.0 < p["normalization"]["elevation_min_m"] < p["normalization"]["elevation_max_m"] < 12_000.0
    assert p["normalization"]["masked_cells"] < 0.01 * p["normalization"]["total_cells"]

    # 4. The existing Phase 5 engine on the real DEM.
    ctx = open_dem_context(acquired.dem_path)
    assert ctx.metadata.crs_wkt and not ctx.is_geographic
    pixel_m = ctx.pixel_size_m(LAT, LON)
    assert pixel_m[0] == pytest.approx(p["raster"]["native_pixel_size"][0], rel=0.05)

    stats = analyze_terrain_bbox(acquired.dem_path, -89.95, -89.85, -10.0, 10.0)
    assert stats.terrain_available and stats.coverage_fraction > 0.9
    assert -12_000.0 < stats.elevation.min_m <= stats.elevation.max_m < 12_000.0
    assert 0.0 <= stats.slope.mean_slope_deg <= stats.slope.max_slope_deg < 90.0
    assert stats.dataset.provenance == "sidecar" and stats.dataset.product_type == "GDRDEM"

    route = check_rover_safety([(-89.5, 0.0), (-89.6, 30.0), (-89.7, 60.0)], 15.0, dem_path=acquired.dem_path)
    assert route.terrain_available and route.coverage_fraction == 1.0
    assert route.status in (SafetyStatus.PASS, SafetyStatus.FAIL, SafetyStatus.REVIEW_REQUIRED)
    assert route.max_slope_deg is not None and route.risk_score is not None
    assert route.dataset.product_id == p["product_id"] and route.dataset.cache_id == acquired.cache_id
    assert "not a geoid" in route.dataset.elevation_reference

    site = analyze_landing_site(acquired.dem_path, -89.5, 0.0, radius_m=1000.0, maximum_slope_deg=10.0, min_flat_radius_m=500.0)
    assert site.terrain_available and site.coverage_fraction == 1.0 and site.max_slope_deg is not None

    regions = find_safe_regions(acquired.dem_path, -89.5, 0.0, 5000.0, maximum_slope_deg=10.0, min_area_m2=100_000.0)
    assert regions.terrain_available and regions.assessed_fraction == 1.0

    # A second request is served locally with no network use.
    cached = service.find_cached(request, ["GDRDEM"])
    assert cached is not None and cached.from_cache and cached.cache_id == acquired.cache_id

    summary = {
        "product_id": p["product_id"],
        "product_type": p["product_type"],
        "product_lid": p["product_lid"],
        "mission_instrument": f"{p['mission']} / {p['instrument']}",
        "crs": p["raster"]["crs_proj"],
        "native_pixel_size_m": p["raster"]["native_pixel_size"],
        "raster_dimensions": [p["raster"]["width"], p["raster"]["height"]],
        "product_bounds_lat": p["raster"]["product_bounds_lat"],
        "product_bounds_lon_east": p["raster"]["product_bounds_lon_east"],
        "elevation_range_m": [p["normalization"]["elevation_min_m"], p["normalization"]["elevation_max_m"]],
        "downloaded": not acquired.from_cache,
    }
    with capsys.disabled():
        print("\nREAL NASA DATA VERIFIED")
        print(json.dumps(summary, indent=2))
