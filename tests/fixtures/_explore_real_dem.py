import sys, json, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from terrain_agent.terrain import open_dem_context
from terrain_agent.tools.terrain_stats import analyze_terrain_bbox
from terrain_agent.safety import check_rover_safety, analyze_landing_site, find_safe_regions, SafetyStatus
from terrain_agent.terrain.resource_safety import InvalidCoordinateError, InvalidThresholdError

DEM = Path("tests/fixtures/nasa_real_dem_cache/nasa/ldem_75s_240m-6d162ce24928.tif")
LAT, LON = -89.9, 0.0

ctx = open_dem_context(DEM)
print("CRS geographic:", ctx.is_geographic, "projection:", ctx.projection)
print("pixel_size_m at LAT,LON:", ctx.pixel_size_m(LAT, LON))
print("width,height:", ctx.width, ctx.height)

print("\n--- analyze_terrain_bbox ---")
stats = analyze_terrain_bbox(DEM, -89.95, -89.85, -10.0, 10.0)
print("terrain_available", stats.terrain_available, "coverage", stats.coverage_fraction, "resolution_m", stats.resolution_m)
print("elevation", stats.elevation)
print("slope mean/max", stats.slope.mean_slope_deg, stats.slope.max_slope_deg)
print("roughness mean/max", stats.roughness.mean_tri_m, stats.roughness.max_tri_m)

print("\n--- check_rover_safety: fully covered route ---")
route = [(-89.9, 0.0), (-89.85, 30.0), (-89.8, 60.0)]
r = check_rover_safety(route, 15.0, dem_path=DEM)
print("status", r.status, "terrain_available", r.terrain_available, "coverage", r.coverage_fraction, "risk", r.risk_score, "basis", r.risk_score_basis)

print("\n--- check_rover_safety: missing DEM (dem_path=None) ---")
r2 = check_rover_safety(route, 15.0, dem_path=None)
print("status", r2.status, "terrain_available", r2.terrain_available)
assert r2.status != SafetyStatus.PASS, "BUG: missing DEM produced PASS"

print("\n--- check_rover_safety: partially covered route (one point way outside DEM) ---")
route_partial = [(-89.9, 0.0), (-60.0, 0.0)]
r3 = check_rover_safety(route_partial, 15.0, dem_path=DEM)
print("status", r3.status, "terrain_available", r3.terrain_available, "coverage", r3.coverage_fraction, "incomplete_segments", r3.incomplete_segments)
for seg in r3.segments:
    print("  seg", seg.segment_index, seg.status, seg.coverage_fraction, seg.data_issues)
assert r3.status != SafetyStatus.PASS, "BUG: partially covered route produced PASS"

print("\n--- check_rover_safety: invalid coordinate ---")
try:
    check_rover_safety([(95.0, 0.0), (-89.9, 10.0)], 15.0, dem_path=DEM)
    print("BUG: no exception raised for invalid latitude")
except InvalidCoordinateError as e:
    print("OK raised InvalidCoordinateError:", e)

print("\n--- check_rover_safety: invalid threshold ---")
for bad in [-5.0, 0.0, 91.0, float("nan"), "15"]:
    try:
        check_rover_safety(route, bad, dem_path=DEM)
        print(f"BUG: no exception for maximum_slope_deg={bad!r}")
    except InvalidThresholdError as e:
        print(f"OK raised for {bad!r}:", e)
    except Exception as e:
        print(f"UNEXPECTED exception type for {bad!r}: {type(e).__name__}: {e}")

print("\n--- analyze_landing_site: normal ---")
site = analyze_landing_site(DEM, -89.9, 0.0, radius_m=1000.0, maximum_slope_deg=10.0, min_flat_radius_m=200.0)
print("status", site.status, "terrain_available", site.terrain_available, "coverage", site.coverage_fraction, "rankable", site.rankable)

print("\n--- analyze_landing_site: missing DEM ---")
site2 = analyze_landing_site(None, -89.9, 0.0)
print("status", site2.status, "terrain_available", site2.terrain_available, "rankable", site2.rankable)
assert site2.status != SafetyStatus.PASS, "BUG: missing DEM landing site produced PASS"

print("\n--- analyze_landing_site: outside DEM coverage ---")
site3 = analyze_landing_site(DEM, -60.0, 0.0, radius_m=1000.0)
print("status", site3.status, "terrain_available", site3.terrain_available, "coverage", site3.coverage_fraction, "rankable", site3.rankable, "issues", site3.data_issues)
assert site3.status != SafetyStatus.PASS, "BUG: outside-coverage landing site produced PASS"

print("\n--- find_safe_regions: generous threshold ---")
sr = find_safe_regions(DEM, -89.9, 0.0, 3000.0, maximum_slope_deg=25.0, min_area_m2=50000.0)
print("outcome", sr.outcome, "terrain_available", sr.terrain_available, "assessed_fraction", sr.assessed_fraction, "n_regions", len(sr.regions))

print("\n--- find_safe_regions: tiny threshold ---")
sr2 = find_safe_regions(DEM, -89.9, 0.0, 3000.0, maximum_slope_deg=0.05, min_area_m2=50000.0)
print("outcome", sr2.outcome, "terrain_available", sr2.terrain_available, "n_regions", len(sr2.regions))

print("\n--- find_safe_regions: missing DEM ---")
sr3 = find_safe_regions(None, -89.9, 0.0, 3000.0, maximum_slope_deg=15.0)
print("outcome", sr3.outcome, "terrain_available", sr3.terrain_available)
assert sr3.outcome == "no_terrain_data"

print("\n--- determinism check ---")
r_a = check_rover_safety(route, 15.0, dem_path=DEM)
r_b = check_rover_safety(route, 15.0, dem_path=DEM)
print("equal:", r_a.model_dump() == r_b.model_dump())

print("\nALL EXPLORATION CHECKS COMPLETED OK")
