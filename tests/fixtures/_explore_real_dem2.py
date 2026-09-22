import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from terrain_agent.safety import check_rover_safety, analyze_landing_site, find_safe_regions, SafetyStatus
from terrain_agent.terrain.resource_safety import InvalidCoordinateError, InvalidThresholdError

DEM = Path("tests/fixtures/nasa_real_dem_cache/nasa/ldem_75s_240m-6d162ce24928.tif")

print("--- check_rover_safety: route crossing the DEM edge (partial coverage) ---")
route_partial = [(-75.8, 0.0), (-74.8, 0.0)]  # inside -> outside, ~111 km, within 250km limit
r3 = check_rover_safety(route_partial, 15.0, dem_path=DEM)
print("status", r3.status, "terrain_available", r3.terrain_available, "coverage", r3.coverage_fraction, "incomplete_segments", r3.incomplete_segments)
for seg in r3.segments:
    print("  seg", seg.segment_index, seg.status, "cov=", seg.coverage_fraction, seg.data_issues)
assert r3.status != SafetyStatus.PASS, "BUG: partially covered route produced PASS"
assert r3.coverage_fraction < 1.0, "expected partial coverage"

print("\n--- analyze_landing_site: footprint straddling the DEM edge ---")
site3 = analyze_landing_site(DEM, -75.3, 0.0, radius_m=30000.0, maximum_slope_deg=25.0, min_flat_radius_m=100.0)
print("status", site3.status, "terrain_available", site3.terrain_available, "coverage", site3.coverage_fraction, "rankable", site3.rankable, "issues", site3.data_issues)
assert site3.status != SafetyStatus.PASS, "BUG: partial-coverage landing site produced PASS"

print("\n--- find_safe_regions: footprint straddling the DEM edge ---")
sr = find_safe_regions(DEM, -75.3, 0.0, 30000.0, maximum_slope_deg=25.0, min_area_m2=50000.0)
print("outcome", sr.outcome, "terrain_available", sr.terrain_available, "assessed_fraction", sr.assessed_fraction, "n_regions", len(sr.regions))
assert sr.assessed_fraction < 1.0, "expected partial coverage in assessed_fraction"

print("\n--- analyze_landing_site: entirely outside DEM (beyond raster extent) ---")
site4 = analyze_landing_site(DEM, -60.0, 0.0, radius_m=1000.0)
print("status", site4.status, "terrain_available", site4.terrain_available, "coverage", site4.coverage_fraction, "issues", site4.data_issues)
assert site4.status != SafetyStatus.PASS

print("\n--- find_safe_regions: entirely outside DEM ---")
sr2 = find_safe_regions(DEM, -60.0, 0.0, 1000.0, maximum_slope_deg=15.0)
print("outcome", sr2.outcome, "terrain_available", sr2.terrain_available)

print("\n--- analyze_landing_site: invalid coordinate ---")
try:
    analyze_landing_site(DEM, 999.0, 0.0)
    print("BUG: no exception for invalid lat")
except InvalidCoordinateError as e:
    print("OK:", e)

print("\n--- find_safe_regions: invalid threshold ---")
for bad in [-1.0, 0.0, 95.0, float("nan")]:
    try:
        find_safe_regions(DEM, -89.9, 0.0, 3000.0, maximum_slope_deg=bad)
        print(f"BUG: no exception for maximum_slope_deg={bad!r}")
    except InvalidThresholdError as e:
        print(f"OK for {bad!r}:", e)

print("\n--- analyze_landing_site: invalid threshold ---")
for bad in [-1.0, 0.0, 91.0]:
    try:
        analyze_landing_site(DEM, -89.9, 0.0, maximum_slope_deg=bad)
        print(f"BUG: no exception for maximum_slope_deg={bad!r}")
    except InvalidThresholdError as e:
        print(f"OK for {bad!r}:", e)

print("\n--- determinism: analyze_landing_site / find_safe_regions ---")
a1 = analyze_landing_site(DEM, -89.9, 0.0, radius_m=1000.0, maximum_slope_deg=10.0)
a2 = analyze_landing_site(DEM, -89.9, 0.0, radius_m=1000.0, maximum_slope_deg=10.0)
print("landing equal:", a1.model_dump() == a2.model_dump())
s1 = find_safe_regions(DEM, -89.9, 0.0, 3000.0, maximum_slope_deg=25.0, min_area_m2=50000.0)
s2 = find_safe_regions(DEM, -89.9, 0.0, 3000.0, maximum_slope_deg=25.0, min_area_m2=50000.0)
print("safe_regions equal:", s1.model_dump() == s2.model_dump())

print("\nALL CHECKS 2 COMPLETED OK")
