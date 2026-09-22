import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from terrain_agent.safety import check_rover_safety, analyze_landing_site, find_safe_regions, SafetyStatus

DEM = Path("tests/fixtures/nasa_real_dem_cache/nasa/ldem_75s_240m-6d162ce24928.tif")

print("--- strict slope threshold safe-region ---")
strict = find_safe_regions(DEM, -89.9, 0.0, 3000.0, maximum_slope_deg=0.05, min_area_m2=50000.0)
print("outcome", strict.outcome, "n_regions", len(strict.regions), "assessed_fraction", strict.assessed_fraction, "safe_fraction", strict.safe_fraction_of_assessed)

print("\n--- generous slope threshold safe-region ---")
gen = find_safe_regions(DEM, -89.9, 0.0, 3000.0, maximum_slope_deg=25.0, min_area_m2=50000.0)
print("outcome", gen.outcome, "n_regions", len(gen.regions), "assessed_fraction", gen.assessed_fraction, "safe_fraction", gen.safe_fraction_of_assessed)

print("\n--- landing site: insufficient flat-radius evidence, generous slope ---")
site = analyze_landing_site(DEM, -89.9, 0.0, radius_m=1000.0, maximum_slope_deg=90.0, min_flat_radius_m=50.0)
print("status", site.status, "coverage", site.coverage_fraction, "max_slope", site.max_slope_deg, "flat_radius", site.flat_radius_m, "issues", site.data_issues, "rankable", site.rankable)

print("\n--- rover: roughness-only violation, generous slope ---")
route = [(-89.9, 0.0), (-89.85, 30.0), (-89.8, 60.0)]
result = check_rover_safety(route, 89.0, maximum_roughness=5.0, dem_path=DEM)
print("status", result.status, "max_slope", result.max_slope_deg, "mean_tri", result.mean_tri_m, "violated_segments", result.violated_segments)
for seg in result.segments:
    print("  seg", seg.segment_index, seg.status, "max_slope", seg.max_slope_deg, "mean_tri", seg.mean_tri_m, [v.kind for v in seg.violations])

print("\n--- disclaimer / risk method text ---")
print(repr(result.disclaimer))
print(repr(result.risk_score_method))
print(repr(site.disclaimer))
