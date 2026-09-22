# Phase 5: Deterministic Route, Landing-Site and Safe-Region Analysis

> [!CAUTION]
> **Research and demonstration only.** Every threshold below is a configured analysis
> parameter, not a certified safety limit. Results are not mission approval, landing
> approval, or guaranteed rover safety.

All numbers come from deterministic Python code in `src/terrain_agent/`. The language model
only selects tools and explains their output. Statuses are computed from measured terrain
and configured thresholds, never from text.

## Entry points

| Function | Module | Purpose |
| :--- | :--- | :--- |
| `check_rover_safety(waypoints, maximum_slope_deg, maximum_roughness=None, no_go_zones=None, *, dem_path=None)` | `safety/rover.py` | Route analysis |
| `analyze_landing_site(dem_path, lat, lon, ...)` | `safety/landing.py` | One candidate site |
| `compare_landing_candidates(dem_path, sites, ...)` | `safety/landing.py` | Analyse and rank sites |
| `rank_landing_sites(analyses)` | `safety/landing.py` | Rank existing analyses |
| `find_safe_regions(dem_path, lat, lon, radius_m, ...)` | `safety/safe_regions.py` | Connected safe areas |
| `analyze_terrain_bbox(dem_path, min_lat, max_lat, min_lon, max_lon)` | `tools/terrain_stats.py` | Elevation, slope, roughness statistics |

The agent exposes these as `evaluate_traverse_route`, `evaluate_landing_sites`,
`find_safe_regions`, `get_elevation_stats`, `get_slope_stats` and `get_roughness_stats`. The
model passes a DEM file name. It cannot choose arbitrary file system paths. Files must
resolve inside the configured cache or sample directory.

## How terrain is measured

1. **Georeferencing.** Latitude and longitude are converted with the reference sphere
   declared by the DEM itself. Converting from Earth EPSG:4326 to a lunar projection is not a
   valid coordinate operation, so it is not used. Longitude conventions of -180 to 180 and
   0 to 360 are both handled.
2. **Cell size in metres.** Projected DEMs use the cell size divided by the local map scale
   factor, measured numerically at the analysis location. For polar stereographic this matches
   `2 / (1 + sin(latitude))`, which is 0.77% at 80 degrees. Geographic DEMs use degrees times
   the lunar radius, with the cosine of latitude in the east-west direction. Geographic DEMs
   within about 3 degrees of a pole are refused because metric slope is unreliable there.
3. **Slope** is Horn's gradient magnitude. **Roughness** is the Terrain Ruggedness Index
   (Riley et al. 1999), the mean absolute elevation difference to the eight neighbours, in
   metres.
4. **Measured cell.** A cell counts as measured only when its whole 3x3 neighbourhood holds
   valid elevations. Cells outside the raster, on nodata, or on the raster edge are
   unmeasured. Unmeasured cells are reported and never filled in.
5. **Read limits.** No read exceeds 2048 x 2048 cells. Long routes are processed in chunks of
   at most 1024 cells with a 2 cell margin.

## Rover route analysis

The route is sampled along great circles at half the DEM cell size (or at the requested
spacing). Each distinct cell touched by a segment is counted once for that segment.
Measurement is along the route centreline only.

### Status rules

Per segment, the worst applicable rule wins. The route status is the worst segment status.

| Status | Condition |
| :--- | :--- |
| `FAIL` | Maximum measured slope on the segment exceeds `maximum_slope_deg`, or the segment touches a no-go zone |
| `REVIEW_REQUIRED` | Mean TRI exceeds `maximum_roughness` (only if a limit was given), or any cell on the segment is unmeasured, or no terrain data exists |
| `PASS` | Every cell measured and none of the rules above applies |

**Missing terrain data can never produce `PASS`.** A missing, unreadable or non-covering DEM
gives `REVIEW_REQUIRED`. A no-go zone crossing gives `FAIL` even without a DEM, because it
is geometric. A measured violation still gives `FAIL` when coverage is incomplete.

Invalid coordinates, malformed waypoint lists, duplicate consecutive waypoints, invalid
thresholds and malformed zones raise typed errors instead of returning a status.

### Risk score

A heuristic from 0 to 100. It is not a probability and is not calibrated against real rover
performance.

```
slope_component     = min(max_slope / maximum_slope_deg, 2) / 2
roughness_component = min(mean_tri / maximum_roughness, 2) / 2      (only if a limit is given)
segment_risk        = 100 * max(slope_component, roughness_component)
segment_risk        = 100 if the segment touches a no-go zone
route_risk          = 0.5 * length_weighted_mean(segment_risk) + 0.5 * max(segment_risk)
```

A segment exactly at its limit scores 50. At twice its limit or worse it scores 100. Only
segments with measurements, or a no-go crossing, contribute. Unmeasured terrain is reported
through coverage and status and is not scored. With no basis at all the score is `null`.
`risk_score_basis` in the result says whether the score is complete or partial.

### No-go zones

Circles (`lat`, `lon`, `radius_m` or `radius_km`) and polygons (`polygon` or `vertices`, a
list of `[lat, lon]`, at least three, not self-intersecting). Tests run in a local
azimuthal equidistant plane on route chords, so zones smaller than the sample spacing are
still detected. Zones up to 500 km across are supported.

### Result fields

Status, risk score and its basis, segments analysed, coverage, maximum and mean slope,
roughness, violated segments, incomplete segments, per-segment details, configured
thresholds with the required threshold statement, dataset information, warnings and
limitations.

## Landing-site analysis

Each candidate is measured inside a circular footprint (default radius 500 m). Measurements
are elevation (mean, minimum, maximum), mean and maximum slope, mean and maximum TRI, DEM
resolution, data coverage, and the flat radius, which is the distance to the nearest cell
that is steeper than the limit or unmeasured.

| Status | Condition |
| :--- | :--- |
| `FAIL` | Maximum slope in the footprint exceeds `maximum_slope_deg` |
| `REVIEW_REQUIRED` | Mean TRI exceeds the limit (if given), or any footprint cell is unmeasured, or the DEM is too coarse to verify `min_flat_radius_m` (fewer than 4 cells inside it), or nothing could be measured |
| `PASS` | Everything measured, resolution adequate, no rule above applies |

### Comparison

Only sites whose required measurements are complete are ranked: maximum slope, mean slope,
mean TRI, and full footprint coverage. Other sites are listed as unranked with reasons. No
rank, score or status is invented for missing data.

Order is lexicographic and has no weighted score: status (`PASS`, `REVIEW_REQUIRED`,
`FAIL`), maximum slope ascending, mean TRI ascending, mean slope ascending, flat radius
descending, then site id. Warnings are added when ranked sites came from different DEM
files, resolutions or thresholds.

## Safe-region detection

A cell is safe only if it was measured and its slope is within the limit and, when given,
its own TRI is within the roughness limit. Safe cells form 4-connected regions. Regions
smaller than `min_area_m2` are dropped. Each region reports area, centroid, its best point,
the largest inscribed circle radius (accurate to about one cell), and slope, TRI and
elevation statistics.

The outcome separates `no_safe_region_in_assessed_area` from `no_terrain_data`. Unmeasured
cells are never treated as safe, and an unmeasured area is never reported as unsafe.

## Provenance

Dataset information is read from the JSON sidecar written by the secure downloader. Each
field is validated against a strict pattern and dropped with a warning if it does not match.
A DEM without a sidecar is reported as having unknown provenance and must not be presented
as a specific NASA product. The recorded SHA-256 is reported as recorded and is not
re-verified. Text that passes validation is still untrusted data.

## Known limitations

- Rover width, wheel-soil interaction, cross-track slope and direction of travel are not
  modelled. Slope is a magnitude.
- Hazards smaller than one cell are not resolved. Coarse or smoothed DEMs understate steep
  local slopes.
- TRI depends on DEM resolution. A roughness limit chosen for one DEM may not suit another.
- Vertical accuracy of the DEM is not evaluated.
- Metric cell size for slope is evaluated at each chunk centre. Results for one input are
  identical between runs and differ by about 1e-4 degrees between chunk sizes.
- Terrain statistics for a latitude/longitude box describe the smallest enclosing window in
  the DEM coordinates. Boxes crossing the antimeridian are not supported.
- Rerouting around unsafe segments is not implemented.
- The Streamlit interface does not yet display these structured results.
