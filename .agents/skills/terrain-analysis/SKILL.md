---
name: terrain-analysis
description: Instructions for selecting terrain tools, validating lunar coordinates, loading bounded DEM windows, calculating deterministic metrics, interpreting results, and reporting evidence.
---

# Skill: Terrain Analysis Operations

This skill guides the agent in performing deterministic terrain analysis in TALUS.

## 1. Tool Selection Strategy

Select the appropriate tool based on the user's intent:

| User Intent | Appropriate Tool | Required Parameters |
| :--- | :--- | :--- |
| Elevation or slope statistics for an area | `get_dem_stats` | `center_lat`, `center_lon`, `radius_km` |
| Finding safe landing zones | `find_safe_regions` | `center_lat`, `center_lon`, `search_radius_km`, `max_slope_deg` |
| Rover path or traverse assessment | `check_rover_safety` | `waypoints`, `max_slope_threshold_deg` |
| Querying dataset details or limitations | `dataset_information` | `dataset_name` or `product_id` |
| Discovering available DEMs | `search_lunar_dem` | `min_lat`, `max_lat`, `min_lon`, `max_lon` |
| Downloading and caching a DEM | `download_dem_tile` | `product_id` |

## 2. Coordinate Validation Protocol

Before invoking any spatial tool, strictly validate that coordinates fall within physical lunar domains:
- **Latitude**: Must be between $-90.0^\circ$ and $+90.0^\circ$.
- **Longitude**: Either $[-180.0^\circ, +180.0^\circ]$ or $[0.0^\circ, 360.0^\circ]$.
- **Polar Regions**: If latitude $\le -80^\circ$ or $\ge +80^\circ$, inform the user that South/North Polar Stereographic projection is utilized for accurate planar meter distances.

## 3. Loading DEM Windows

Never attempt to load global DEM datasets into system RAM.
- Use windowed raster reads via Rasterio.
- Limit the window size to $\le 2048 \times 2048$ cells.
- Calculate bounding box in projected meters and clip using `from_bounds`.

```python
import rasterio
from rasterio.windows import from_bounds

def load_bounded_dem_window(raster_path: str, bounds: tuple[float, float, float, float]):
    min_x, min_y, max_x, max_y = bounds
    with rasterio.open(raster_path) as src:
        win = from_bounds(min_x, min_y, max_x, max_y, transform=src.transform)
        # Enforce maximum 2048x2048 cells limit
        w = min(int(win.width), 2048)
        h = min(int(win.height), 2048)
        clamped_win = rasterio.windows.Window(win.col_off, win.row_off, w, h)
        elevation = src.read(1, window=clamped_win)
        transform = rasterio.windows.transform(clamped_win, src.transform)
        return elevation, transform
```

## 4. Deterministic Terrain Statistics Calculation

1. **Elevation Statistics**: Compute `min`, `max`, `mean`, and `std` on valid non-nodata cells.
2. **Slope Calculation**: Compute slope gradients using Horn's $3 \times 3$ finite-difference stencil on projected metric grid cells ($dx, dy$ in meters).
3. **Roughness & TRI**: Compute Terrain Ruggedness Index (TRI) as the mean absolute difference from center to 8 neighboring cells.
4. **Traverse Segmentation**: Sample elevation and slope along great-circle waypoint tracks at nominal 10-meter intervals.

## 5. Result Interpretation & Evidence Reporting

Every agent response conveying terrain analysis results must:
1. **Report Numerical Evidence**:
   - Total area or traverse segments analyzed
   - Mean and maximum observed slope
   - Configured mission threshold (e.g. `"Configured analysis threshold: 15.0°"`)
   - Explicit status: `PASS`, `REVIEW_REQUIRED`, or `FAIL`
2. **Cite Data Provenance**:
   - Underlying dataset name (e.g., LRO LOLA, SLDEM2015, LROC NAC)
   - Spatial resolution (e.g., 60 m/pixel)
3. **State Limitations**:
   - Sub-pixel boulder hazards not resolved by dataset
   - Interpolation artifacts or shadow voids in polar craters
4. **Disclose Research Status**:
   - Reiterate that TALUS provides research demonstration data and not certified flight safety.
