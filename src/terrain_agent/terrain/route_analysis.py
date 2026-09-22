"""Measure terrain along a densified route.

Method
------
1. Route samples (see :func:`terrain_agent.terrain.geometry.densify_route`) are converted to
   raster cells using the DEM own georeferencing.
2. Each distinct cell touched by a segment is counted once for that segment, so a coarse DEM
   is not over-weighted by many samples falling in one cell.
3. Slope (Horn), terrain ruggedness (TRI) and elevation are read at those cells from windowed
   grids. Samples are processed in chunks so no read exceeds the raster window limit.
4. A cell counts as measured only if its slope is valid, which requires a complete 3x3
   neighbourhood of valid elevations. Cells outside the raster, on nodata, or on the raster
   edge are unmeasured. Unmeasured cells are reported, never filled in.

The measurement is along the route centreline only. Rover width and cross-track terrain
are not evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from terrain_agent.terrain.geometry import RouteSamples
from terrain_agent.terrain.georef import DemGeoContext
from terrain_agent.terrain.resource_safety import MAX_RASTER_DIM
from terrain_agent.terrain.terrain_grids import load_terrain_grids

#: Extra cells read around each chunk so Horn slope and TRI are valid at the chunk edge.
PAD_CELLS = 2

#: Default largest chunk extent in cells. Well inside the 2048 cell read limit.
DEFAULT_MAX_CHUNK_CELLS = 1024


@dataclass(frozen=True)
class SegmentTerrainStats:
    """Terrain measured along one route segment."""

    segment_index: int
    cells_total: int
    cells_valid: int
    max_slope_deg: Optional[float]
    mean_slope_deg: Optional[float]
    mean_tri_m: Optional[float]
    max_tri_m: Optional[float]
    min_elevation_m: Optional[float]
    max_elevation_m: Optional[float]
    elevation_change_m: Optional[float]

    @property
    def coverage_fraction(self) -> float:
        return self.cells_valid / self.cells_total if self.cells_total else 0.0


@dataclass(frozen=True)
class RouteTerrainMeasurement:
    """Result of measuring a route against a DEM."""

    segments: tuple[SegmentTerrainStats, ...]
    pixel_size_m: tuple[float, float]
    chunks_read: int


def _chunk_ranges(rows: list[int], cols: list[int], max_extent: int) -> list[tuple[int, int]]:
    """Split ordered cell indices into runs whose bounding box stays within *max_extent*."""
    ranges: list[tuple[int, int]] = []
    start = 0
    rmin = rmax = rows[0]
    cmin = cmax = cols[0]
    for i in range(1, len(rows)):
        nrmin, nrmax = min(rmin, rows[i]), max(rmax, rows[i])
        ncmin, ncmax = min(cmin, cols[i]), max(cmax, cols[i])
        if nrmax - nrmin + 1 > max_extent or ncmax - ncmin + 1 > max_extent:
            ranges.append((start, i))
            start = i
            rmin = rmax = rows[i]
            cmin = cmax = cols[i]
        else:
            rmin, rmax, cmin, cmax = nrmin, nrmax, ncmin, ncmax
    ranges.append((start, len(rows)))
    return ranges


def measure_route_terrain(
    ctx: DemGeoContext,
    samples: RouteSamples,
    n_segments: int,
    *,
    max_chunk_cells: int = DEFAULT_MAX_CHUNK_CELLS,
) -> RouteTerrainMeasurement:
    """Measure slope, roughness and elevation along a sampled route.

    Raises
    ------
    UnsupportedCRSError
        If metric cell size cannot be determined (for example a geographic DEM near a pole).
    MalformedRasterError, OversizedRequestError
        On unreadable rasters or window-limit violations.
    """
    if not (8 <= max_chunk_cells <= MAX_RASTER_DIM - 2 * PAD_CELLS):
        raise ValueError(
            f"max_chunk_cells must be between 8 and {MAX_RASTER_DIM - 2 * PAD_CELLS}."
        )

    n = samples.count
    row_f, col_f, finite = ctx.latlon_to_rowcol_float(samples.lat, samples.lon)
    outside_marker = -10**9 - np.arange(n, dtype=np.int64)
    rows = np.where(
        finite, np.floor(np.clip(np.nan_to_num(row_f), -1e9, 1e9)).astype(np.int64), outside_marker
    )
    cols = np.where(
        finite, np.floor(np.clip(np.nan_to_num(col_f), -1e9, 1e9)).astype(np.int64), outside_marker
    )
    inside = finite & (rows >= 0) & (rows < ctx.height) & (cols >= 0) & (cols < ctx.width)

    slope = np.full(n, np.nan, dtype=np.float64)
    tri = np.full(n, np.nan, dtype=np.float64)
    elev = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    mid = n // 2
    pixel_size = ctx.pixel_size_m(float(samples.lat[mid]), float(samples.lon[mid]))

    chunks_read = 0
    inside_idx = np.flatnonzero(inside)
    if inside_idx.size:
        ranges = _chunk_ranges(
            rows[inside_idx].tolist(), cols[inside_idx].tolist(), max_chunk_cells
        )
        for start, end in ranges:
            ids = inside_idx[start:end]
            r_min, r_max = int(rows[ids].min()), int(rows[ids].max())
            c_min, c_max = int(cols[ids].min()), int(cols[ids].max())
            row0, col0 = r_min - PAD_CELLS, c_min - PAD_CELLS
            height = r_max - r_min + 1 + 2 * PAD_CELLS
            width = c_max - c_min + 1 + 2 * PAD_CELLS
            grids = load_terrain_grids(
                ctx,
                row0,
                col0,
                height,
                width,
                ref_lat=float(np.mean(samples.lat[ids])),
                ref_lon=float(samples.lon[ids[len(ids) // 2]]),
            )
            rr = rows[ids] - row0
            cc = cols[ids] - col0
            valid[ids] = grids.slope_valid[rr, cc]
            slope[ids] = grids.slope_deg[rr, cc]
            tri[ids] = grids.tri[rr, cc]
            elev[ids] = grids.elevation[rr, cc]
            chunks_read += 1

    # One record per distinct (segment, cell) pair, in a deterministic order.
    keys = np.stack([samples.segment, rows, cols], axis=1)
    _unique, first = np.unique(keys, axis=0, return_index=True)
    seg_u = samples.segment[first]
    valid_u = valid[first]
    slope_u = slope[first]
    tri_u = tri[first]
    elev_u = elev[first]

    stats: list[SegmentTerrainStats] = []
    for index in range(n_segments):
        in_segment = seg_u == index
        measured = in_segment & valid_u
        cells_total = int(in_segment.sum())
        cells_valid = int(measured.sum())

        seg_samples = np.flatnonzero(samples.segment == index)
        first_elev = elev[seg_samples[0]] if seg_samples.size else np.nan
        last_elev = elev[seg_samples[-1]] if seg_samples.size else np.nan
        change: Optional[float] = None
        if valid[seg_samples[0]] and valid[seg_samples[-1]]:
            change = float(last_elev - first_elev)

        if cells_valid:
            stats.append(
                SegmentTerrainStats(
                    segment_index=index,
                    cells_total=cells_total,
                    cells_valid=cells_valid,
                    max_slope_deg=float(np.max(slope_u[measured])),
                    mean_slope_deg=float(np.mean(slope_u[measured])),
                    mean_tri_m=float(np.mean(tri_u[measured])),
                    max_tri_m=float(np.max(tri_u[measured])),
                    min_elevation_m=float(np.min(elev_u[measured])),
                    max_elevation_m=float(np.max(elev_u[measured])),
                    elevation_change_m=change,
                )
            )
        else:
            stats.append(
                SegmentTerrainStats(
                    segment_index=index,
                    cells_total=cells_total,
                    cells_valid=0,
                    max_slope_deg=None,
                    mean_slope_deg=None,
                    mean_tri_m=None,
                    max_tri_m=None,
                    min_elevation_m=None,
                    max_elevation_m=None,
                    elevation_change_m=None,
                )
            )

    return RouteTerrainMeasurement(
        segments=tuple(stats), pixel_size_m=pixel_size, chunks_read=chunks_read
    )
