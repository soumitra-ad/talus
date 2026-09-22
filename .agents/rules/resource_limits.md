---
description: Enforces CPU-based execution boundaries, raster read limits, and timeout parameters
---

# Rule: Resource Bounds & Execution Limits

1. Target environment is CPU-based execution:
   - Do NOT load entire global lunar DEM mosaics into system memory.
   - All spatial sampling and raster reads MUST use windowed reads via Rasterio.
   - Window bounding boxes must be clamped to approximately $2048 \times 2048$ cells or smaller.

2. Caching and Storage Limits:
   - Downloaded DEM tiles must be cached locally in `data/cache/`.
   - Maximum single tile download size: $100\text{MB}$.
   - Cache directory quota: $2\text{GB}$.
   - Number of traverse waypoints per evaluation: $\le 100$.

3. Timeouts:
   - Tool execution timeout: 30 seconds.
   - External HTTP connection/read timeout: 30 seconds.
   - Max retry limit for remote data requests: 3 retries with exponential backoff.
