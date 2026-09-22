---
name: nasa-data
description: Procedures for discovering lunar DEM products using NASA ODE REST API, downloading from allowlisted hosts, streaming with size caps, calculating SHA-256 checksums, and caching.
---

# Skill: NASA Lunar Data Acquisition & Caching

This skill outlines the standard operating procedures for querying, downloading, caching, and handling lunar elevation datasets from official NASA archives.

## 1. Product Discovery via NASA Lunar ODE

Use only what the ODE REST Interface Manual (version 2.1.6, https://oderest.rsl.wustl.edu/) documents.
The implementation is `src/terrain_agent/acquisition/ode_provider.py`.

- **Address**: `https://oderest.rsl.wustl.edu/live2` (GET only).
- **Product search**: `query=product` with `target=moon`, `output=JSON`, `ihid=LRO`, `iid=LOLA`,
  `pt=GDRDEM` or `pt=SLDEM`, `results=opm`, a bounding box (`minlat`, `maxlat`, `westernlon`,
  `easternlon`) with `loc=b`, and `limit`.
- **Product files**: `query=product&results=opf&odeid=id1|id2|...`.
- **Valid product types**: `query=iipt&odemetadb=moon`.
- Latitudes are planetocentric. Longitudes are degrees east from 0 to 360.
- Product file URLs come only from the `Product_files` list. Never use `External_url`, which can
  point at a non-NASA mirror.
- Do not invent parameters or endpoints. The earlier `ode.rsl.wustl.edu/moon/service/search`
  endpoint and `pt=DEM` product type are not in the manual.

See `docs/nasa_data_acquisition.md` for the full workflow, supported products and limits.

## 2. Host Allowlisting & URL Validation

Before initiating any remote network connection, ensure the destination URL's hostname exactly matches the allowlist:
- `ode.rsl.wustl.edu`
- `pds-geosciences.wustl.edu`
- `wac.lroc.asu.edu`
- `lroc.sese.asu.edu`

```python
from urllib.parse import urlparse

ALLOWED_HOSTS = {
    "ode.rsl.wustl.edu",
    "pds-geosciences.wustl.edu",
    "wac.lroc.asu.edu",
    "lroc.sese.asu.edu",
}

def validate_remote_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Insecure protocol rejected: {parsed.scheme}")
    if parsed.hostname not in ALLOWED_HOSTS:
        raise PermissionError(f"Host '{parsed.hostname}' is not in the NASA/LROC allowlist.")
```

## 3. Streaming Downloads & File-Size Limits

- Never buffer entire large files into memory.
- Enforce a strict file-size limit ($\le 100\text{MB}$) by inspecting `Content-Length` headers.
- Abort immediately if the content exceeds the threshold.
- Stream chunks into a temporary file (`.tmp`) before finalizing.

```python
import httpx

MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

async def stream_download_product(url: str, dest_path: Path):
    validate_remote_url(url)
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            content_length = int(response.headers.get("content-length", 0))
            if content_length > MAX_DOWNLOAD_BYTES:
                raise ValueError(f"Remote file size ({content_length} bytes) exceeds limit ({MAX_DOWNLOAD_BYTES} bytes).")
            
            bytes_written = 0
            temp_path = dest_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    bytes_written += len(chunk)
                    if bytes_written > MAX_DOWNLOAD_BYTES:
                        raise ValueError("Download exceeded maximum allowed file size.")
                    f.write(chunk)
            temp_path.rename(dest_path)
```

## 4. Checksums & Tile Caching

- Compute and store the SHA-256 checksum for all cached tiles to ensure file integrity:
  ```python
  import hashlib

  def compute_sha256(file_path: Path) -> str:
      hasher = hashlib.sha256()
      with open(file_path, "rb") as f:
          for chunk in iter(lambda: f.read(65536), b""):
              hasher.update(chunk)
      return hasher.hexdigest()
  ```
- Save metadata sidecars (`.json`) alongside downloaded rasters preserving product ID, source URL, spatial bounds, resolution, and SHA-256 hash.

## 5. Bounded Fetch Rule

- **NEVER download an entire global dataset for a localized query**:
  - If a user asks for a 5 km region around Shackleton Crater, search for regional or tiled polar DEMs rather than pulling a multi-gigabyte global LOLA grid.
  - When regional high-resolution DEMs are not available, use pre-bundled sample DEMs or local window extracts.
