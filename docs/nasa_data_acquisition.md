# NASA Lunar DEM Acquisition (Phase 4)

> [!CAUTION]
> **Research and demonstration only.** DEMs obtained here are research inputs. Results computed
> from them are not certified for flight, landing approval, or mission use.

TALUS obtains real NASA lunar DEMs on demand and hands them to the same terrain and safety
analysis used for every other DEM. The acquisition layer lives in `src/terrain_agent/acquisition/`.
It provides data to the terrain engine. It does not replace or duplicate any analysis code.

## Data source

| Item | Value |
| :--- | :--- |
| Discovery service | NASA PDS Geosciences Node, Orbital Data Explorer (ODE) REST interface |
| Documentation used | ODE REST Interface Manual, version 2.1.6 |
| ODE address | `https://oderest.rsl.wustl.edu/live2/` (GET only) |
| File server | `https://pds-geosciences.wustl.edu/` (the only host files are downloaded from) |

Only documented ODE capabilities are used:

- `query=product` with `target=moon`, `output=JSON`, `ihid`, `iid`, `pt`, a bounding box
  (`minlat`, `maxlat`, `westernlon`, `easternlon`) with `loc=b`, `limit` and `offset`.
- Results letters `o` (ODE id), `p` (PDS identifiers), `m` (metadata) and `f` (product files).
- `odeid=id1|id2|...` to fetch the file lists of several products in one request.

ODE latitudes are planetocentric and longitudes are degrees east from 0 to 360. ODE ids can
change when a data set is rebuilt, so they are used immediately and never stored as identity.
The stable identifiers are the PDS product id and the PDS4 logical identifier.

The hosts are constants in the code. Configuration cannot point downloads elsewhere. The MIT
mirror that ODE lists in `External_url` is never used.

## Supported products

| Product type | ODE product type | Format | Notes |
| :--- | :--- | :--- | :--- |
| LOLA gridded DEM (shape map) | `LRO` / `LOLA` / `GDRDEM` | PDS4 label with raw binary image | Global, regional and polar products |
| SLDEM2015 | `LRO` / `LOLA` / `SLDEM` | PDS4 label with raw binary image | LOLA with SELENE Terrain Camera data |

Accepted data layouts, taken from the real product labels:

- 16-bit integer values in metres with scaling factor 0.5 (for example `ldem_75s_240m`).
- 32-bit floating point values in kilometres with scaling factor 1 (the `_float` products and
  SLDEM2015 tiles).
- Polar stereographic and equirectangular projections on the lunar reference sphere.

Not supported: LROC NAC and WAC digital terrain models, other ODE products, PDS3-only products,
and any label using a different height convention. These are rejected, not guessed.

### What ODE lists, and what fits

Measured from the live ODE metadata on 2026-09-21. Sizes are the data file sizes ODE reports.

| Family | Products | Sizes |
| :--- | ---: | :--- |
| LOLA `GDRDEM` | 270 | 2 MB to 4.1 GB |
| `SLDEM` | 40 | 0.2 MB, then 1.4 GB to 2.8 GB |

The default download limit is 100 MiB (about 105 MB). Under it are the coarse global products
(`ldem_4`, `ldem_8_float`, `ldem_16`, `ldem_16_float`) and the 240 m polar caps
(`ldem_75s_240m`, `ldem_75n_240m` and their float twins). The finer polar products that suit
landing analysis are just over the limit. For example `ldem_875s_20m` (87.5 to 90 degrees south
at 20 m per pixel) is about 115 MB. Raise the limit to use it:

```
TALUS_MAX_DOWNLOAD_BYTES=200000000
```

All SLDEM2015 tiles are above 1.4 GB. They are not downloadable under any sensible limit today.

### What has been verified against real NASA data

| Item | Status |
| :--- | :--- |
| Search, file lists, download, validation, conversion, cache and Phase 5 analysis | Run end to end on the real product `ldem_75s_240m` (LOLA GDRDEM, south polar stereographic, 240 m, 29 MB) |
| Label parsing for the float and SLDEM2015 layouts | The parser accepts the real labels of `ldem_16_float` and two SLDEM2015 tiles, and the sizes it computes match the sizes ODE lists. Their data files were not downloaded |
| Equirectangular georeferencing | Verified on synthetic rasters shaped like the real labels, and on the real label projection as reported by GDAL. No real cylindrical data file was analysed |
| Products above the size limit | Not downloaded |

When a finer product exists but exceeds the limit, the acquisition error lists it with its size
and resolution, so the trade-off is visible.

## Acquisition workflow

```
request (area)
  -> local cache lookup, no network
  -> ODE product search (paged)
  -> keep products that fully cover the area, finest first
  -> ODE file lists for the shortlist
  -> pick the finest product whose data file fits the size limit
  -> download the PDS4 label, then the data file, into a private work directory
  -> validate
  -> convert to a GeoTIFF of elevation in metres
  -> write provenance, commit to the cache
  -> path handed to the terrain engine
```

Use it from code:

```python
from terrain_agent.acquisition.service import build_default_service
from terrain_agent.acquisition.models import CoverageRequest

service = build_default_service()
request = CoverageRequest.from_point(-89.9, 0.0, 5000.0)   # latitude, longitude, radius in metres
acquired = service.acquire(request, product_types=["GDRDEM"])
acquired.dem_path            # analysis-ready GeoTIFF inside the cache
acquired.provenance          # structured provenance
```

Or from the agent with the `fetch_nasa_dem` tool, which takes a location and radius only. It
returns a file name such as `nasa/<cache_id>.tif`, which the existing analysis tools accept as
`dem_path`. `CoverageRequest` also accepts a bounding box (`from_bbox`) and a route
(`from_waypoints`).

One acquisition runs at a time per process.

### Download controls

- Streams to disk. Nothing is held in memory.
- https only, no user information, default port only, exact host allowlist.
- The host name is resolved before every request and every redirect hop. Every address must be
  a global unicast address. Private, loopback, link-local, carrier-grade NAT, reserved, multicast
  and IPv6-wrapped private addresses are refused. If the name cannot be resolved the request is
  refused.
- Redirects are followed manually, at most three, and every hop is re-checked.
- Separate connect and read timeouts, plus a wall-clock budget per attempt.
- Bounded retries with exponential backoff, for timeouts, connection failures and retryable
  HTTP status codes only.
- A size limit, checked from the declared length and again while streaming.
- Identity encoding only, so compressed responses and decompression bombs are not possible.
- The received size must equal the declared length and match the size ODE lists. ODE lists sizes
  in kilobytes of 1000 bytes, rounded up.
- ODE provides no checksum for product files. The SHA-256 is computed on receipt and recorded.
  It is not verified against a NASA value.

Environment variables (all optional):

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `TALUS_NASA_DOWNLOADS` | on, off when `TALUS_ENV=production` | Allow network acquisition |
| `TALUS_MAX_DOWNLOAD_BYTES` | 104857600 | Largest data file to download |
| `TALUS_MAX_CACHE_MB` | 2048 | Cache quota |
| `TALUS_CACHE_DIR` | `data/cache` | Cache location. Entries live in a `nasa` subfolder |
| `TALUS_NASA_CONNECT_TIMEOUT_SEC` | 10 | Connect timeout |
| `TALUS_NASA_READ_TIMEOUT_SEC` | 60 | Timeout for each read |
| `TALUS_NASA_TOTAL_TIMEOUT_SEC` | 900 | Time budget for one file |
| `TALUS_NASA_MAX_RETRIES` | 3 | Retries after the first attempt |

## Validation workflow

A downloaded product enters the cache only if every check passes, in this order. The first
failure stops the process and names the check.

1. **Label safety.** At most 256 KiB, valid UTF-8, no DTD or entity declaration, well-formed XML,
   a PDS4 observational product with exactly one file area and one image array.
2. **Declared file.** The data file named in the label must be the file that was downloaded. This
   stops a label from making the raster driver read another local file.
3. **Supported layout.** Data type, height unit, positive scaling factor, zero byte offset, sane
   dimensions.
4. **Exact size.** The data file must be exactly the size the label describes. This catches
   truncated and padded files.
5. **Raster open.** GDAL opens the label with the PDS4 driver. Dimensions, data type, scale and
   offset must agree with the label.
6. **CRS.** Read from the product. It must declare a reference sphere within 1% of the lunar
   radius. It is never assumed to be WGS84 or any Earth datum.
7. **Height convention.** The label offset must equal the CRS reference radius, which is the
   convention the NASA labels state. Anything else is rejected.
8. **Georeferencing.** Finite positive pixel size, valid bounds, a supported projection.
9. **Resolution.** The label pixel size must agree with the resolution ODE lists, within 2%.
10. **Coverage.** Sample points across the requested area must lie inside the raster.
11. **Elevation values.** A decimated read, the last rows, and a full-resolution window at the
    requested location must be readable. Values must be finite, within plus or minus 30 km, and
    not constant.

After conversion the output is re-opened with the terrain engine reader and compared with the
source.

### Conversion to metres

NASA labels give height as `raw value x scaling factor`, in the label unit, above a reference
sphere of radius 1737.4 km. The terrain engine reads values directly and ignores raster scale
metadata, so the product is converted once:

```
elevation (m) = raw value x scaling factor x unit factor   (unit factor 1 for metres, 1000 for km)
```

Without this the engine would read half-metres as metres. The output is a tiled, compressed
float32 GeoTIFF with the source CRS and geotransform. Invalid values are written as nodata and
counted. Elevations are heights above a sphere, not above a geoid.

### Terrain engine changes needed for real data

Real cylindrical products are equirectangular grids in metres, where the east-west cell shrinks
with the cosine of latitude. The georeferencing layer now measures the east-west and north-south
scale separately for equirectangular grids and refuses projections it cannot interpret. Cylindrical
products cannot be used within about 3 degrees of a pole. Use a polar product there.

## Cache

```
<cache_dir>/nasa/<cache_id>.tif          elevation in metres
<cache_dir>/nasa/<cache_id>.json         provenance
<cache_dir>/nasa/<cache_id>.label.xml    the label as received, for audit only
<cache_dir>/nasa/.tmp/acq-*/             work directories
```

- `cache_id` is the product id plus a 12 character digest of provider, product id, data URL and
  conversion version. No caller supplied text becomes a path. Every resolved path must lie
  directly inside the cache directory, which also stops symbolic link escapes.
- A quota is enforced with least-recently-used eviction. A product larger than the whole quota
  is refused.
- Entries are verified on use by recomputing the recorded SHA-256. A damaged entry is removed and
  downloaded again.
- Provenance is written last, so an entry without it is incomplete and ignored.
- A cached DEM that covers the request is used without any network access, even if a finer
  product exists. Pass `max_pixel_size_m` (or `refresh=True`) to require better.
- Work directories left by an interrupted run are removed after 24 hours.

## Provenance

Each cached DEM carries a structured record in the `nasa_provenance` block of its JSON file:
provider, mission, instrument, product type and identifier, PDS4 logical identifier, data set,
version, discovery endpoint and the queries used, request area, acquisition time, cache id,
source URLs, received and expected sizes, checksums and how they were obtained, CRS, projection,
native and metric pixel size, dimensions, bounds, the height convention, conversion details
(elevation range, masked cell count), the validation checks that passed, and warnings.

The analysis layer reads a validated subset into its `dataset` field, so every rover, landing and
safe-region result names the product it used.

Provenance is display data. Every field is checked against a strict pattern when it is read and
dropped with a warning if it does not match. Nothing in it can change an analysis result or a
safety status. Statuses come only from measured terrain and configured thresholds.

## Security summary

Everything from NASA is treated as untrusted: ODE JSON, file lists, labels and data files.

| Threat | Control |
| :--- | :--- |
| Arbitrary URL fetching | No URL parameter anywhere. Fixed API host, single file host, exact allowlist |
| SSRF | Address check before every request and hop, fail closed, redirects re-checked |
| Path traversal, malicious names | Names come from validated fields, cache ids are derived and pattern checked, resolved paths must stay inside the cache |
| Label pointing at another file | The declared file name must equal the downloaded file |
| Oversized downloads and responses | Declared and streamed size limits, response size limits, result caps |
| Decompression and resource exhaustion | Identity encoding only, no compressed formats accepted, bounded reads, bounded windows |
| Unexpected file types | Only PDS4 labels with raw binary images of two data types |
| Malformed raster metadata | Label and CRS validation, size and dimension checks, read checks |
| XML attacks | Labels with a DTD or entity are rejected before parsing |
| Hostile text in metadata | Strict patterns, short text, never stored in free form, cannot affect results |
| Secrets | None are used. The acquisition code reads no credentials and sends none |

Known residual risk: the HTTP client resolves the host name again when it connects. Exploiting
that gap would require control of DNS for an allowlisted NASA host name.

## Limitations

- Only LOLA `GDRDEM` and `SLDEM` products in PDS4 format. No LROC DTMs.
- Whole files are downloaded. Products above the size limit cannot be used, and that includes the
  finest polar DEMs at the default limit. Windowed remote reads are not implemented.
- Slow server. The NASA data server delivered about 230 kB per second in testing, so a 29 MB
  product took about two minutes.
- Elevations are above a reference sphere, not a geoid, and their vertical accuracy is not
  evaluated here.
- No checksum is available from NASA to verify against.
- Polar products are square rasters. Their corners extend beyond the latitude bounds ODE lists, so
  coverage is judged on the raster itself.
- One acquisition at a time per process.

## Optional live NASA test

`tests/live/test_nasa_live.py` is the only test that uses the network. It is skipped unless
`TALUS_RUN_LIVE_NASA=1`, so the normal suite stays offline and deterministic.

It queries the documented IIPT list, searches ODE for LOLA products covering the Shackleton
crater area, downloads the product that fits the limit, validates and converts it, then runs the
Phase 5 analyses on the real DEM. It asserts physical plausibility and provenance structure, not
specific terrain values.

PowerShell:

```
$env:TALUS_RUN_LIVE_NASA = "1"; python -m pytest tests/live -m live_nasa -s -p no:cacheprovider
```

bash:

```
TALUS_RUN_LIVE_NASA=1 python -m pytest tests/live -m live_nasa -s -p no:cacheprovider
```

Set `TALUS_LIVE_CACHE_DIR` to keep the downloaded product between runs. Allow several minutes.
