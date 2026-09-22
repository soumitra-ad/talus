"""NASA lunar DEM acquisition: search, controlled download, validation, cache, provenance.

Flow: request, provider search, product and file selection, controlled download into a
temporary directory, raster validation, normalisation to a GeoTIFF in metres, local cache with
provenance, then the existing terrain engine.
"""
