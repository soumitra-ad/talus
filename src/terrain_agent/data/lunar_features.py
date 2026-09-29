"""Deterministic resolution of named lunar features to coordinates and an analysis area.

Feature coordinates come only from the curated gazetteer shipped with the repository
(``data/metadata/lunar_features.json``). The language model never supplies coordinates for a
named feature: it calls this resolver, and an unknown name is reported as unknown rather than
guessed. The analysis bounding box is derived from the feature centre and radius by the same
pole-aware ``CoverageRequest`` geometry the NASA acquisition layer uses.

Research/Demo Disclaimer
------------------------
TALUS is a research and demonstration system. Gazetteer coordinates are approximate feature
centres for analysis-area selection, not survey-grade positions.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from terrain_agent.terrain.coordinates import validate_analysis_radius, validate_lunar_coordinate

GAZETTEER_SOURCE = "TALUS curated lunar feature gazetteer (data/metadata/lunar_features.json)"
_GAZETTEER_NAME = "lunar_features.json"
_MAX_GAZETTEER_BYTES = 256 * 1024
_MAX_QUERY_CHARS = 100
#: Radius used when a feature has no recorded diameter and the caller gives none.
DEFAULT_FEATURE_RADIUS_KM = 5.0
#: Generic words ignored when matching, so "Shackleton" and "shackleton crater" both match
#: "Shackleton Crater".
_GENERIC_WORDS = frozenset({"the", "crater", "mountain", "mount", "mons", "plateau", "region", "area"})
_WORD_RE = re.compile(r"[a-z0-9]+")


def _key(name: str) -> str:
    words = [w for w in _WORD_RE.findall(name.lower()) if w not in _GENERIC_WORDS]
    return " ".join(words)


def _gazetteer_path() -> Path:
    from terrain_agent.config import settings

    configured = settings.paths.metadata_dir / _GAZETTEER_NAME
    if configured.is_file():
        return configured
    # Fall back to the copy in the repository, independent of the working directory.
    return Path(__file__).resolve().parents[3] / "data" / "metadata" / _GAZETTEER_NAME


@lru_cache(maxsize=1)
def load_features() -> tuple[dict[str, Any], ...]:
    """Load and validate the gazetteer. Entries that fail validation are skipped."""
    path = _gazetteer_path()
    if not path.is_file() or path.stat().st_size > _MAX_GAZETTEER_BYTES:
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    features: list[dict[str, Any]] = []
    for entry in raw.get("features", []) if isinstance(raw, dict) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            continue
        try:
            lat, lon = validate_lunar_coordinate(entry.get("center_lat"), entry.get("center_lon"))
        except Exception:  # noqa: BLE001 - a malformed entry is skipped, never guessed
            continue
        diameter = entry.get("diameter_km")
        features.append(
            {
                "name": entry["name"][:80],
                "center_lat": lat,
                "center_lon": lon,
                "diameter_km": float(diameter) if isinstance(diameter, (int, float)) and diameter > 0 else None,
                "region": str(entry.get("region", ""))[:60] or None,
            }
        )
    return tuple(features)


def known_feature_names() -> list[str]:
    return [f["name"] for f in load_features()]


def find_feature(name: Any) -> Optional[dict[str, Any]]:
    """Exact (case- and generic-word-insensitive) gazetteer match, or ``None``."""
    if not isinstance(name, str) or not name.strip() or len(name) > _MAX_QUERY_CHARS:
        return None
    wanted = _key(name)
    if not wanted:
        return None
    for feature in load_features():
        if _key(feature["name"]) == wanted:
            return dict(feature)
    return None


def analysis_bbox(lat: float, lon: float, radius_km: float) -> dict[str, Any]:
    """Latitude/longitude box enclosing a circle, in the -180..180 convention the terrain
    statistics tools take. A circle containing a pole spans every longitude."""
    from terrain_agent.acquisition.models import CoverageRequest

    radius_km = validate_analysis_radius(radius_km)
    request = CoverageRequest.from_point(lat, lon, radius_km * 1000.0)
    if request.full_longitude:
        min_lon, max_lon = -180.0, 180.0
    else:
        min_lon = ((request.west_lon + 180.0) % 360.0) - 180.0
        max_lon = ((request.east_lon + 180.0) % 360.0) - 180.0
        if min_lon >= max_lon:  # crosses the antimeridian, which the stats tools reject
            min_lon, max_lon = -180.0, 180.0
    return {
        "min_lat": round(request.min_lat, 6),
        "max_lat": round(request.max_lat, 6),
        "min_lon": round(min_lon, 6),
        "max_lon": round(max_lon, 6),
        "radius_km": round(radius_km, 3),
        "spans_all_longitudes": request.full_longitude,
    }


def resolve_feature(name: Any, radius_km: Any = None) -> dict[str, Any]:
    """Resolve a feature name to its centre and a deterministic analysis area.

    Returns ``{"status": "ok", "feature": ..., "analysis_area": ..., "source": ...}`` or
    ``{"status": "not_found", "known_features": [...]}``. Never guesses coordinates.
    """
    feature = find_feature(name)
    if feature is None:
        return {
            "status": "not_found",
            "error": "That feature is not in the TALUS gazetteer, so its coordinates are unknown.",
            "known_features": known_feature_names(),
            "source": GAZETTEER_SOURCE,
        }
    if radius_km is None:
        radius_km = feature["diameter_km"] / 2.0 if feature["diameter_km"] else DEFAULT_FEATURE_RADIUS_KM
    if isinstance(radius_km, bool) or not isinstance(radius_km, (int, float)):
        raise ValueError("radius_km must be a number")
    return {
        "status": "ok",
        "feature": feature,
        "analysis_area": analysis_bbox(feature["center_lat"], feature["center_lon"], float(radius_km)),
        "source": GAZETTEER_SOURCE,
    }


__all__ = [
    "GAZETTEER_SOURCE",
    "analysis_bbox",
    "find_feature",
    "known_feature_names",
    "load_features",
    "resolve_feature",
]
