"""System health diagnostics for TALUS deployments.

``check_system_health()`` reports on everything a deployment needs, each as ``ok`` / ``warn`` /
``fail`` with a short, safe detail string:

* required secrets / configuration (names only, never values)
* Gemini API (configuration and last observed state; a live request only when asked, because
  the Gemini free tier allows very few requests per day)
* NASA ODE search API reachability
* internet access to the NASA download host
* DEM cache contents
* cache folder writability

Every check has a short timeout and catches its own errors, so a diagnostic can never stop the
application from starting. Only the fixed NASA hosts the acquisition layer already uses are
contacted -- no user-supplied or arbitrary URL.
"""

from __future__ import annotations

import os
import socket
import time
import uuid
from typing import Any, Optional

import httpx

OK, WARN, FAIL = "ok", "warn", "fail"

#: A tiny documented ODE query (``results=c``: product count only), so the probe transfers a
#: few hundred bytes instead of the ~145 kB-per-product metadata of a real search.
_ODE_PROBE_PARAMS = {
    "query": "product",
    "target": "moon",
    "ihid": "LRO",
    "iid": "LOLA",
    "pt": "GDRDEM",
    "results": "c",
    "output": "JSON",
    "loc": "b",
    "minlat": "-90",
    "maxlat": "-89",
    "westernlon": "0",
    "easternlon": "360",
}
_DOWNLOAD_HOST = ("pds-geosciences.wustl.edu", 443)
_PROBE_MAX_BYTES = 64 * 1024
_USER_AGENT = "TALUS-terrain-agent/0.1 (research; health check)"


def _check(name: str, state: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "state": state, "detail": detail, **extra}


def check_secrets() -> dict[str, Any]:
    """Which required settings are present. Reports names only, never values."""
    from terrain_agent.config import settings

    missing: list[str] = []
    notes: list[str] = []
    if not settings.model.api_key and not settings.model.vertex_project_id:
        missing.append("GEMINI_API_KEY")
    if not settings.nasa.downloads_enabled:
        notes.append(
            "TALUS_NASA_DOWNLOADS is not \"true\""
            + (" (TALUS_ENV=production turns downloads off by default)" if settings.environment == "production" else "")
        )
    if missing or notes:
        parts = [f"Missing: {', '.join(missing)} (chat runs in deterministic mode)"] if missing else []
        parts += [f"{n}: NASA DEMs cannot be downloaded" for n in notes]
        return _check("Required secrets", WARN, "; ".join(parts), missing=missing)
    return _check("Required secrets", OK, "GEMINI_API_KEY and NASA download settings are present.", missing=[])


def check_gemini(*, live: bool = False, observed_block: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Gemini configuration plus the last state the agent observed; a live call only if asked."""
    from terrain_agent.config import settings

    if not (settings.model.api_key or settings.model.vertex_project_id):
        return _check("Gemini API", WARN, "Not configured: set GEMINI_API_KEY. Terrain tools still work.")
    if observed_block:
        return _check("Gemini API", FAIL, observed_block.get("message") or "Temporarily unavailable.",
                      category=observed_block.get("category"))
    if not live:
        return _check("Gemini API", OK, f"Configured ({settings.model.model_name}). Live check runs on request.")
    from terrain_agent.agent.agent import check_gemini_health

    report = check_gemini_health()
    if report["request"] == "SUCCESS":
        return _check("Gemini API", OK, f"Responded in {report['latency_s']} s ({report['model']}).")
    category = report.get("error_category")
    if category == "quota_daily":
        return _check("Gemini API", FAIL, "Gemini daily quota reached. Terrain tools still available.", category=category)
    return _check("Gemini API", FAIL, f"Request failed ({category}).", category=category)


def check_nasa_ode(timeout_s: float = 6.0, *, transport: Optional[httpx.BaseTransport] = None) -> dict[str, Any]:
    """Reachability of the documented NASA ODE REST endpoint, with a count-only query.
    ``transport`` exists for offline tests only."""
    from terrain_agent.acquisition.ode_provider import ODE_LIVE2_URL

    started = time.monotonic()
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_s), follow_redirects=False,
                          headers={"User-Agent": _USER_AGENT}, transport=transport) as client:
            with client.stream("GET", ODE_LIVE2_URL, params=_ODE_PROBE_PARAMS) as response:
                body = b""
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) > _PROBE_MAX_BYTES:
                        break
                status = response.status_code
    except httpx.TimeoutException:
        return _check("NASA ODE", FAIL, f"No response within {timeout_s:.0f} s.")
    except httpx.HTTPError as exc:
        return _check("NASA ODE", FAIL, f"Unreachable ({type(exc).__name__}).")
    latency = round(time.monotonic() - started, 2)
    if status != 200:
        return _check("NASA ODE", FAIL, f"Answered HTTP {status}.")
    if b"ODEResults" not in body:
        return _check("NASA ODE", WARN, "Reachable, but the response was not a recognised ODE result.")
    return _check("NASA ODE", OK, f"Reachable ({latency} s).", latency_s=latency)


def check_internet(timeout_s: float = 4.0) -> dict[str, Any]:
    """DNS + TCP reachability of the NASA PDS download host (where DEM files come from)."""
    host, port = _DOWNLOAD_HOST
    started = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            pass
    except socket.gaierror:
        return _check("Internet access", FAIL, f"DNS lookup failed for {host}.")
    except OSError as exc:
        return _check("Internet access", FAIL, f"Cannot connect to {host} ({type(exc).__name__}).")
    return _check("Internet access", OK, f"{host} reachable ({round(time.monotonic() - started, 2)} s).")


def _cached_dem_names() -> list[str]:
    from terrain_agent.config import settings

    names: list[str] = []
    for root in (settings.paths.cache_dir, settings.paths.sample_dir):
        if root.is_dir():
            names += [p.name for pattern in ("*.tif", "*.tiff") for p in root.rglob(pattern)]
    return sorted(names)


def check_dem_cache() -> dict[str, Any]:
    names = _cached_dem_names()
    if names:
        return _check("DEM cache", OK, f"{len(names)} DEM(s) cached.", count=len(names))
    return _check(
        "DEM cache", WARN,
        "Empty. The first place question downloads a NASA DEM automatically (the south-pole "
        "LOLA product is ~29 MB and can take several minutes).",
        count=0,
    )


def check_cache_writable() -> dict[str, Any]:
    from terrain_agent.config import settings

    root = settings.paths.cache_dir
    probe = root / f".talus-write-probe-{uuid.uuid4().hex[:8]}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        return _check("Cache folder writable", FAIL, f"Cannot write to the DEM cache folder ({type(exc).__name__}).")
    return _check("Cache folder writable", OK, "DEM cache folder is writable.")


def check_system_health(
    *,
    network: bool = True,
    gemini_live: bool = False,
    observed_gemini_block: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Run every check. ``network=False`` skips the NASA/internet probes (e.g. offline tests);
    ``gemini_live=True`` spends one Gemini request on a live check."""
    checks = [
        check_secrets,
        lambda: check_gemini(live=gemini_live, observed_block=observed_gemini_block),
    ]
    if network:
        checks += [check_nasa_ode, check_internet]
    checks += [check_dem_cache, check_cache_writable]
    results = []
    for run in checks:
        try:
            results.append(run())
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never break the app
            results.append(_check("Diagnostic", WARN, f"A check could not run ({type(exc).__name__})."))
    return results


def network_checks_enabled() -> bool:
    """Startup network probes can be switched off (the offline test suite does this)."""
    return os.getenv("TALUS_HEALTH_NETWORK_CHECK", "true").strip().lower() not in ("0", "false", "no", "off")


__all__ = [
    "FAIL", "OK", "WARN",
    "check_cache_writable", "check_dem_cache", "check_gemini", "check_internet",
    "check_nasa_ode", "check_secrets", "check_system_health", "network_checks_enabled",
]
