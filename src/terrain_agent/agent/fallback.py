"""Deterministic analysis when Gemini is unavailable (no key, quota exhausted, outage).

Without a language model there is no free-form intent interpretation, so this handles the
common case deterministically: a question that names a feature in the TALUS gazetteer gets
the standard pipeline -- resolve the feature, fetch (or reuse) a NASA DEM, and compute
elevation, slope and roughness statistics -- plus a safe-region search when the question
mentions rover or landing safety. Every number in the reply is formatted from a tool result;
nothing is estimated. Questions that name no known feature get an honest explanation.

Research/Demo Disclaimer
------------------------
TALUS is a research and demonstration system. It must NEVER claim to provide certified flight
safety, operational landing approval, autonomous spacecraft control, or guaranteed rover safety.
"""

from __future__ import annotations

import re
from typing import Any, Callable

_SAFETY_WORDS = re.compile(r"\b(rover|safe|safety|traverse|landing|land|hazard|slope limit)\b", re.I)
_MANDATORY_DISCLAIMER = (
    "TALUS is a research and demonstration system. Results are not certified flight safety, "
    "operational landing approval, autonomous spacecraft control, or guaranteed rover safety."
)


def _num(value: Any, unit: str, decimals: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{decimals}f}{unit}"


def deterministic_analysis(
    user_message: str,
    *,
    max_slope_deg: float,
    notice: str,
    dem_cache_dir: str | None = None,
    emit: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the fixed resolve -> fetch -> statistics pipeline for a named feature.

    Returns the same shape as ``TALUSAgent.chat``: ``status`` is ``"fallback"`` when an analysis
    was attempted, or ``"model_error"`` when the question names no known feature.
    """
    from terrain_agent.agent.agent import dispatch_tool_call, summarize_tool_call
    from terrain_agent.data.lunar_features import find_feature_in_text, known_feature_names

    emit = emit or (lambda kind, info: None)
    feature = find_feature_in_text(user_message)
    if feature is None:
        return {
            "status": "model_error",
            "notice": notice,
            "text": (
                f"{notice} Without the language model TALUS can only analyse a named feature "
                f"from its gazetteer ({', '.join(known_feature_names())}), or you can use the "
                "Direct Structured Analysis tools below."
            ),
            "tool_calls": [],
            "is_demo": False,
        }

    tool_calls: list[dict[str, Any]] = []

    def run(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        emit("tool_start", {"tool": tool})
        result = dispatch_tool_call(tool, args, dem_cache_dir=dem_cache_dir)
        summary = summarize_tool_call(tool, args, result)
        emit("tool_end", {"tool": tool, "status": result.get("status"), "summary": summary})
        tool_calls.append({"tool": tool, "result_status": result.get("status"), "summary": summary, "args": args, "result": result})
        return result

    resolved = run("resolve_lunar_feature", {"name": feature["name"]})
    lines: list[str] = []
    if resolved.get("status") != "ok":
        lines.append("The feature could not be resolved, so no terrain analysis was run.")
        return _result(lines, tool_calls, notice)
    area = resolved["analysis_area"]
    fetched = run(
        "fetch_nasa_dem",
        {"lat": feature["center_lat"], "lon": feature["center_lon"], "radius_km": area["radius_km"]},
    )
    if fetched.get("status") != "ok":
        lines.append(
            f"Deterministic analysis of **{feature['name']}** could not be completed: no NASA DEM "
            f"could be obtained ({fetched.get('error') or fetched.get('status')}). No terrain "
            "values were produced."
        )
        return _result(lines, tool_calls, notice)

    box = {"dem_path": fetched["dem_path"], **{k: area[k] for k in ("min_lat", "max_lat", "min_lon", "max_lon")}}
    elevation = run("get_elevation_stats", box)
    slope = run("get_slope_stats", box)
    roughness = run("get_roughness_stats", box)
    prov = fetched.get("provenance") or {}
    lines.append(
        f"Deterministic analysis of **{feature['name']}** (radius {_num(area['radius_km'], ' km', 1)}) "
        f"from NASA {prov.get('mission')} / {prov.get('instrument')} product `{prov.get('product_id')}`:"
    )
    e = (elevation.get("analysis") or {}).get("elevation") or {}
    s = (slope.get("analysis") or {}).get("slope") or {}
    r = (roughness.get("analysis") or {}).get("roughness") or {}
    if e:
        lines.append(f"- Mean elevation: **{_num(e.get('mean_m'), ' m')}** (min {_num(e.get('min_m'), ' m')}, max {_num(e.get('max_m'), ' m')})")
    if s:
        lines.append(f"- Mean slope: **{_num(s.get('mean_slope_deg'), '°')}** (max {_num(s.get('max_slope_deg'), '°')})")
    if r:
        lines.append(f"- Mean roughness (TRI): **{_num(r.get('mean_tri_m'), ' m')}**")
    if not (e or s or r):
        lines.append("- No terrain statistics could be computed for this area.")
    lines.append(f"- Cell size: {_num(elevation.get('resolution_m'), ' m', 1)}; coverage "
                 f"{_num(100 * float((elevation.get('analysis') or {}).get('coverage_fraction') or 0), '%', 0)}")

    if _SAFETY_WORDS.search(user_message):
        regions = run(
            "find_safe_regions",
            {
                "center_lat": feature["center_lat"], "center_lon": feature["center_lon"],
                "radius_m": area["radius_km"] * 1000.0, "dem_path": fetched["dem_path"],
                "max_slope_deg": max_slope_deg,
            },
        )
        analysis = regions.get("analysis") or {}
        lines.append(f"- Configured analysis threshold: {max_slope_deg:g}°. This is NOT a certified safety limit.")
        lines.append(f"- Safe-region search: {summarize_tool_call('find_safe_regions', {}, regions)}")
        if analysis.get("safe_fraction_of_assessed") is not None:
            lines.append(f"- Share of assessed cells within the threshold: {_num(100 * analysis['safe_fraction_of_assessed'], '%', 1)}")
    return _result(lines, tool_calls, notice)


def _result(lines: list[str], tool_calls: list[dict[str, Any]], notice: str) -> dict[str, Any]:
    lines += ["", "_Generated without the language model: every value above is a direct tool result._", "", _MANDATORY_DISCLAIMER]
    return {"status": "fallback", "notice": notice, "text": "\n".join(lines), "tool_calls": tool_calls, "is_demo": False}


__all__ = ["deterministic_analysis"]
