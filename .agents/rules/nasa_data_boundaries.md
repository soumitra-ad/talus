---
description: Enforces official NASA/PDS domain boundaries and prohibited remote inputs
---

# Rule: NASA Data Acquisition Boundaries

1. Primary lunar data discovery MUST use the official NASA Lunar Orbital Data Explorer (ODE) REST API endpoint specifications.
2. Remote data discovery and acquisition must be restricted to approved official domains:
   - `ode.rsl.wustl.edu`
   - `pds-geosciences.wustl.edu`
   - `wac.lroc.asu.edu`
   - `lroc.sese.asu.edu`
3. Prohibited operations:
   - Never download, parse, or request arbitrary user-supplied URLs.
   - Never invent undocumented request parameters for the ODE API.
   - Never query private IP addresses, loopback interfaces, or cloud metadata endpoints.
4. Fallback:
   - When external network access is unavailable, the application must gracefully use bundled local sample DEMs and mock data catalog entries.
