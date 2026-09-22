# TALUS Agent Operational Directives

You are operating within the codebase of **Terrain Analysis for Landing and Uncrewed Systems (TALUS)**.

You MUST strictly comply with the following architectural and security invariants:

1. **Deterministic Computation Boundary**:
   - The LLM must NEVER calculate terrain values itself.
   - All numerical elevations, slopes, roughness metrics, path segments, and risk scores MUST originate from deterministic Python functions in `src/terrain_agent/`.
   - If a numerical calculation is required, you must invoke the corresponding deterministic tool.

2. **Mandatory Research & Demo Disclaimer**:
   - TALUS is strictly a research and demonstration system.
   - It must NEVER claim to provide certified flight safety, operational landing approval, autonomous spacecraft control, or guaranteed rover safety.

3. **Data Source Boundaries**:
   - Primary data discovery: NASA Lunar Orbital Data Explorer (ODE) REST API.
   - Use official endpoint specifications.
   - Restrict all remote data acquisition to official NASA, PDS Geosciences, and LROC domains (`ode.rsl.wustl.edu`, `pds-geosciences.wustl.edu`, `wac.lroc.asu.edu`, `lroc.sese.asu.edu`).
   - Never accept, parse, or download arbitrary user-supplied URLs.

4. **Safety Evaluation Framing**:
   - Rover safety thresholds (e.g. `maximum_slope = 15.0`) must always be framed as configured mission parameters, never as universal scientific safety limits.
   - The output must explicitly state: `"Configured analysis threshold: {X}°"`.
   - Allowed statuses: `PASS`, `REVIEW_REQUIRED`, `FAIL`.

5. **Resource Limits**:
   - Design for CPU execution.
   - Never load global lunar mosaics into memory.
   - Target raster read windows must not exceed $2048 \times 2048$ cells.
   - Enforce timeouts and bounded arrays.

6. **Zero Secrets Requirement**:
   - The repository must run out of the box with zero mandatory API keys or credentials.
