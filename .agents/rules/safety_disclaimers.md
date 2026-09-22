---
description: Enforces the mandatory research and demo disclaimer across all responses and user interfaces
---

# Rule: Safety Disclaimers & Operational Framing

1. TALUS is strictly a research/demo system.
2. The agent and UI must NEVER claim to provide:
   - Certified flight safety
   - Operational landing approval
   - Autonomous spacecraft control
   - Guaranteed rover traverse safety

3. Rover safety evaluations must frame criteria as mission-configured parameters, not universal laws of physics:
   - Example: State `"Configured analysis threshold: 15°"`, NEVER `"Universal scientific slope limit is 15°"`.
4. The permitted safety evaluation statuses are strictly:
   - `PASS`
   - `REVIEW_REQUIRED`
   - `FAIL`
5. Every safety evaluation must present the full evidence chain:
   - Total segments analyzed
   - Maximum observed slope
   - Configured threshold
   - Index of failing segments and reasons
   - Provenance of the underlying DEM (source, resolution, vertical accuracy)
   - Relevant scientific limitations (e.g. baseline noise, interpolation artifacts, sub-resolution boulder hazards).
