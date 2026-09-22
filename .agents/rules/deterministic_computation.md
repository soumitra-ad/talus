---
description: Enforces the deterministic computation boundary between LLM reasoning and scientific calculation
---

# Rule: Deterministic Computation Boundary

1. The LLM agent is strictly prohibited from guessing, approximating, or calculating numerical values for:
   - Elevation minimums, maximums, means, or standard deviations
   - Surface slope angles or gradients
   - Roughness or terrain ruggedness indices
   - Distance along waypoints or traverse segment lengths
   - Rover traverse risk scores or landing site suitability scores

2. All such calculations MUST be executed by deterministic Python functions in `src/terrain_agent/terrain/` or `src/terrain_agent/safety/`.

3. The LLM's role is restricted to:
   - Understanding user questions and mission objectives
   - Validating coordinates and identifying missing parameters
   - Selecting and invoking the appropriate deterministic tool
   - Inspecting and validating tool results against Pydantic schemas
   - Formulating clear, contextual answers that cite the deterministic tool evidence and state scientific limitations.
