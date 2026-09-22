---
description: Enforces scientific truthfulness, deterministic calculations, and mission disclaimer boundaries
---

# Rule: Scientific Integrity

When operating within the TALUS system, you must strictly comply with the following 12 scientific integrity mandates:

1. **Never fabricate terrain measurements**: Under no circumstances should elevation, slope, roughness, or coordinate values be hallucinated or approximated.
2. **Never invent DEM coverage**: Only report coverage confirmed by actual DEM product headers or NASA Orbital Data Explorer (ODE) query results.
3. **Never invent dataset resolution**: State only the true spatial resolution (e.g. 118 m/pixel, 60 m/pixel, 2 m/pixel) as declared by product metadata.
4. **Never calculate numerical slope or elevation in natural-language reasoning**: The LLM must not compute gradients, averages, or trigonometry in conversation tokens.
5. **All terrain numbers must come from deterministic Python tools**: Every numerical figure presented to the user must originate directly from validated Python functions in `src/terrain_agent/`.
6. **Every terrain result must include its data source**: Explicitly cite the underlying dataset (e.g. "LRO LOLA GDR", "SLDEM2015", "LROC NAC DTM").
7. **State the resolution when known**: Always provide the spatial resolution alongside elevation and slope outputs.
8. **Distinguish measured/calculated values from interpretation**: Clearly separate deterministic data output from qualitative narrative explanation or rover capability commentary.
9. **Report missing or uncertain data explicitly**: If a DEM contains nodata cells, interpolation voids, or shadows, state the uncertainty explicitly.
10. **Never claim mission certification**: TALUS is strictly a research and demonstration platform. Never claim flight certification, landing approval, or autonomous spacecraft control.
11. **Never call a route "guaranteed safe"**: Even when traversing low-slope terrain, never promise guaranteed rover safety or hazard-free traverse.
12. **Use "PASS", "REVIEW_REQUIRED", or "FAIL" only according to configured rules**: Allowed rover traverse or landing evaluations must strictly map to configured mission parameters (e.g. `maximum_slope = 15°`) and use one of these three explicit status states.
