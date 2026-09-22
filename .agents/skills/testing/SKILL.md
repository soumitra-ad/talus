---
name: testing
description: Testing requirements and procedures covering unit tests, integration tests, security tests, agent evaluation tests, regression tests, and deterministic terrain tests.
---

# Skill: Comprehensive Testing Standards

This skill defines the mandatory test categories, invariants, and procedures required for TALUS.

## 1. Required Testing Tiers

Every contribution must be validated across 6 distinct testing domains:

### 1. Unit Tests (`tests/unit/`)
- Test individual functions in isolation.
- Verify configuration defaults, environment variable overrides, and zero-secret instantiation.
- Verify coordinate bounds validation and projection transformations.

### 2. Deterministic Terrain Tests (`tests/unit/test_terrain.py`)
- Test slope calculation using Horn's method against known mathematical fixtures:
  - Flat horizontal surface ($z = c$): slope must be $0.0^\circ \pm 0.001^\circ$.
  - 45° planar ramp (100m rise over 100m run): slope must be $45.0^\circ \pm 0.05^\circ$.
  - Uniform vertical step / ridge: check directional gradients.
- Test Terrain Ruggedness Index (TRI) against uniform vs. rugged matrices.
- Verify that elevation stats (min, max, mean, std) accurately reflect underlying raster cells.

### 3. Integration Tests (`tests/integration/`)
- Test tool orchestration and data pipeline end-to-end using local synthetic DEM fixtures.
- Verify windowed raster loading (`rasterio.windows`) and ensure array dimensions remain $\le 2048 \times 2048$.
- Verify tile caching, SHA-256 computation, and metadata JSON serialization.

### 4. Security Tests (`tests/security/`)
- **SSRF Prevention**: Assert that queries targeting `http://169.254.169.254`, `http://localhost`, or unapproved hosts raise strict `SecurityException`.
- **Path Traversal**: Assert that filenames containing `../../` or absolute paths are rejected.
- **Input Bounds & Fuzzing**: Test invalid coordinates, negative window sizes, NaN elevations, and oversized waypoint lists.
- **Secret Scanner**: Test that no credentials or private tokens exist in git tracking.

### 5. Agent Evaluation Tests (`tests/evaluation/`)
- Benchmark intent parsing and tool selection against the 12 core mission questions.
- Verify that the agent handles missing parameters by generating clarification prompts rather than guessing.
- Assert that numerical values in the final agent response match tool outputs with zero hallucination.

### 6. Regression Tests (`tests/regression/`)
- Ensure previously identified defects (e.g. edge-cell slope boundary artifacts, nodata handling, projection distortions) have regression assertions.

## 2. Test Execution Commands

```bash
# Run all tests
pytest -v

# Run with coverage threshold
pytest --cov=terrain_agent --cov-report=term-missing --cov-fail-under=80 tests/

# Run security tests exclusively
pytest tests/security/ -v

# Run deterministic terrain tests
pytest tests/unit/test_terrain.py -v
```

## 3. Invariants & Acceptance Criteria
- Zero test failures permitted.
- Deterministic terrain math must never exhibit floating-point divergence across platforms.
- Safety evaluations must strictly output `PASS`, `REVIEW_REQUIRED`, or `FAIL`.
