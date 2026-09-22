# Contributing to TALUS

Terrain Analysis for Landing and Uncrewed Systems (TALUS) is an open research and demonstration application. We welcome contributions that improve the accuracy, safety evaluation methods, and agentic workflows of the project.

## Development Principles

1. **Deterministic Computation Boundary**:
   - The LLM must NEVER perform numerical calculations for slopes, elevations, roughness, or risk.
   - All numerical evaluations must be executed by validated Python functions under `src/terrain_agent/terrain/` or `src/terrain_agent/safety/`.
2. **Safety Disclaimer Integrity**:
   - Never remove or weaken the non-certified research/demo disclaimer.
   - TALUS must never be presented as flight-certified or operational mission control software.
3. **Security by Default**:
   - Never commit API keys, service account JSON files, or `.env` files.
   - All network interactions with external data sources must adhere to the official NASA ODE / PDS / LROC domain allowlist.
   - No user-supplied URLs or shell execution paths.
4. **Testing and Code Quality**:
   - All new features must include unit tests under `tests/unit/`.
   - Security-sensitive code must be validated with tests under `tests/security/`.
   - Code should conform to `ruff` and `black` standards (line length: 100).

## Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/example/terrain-analysis-agent.git
cd terrain-analysis-agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# 3. Install in editable mode with development dependencies
pip install -e ".[dev]"

# 4. Run tests
pytest
```
