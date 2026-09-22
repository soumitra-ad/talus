## Description

Brief summary of changes and why they are needed.

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Security enhancement

## Architectural Checklist

- [ ] All numerical terrain calculations reside in deterministic Python functions (`terrain/` or `safety/`).
- [ ] No LLM-generated numerical values are treated as ground truth.
- [ ] No secrets or credentials are hardcoded or committed.
- [ ] External data fetches strictly adhere to official NASA ODE / PDS domain allowlists.
- [ ] Raster operations enforce bounded window limits ($\le 2048 \times 2048$).
- [ ] Research/demo safety disclaimer is preserved.
- [ ] Unit and security tests pass locally.
