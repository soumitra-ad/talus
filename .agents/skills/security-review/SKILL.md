---
name: security-review
description: Comprehensive security audit procedures for secrets, SSRF, path traversal, prompt injection, command injection, dependency risks, unsafe logging, and unbounded resources.
---

# Skill: Security Review & Threat Verification

This skill establishes the security audit procedure to verify that all TALUS code, configurations, and tools comply with defensive engineering standards.

## 1. Security Audit Checklist

Run through these 9 verification domains during code reviews or before merging changes:

### 1. Secrets & Credential Exposure
- [ ] Scan the codebase for hardcoded keys, passwords, or tokens.
- [ ] Verify that `.gitignore` and `.dockerignore` exclude `.env`, `*.json` (except static metadata), `*.pem`, `*.key`.
- [ ] Confirm no backend credentials or API keys are exposed to the browser or Streamlit session state.

### 2. Server-Side Request Forgery (SSRF)
- [ ] Verify that all outbound HTTP requests validate destination URLs against the official host allowlist (`ode.rsl.wustl.edu`, `pds-geosciences.wustl.edu`, `wac.lroc.asu.edu`, `lroc.sese.asu.edu`).
- [ ] Ensure user input is never directly used to construct target IP addresses or hosts.
- [ ] Confirm requests to localhost (`127.0.0.1`, `::1`), link-local metadata endpoints (`169.254.169.254`), and private subnets are blocked.

### 3. Path Traversal & Filesystem Hardening
- [ ] Confirm that all local file operations use sanitized identifiers or hashes.
- [ ] Verify that directory path resolutions are restricted to `data/cache/` or `outputs/` using `Path.resolve().is_relative_to()`.
- [ ] Reject any filenames containing `../`, null bytes, or backslashes.

### 4. Prompt Injection Resistance
- [ ] Ensure clear delimiters separate system instructions, tool outputs, and user query strings.
- [ ] Verify that external DEM metadata or remote descriptions are treated as untrusted text, not executable instructions.
- [ ] Confirm that safety statuses (`PASS`, `REVIEW_REQUIRED`, `FAIL`) are generated strictly by Python logic, never by the LLM.

### 5. Command & Code Injection
- [ ] Ensure complete absence of `eval()`, `exec()`, `os.system()`, or unescaped `subprocess` calls.
- [ ] Verify that all tool parameters are parsed and strictly typed via Pydantic models.

### 6. Dependency Risks & Vulnerabilities
- [ ] Check `pyproject.toml` dependencies against known vulnerability advisories.
- [ ] Ensure container base image uses pinned, minimal distributions (e.g. `python:3.12-slim`).

### 7. Unsafe Logging & Data Leakage
- [ ] Verify that raw authorization headers (`Bearer`, `x-goog-api-key`) are redacted before logging.
- [ ] Ensure that internal system paths and raw exception tracebacks are sanitized before rendering in user-facing UI.

### 8. Excessive Permissions & Principle of Least Privilege
- [ ] Confirm that the Docker container runs under a non-root user (`talususer`, UID 1001).
- [ ] Ensure Cloud Run service account permissions are restricted to `roles/aiplatform.user` without broad administrative roles.

### 9. Unbounded Resource Consumption
- [ ] Verify that raster read windows are capped at $\le 2048 \times 2048$ cells.
- [ ] Confirm that file downloads enforce a $100\text{MB}$ size ceiling.
- [ ] Confirm that network and tool execution timeouts ($\le 30$ seconds) are configured on all async clients.
- [ ] Verify that agent reasoning loops enforce a maximum iteration limit ($\le 5$ tool steps per turn).

## 2. Review Verdicts
- **CLEAN**: All checks pass; no vulnerabilities detected.
- **FLAGGED**: One or more checks fail; remediation required before deployment.
