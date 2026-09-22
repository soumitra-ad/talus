---
description: Comprehensive security invariants and defensive programming enforcement
---

# Rule: Security & System Hardening

To safeguard users and infrastructure, you must strictly uphold these security rules across all operations and code changes:

## 1. Secrets & Credentials Management
- **Never hardcode secrets**: No API keys, passwords, or tokens in source code or documentation.
- **Never read secret files unnecessarily**: Do not inspect `.env` or credential files unless required for explicit authentication debugging.
- **Never commit `.env` or service-account JSON**: Verify repository exclusions and `.gitignore` prior to file operations.
- **Never expose backend credentials to browser code**: Ensure Streamlit session state, client templates, and DOM components never receive Gemini keys or Google Cloud credentials.
- **Do not log credentials**: Strip or mask any tokens, secrets, or keys before logging.
- **Do not log raw authentication headers**: Redact `Authorization`, `Bearer`, or `x-goog-api-key` headers from all HTTP logs.

## 2. Remote Data & Network Boundaries
- **Only allow downloads from configured NASA/LROC hosts**: Restrict remote network acquisition exclusively to `ode.rsl.wustl.edu`, `pds-geosciences.wustl.edu`, `wac.lroc.asu.edu`, and `lroc.sese.asu.edu`.
- **Block arbitrary URLs**: Never accept, parse, or download user-supplied URLs.
- **Reject private and internal addresses**: Prohibit localhost (`127.0.0.1`, `::1`), link-local metadata endpoints (`169.254.169.254`), and RFC 1918 subnets.

## 3. Code & Command Execution Restrictions
- **Never execute user input as shell**: Disallow `subprocess`, `os.system`, or shell commands parameterized by user query strings.
- **Never execute user input as Python**: Total prohibition of `eval()`, `exec()`, or dynamic code evaluation on user text.

## 4. Input & Path Validation
- **Validate all coordinates**: Enforce lunar latitude bounds $[-90^\circ, +90^\circ]$ and longitude bounds $[-180^\circ, +180^\circ]$ or $[0^\circ, 360^\circ]$.
- **Validate all paths**: Block path traversal attempts (`../`, absolute paths); ensure all writes remain strictly within designated directories (`data/cache/`, `outputs/`).
- **Validate all file sizes**: Reject unvalidated file streams; inspect `Content-Length` headers before streaming downloads.

## 5. Resource Limits & Denial of Service Protection
- **Cap downloads**: Maximum single download size bounded at $100\text{MB}$.
- **Cap raster windows**: Maximum DEM window dimensions bounded at $2048 \times 2048$ cells.
- **Cap request size**: Limit waypoint arrays to $\le 100$ points and search radius to reasonable operational scales.
- **Cap agent tool iterations**: Bound the agent loop to a maximum of 5 tool execution steps per user query to prevent runaways.

## 6. Error Handling
- **Sanitize displayed errors**: Never expose internal tracebacks, system file paths, or infrastructure details to user-facing UI screens.
