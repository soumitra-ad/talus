# Security Policy

## Reporting Security Vulnerabilities

If you identify a potential security vulnerability in TALUS (Terrain Analysis for Landing and Uncrewed Systems), please report it responsibly by contacting the project maintainers via email or private security advisory rather than opening a public issue.

## Security Architecture & Core Tenets

TALUS processes untrusted user queries and fetches lunar terrain data from external scientific archives. To safeguard systems and users, the following security guarantees are strictly enforced:

1. **No Arbitrary Code / Command Execution**:
   - User inputs are never passed to `eval()`, `exec()`, or subshells.
   - All tool arguments are parsed and validated via strict Pydantic schemas.

2. **Server-Side Request Forgery (SSRF) Prevention**:
   - The application strictly forbids downloading arbitrary URLs provided by users or parsed from untrusted metadata.
   - Remote queries are restricted exclusively to official NASA ODE and PDS endpoints:
     - `ode.rsl.wustl.edu`
     - `pds-geosciences.wustl.edu`
     - `wac.lroc.asu.edu`
     - `lroc.sese.asu.edu`

3. **Path Traversal Protection**:
   - All local file writes (DEM tile cache, temporary files) use cryptographically hashed identifiers or strictly sanitized alphanumeric filenames.
   - Path operations verify that target paths resolve within designated `data/cache/` or `outputs/` directories.

4. **Resource Exhaustion & Denial of Service Protection**:
   - DEM raster window reads are constrained to $\le 2048 \times 2048$ cells to prevent out-of-memory errors.
   - File downloads are bounded to $\le 100\text{MB}$ with strict HTTP timeouts (default: 30 seconds).
   - Rover path waypoints are capped at 100 points per analysis.

5. **Credential & Secret Protection**:
   - The repository operates out of the box with zero mandatory secrets.
   - When Google Gemini or Vertex AI is enabled, credentials must be supplied via environment variables (`GEMINI_API_KEY`) or Google Cloud Application Default Credentials (ADC).
   - No credentials or API tokens are ever exposed to the client browser or logged in stdout/log files.
