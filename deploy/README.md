# TALUS — Cloud Run Deployment Runbook

TALUS (Terrain Analysis for Landing and Uncrewed Systems) is a **research and technology
demonstration system**. It must never be described or operated as flight-certified,
providing operational landing approval, autonomous spacecraft control, or guaranteed rover
safety.

This directory contains a ready-to-execute deployment pipeline for Google Cloud Run. It was
prepared and locally verified in an environment with **no `gcloud` CLI and no GCP
credentials**, so the actual `gcloud`/cloud-side steps below have not been executed. Everything
that could be verified without GCP access has been (see "Local verification performed", below).

## Why this wasn't deployed automatically

Deploying to Cloud Run creates real, billed cloud infrastructure (a running service, a service
account, IAM bindings, a public or private URL) tied to *your* GCP project and billing account.
That requires credentials and a project choice only you can provide, and it's the kind of
action that should not happen without your explicit execution. Run the scripts below yourself;
each one prints what it did and the next step.

## 0. Prerequisites

- A GCP project with billing enabled.
- `gcloud` CLI installed and authenticated: `gcloud auth login`.
- An identity with `roles/owner` or equivalent broad access **for the one-time bootstrap only**
  (`00_setup_gcp.sh`); day-to-day deploys need much less.

## 1. Run in order

```bash
export PROJECT_ID=your-gcp-project
export REGION=us-central1        # any Cloud Run region

./deploy/00_setup_gcp.sh          # one-time: APIs, Artifact Registry, service account, IAM
# then create secrets as instructed in that script's output (GEMINI_API_KEY optional,
# TALUS_ACCESS_TOKEN strongly recommended)

./deploy/01_build_and_push.sh     # Cloud Build -> Artifact Registry, tagged by git commit SHA
./deploy/02_deploy.sh             # gcloud run services replace deploy/service.yaml (private by default)

SERVICE_URL=$(gcloud run services describe talus --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)')
ID_TOKEN=$(gcloud auth print-identity-token)   # only needed if the service is private
SERVICE_URL="$SERVICE_URL" ID_TOKEN="$ID_TOKEN" ./deploy/03_smoke_test.sh
```

To make the service reachable without a Cloud Run IAM grant (e.g. for an open demo link),
re-run `02_deploy.sh` with `PUBLIC=true`. The app-level `TALUS_ACCESS_TOKEN` gate (see below)
still applies if you created that secret — the frontend is never the only gate.

## 2. Requirement-by-requirement mapping

| # | Requirement | How it's met |
|---|---|---|
| 1 | Build production container | `deploy/01_build_and_push.sh` via Cloud Build; multi-stage `Dockerfile` (hardened in a prior pass — build toolchain excluded from the runtime image) |
| 2 | Deploy to Cloud Run | `deploy/02_deploy.sh` (`gcloud run services replace`) |
| 3 | No secrets in the image | Dockerfile never `COPY`s `.env` or credentials (`.dockerignore` excludes it); confirmed no secret strings anywhere in `src/`/`app/` |
| 4 | Env vars / Secret Manager | `GEMINI_API_KEY` and `TALUS_ACCESS_TOKEN` injected via `valueFrom.secretKeyRef` in `service.yaml`, never as plain env values or build args |
| 5 | Dedicated service identity | `talus-run-sa` created in `00_setup_gcp.sh`, used as `serviceAccountName` in `service.yaml` — not the default Compute Engine service account |
| 6 | Least-privilege permissions | Runtime SA gets only `roles/logging.logWriter` + `roles/monitoring.metricWriter` at project level, plus `roles/secretmanager.secretAccessor` on the two *specific* secrets (never a project-wide grant, never `roles/editor`/`roles/owner`) |
| 7 | Resource limits | `service.yaml`: `cpu: "2"`, `memory: 2Gi`, `maxScale: 3` (bounds cost/blast radius) |
| 8 | Request timeout | `timeoutSeconds: 900`, matched to `TALUS_NASA_TOTAL_TIMEOUT_SEC`'s default (a synchronous NASA download can legitimately take that long) |
| 9 | Concurrency | `containerConcurrency: 4` — kept low because each session can hold a bounded-but-nontrivial raster window in memory; horizontal autoscaling (up to `maxScale`) absorbs load instead of overloading one instance |
| 10 | Auth/authz for non-public backend | TALUS is a single Streamlit service (no separate backend API). Default: **private** — no `allUsers` invoker binding; `02_deploy.sh` requires `PUBLIC=true` to open it. Independently, the app itself gates on `TALUS_ACCESS_TOKEN` (`app/streamlit_app.py::_check_access`, constant-time comparison) whenever that secret is set |
| 11 | Frontend is not the security boundary | SSRF allowlisting, resource limits, and input validation all live in `src/terrain_agent/` (deterministic layer) and are enforced regardless of who can reach the Streamlit UI — verified in `tests/security/` (100 tests) and the adversarial smoke test below, independent of any UI gate |
| 12 | Logging | No explicit log sink is configured, so Python's standard `logging` output (stderr) is used — Cloud Run captures container stdout/stderr into Cloud Logging automatically. Runtime SA has `roles/logging.logWriter` |
| 13 | Health checks | `/_stcore/health` (Streamlit's built-in endpoint), used both by the existing Dockerfile `HEALTHCHECK` and by `service.yaml`'s `startupProbe`/`livenessProbe` |
| 14 | No arbitrary outbound requests | `src/terrain_agent/acquisition/net_policy.py`: https-only, exact-host allowlist (no wildcards/subdomains), DNS-resolved address must be global-unicast (blocks loopback/private/link-local — including the `169.254.169.254` cloud metadata address), checked on every request *and* every redirect hop. Verified live, see below |
| 15 | Bounded NASA downloads | `TALUS_MAX_DOWNLOAD_BYTES` (100 MiB default) enforced against the declared `Content-Length` and against actual bytes streamed in `acquisition/download.py`; independently, `terrain/resource_safety.py` hard-caps raster reads at 2048×2048 cells. Verified live, see below |

## 3. Local verification performed (no GCP access available)

Everything below ran directly against this repository's actual code — the same code the
container image is built from — not a simulation.

**Phases 4/5 (real NASA DEM integration, real-data validation):**
- `pytest -m real_dem` — 22 tests, **passed**, against an already-cached real LOLA DEM
  (`tests/fixtures/nasa_real_dem_cache/`, downloaded from `pds-geosciences.wustl.edu`).
- A fresh, non-mocked, live end-to-end run (`tests/live/test_nasa_live.py`, no mocks: NASA ODE
  discovery → PDS download → validation → terrain/rover/landing analysis on real data) —
  see the note in the final chat report for this run's outcome, since it depends on live
  network conditions at the time this runbook was generated.

**Phase 6 (agent) / Phase 7 (UI):**
- `pytest tests/unit/test_agent_chat_loop.py tests/unit/test_agent_tools_phase5.py tests/integration/test_streamlit_app.py` —
  part of the 599-test full suite, **all passed**. UI tests use Streamlit's `AppTest` framework
  to drive the real widget tree (not a mock UI), including both a normal query and an
  invalid/unsafe one, against a mocked deterministic backend.

**Phase 8 (security audit):**
- `pytest tests/security` — 100 tests, **passed**.
- Adversarial smoke test run directly against the real (non-mocked) library code:
  - Non-allowlisted host (`evil.example.com`) → **rejected** (`HostPolicyError`).
  - Allowlisted host name forced to resolve to `169.254.169.254` (cloud metadata address) via
    an injected resolver → **rejected** (`HostPolicyError`, address-safety check).
  - 5000×5000 raster window request (cap is 2048×2048) → **rejected** (`OversizedRequestError`).
  - 500-waypoint route (cap is 100) → **rejected** (`OversizedRequestError`).
  - Latitude −999° → **rejected** (`InvalidCoordinateError`).

**Phase 9 (Docker/CI):**
- Dockerfile hardened to a multi-stage build in a prior pass (build toolchain excluded from
  the runtime image; non-root `talususer`; healthcheck; no secrets copied in).
- CI (`.github/workflows/ci.yml`) runs lint, the full non-network test suite, a `pip-audit`
  dependency scan, a `bandit` static-analysis scan, and a Docker build-verification job.
- **Not verified locally**: an actual `docker build` — no Docker daemon is available in this
  environment. The CI `docker-build` job will validate this on the next push/PR.

## 4. Known limitations

- **Not yet deployed.** The scripts in this directory are ready to run but have not been
  executed against a real GCP project — see "Why this wasn't deployed automatically" above.
- **Docker build is CI-verified, not locally verified**, for the reason above.
- **No infrastructure-level egress lockdown.** The NASA-domain allowlist is enforced entirely
  in the application layer (`net_policy.py`) and is well-tested, but Cloud Run's default
  networking allows the container to reach the general internet at the OS/socket level. For
  defense-in-depth beyond the app layer, an optional next step is a Serverless VPC Connector +
  Cloud NAT + firewall/Secure Web Proxy policy restricting egress to the four NASA/PDS
  hostnames; not implemented here as it adds real infra cost/complexity that should be a
  deliberate choice, not a default.
- **Single-service architecture.** Requirement 10 (auth for "backend services that should not
  be public") is addressed by treating the one Cloud Run service as private-by-default; there
  is no separate internal API to isolate further.
- **Cold starts.** `minScale: 0` means the first request after idle will see Cloud Run + Python
  + rasterio/GDAL cold-start latency (rough order: several seconds to ~20s). Set `minScale: 1`
  if that's unacceptable for a demo; it removes the cost benefit of scale-to-zero.
- **TALUS remains a research/demo system.** Nothing in this deployment changes that; it must
  never be presented as certified, operational, or safety-guaranteeing.
