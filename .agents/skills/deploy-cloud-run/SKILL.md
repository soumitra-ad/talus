---
name: deploy-cloud-run
description: Secure multi-stage deployment process to Google Cloud Run covering development, tests, build, image validation, staging, smoke testing, and production gating.
---

# Skill: Secure Cloud Run Deployment Pipeline

This skill outlines the strict, phased promotion pipeline for deploying TALUS to Google Cloud Run.

> [!IMPORTANT]
> **Production Protection Rule**: Never deploy directly to production from pull requests or unverified branch merges. Production deployments require passing all previous stages, explicit review, and smoke test verification on staging.

---

## 1. Deployment Pipeline Architecture

```
[ 1. Development ]
       │
       ▼
[ 2. Tests (Unit, Integration, Security) ]
       │
       ▼
[ 3. Container Build (Non-root, minimal) ]
       │
       ▼
[ 4. Image Validation & Vulnerability Scan ]
       │
       ▼
[ 5. Staging Deployment ]
       │
       ▼
[ 6. Automated Smoke Testing ]
       │
       ▼
[ 7. Production Gate (Manual Approval) ]
       │
       ▼
[ 8. Production Deployment ]
```

---

## 2. Phased Step-by-Step Procedure

### Stage 1: Development & Local Verification
- Ensure code conforms to `ruff` linting and formatting.
- Confirm zero secrets exist in working files or configuration.
- Validate local execution without external credentials.

### Stage 2: Automated Tests
Execute the full test suite before any containerization:
```bash
pytest --cov=terrain_agent --cov-fail-under=80 tests/
```
All unit, integration, and security tests must pass.

### Stage 3: Container Build
Build a secure, minimal container image using Google Cloud Build or local Docker:
```bash
docker build -t gcr.io/${PROJECT_ID}/talus:${COMMIT_SHA} -f Dockerfile .
```
- Multi-stage build with `python:3.12-slim`.
- Dedicated unprivileged user (`talususer`, UID 1001).

### Stage 4: Image Validation & Security Scan
Inspect the container image for security flaws:
```bash
# Scan image for known vulnerabilities
gcloud artifacts docker images scan gcr.io/${PROJECT_ID}/talus:${COMMIT_SHA}
# Verify container runs as non-root
docker run --rm gcr.io/${PROJECT_ID}/talus:${COMMIT_SHA} whoami  # Must output: talususer
```

### Stage 5: Staging Deployment
Deploy the verified image to the staging environment:
```bash
gcloud run deploy talus-staging \
    --image gcr.io/${PROJECT_ID}/talus:${COMMIT_SHA} \
    --platform managed \
    --region us-central1 \
    --no-allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --service-account talus-staging-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --set-env-vars "TALUS_ENV=staging,GEMINI_MODEL=gemini-2.5-flash"
```

### Stage 6: Staging Smoke Test
Execute automated health and functional checks on the staging instance:
```bash
# Verify healthcheck endpoint
curl -f https://talus-staging-uc.a.run.app/_stcore/health

# Verify zero-secret demonstration mode and configuration initialization
```

### Stage 7: Production Gate
- Automated pull request builds must **NEVER** deploy to production.
- Production deployment requires:
  1. Passing all unit, integration, and security tests.
  2. Successful smoke tests in staging.
  3. Explicit approval by a codeowner/maintainer.

### Stage 8: Production Deployment
Once approved, promote the identical validated image SHA to the production service:
```bash
gcloud run deploy talus-prod \
    --image gcr.io/${PROJECT_ID}/talus:${COMMIT_SHA} \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --service-account talus-prod-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --set-env-vars "TALUS_ENV=production,GEMINI_MODEL=gemini-2.5-flash"
```
