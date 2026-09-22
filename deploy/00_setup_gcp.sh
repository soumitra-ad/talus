#!/usr/bin/env bash
# One-time GCP project bootstrap for TALUS on Cloud Run.
#
# Prepares a project to host TALUS: enables required APIs, creates an Artifact Registry
# repository, and creates a dedicated, least-privilege runtime service account. Secrets are
# created separately (see the notes at the bottom) and are never echoed or logged.
#
# Usage:
#   PROJECT_ID=my-project REGION=us-central1 ./deploy/00_setup_gcp.sh
#
# Requires: gcloud CLI, authenticated (`gcloud auth login`), with an owner/editor-level
# identity on PROJECT_ID for this one-time setup only. The identity used for day-to-day
# deploys afterwards does not need this level of access.

set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to your GCP project id}"
REGION="${REGION:-us-central1}"
REPO_NAME="${REPO_NAME:-talus}"
RUNTIME_SA_NAME="${RUNTIME_SA_NAME:-talus-run-sa}"
RUNTIME_SA_EMAIL="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "== Project: ${PROJECT_ID}  Region: ${REGION} =="

echo "-- Enabling required APIs --"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com \
  cloudbuild.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  --project "${PROJECT_ID}"

echo "-- Creating Artifact Registry repository (idempotent) --"
gcloud artifacts repositories describe "${REPO_NAME}" \
  --project "${PROJECT_ID}" --location "${REGION}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPO_NAME}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --repository-format=docker \
  --description="TALUS container images"

echo "-- Creating dedicated runtime service account (idempotent) --"
gcloud iam service-accounts describe "${RUNTIME_SA_EMAIL}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud iam service-accounts create "${RUNTIME_SA_NAME}" \
  --project "${PROJECT_ID}" \
  --display-name="TALUS Cloud Run runtime identity" \
  --description="Least-privilege identity for the TALUS Cloud Run service. No project-level roles."

echo "-- Granting least-privilege project-level roles (logging + metrics only) --"
# Deliberately NOT granting roles/editor, roles/owner, or any broad secretAccessor role here.
# Secret access is granted per-secret, below, after the secrets themselves exist.
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_SA_EMAIL}" \
  --role="roles/logging.logWriter" --condition=None >/dev/null
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_SA_EMAIL}" \
  --role="roles/monitoring.metricWriter" --condition=None >/dev/null

cat <<EOF

== Setup complete ==

Runtime service account: ${RUNTIME_SA_EMAIL}
Artifact Registry repo:  ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}

Next: create secrets (values are never printed or logged; nothing is baked into the image).

  GEMINI_API_KEY is OPTIONAL -- TALUS runs in deterministic demo mode without it.
  TALUS_ACCESS_TOKEN is STRONGLY RECOMMENDED for any deployment reachable from the public
  internet -- it gates the Streamlit UI itself (see app/streamlit_app.py::_check_access),
  independent of Cloud Run's own IAM invoker check.

  Create a secret by piping the value in on stdin (never as a CLI argument, which would land
  in shell history and process listings):

    printf '%s' "<your-gemini-api-key>" | gcloud secrets create GEMINI_API_KEY \\
      --project "${PROJECT_ID}" --data-file=- --replication-policy=automatic

    printf '%s' "<a-long-random-token>" | gcloud secrets create TALUS_ACCESS_TOKEN \\
      --project "${PROJECT_ID}" --data-file=- --replication-policy=automatic

  Then grant the runtime service account access to ONLY those specific secrets (not a
  project-wide secretAccessor role):

    gcloud secrets add-iam-policy-binding GEMINI_API_KEY \\
      --project "${PROJECT_ID}" \\
      --member="serviceAccount:${RUNTIME_SA_EMAIL}" --role="roles/secretmanager.secretAccessor"

    gcloud secrets add-iam-policy-binding TALUS_ACCESS_TOKEN \\
      --project "${PROJECT_ID}" \\
      --member="serviceAccount:${RUNTIME_SA_EMAIL}" --role="roles/secretmanager.secretAccessor"

  If you skip GEMINI_API_KEY entirely, remove its secretKeyRef from deploy/service.yaml before
  deploying -- Cloud Run will fail to start if a referenced secret does not exist.

Next script: deploy/01_build_and_push.sh
EOF
