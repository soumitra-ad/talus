#!/usr/bin/env bash
# Deploy (or update) the TALUS Cloud Run service from deploy/service.yaml.
#
# Usage:
#   PROJECT_ID=my-project REGION=us-central1 ./deploy/02_deploy.sh
#
# By default the service is PRIVATE: only principals you explicitly grant roles/run.invoker
# can reach it (requirement: do not assume an unguessable URL is a security boundary). To make
# it reachable by anyone with the URL (still gated by the app's own TALUS_ACCESS_TOKEN if you
# created that secret), re-run with PUBLIC=true. Either way, the frontend is never treated as
# the sole security boundary: SSRF/allowlist enforcement, resource limits, and input validation
# all live in the deterministic backend layer regardless of who can reach the UI.

set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to your GCP project id}"
REGION="${REGION:-us-central1}"
RUNTIME_SA_NAME="${RUNTIME_SA_NAME:-talus-run-sa}"
RUNTIME_SA_EMAIL="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
PUBLIC="${PUBLIC:-false}"

if [[ ! -f deploy/.last_image ]]; then
  echo "deploy/.last_image not found -- run deploy/01_build_and_push.sh first." >&2
  exit 1
fi
IMAGE="$(cat deploy/.last_image)"

echo "-- Rendering service.yaml (image=${IMAGE}, sa=${RUNTIME_SA_EMAIL}) --"
RENDERED="$(mktemp)"
sed \
  -e "s|__IMAGE__|${IMAGE}|g" \
  -e "s|__RUNTIME_SA_EMAIL__|${RUNTIME_SA_EMAIL}|g" \
  deploy/service.yaml > "${RENDERED}"

echo "-- Deploying to Cloud Run (region=${REGION}) --"
gcloud run services replace "${RENDERED}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}"
rm -f "${RENDERED}"

if [[ "${PUBLIC}" == "true" ]]; then
  echo "-- PUBLIC=true: allowing unauthenticated invocations (app-level TALUS_ACCESS_TOKEN still applies if configured) --"
  gcloud run services add-iam-policy-binding talus \
    --project "${PROJECT_ID}" --region "${REGION}" \
    --member="allUsers" --role="roles/run.invoker"
else
  echo "-- Service is private. Grant access with, e.g.:"
  echo "     gcloud run services add-iam-policy-binding talus --project ${PROJECT_ID} --region ${REGION} \\"
  echo "       --member=\"user:someone@example.com\" --role=\"roles/run.invoker\""
fi

URL="$(gcloud run services describe talus --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)')"
echo ""
echo "== Deployed: ${URL} =="
echo "Next: deploy/03_smoke_test.sh (set SERVICE_URL=${URL})"
