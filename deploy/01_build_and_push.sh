#!/usr/bin/env bash
# Build the production TALUS image with Cloud Build and push it to Artifact Registry.
#
# Uses Cloud Build so the build environment matches the deployed environment exactly and
# no local Docker daemon is required. The image is tagged with the current git commit SHA so
# every deployment maps back to an exact, auditable source revision -- never ":latest".
#
# Usage:
#   PROJECT_ID=my-project REGION=us-central1 ./deploy/01_build_and_push.sh

set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID to your GCP project id}"
REGION="${REGION:-us-central1}"
REPO_NAME="${REPO_NAME:-talus}"

COMMIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo "nogit-$(date +%s)")"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/talus:${COMMIT_SHA}"

echo "-- Building ${IMAGE} via Cloud Build --"
gcloud builds submit \
  --project "${PROJECT_ID}" \
  --tag "${IMAGE}" \
  .

echo "-- Image built and pushed: ${IMAGE} --"
echo "${IMAGE}" > deploy/.last_image
echo "(written to deploy/.last_image for the next script)"
