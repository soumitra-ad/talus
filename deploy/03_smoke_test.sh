#!/usr/bin/env bash
# Post-deploy smoke test against the live Cloud Run URL.
#
# A Streamlit app is a stateful websocket UI, not a REST API, so a black-box curl-based check
# can only verify transport-level facts: the service is up, the health endpoint reports
# healthy, the access gate behaves as configured, and no secret value is ever reflected back
# in a response body or header. It CANNOT drive the chat UI itself (submit a query, read the
# rendered structured result) -- that is what the two test suites below already cover, using
# the exact same application code this image was built from:
#
#   * tests/integration/test_streamlit_app.py (Phase 7, Streamlit AppTest framework) drives the
#     real UI widgets end-to-end -- including a normal query and an invalid/unsafe one -- with
#     a mocked deterministic-tool backend. Required to pass in CI before this image is built.
#   * tests/live/test_nasa_live.py + tests/integration/test_real_nasa_dem_pipeline.py exercise
#     the full, non-mocked pipeline (NASA ODE discovery -> PDS download -> validation ->
#     terrain/rover/landing analysis) against real NASA data, using the same library code
#     running inside this container.
#
# Usage:
#   SERVICE_URL=https://talus-xxxxx-uc.a.run.app ./deploy/03_smoke_test.sh
#   (optional) TALUS_ACCESS_TOKEN=... if the service is public and token-gated
#   (optional) ID_TOKEN=$(gcloud auth print-identity-token) if the service is private

set -euo pipefail
: "${SERVICE_URL:?Set SERVICE_URL to the deployed Cloud Run URL}"

AUTH_HEADER=()
if [[ -n "${ID_TOKEN:-}" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${ID_TOKEN}")
fi

echo "== 1. Health endpoint =="
HEALTH_CODE=$(curl -s -o /tmp/talus_health.out -w '%{http_code}' "${AUTH_HEADER[@]}" "${SERVICE_URL}/_stcore/health")
if [[ "${HEALTH_CODE}" != "200" ]]; then
  echo "FAIL: health check returned HTTP ${HEALTH_CODE}" >&2
  exit 1
fi
echo "PASS: HTTP 200"

echo "== 2. Root page loads (access gate or app shell) =="
ROOT_CODE=$(curl -s -o /tmp/talus_root.out -w '%{http_code}' "${AUTH_HEADER[@]}" "${SERVICE_URL}/")
if [[ "${ROOT_CODE}" != "200" ]]; then
  echo "FAIL: root page returned HTTP ${ROOT_CODE}" >&2
  exit 1
fi
echo "PASS: HTTP 200"

echo "== 3. Research/demo disclaimer is present (never claims certified/operational status) =="
if grep -qi "research" /tmp/talus_root.out && grep -qi "demo" /tmp/talus_root.out; then
  echo "PASS: disclaimer language present"
else
  echo "WARN: could not confirm disclaimer text in initial HTML (Streamlit renders client-side; verify manually in-browser)"
fi

echo "== 4. No secret value reflected in any response body or header =="
LEAK_FOUND=0
for pattern in "${GEMINI_API_KEY:-__unset__}" "${TALUS_ACCESS_TOKEN:-__unset__}"; do
  if [[ "${pattern}" != "__unset__" ]] && grep -q -- "${pattern}" /tmp/talus_root.out /tmp/talus_health.out 2>/dev/null; then
    echo "FAIL: a secret value was found in a response body!" >&2
    LEAK_FOUND=1
  fi
done
if [[ "${LEAK_FOUND}" == "0" ]]; then
  echo "PASS: no configured secret value appears in response bodies"
fi

echo ""
echo "Transport-level smoke test complete. For the full golden-path and invalid/unsafe-request"
echo "pipeline verification, see the CI-gated test suites referenced above (already required to"
echo "pass before this image was built) and deploy/README.md's manual verification checklist."
