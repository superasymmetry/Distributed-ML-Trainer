#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
NAMESPACE="${NAMESPACE:-default}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
KUBECTL_TIMEOUT_SECONDS="${KUBECTL_TIMEOUT_SECONDS:-15}"

ensure_kubernetes_access() {
  local current_context
  current_context="$(kubectl config current-context 2>/dev/null || echo "unknown")"
  if ! kubectl cluster-info --request-timeout="${KUBECTL_TIMEOUT_SECONDS}s" >/dev/null 2>&1; then
    echo "kubectl cannot reach a cluster (current context: '$current_context')."
    echo "Enable Kubernetes in Docker Desktop and switch context before running this script."
    exit 1
  fi
}

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl not found. Install kubectl and configure cluster access first."
  exit 1
fi
ensure_kubernetes_access

create_job_response="$(mktemp)"
trap 'rm -f "$create_job_response"' EXIT
create_code="$(
  curl -sS -o "$create_job_response" -w "%{http_code}" \
    -X POST "$BASE_URL/jobs" \
    -H "Content-Type: application/json" \
    -d '{"model":"test_efficientnet.r160_in1k","dataset":"mnist","epochs":4,"lr":0.01,"code":""}'
)"

if [[ "$create_code" != "200" ]]; then
  echo "Failed to create job. HTTP $create_code"
  cat "$create_job_response"
  exit 1
fi

job_id="$(python - "$create_job_response" <<'PY'
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    body = json.load(f)

print(body['job_id'])
PY
)"

pod_name="job-${job_id}"
echo "Created job: $job_id"

echo "Waiting for worker pod '$pod_name' to appear..."
for ((i=1; i<=WAIT_SECONDS; i++)); do
  phase="$(kubectl get pod "$pod_name" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  if [[ "$phase" == "Running" || "$phase" == "Pending" ]]; then
    echo "Pod phase before chaos: $phase"
    break
  fi
  sleep 1
done

kubectl delete pod "$pod_name" -n "$NAMESPACE" --ignore-not-found >/dev/null
echo "Deleted pod '$pod_name'. Waiting for job status transition..."

for ((i=1; i<=WAIT_SECONDS; i++)); do
  row_file="$(mktemp)"
  row_code="$(curl -sS -o "$row_file" -w "%{http_code}" "$BASE_URL/jobs/$job_id")"

  if [[ "$row_code" == "200" ]]; then
    read -r status retries <<<"$(python - "$row_file" <<'PY'
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    body = json.load(f)

if isinstance(body, list) and len(body) >= 5:
    print(body[1], body[4])
else:
    print('unknown', '-1')
PY
)"

    if [[ "$status" == "queued" || "$status" == "failed" || "$status" == "complete" ]]; then
      echo "PASS: job transitioned to status='$status' retries='$retries'"
      exit 0
    fi
  fi

  sleep 1
done

echo "FAIL: job '$job_id' stayed in non-terminal/non-requeued state after pod kill"
exit 1
