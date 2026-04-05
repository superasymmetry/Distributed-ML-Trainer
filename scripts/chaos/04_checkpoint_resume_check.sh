#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
NAMESPACE="${NAMESPACE:-default}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"
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
    -d '{"model":"test_efficientnet.r160_in1k","dataset":"mnist","epochs":8,"lr":0.01,"code":""}'
)"

if [[ "$create_code" != "200" ]]; then
  echo "Failed to create checkpoint test job. HTTP $create_code"
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
checkpoint_file="/data/checkpoints/job-${job_id}.pt"
echo "Created job: $job_id"

echo "Waiting for pod '$pod_name' to reach Running..."
for ((i=1; i<=WAIT_SECONDS; i++)); do
  phase="$(kubectl get pod "$pod_name" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || true)"
  if [[ "$phase" == "Running" ]]; then
    break
  fi
  sleep 1
done

echo "Waiting for checkpoint file $checkpoint_file to exist..."
for ((i=1; i<=WAIT_SECONDS; i++)); do
  if kubectl exec -n "$NAMESPACE" "$pod_name" -- sh -lc "test -f '$checkpoint_file'" >/dev/null 2>&1; then
    echo "Checkpoint detected. Killing pod to force resume."
    break
  fi
  sleep 1
done

kubectl delete pod "$pod_name" -n "$NAMESPACE" --ignore-not-found >/dev/null

echo "Waiting for resumed-from-checkpoint signal in worker logs..."
for ((i=1; i<=WAIT_SECONDS; i++)); do
  if kubectl logs -n "$NAMESPACE" "$pod_name" --tail=300 2>/dev/null | grep -q "resumed from checkpoint"; then
    echo "PASS: checkpoint resume detected for job $job_id"
    exit 0
  fi
  sleep 1
done

echo "FAIL: did not detect checkpoint resume signal for job $job_id"
exit 1
