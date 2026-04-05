#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SERVICE="${SERVICE:-api}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-90}"

resolve_compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
    return
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    echo "docker-compose"
    return
  fi
  echo ""
}

compose_cmd="$(resolve_compose_cmd)"
if [[ -z "$compose_cmd" ]]; then
  echo "Docker Compose not found. Install Docker Compose v2 or docker-compose."
  exit 1
fi

if ! curl -fsS "$BASE_URL/health" >/dev/null; then
  echo "API health check failed before chaos test."
  exit 1
fi

container_id="$($compose_cmd ps -q "$SERVICE")"
if [[ -z "$container_id" ]]; then
  echo "No container found for service '$SERVICE'."
  exit 1
fi

echo "Killing container $container_id for service '$SERVICE'"
docker kill "$container_id" >/dev/null

echo "Waiting for API recovery..."
for ((i=1; i<=TIMEOUT_SECONDS; i++)); do
  if curl -fsS "$BASE_URL/health" >/dev/null; then
    echo "PASS: API recovered in ${i}s after forced kill"
    exit 0
  fi
  sleep 1
done

echo "FAIL: API did not recover within ${TIMEOUT_SECONDS}s"
exit 1
