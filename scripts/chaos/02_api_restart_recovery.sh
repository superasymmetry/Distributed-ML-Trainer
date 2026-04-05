#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SERVICE="${SERVICE:-api}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-90}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-30}"
AUTO_RESTART_DETECT_SECONDS="${AUTO_RESTART_DETECT_SECONDS:-20}"

compose_cmd=()

resolve_compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    compose_cmd=(docker compose)
    return
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    compose_cmd=(docker-compose)
    return
  fi
  return 1
}

wait_for_health() {
  local timeout_seconds="$1"
  for ((i=1; i<=timeout_seconds; i++)); do
    if curl -fsS "$BASE_URL/health" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

get_container_id() {
  "${compose_cmd[@]}" ps -q "$SERVICE"
}

get_started_at() {
  local container_id="$1"
  docker inspect -f '{{.State.StartedAt}}' "$container_id"
}

get_restart_count() {
  local container_id="$1"
  docker inspect -f '{{.RestartCount}}' "$container_id"
}

ensure_service_running() {
  local container_id
  container_id="$(get_container_id)"
  if [[ -n "$container_id" ]]; then
    echo "$container_id"
    return
  fi

  echo "Starting service '$SERVICE' with Docker Compose..."
  "${compose_cmd[@]}" up -d "$SERVICE" >/dev/null
  container_id="$(get_container_id)"
  if [[ -z "$container_id" ]]; then
    echo "Unable to start or locate service '$SERVICE'."
    exit 1
  fi
  echo "$container_id"
}

if ! resolve_compose_cmd; then
  echo "Docker Compose not found. Install Docker Compose v2 or docker-compose."
  exit 1
fi

container_id="$(ensure_service_running)"
if ! wait_for_health "$STARTUP_TIMEOUT_SECONDS"; then
  echo "API health check failed before chaos test."
  exit 1
fi

started_before="$(get_started_at "$container_id")"
restart_count_before="$(get_restart_count "$container_id")"
restart_policy="$(docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' "$container_id")"
if [[ "$restart_policy" != "always" && "$restart_policy" != "unless-stopped" ]]; then
  echo "WARN: restart policy is '$restart_policy' (expected 'always' or 'unless-stopped')."
fi

echo "Crashing PID 1 in container $container_id for service '$SERVICE'"
docker exec "$container_id" sh -lc 'kill -9 1' >/dev/null 2>&1 || true

restart_mode="auto"
for ((i=1; i<=AUTO_RESTART_DETECT_SECONDS; i++)); do
  current_container_id="$(get_container_id)"
  if [[ -n "$current_container_id" ]]; then
    current_started="$(get_started_at "$current_container_id" 2>/dev/null || true)"
    current_restart_count="$(get_restart_count "$current_container_id" 2>/dev/null || true)"
    if [[ "$current_started" != "$started_before" || "$current_restart_count" != "$restart_count_before" ]]; then
      break
    fi
  fi
  sleep 1
done

current_container_id="$(get_container_id)"
current_started="$( [[ -n "$current_container_id" ]] && get_started_at "$current_container_id" 2>/dev/null || echo "" )"
current_restart_count="$( [[ -n "$current_container_id" ]] && get_restart_count "$current_container_id" 2>/dev/null || echo "" )"
if [[ "$current_started" == "$started_before" && "$current_restart_count" == "$restart_count_before" ]]; then
  restart_mode="controlled"
  echo "Auto-restart signal not detected. Performing controlled service restart..."
  "${compose_cmd[@]}" restart "$SERVICE" >/dev/null
fi

if wait_for_health "$TIMEOUT_SECONDS"; then
  echo "PASS: API recovered after ${restart_mode} restart"
  exit 0
fi

echo "FAIL: API did not recover within ${TIMEOUT_SECONDS}s"
"${compose_cmd[@]}" ps "$SERVICE" || true
exit 1
