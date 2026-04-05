#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
REQUESTS="${REQUESTS:-50}"
CONCURRENCY="${CONCURRENCY:-10}"

tmp_dir="$(mktemp -d)"
results_file="$tmp_dir/results.tsv"
responses_dir="$tmp_dir/responses"
mkdir -p "$responses_dir"
export BASE_URL results_file responses_dir

echo "Running load test: requests=$REQUESTS concurrency=$CONCURRENCY"

seq 1 "$REQUESTS" | xargs -P "$CONCURRENCY" -I{} bash -c '
  idx="$1"
  octet=$(( (idx % 250) + 1 ))
  subnet=$(( (idx / 250) + 1 ))
  forwarded_ip="10.${subnet}.0.${octet}"
  payload="{\"model\":\"test_efficientnet.r160_in1k\",\"dataset\":\"mnist\",\"epochs\":1,\"lr\":0.01,\"code\":\"pass\"}"
  response_file="${responses_dir}/response_${idx}.json"
  line="$(curl -sS -o "${response_file}" -w "%{http_code}\t%{time_total}" \
    -X POST "${BASE_URL}/jobs" \
    -H "Content-Type: application/json" \
    -H "X-Forwarded-For: ${forwarded_ip}" \
    -d "${payload}")"
  printf "%s\t%s\n" "${idx}" "${line}" >> "${results_file}"
' _ {}

total_count="$(wc -l < "$results_file" | tr -d ' ')"
if [[ "$total_count" != "$REQUESTS" ]]; then
  echo "FAIL: expected $REQUESTS responses but captured $total_count"
  exit 1
fi

non_200_count="$(awk -F '\t' '$2 != "200" {count++} END {print count+0}' "$results_file")"
if [[ "$non_200_count" != "0" ]]; then
  echo "FAIL: found $non_200_count non-200 responses"
  awk -F '\t' '$2 != "200" {print $1, $2}' "$results_file" | head -n 10
  exit 1
fi

python - "$results_file" <<'PY'
import math
import sys

path = sys.argv[1]
times = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        times.append(float(parts[2]))

if not times:
    raise SystemExit("No timing samples captured.")

times.sort()

def percentile(values, p):
    idx = int(math.ceil((p / 100.0) * len(values))) - 1
    idx = max(0, min(idx, len(values) - 1))
    return values[idx]

p50 = percentile(times, 50) * 1000
p99 = percentile(times, 99) * 1000
print(f"Latency summary: p50={p50:.2f}ms p99={p99:.2f}ms")
PY

health_code="$(curl -sS -o /dev/null -w "%{http_code}" "${BASE_URL}/health")"
if [[ "$health_code" != "200" ]]; then
  echo "FAIL: API health degraded after load (HTTP $health_code)"
  exit 1
fi

echo "PASS: all ${REQUESTS} requests succeeded and API stayed healthy"

