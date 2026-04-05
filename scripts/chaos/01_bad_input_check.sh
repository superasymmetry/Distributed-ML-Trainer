#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

has_detail() {
  local file_path="$1"
  python - "$file_path" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    body = json.load(f)

if "detail" not in body:
    raise SystemExit(1)
PY
}

echo "[1/2] Invalid model payload should return JSON error"
resp_a="$(mktemp)"
code_a="$(
  curl -sS -o "$resp_a" -w "%{http_code}" \
    -X POST "$BASE_URL/jobs" \
    -H "Content-Type: application/json" \
    -d '{"model":"invalid_model_name_for_training","dataset":"mnist","epochs":1,"lr":0.01,"code":"pass"}'
)"

if [[ "$code_a" != "400" ]]; then
  echo "Expected HTTP 400 for invalid model, got $code_a"
  cat "$resp_a"
  exit 1
fi
has_detail "$resp_a"


echo "[2/2] Malformed JSON should return validation JSON error"
resp_b="$(mktemp)"
code_b="$(
  curl -sS -o "$resp_b" -w "%{http_code}" \
    -X POST "$BASE_URL/jobs" \
    -H "Content-Type: application/json" \
    -d '{"model":"test_efficientnet.r160_in1k",'
)"

if [[ "$code_b" != "422" ]]; then
  echo "Expected HTTP 422 for malformed JSON, got $code_b"
  cat "$resp_b"
  exit 1
fi
has_detail "$resp_b"

echo "PASS: bad input paths return clean JSON errors"
