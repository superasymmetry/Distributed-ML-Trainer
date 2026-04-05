# Chaos Scripts

These scripts provide repeatable checks for reliability-demo scenarios.

## Prerequisites
- API reachable at `http://127.0.0.1:8000` (override with `BASE_URL`).
- `docker` and Docker Compose for API restart test.
- `kubectl` configured for worker/pod chaos tests.
- Controller process running for pod lifecycle checks.

## Scripts
- `01_bad_input_check.sh`: verifies bad input returns JSON error payloads.
- `02_api_restart_recovery.sh`: attempts in-container process crash, then verifies recovery (falls back to controlled restart when auto-restart cannot be observed on local runtime).
- `03_worker_pod_kill_recovery.sh`: kills worker pod and checks job status transition.
- `04_checkpoint_resume_check.sh`: verifies checkpoint-resume behavior after pod kill.
- `05_load_test.sh`: runs concurrent `POST /jobs` traffic, checks p50/p99 latency, and verifies API health remains OK.

## Usage
```bash
scripts/chaos/01_bad_input_check.sh
scripts/chaos/02_api_restart_recovery.sh
scripts/chaos/03_worker_pod_kill_recovery.sh
scripts/chaos/04_checkpoint_resume_check.sh
scripts/chaos/05_load_test.sh
```

Optional overrides:
```bash
BASE_URL=http://127.0.0.1:8000 NAMESPACE=default scripts/chaos/03_worker_pod_kill_recovery.sh
BASE_URL=http://127.0.0.1:8000 REQUESTS=50 CONCURRENCY=10 scripts/chaos/05_load_test.sh
```
