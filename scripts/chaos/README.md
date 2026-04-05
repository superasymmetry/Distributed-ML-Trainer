# Chaos Scripts

These scripts provide repeatable checks for reliability-demo scenarios.

## Prerequisites
- API reachable at `http://127.0.0.1:8000` (override with `BASE_URL`).
- `docker` and Docker Compose for API restart test.
- `kubectl` configured for worker/pod chaos tests.
- Controller process running for pod lifecycle checks.

## Scripts
- `01_bad_input_check.sh`: verifies bad input returns JSON error payloads.
- `02_api_restart_recovery.sh`: kills API container and confirms health recovery.
- `03_worker_pod_kill_recovery.sh`: kills worker pod and checks job status transition.
- `04_checkpoint_resume_check.sh`: verifies checkpoint-resume behavior after pod kill.

## Usage
```bash
scripts/chaos/01_bad_input_check.sh
scripts/chaos/02_api_restart_recovery.sh
scripts/chaos/03_worker_pod_kill_recovery.sh
scripts/chaos/04_checkpoint_resume_check.sh
```

Optional overrides:
```bash
BASE_URL=http://127.0.0.1:8000 NAMESPACE=default scripts/chaos/03_worker_pod_kill_recovery.sh
```
