# Demo Runbook (2 Minutes)

This runbook is optimized for the hackathon reliability demo.

## Pre-Demo Setup
1. Start API + controller:
```bash
fastapi dev main.py
python -m controller.controller
```
2. Ensure worker image exists:
```bash
docker build -t distributed-trainer-worker:latest .
```
3. Confirm health:
```bash
curl -sS http://127.0.0.1:8000/health
```

## Demo Sequence

### 0:00-0:25 Bad Input -> Clean JSON Error
```bash
scripts/chaos/01_bad_input_check.sh
```
Expected:
- `POST /jobs` returns JSON `detail` errors.
- API remains healthy.

### 0:25-0:55 Crash API Process -> Auto Recovery
```bash
scripts/chaos/02_api_restart_recovery.sh
```
Expected:
- API process is force-crashed inside the running container.
- Docker runtime recovers service (auto-restart where detectable, otherwise controlled restart fallback).
- `/health` returns `200` again within timeout.

### 0:55-1:30 Kill Worker Pod -> Controller Handles Transition
```bash
scripts/chaos/03_worker_pod_kill_recovery.sh
```
Expected:
- Worker pod is deleted mid-job.
- Job status transitions cleanly (re-queued/retried or terminal failure), not stuck forever.

### 1:30-2:00 Checkpoint Resume Proof
```bash
scripts/chaos/04_checkpoint_resume_check.sh
```
Expected:
- Checkpoint file exists before pod kill.
- After restart, worker logs show `resumed from checkpoint ...`.

## Demo Artifacts to Capture
- Terminal output from each script.
- `docker ps` evidence of API restart.
- `kubectl get pods` and `kubectl logs` snippets for worker recovery.
- Link to `FAILURE_MODES.md` in submission.
