# Demo Runbook (2 Minutes)

This runbook is optimized for the hackathon reliability demo.

## Before Recording
- Docker Desktop is running.
- Kubernetes is enabled if you plan to run pod-chaos scripts (`03`/`04`).
- `kubectl get nodes` shows a ready node (for Kubernetes demos).
- `python -m controller.controller` will run in a second terminal.

For full setup, see [README](https://github.com/superasymmetry/Distributed-ML-Trainer/blob/main/README.md).

## Pre-Demo Setup (Off Camera)
1. Start API and controller:
```bash
fastapi dev main.py
python -m controller.controller
```
2. Build worker image:
```bash
docker build -t distributed-trainer-worker:latest .
```
CPU-first build is default for faster local setup. For CUDA wheels explicitly:
```bash
docker build --build-arg REQUIREMENTS_FILE=requirements-gpu-cu124.txt -t distributed-trainer-worker:latest .
```
3. Confirm health:
```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/readyz
```

## Demo Sequence

### 0:00-0:10 Intro
State the required intro clearly:
`"Hi, this is my demo for the Production Engineering Hackathon."`

### 0:10-0:30 Submit Valid Job (UI)
1. Open `http://127.0.0.1:8000/submit`.
2. Use defaults (`test_efficientnet.r160_in1k`, `mnist`, `epochs=10`, `lr=0.01`).
3. Click **Launch Pod** and show redirect to dashboard.
4. Show status moving from `QUEUED` to `RUNNING`.

### 0:30-0:55 Bad Input -> Clean JSON Error
```bash
scripts/chaos/01_bad_input_check.sh
```
Expected:
- Bad requests return clean JSON `detail` errors.
- API remains healthy.

### 0:55-1:20 Crash API -> Recovery
```bash
scripts/chaos/02_api_restart_recovery.sh
```
Expected:
- API process is force-crashed.
- Service recovers (auto-restart or controlled restart fallback).
- Health endpoint becomes OK again.

### 1:20-1:50 Load Resilience
```bash
REQUESTS=50 CONCURRENCY=10 scripts/chaos/05_load_test.sh
```
Expected:
- 50 concurrent `POST /jobs` requests succeed.
- Script prints p50/p99 latency.
- API remains healthy after load.

### Optional Kubernetes Clip (if available)
```bash
scripts/chaos/03_worker_pod_kill_recovery.sh
scripts/chaos/04_checkpoint_resume_check.sh
```
Expected:
- Worker pod kill triggers clean status transition.
- Checkpoint resume is visible in logs.

## Artifacts to Capture
- Terminal output from each script.
- `docker ps` evidence of API restart.
- `kubectl get pods` and `kubectl logs` (for pod-chaos clips).
- p50/p99 output from load test.
- Link to `FAILURE_MODES.md`.
