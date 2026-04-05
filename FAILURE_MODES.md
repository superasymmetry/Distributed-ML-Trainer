# Failure Modes

This document defines expected behavior and recovery steps for common failure paths in `TrainCTL`.

## Assumptions
- API is running and reachable at `http://127.0.0.1:8000`.
- `controller` process is running when validating pod lifecycle behavior.
- Docker Desktop and Kubernetes are enabled for chaos checks that require them.

## Failure Matrix
| Failure Mode | Trigger | Expected Behavior | Recovery Path | Verification |
| --- | --- | --- | --- | --- |
| Invalid job payload | `POST /jobs` with invalid model/fields | API returns JSON error (`400` or `422`) with `detail`; no crash | Caller fixes payload and retries | `scripts/chaos/01_bad_input_check.sh` |
| Unknown job ID | `GET /jobs/{id}` for missing ID | API returns `404` JSON (`{"detail":"Job not found"}`) | None required | `curl -sS http://127.0.0.1:8000/jobs/notfound` |
| Invalid metrics JSON row | Corrupt `jobs.metrics` in DB | `GET /api/dashboard_data` still returns `200` and uses safe placeholders (`"-"`) | Job can continue; DB can be repaired separately | Covered by integration tests |
| API process crash | Crash API process inside container | API becomes temporarily unavailable, then recovers via runtime/container restart | Docker restart policy handles recovery when observable; script uses controlled restart fallback when local runtime blocks direct crash signaling | `scripts/chaos/02_api_restart_recovery.sh` |
| Worker pod failure | Pod enters `Failed` or is deleted while job is `running` | Controller transitions job away from stale state and applies retry policy | Controller re-queues up to retry budget, then marks failed | `scripts/chaos/03_worker_pod_kill_recovery.sh` |
| Checkpoint load failure | Corrupt or incompatible checkpoint file | Worker falls back to clean start (`epoch=0`) instead of crashing on load | New checkpoint is written on subsequent epochs | Unit test coverage in `tests/unit/test_worker.py` |
| Mid-run worker restart | Delete worker pod during training after checkpoint exists | New worker resumes from checkpoint (`resumed from checkpoint ...`) | Controller relaunches worker pod | `scripts/chaos/04_checkpoint_resume_check.sh` |

## Alerting and Triage Notes
- Prioritize stuck `running` jobs and repeated retries with no progress.
- Inspect API logs (`/tmp/trainctl-api.log` in CI/local workflows).
- Inspect controller logs and worker pod logs under `logs/job-<id>.log`.

## Recovery Invariants
- API failure paths must return JSON errors (no raw stack traces to clients).
- Job state transitions must be deterministic and bounded by retry limits.
- Training progress must be recoverable from checkpoint artifacts where available.
