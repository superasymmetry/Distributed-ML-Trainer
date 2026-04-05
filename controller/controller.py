"""
Controller: runs every few seconds and reconciles desired vs actual state.
- Schedules queued jobs as K8s pods
- Tracks running pod states and transitions jobs deterministically
- Retries failed/missing pods up to a configured limit
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from kubernetes import client, config as k8s_config

from logging_utils import configure_json_logging
from worker.fixworker import fix

DB_PATH = os.getenv("DB_PATH", "jobs.db")
WORKER_IMAGE = os.getenv("WORKER_IMAGE", "distributed-trainer-worker:latest")
NAMESPACE = os.getenv("K8S_NAMESPACE", "default")
IMAGE_PULL_POLICY = os.getenv("K8S_IMAGE_PULL_POLICY", "IfNotPresent")
POLL_INTERVAL_SECONDS = int(os.getenv("CONTROLLER_POLL_SECONDS", "5"))
MAX_JOB_RETRIES = int(os.getenv("MAX_JOB_RETRIES", "2"))
TRAIN_CODE_PATH = Path("worker") / "train_code.txt"
LOG_DIR = Path("logs")

configure_json_logging(service="controller")
log = logging.getLogger("trainctl.controller")


def _init_core_client() -> client.CoreV1Api:
    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()
    return client.CoreV1Api()


core = _init_core_client()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def read_training_code() -> str:
    try:
        return TRAIN_CODE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning(
            "Unable to read training code",
            extra={"event": "training_code_read_error"},
        )
        return ""


def write_pod_log(job_id: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"job-{job_id}.log"

    try:
        pod_logs = core.read_namespaced_pod_log(name=f"job-{job_id}", namespace=NAMESPACE)
    except Exception as exc:
        pod_logs = f"unable to read pod logs: {exc}"

    log_path.write_text(pod_logs, encoding="utf-8")


# launch a pod
def launch_pod(job_id, image, config: dict):
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(name=f"job-{job_id}"),
        spec=client.V1PodSpec(
            restart_policy="Never",
            volumes=[
                client.V1Volume(
                    name="jobdata",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name="trainctl-data"
                    ),
                ),
                client.V1Volume(
                    name="dshm",
                    empty_dir=client.V1EmptyDirVolumeSource(medium="Memory", size_limit="2Gi"),
                ),
            ],
            containers=[
                client.V1Container(
                    name="worker",
                    image=image,
                    image_pull_policy=IMAGE_PULL_POLICY,
                    volume_mounts=[
                        client.V1VolumeMount(
                            name="jobdata",
                            mount_path="/data",
                        )
                    ],
                    env=[
                        client.V1EnvVar("JOB_ID", job_id),
                        client.V1EnvVar("DB_PATH", "/data/jobs.db"),
                        client.V1EnvVar("MODEL", config.get("model", "test_efficientnet.r160_in1k")),
                        client.V1EnvVar("EPOCHS", str(config.get("epochs", 10))),
                        client.V1EnvVar("LR", str(config.get("lr", 0.01))),
                        client.V1EnvVar("DATASET", config.get("dataset", "mnist")),
                    ],
                )
            ],
        ),
    )
    core.create_namespaced_pod(NAMESPACE, pod)


# check if it's alive
def pod_phase(job_id):
    try:
        pod = core.read_namespaced_pod(f"job-{job_id}", NAMESPACE)
        return pod.status.phase
    except client.exceptions.ApiException as exc:
        if exc.status == 404:
            return None  # pod is gone
        raise


# delete it
def delete_pod(job_id):
    try:
        core.delete_namespaced_pod(f"job-{job_id}", NAMESPACE)
    except client.exceptions.ApiException:
        pass  # already gone, that's fine


def _mark_running(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute(
        "UPDATE jobs SET status='running', pod_name=?, updated_at=? WHERE id=?",
        (f"job-{job_id}", utc_now_iso(), job_id),
    )


def _mark_complete(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute(
        "UPDATE jobs SET status='complete', updated_at=? WHERE id=?",
        (utc_now_iso(), job_id),
    )


def _mark_failed(conn: sqlite3.Connection, job_id: str, retries: int) -> None:
    conn.execute(
        "UPDATE jobs SET status='failed', retries=?, pod_name=NULL, updated_at=? WHERE id=?",
        (retries, utc_now_iso(), job_id),
    )


def _requeue_or_fail(conn: sqlite3.Connection, job: sqlite3.Row, reason: str) -> None:
    job_id = job["id"]
    retries = int(job["retries"] or 0) + 1

    if retries <= MAX_JOB_RETRIES:
        log.warning(
            "Re-queueing job",
            extra={
                "event": "job_requeued",
                "job_id": job_id,
            },
        )
        conn.execute(
            "UPDATE jobs SET status='queued', retries=?, pod_name=NULL, updated_at=? WHERE id=?",
            (retries, utc_now_iso(), job_id),
        )
        return

    log.error(
        "Marking job as failed after retry budget exhausted",
        extra={"event": "job_failed", "job_id": job_id},
    )
    _mark_failed(conn, job_id, retries)


def _attempt_fix(job_id: str) -> None:
    training_code = read_training_code()
    if not training_code:
        return

    try:
        fix(job_id, training_code)
    except Exception as exc:
        log.error(
            "Auto-fix failed",
            extra={"event": "autofix_failed", "job_id": job_id},
        )


def _handle_queued_job(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    job_id = job["id"]
    try:
        config = json.loads(job["config"] or "{}")
    except json.JSONDecodeError:
        _mark_failed(conn, job_id, int(job["retries"] or 0))
        log.error(
            "Job has invalid config JSON and was marked failed",
            extra={"event": "job_invalid_config", "job_id": job_id},
        )
        return

    launch_pod(job_id, WORKER_IMAGE, config)
    _mark_running(conn, job_id)
    log.info("Pod launched", extra={"event": "pod_launched", "job_id": job_id})


def _handle_running_job(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    job_id = job["id"]
    write_pod_log(job_id)
    phase = pod_phase(job_id)

    if phase == "Succeeded":
        delete_pod(job_id)
        _mark_complete(conn, job_id)
        return

    if phase == "Failed":
        delete_pod(job_id)
        _attempt_fix(job_id)
        _requeue_or_fail(conn, job, reason="pod_failed")
        return

    if phase is None:
        _requeue_or_fail(conn, job, reason="pod_missing")


def process_job(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    if job["status"] == "queued":
        _handle_queued_job(conn, job)
    elif job["status"] == "running":
        _handle_running_job(conn, job)


def run_once() -> None:
    with db() as conn:
        jobs = conn.execute("SELECT * FROM jobs").fetchall()

        for job in jobs:
            try:
                process_job(conn, job)
            except Exception as exc:
                job_id = job["id"]
                log.error(
                    "Error while processing job",
                    extra={"event": "job_process_error", "job_id": job_id},
                )
                _requeue_or_fail(conn, job, reason="controller_exception")

                if job["status"] == "running":
                    _attempt_fix(job_id)


def run() -> None:
    log.info("Controller started", extra={"event": "controller_started"})
    while True:
        try:
            run_once()
        except Exception as exc:
            log.error("Controller loop error", extra={"event": "controller_loop_error"})
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()
