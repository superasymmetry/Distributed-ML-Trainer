"""
Controller: runs every 5s, reconciles desired vs actual state.
- Schedules queued jobs as K8s pods
- Detects dead/stuck pods and requeues
- Detects loss anomalies and retries with adjusted LR
"""

import logging
import os
import time
from kubernetes import client, config as k8s_config
import json
import sys
from worker.fixworker import fix

import sqlite3
def db():
    conn = sqlite3.connect("jobs.db")
    conn.row_factory = sqlite3.Row
    return conn

k8s_config.load_kube_config()
core = client.CoreV1Api()
print("core namespace", core.list_namespace())
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("controller")

# launch a pod
def launch_pod(job_id, image, config: dict):
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(name=f"job-{job_id}"),
        spec=client.V1PodSpec(
            restart_policy="Never",
            volumes=[client.V1Volume(
                name="jobdata",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name="trainctl-data"
                )
            )],
            containers=[client.V1Container(
                name="worker",
                image=image,
                image_pull_policy="Never",
                volume_mounts=[client.V1VolumeMount(
                    name="jobdata",
                    mount_path="/data"
                )],
                env=[
                    client.V1EnvVar("JOB_ID", job_id),
                    client.V1EnvVar("DB_PATH", "/data/jobs.db"),
                    client.V1EnvVar("MODEL", config.get("model", "resnet18")),
                    client.V1EnvVar("EPOCHS", str(config.get("epochs", 10))),
                    client.V1EnvVar("LR", str(config.get("lr", 0.01))),
                    client.V1EnvVar("DATASET", config.get("dataset", "mnist")),
                ]
            )]
        )
    )
    core.create_namespaced_pod("default", pod)

# check if it's alive
def pod_phase(job_id):
    try:
        pod = core.read_namespaced_pod(f"job-{job_id}", "default")
        return pod.status.phase  # Pending / Running / Succeeded / Failed
    except client.exceptions.ApiException as e:
        if e.status == 404:
            return None  # pod is gone
        raise

# delete it
def delete_pod(job_id):
    try:
        core.delete_namespaced_pod(f"job-{job_id}", "default")
    except client.exceptions.ApiException:
        pass  # already gone, that's fine

def run():
    log.info("Controller started")
    while(True):
        try: 
            # 1. fetch jobs from DB
            # 2. for each job:
            #    if status=queued -> launch pod, set status=running
            #    if status=running -> check pod phase
            #       if Succeeded -> set status=complete
            #       if Failed -> set status=failed, maybe retry
            with db() as conn:
                jobs = conn.execute("SELECT * FROM jobs").fetchall()
                print(f"fetched {len(jobs)} jobs from database")
                for job in jobs:
                    print(f"job {job['id']} status={job['status']}")
                    if job["status"] == "queued":
                        launch_pod(job["id"], "distributed-trainer-worker:latest", json.loads(job["config"]))
                        conn.execute("UPDATE jobs SET status='running', pod_name=? WHERE id=?", (f"job-{job['id']}", job["id"]))
                    elif job["status"] == "running":
                        pod_logs = core.read_namespaced_pod_log(name=f"job-{job['id']}", namespace="default")
                        print(pod_logs)
                        with open(f"logs/job-{job['id']}.log", "w") as f:
                            f.write(pod_logs)

                        phase = pod_phase(job["id"])
                        if phase == "Succeeded":
                            conn.execute("UPDATE jobs SET status='complete' WHERE id=?", (job["id"],))
                            # delete_pod(job["id"])
                        elif phase == "Failed":
                            fix(job["id"])
                            conn.execute("UPDATE jobs SET status='failed' WHERE id=?", (job["id"],))
                            # delete_pod(job["id"])
                        
            
        except Exception as e:
            log.error(f"Error in controller loop: {e}")

        time.sleep(5)

if __name__ == "__main__":
    run()