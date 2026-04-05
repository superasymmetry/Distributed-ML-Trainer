import json
import sqlite3
from pathlib import Path

import pytest


class CoreCapture:
    def __init__(self):
        self.created = []
        self.deleted = []

    def create_namespaced_pod(self, namespace, pod):
        self.created.append((namespace, pod))

    def read_namespaced_pod(self, _name=None, _namespace=None, **_kwargs):
        raise NotImplementedError

    def delete_namespaced_pod(self, name=None, namespace=None, **_kwargs):
        self.deleted.append((name, namespace))

    def read_namespaced_pod_log(self, _name=None, _namespace=None, **_kwargs):
        return "pod logs"


def _create_controller_db(db_path: Path, rows):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY,
            status TEXT,
            config TEXT,
            metrics TEXT,
            retries INTEGER,
            pod_name TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    for row in rows:
        conn.execute(
            "INSERT INTO jobs (id, status, config, metrics, retries, pod_name, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )
    conn.commit()
    conn.close()


def _db_factory(db_path: Path):
    def _db():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    return _db


def test_launch_pod_builds_expected_spec(controller_module, monkeypatch):
    core = CoreCapture()
    monkeypatch.setattr(controller_module, "core", core)

    config = {"model": "resnet", "epochs": 3, "lr": 0.05, "dataset": "mnist"}
    controller_module.launch_pod("abc123", "worker:latest", config)

    assert len(core.created) == 1
    namespace, pod = core.created[0]
    assert namespace == "default"
    assert pod.metadata.name == "job-abc123"

    env = {item.name: item.value for item in pod.spec.containers[0].env}
    assert env["JOB_ID"] == "abc123"
    assert env["MODEL"] == "resnet"
    assert env["EPOCHS"] == "3"
    assert env["LR"] == "0.05"
    assert env["DATASET"] == "mnist"


def test_pod_phase_returns_none_for_missing_pod(controller_module, monkeypatch):
    class MissingPodCore(CoreCapture):
        def read_namespaced_pod(self, _name, _namespace):
            raise controller_module.client.exceptions.ApiException(status=404)

    monkeypatch.setattr(controller_module, "core", MissingPodCore())

    assert controller_module.pod_phase("unknown") is None


def test_delete_pod_ignores_api_errors(controller_module, monkeypatch):
    class FailingDeleteCore(CoreCapture):
        def delete_namespaced_pod(self, _name, _namespace):
            raise controller_module.client.exceptions.ApiException(status=500)

    monkeypatch.setattr(controller_module, "core", FailingDeleteCore())

    # No exception should escape when pod deletion fails.
    controller_module.delete_pod("job-x")


def test_run_transitions_queued_job_to_running(controller_module, monkeypatch, tmp_path):
    db_path = tmp_path / "jobs.db"
    _create_controller_db(
        db_path,
        [
            (
                "queued01",
                "queued",
                json.dumps({"model": "resnet", "epochs": 1, "lr": 0.1, "dataset": "mnist"}),
                "{}",
                0,
                None,
                None,
                None,
            )
        ],
    )

    launched = []

    def fake_launch(job_id, image, config):
        launched.append((job_id, image, config))

    monkeypatch.setattr(controller_module, "db", _db_factory(db_path))
    monkeypatch.setattr(controller_module, "launch_pod", fake_launch)
    monkeypatch.setattr(controller_module.time, "sleep", lambda _seconds: (_ for _ in ()).throw(StopIteration))

    with pytest.raises(StopIteration):
        controller_module.run()

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT status, pod_name FROM jobs WHERE id = 'queued01'").fetchone()
    conn.close()

    assert launched and launched[0][0] == "queued01"
    assert row == ("running", "job-queued01")


def test_run_marks_succeeded_job_complete(controller_module, monkeypatch, tmp_path):
    db_path = tmp_path / "jobs.db"
    _create_controller_db(
        db_path,
        [
            (
                "running01",
                "running",
                json.dumps({"model": "resnet", "epochs": 1, "lr": 0.1, "dataset": "mnist"}),
                "{}",
                0,
                "job-running01",
                None,
                None,
            )
        ],
    )

    deleted = []

    def fake_delete(job_id):
        deleted.append(job_id)

    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir(exist_ok=True)

    core = CoreCapture()
    monkeypatch.setattr(controller_module, "core", core)
    monkeypatch.setattr(controller_module, "db", _db_factory(db_path))
    monkeypatch.setattr(controller_module, "pod_phase", lambda _job_id: "Succeeded")
    monkeypatch.setattr(controller_module, "delete_pod", fake_delete)
    monkeypatch.setattr(controller_module.time, "sleep", lambda _seconds: (_ for _ in ()).throw(StopIteration))

    with pytest.raises(StopIteration):
        controller_module.run()

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT status FROM jobs WHERE id = 'running01'").fetchone()
    conn.close()

    assert row == ("complete",)
    assert deleted == ["running01"]
    assert (tmp_path / "logs" / "job-running01.log").exists()
