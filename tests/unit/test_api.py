import json
import sqlite3
from datetime import datetime

import pytest


@pytest.mark.parametrize("path", ["/", "/dashboard", "/submit", "/manage"])
def test_html_pages_render(api_client, path):
    response = api_client.get(path)

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_submit_job_persists_config(api_client, temp_jobs_db):
    payload = {
        "model": "test_efficientnet.r160_in1k",
        "dataset": "mnist",
        "epochs": 5,
        "lr": 0.01,
        "code": "print('ok')",
    }

    response = api_client.post("/jobs", json=payload)

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert len(job_id) == 8

    conn = sqlite3.connect(temp_jobs_db)
    row = conn.execute("SELECT status, config FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "queued"

    persisted_config = json.loads(row[1])
    assert persisted_config["model"] == payload["model"]
    assert persisted_config["epochs"] == payload["epochs"]
    assert persisted_config["code"] == payload["code"]


def test_submit_job_without_code_uses_default_training_code(api_client, main_module, temp_jobs_db, tmp_path):
    default_code_path = tmp_path / "train_code.py"
    default_code_path.write_text("print('default training code')", encoding="utf-8")
    main_module.DEFAULT_TRAINING_CODE_PATH = default_code_path

    payload = {
        "model": "test_efficientnet.r160_in1k",
        "dataset": "mnist",
        "epochs": 3,
        "lr": 0.01,
    }
    response = api_client.post("/jobs", json=payload)

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    conn = sqlite3.connect(temp_jobs_db)
    row = conn.execute("SELECT config FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()

    assert row is not None
    persisted_config = json.loads(row[0])
    assert persisted_config["code"] == "print('default training code')"


def test_submit_job_writes_created_and_updated_timestamps(api_client, temp_jobs_db):
    payload = {
        "model": "test_efficientnet.r160_in1k",
        "dataset": "mnist",
        "epochs": 1,
        "lr": 0.01,
        "code": "pass",
    }
    response = api_client.post("/jobs", json=payload)
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    conn = sqlite3.connect(temp_jobs_db)
    row = conn.execute("SELECT created_at, updated_at FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()

    assert row is not None
    assert row[0] is not None
    assert row[1] is not None
    datetime.fromisoformat(row[0])
    datetime.fromisoformat(row[1])
    assert row[0] == row[1]


def test_submit_job_invalid_payload_returns_json_error(api_client):
    payload = {
        "model": "test_efficientnet.r160_in1k",
        "dataset": "mnist",
        "epochs": "not-an-int",
        "lr": 0.01,
        "code": "pass",
    }

    response = api_client.post("/jobs", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert isinstance(body["detail"], list)


def test_submit_job_rejects_blank_dataset(api_client):
    payload = {
        "model": "test_efficientnet.r160_in1k",
        "dataset": "   ",
        "epochs": 1,
        "lr": 0.01,
        "code": "pass",
    }
    response = api_client.post("/jobs", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == ["Dataset name must be a non-empty string"]


def test_submit_job_rejects_non_positive_learning_rate(api_client):
    payload = {
        "model": "test_efficientnet.r160_in1k",
        "dataset": "mnist",
        "epochs": 1,
        "lr": 0,
        "code": "pass",
    }
    response = api_client.post("/jobs", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == ["Learning rate must be a positive number"]


def test_submit_job_missing_default_training_code_returns_json_error(api_client, main_module, tmp_path):
    main_module.DEFAULT_TRAINING_CODE_PATH = tmp_path / "missing_train_code.py"

    payload = {
        "model": "test_efficientnet.r160_in1k",
        "dataset": "mnist",
        "epochs": 1,
        "lr": 0.01,
    }
    response = api_client.post("/jobs", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Default training code file is missing"


def test_get_job_missing_returns_404(api_client):
    response = api_client.get("/jobs/doesnotexist")

    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_list_jobs_empty_returns_empty_list(api_client):
    response = api_client.get("/jobs")

    assert response.status_code == 200
    assert response.json() == []


def test_list_jobs_row_shape(api_client):
    payload = {
        "model": "test_efficientnet.r160_in1k",
        "dataset": "mnist",
        "epochs": 1,
        "lr": 0.01,
        "code": "pass",
    }
    job_id = api_client.post("/jobs", json=payload).json()["job_id"]

    response = api_client.get("/jobs")
    assert response.status_code == 200

    jobs = response.json()
    assert len(jobs) == 1
    assert isinstance(jobs[0], dict)
    assert jobs[0]["id"] == job_id
    assert jobs[0]["status"] == "queued"
    assert jobs[0]["retries"] == 0
    assert jobs[0]["config"]["model"] == payload["model"]
    assert jobs[0]["metrics"] == {}


def test_delete_job_removes_record(api_client, temp_jobs_db):
    payload = {
        "model": "test_efficientnet.r160_in1k",
        "dataset": "mnist",
        "epochs": 1,
        "lr": 0.1,
        "code": "pass",
    }
    job_id = api_client.post("/jobs", json=payload).json()["job_id"]

    delete_response = api_client.delete(f"/jobs/{job_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"deleted_job_id": job_id}

    conn = sqlite3.connect(temp_jobs_db)
    row = conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
    conn.close()

    assert row is None


def test_delete_job_missing_returns_404(api_client):
    response = api_client.delete("/jobs/doesnotexist")

    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_delete_running_job_returns_409(api_client, temp_jobs_db):
    conn = sqlite3.connect(temp_jobs_db)
    conn.execute(
        """
        INSERT INTO jobs (id, status, config, metrics, retries, pod_name, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run12345",
            "running",
            json.dumps({"model": "test_efficientnet.r160_in1k", "dataset": "mnist", "epochs": 1, "lr": 0.01, "code": "pass"}),
            "{}",
            0,
            "job-run12345",
            "2026-04-05T10:00:00+00:00",
            "2026-04-05T10:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()

    response = api_client.delete("/jobs/run12345")

    assert response.status_code == 409
    assert "Cannot delete a running job" in response.json()["detail"]


def test_dashboard_data_formats_metrics(api_client, temp_jobs_db):
    metrics = {"loss_history": [0.87, 0.63], "epoch": 2}
    conn = sqlite3.connect(temp_jobs_db)
    conn.execute(
        """
        INSERT INTO jobs (id, status, config, metrics, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "job12345",
            "running",
            json.dumps({"model": "test_efficientnet.r160_in1k", "dataset": "mnist", "epochs": 2, "lr": 0.01, "code": "pass"}),
            json.dumps(metrics),
            "2026-04-05T10:00:00Z",
        ),
    )
    conn.commit()
    conn.close()

    response = api_client.get("/api/dashboard_data")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "job12345"
    assert body[0]["status"] == "RUNNING"
    assert body[0]["epoch"] == 2
    assert body[0]["last_loss"] == "0.6300"


def test_dashboard_data_empty_returns_empty_list(api_client):
    response = api_client.get("/api/dashboard_data")

    assert response.status_code == 200
    assert response.json() == []


def test_dashboard_data_sorted_desc_by_created_at(api_client, temp_jobs_db):
    conn = sqlite3.connect(temp_jobs_db)
    row_config = json.dumps(
        {
            "model": "test_efficientnet.r160_in1k",
            "dataset": "mnist",
            "epochs": 1,
            "lr": 0.01,
            "code": "pass",
        }
    )
    conn.execute(
        """
        INSERT INTO jobs (id, status, config, metrics, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("older000", "queued", row_config, "{}", "2026-04-05T09:00:00Z"),
    )
    conn.execute(
        """
        INSERT INTO jobs (id, status, config, metrics, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("newer000", "queued", row_config, "{}", "2026-04-05T10:00:00Z"),
    )
    conn.commit()
    conn.close()

    response = api_client.get("/api/dashboard_data")
    assert response.status_code == 200

    body = response.json()
    assert [body[0]["id"], body[1]["id"]] == ["newer000", "older000"]


def test_health_uses_kube_fallback(main_module, api_client, monkeypatch):
    calls = {"fallback_used": False}

    def fail_incluster():
        raise main_module.config.ConfigException("incluster config missing")

    def mark_fallback_used():
        calls["fallback_used"] = True

    monkeypatch.setattr(main_module.config, "load_incluster_config", fail_incluster)
    monkeypatch.setattr(main_module.config, "load_kube_config", mark_fallback_used)

    response = api_client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["database"] == "ok"
    assert data["api"] == "ok"
    assert data["kubernetes"] == {"livez": 200, "readyz": 200}
    assert calls["fallback_used"] is True


def test_health_returns_503_when_database_unhealthy(main_module, api_client, monkeypatch):
    def fail_connect(_path: str = "jobs.db"):
        raise sqlite3.OperationalError("db down")

    monkeypatch.setattr(main_module, "connect_db", fail_connect)

    response = api_client.get("/health")

    assert response.status_code == 503
    assert response.json()["database"].startswith("error:")


def test_livez_returns_200(api_client):
    response = api_client.get("/livez")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "api_version" in body


def test_readyz_reports_database_health(main_module, api_client, monkeypatch):
    response = api_client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}

    def fail_connect(_path: str = "jobs.db"):
        raise sqlite3.OperationalError("db down")

    monkeypatch.setattr(main_module, "connect_db", fail_connect)
    unhealthy = api_client.get("/readyz")
    assert unhealthy.status_code == 503
    assert unhealthy.json()["status"] == "error"
    assert unhealthy.json()["database"].startswith("error:")


def test_metrics_returns_job_counts(api_client, temp_jobs_db):
    conn = sqlite3.connect(temp_jobs_db)
    rows = [
        ("job-a", "queued"),
        ("job-b", "running"),
        ("job-c", "failed"),
        ("job-d", "complete"),
        ("job-e", "complete"),
    ]
    for job_id, status_value in rows:
        conn.execute(
            "INSERT INTO jobs (id, status, config, metrics, retries, pod_name, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                status_value,
                json.dumps({"model": "test_efficientnet.r160_in1k", "dataset": "mnist", "epochs": 1, "lr": 0.01, "code": "pass"}),
                "{}",
                0,
                None,
                "2026-04-05T10:00:00+00:00",
                "2026-04-05T10:00:00+00:00",
            ),
        )
    conn.commit()
    conn.close()

    response = api_client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["jobs_total"] == 5
    assert body["jobs_by_status"]["queued"] == 1
    assert body["jobs_by_status"]["running"] == 1
    assert body["jobs_by_status"]["failed"] == 1
    assert body["jobs_by_status"]["complete"] == 2
    assert body["uptime_seconds"] >= 0
    assert "api_version" in body


def test_request_id_header_is_included(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.headers["x-request-id"]

    custom = api_client.get("/health", headers={"X-Request-ID": "req-test-123"})
    assert custom.status_code == 200
    assert custom.headers["x-request-id"] == "req-test-123"


def test_submit_job_rate_limiting(main_module, api_client):
    main_module.RATE_LIMIT_ENABLED = True
    main_module.RATE_LIMIT_MAX_REQUESTS = 2
    main_module.RATE_LIMIT_WINDOW_SECONDS = 60
    main_module.RATE_LIMIT_BUCKETS.clear()

    payload = {
        "model": "test_efficientnet.r160_in1k",
        "dataset": "mnist",
        "epochs": 1,
        "lr": 0.01,
        "code": "pass",
    }
    headers = {"X-Forwarded-For": "10.0.0.5"}

    ok_1 = api_client.post("/jobs", json=payload, headers=headers)
    ok_2 = api_client.post("/jobs", json=payload, headers=headers)
    limited = api_client.post("/jobs", json=payload, headers=headers)

    assert ok_1.status_code == 200
    assert ok_2.status_code == 200
    assert limited.status_code == 429
    assert "Rate limit exceeded" in limited.json()["detail"]


def test_upload_dataset_saves_file(api_client, main_module, tmp_path):
    main_module.UPLOAD_DIR = tmp_path / "uploads"

    response = api_client.post(
        "/api/upload",
        files={"file": ("dataset.csv", b"col_a,col_b\n1,2\n", "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "dataset.csv"
    assert body["size_bytes"] == len(b"col_a,col_b\n1,2\n")

    saved_file = main_module.UPLOAD_DIR / "dataset.csv"
    assert saved_file.exists()
    assert saved_file.read_bytes() == b"col_a,col_b\n1,2\n"


def test_upload_dataset_rejects_invalid_filename(api_client):
    response = api_client.post(
        "/api/upload",
        files={"file": ("../dataset.csv", b"1,2\n", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Filename must not include path separators"


def test_upload_dataset_rejects_empty_file(api_client):
    response = api_client.post(
        "/api/upload",
        files={"file": ("dataset.csv", b"", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty"
