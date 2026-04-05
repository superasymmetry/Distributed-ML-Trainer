import json
import sqlite3

import httpx
import pytest

# Ensure FastAPI server is running on localhost:8000 before running integration tests.
BASE_URL = "http://localhost:8000"
DB_PATH = "jobs.db"
VALID_MODEL = "test_efficientnet.r160_in1k"


@pytest.fixture
def mock_job_id():
    job_id = None
    payload = {
        "model": VALID_MODEL,
        "dataset": "test_data",
        "epochs": 1,
        "lr": 0.01,
        "code": "pass",
    }
    try:
        response = httpx.post(f"{BASE_URL}/jobs", json=payload)
        assert response.status_code == 200, f"Failed to create integration job: {response.text}"
        job_id = response.json()["job_id"]
        yield job_id
    finally:
        if job_id:
            httpx.delete(f"{BASE_URL}/jobs/{job_id}")


def test_api_health_integration():
    """Check that the API is running and the database connection works."""
    try:
        response = httpx.get(f"{BASE_URL}/health")
    except httpx.ConnectError:
        pytest.fail("FastAPI server is not running! Please start it on localhost:8000")

    assert response.status_code == 200
    data = response.json()
    assert data["database"] == "ok"
    assert data["api"] == "ok"
    assert "kubernetes" in data


def test_submit_job_integration():
    """POST /jobs -> Verify it saves to the real SQLite file."""
    payload = {
        "model": VALID_MODEL,
        "dataset": "mnist",
        "epochs": 1,
        "lr": 0.02,
        "code": "print('integration test')",
    }

    response = httpx.post(f"{BASE_URL}/jobs", json=payload)
    assert response.status_code == 200

    job_id = response.json()["job_id"]
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT config FROM jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()

    assert row is not None
    assert json.loads(row[0])["model"] == VALID_MODEL

    httpx.delete(f"{BASE_URL}/jobs/{job_id}")


def test_list_jobs_integration(mock_job_id):
    """GET /jobs -> Verify we can list all jobs (at least our mock job)."""
    response = httpx.get(f"{BASE_URL}/jobs")
    assert response.status_code == 200

    jobs = response.json()
    assert isinstance(jobs, list)
    job_ids = [job[0] for job in jobs]
    assert mock_job_id in job_ids


def test_list_jobs_row_shape_integration(mock_job_id):
    """GET /jobs -> Verify response row shape is stable."""
    response = httpx.get(f"{BASE_URL}/jobs")
    assert response.status_code == 200

    jobs = response.json()
    row = next((job for job in jobs if job[0] == mock_job_id), None)
    assert row is not None
    assert len(row) == 8
    assert row[1] == "queued"


def test_get_job_integration(mock_job_id):
    """GET /jobs/{job_id} -> Verify we can retrieve a specific job."""
    response = httpx.get(f"{BASE_URL}/jobs/{mock_job_id}")
    assert response.status_code == 200

    job_data = response.json()
    assert job_data[0] == mock_job_id
    assert job_data[1] == "queued"


def test_get_job_not_found_integration():
    """GET /jobs/{job_id} -> Verify a fake ID returns a 404."""
    response = httpx.get(f"{BASE_URL}/jobs/fakeid999")
    assert response.status_code == 404


def test_delete_job_integration():
    """DELETE /jobs/{job_id} -> Verify we can delete a job from the real database."""
    payload = {"model": VALID_MODEL, "dataset": "test", "epochs": 1, "lr": 0.1, "code": ""}
    create_response = httpx.post(f"{BASE_URL}/jobs", json=payload)
    assert create_response.status_code == 200, create_response.text
    job_id = create_response.json()["job_id"]

    delete_response = httpx.delete(f"{BASE_URL}/jobs/{job_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == job_id

    get_response = httpx.get(f"{BASE_URL}/jobs/{job_id}")
    assert get_response.status_code == 404


def test_dashboard_data_integration(mock_job_id):
    """GET /api/dashboard_data -> Verify JSON formatting for dashboard logic."""
    response = httpx.get(f"{BASE_URL}/api/dashboard_data")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)

    mock_job_entry = next((j for j in data if j["id"] == mock_job_id), None)
    assert mock_job_entry is not None
    assert mock_job_entry["status"] == "QUEUED"
    assert "last_loss" in mock_job_entry
    assert "epoch" in mock_job_entry
    assert sorted(mock_job_entry.keys()) == [
        "created_at",
        "epoch",
        "id",
        "last_loss",
        "status",
        "status_raw",
    ]


def test_dashboard_html_integration():
    """GET /dashboard -> Verify the HTML page renders without server 500s."""
    response = httpx.get(f"{BASE_URL}/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>Dashboard</title>" in response.text


def test_submit_job_malformed_json_integration():
    """POST /jobs with malformed JSON should return a clean validation error."""
    response = httpx.post(
        f"{BASE_URL}/jobs",
        content='{"model":"test_efficientnet.r160_in1k",',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_submit_job_missing_required_fields_integration():
    """POST /jobs with missing required fields should return 422 JSON."""
    payload = {
        "model": VALID_MODEL,
        "epochs": 1,
        "lr": 0.01,
        "code": "pass",
    }
    response = httpx.post(f"{BASE_URL}/jobs", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_submit_job_invalid_model_rejected_integration():
    """POST /jobs with an invalid model should be rejected with a client error."""
    payload = {
        "model": "invalid_model_name_for_training",
        "dataset": "mnist",
        "epochs": 1,
        "lr": 0.01,
        "code": "pass",
    }
    response = httpx.post(f"{BASE_URL}/jobs", json=payload)

    assert response.status_code == 400
    assert "detail" in response.json()


def test_dashboard_data_handles_invalid_metrics_json_integration():
    """Dashboard data endpoint should gracefully handle invalid metrics blobs."""
    job_id = "badjson01"
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO jobs (id, status, config, metrics, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            job_id,
            "running",
            json.dumps(
                {
                    "model": VALID_MODEL,
                    "dataset": "mnist",
                    "epochs": 1,
                    "lr": 0.01,
                    "code": "pass",
                }
            ),
            "{bad-json}",
            "2026-04-05T10:00:00Z",
        ),
    )
    conn.commit()
    conn.close()

    try:
        response = httpx.get(f"{BASE_URL}/api/dashboard_data")
        assert response.status_code == 200
        data = response.json()
        entry = next((item for item in data if item["id"] == job_id), None)
        assert entry is not None
        assert entry["last_loss"] == "-"
        assert entry["epoch"] == "-"
    finally:
        cleanup = sqlite3.connect(DB_PATH)
        cleanup.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        cleanup.commit()
        cleanup.close()
