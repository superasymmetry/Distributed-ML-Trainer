import pytest
import sqlite3
import httpx
import json

# Ensure your FastAPI server is running on localhost:8000 before running these tests!
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
        "code": "pass"
    }
    try:
        response = httpx.post(f"{BASE_URL}/jobs", json=payload)
        assert response.status_code == 200, f"Failed to create integration job: {response.text}"
        job_id = response.json()["job_id"]
        yield job_id
    finally:
        # Cleanup: Make sure it's gone from the real DB even if the test fails
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


def test_submit_job_integration():
    """POST /jobs -> Verify it saves to the real SQLite file."""
    payload = {
        "model": "test_efficientnet.r160_in1k",
        "dataset": "mnist",
        "epochs": 1,
        "lr": 0.02,
        "code": "print('integration test')"
    }
    
    response = httpx.post(f"{BASE_URL}/jobs", json=payload)
    assert response.status_code == 200
    
    # Prove it's on the hard drive
    job_id = response.json()["job_id"]
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT config FROM jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()

    assert row is not None
    assert json.loads(row[0])["model"] == "test_efficientnet.r160_in1k"
    
    # manual cleanup for this specific test
    httpx.delete(f"{BASE_URL}/jobs/{job_id}")


def test_list_jobs_integration(mock_job_id):
    """GET /jobs -> Verify we can list all jobs (at least our mock job)."""
    response = httpx.get(f"{BASE_URL}/jobs")
    assert response.status_code == 200
    
    jobs = response.json()
    assert isinstance(jobs, list)
    # The database returns tuples by default: e.g. ["abc12345", "queued", "{...}", ...]
    job_ids = [job[0] for job in jobs]
    assert mock_job_id in job_ids


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
    # Create one first
    payload = {"model": VALID_MODEL, "dataset": "test", "epochs": 1, "lr": 0.1, "code": ""}
    create_response = httpx.post(f"{BASE_URL}/jobs", json=payload)
    assert create_response.status_code == 200, create_response.text
    job_id = create_response.json()["job_id"]
    
    # Delete it
    delete_response = httpx.delete(f"{BASE_URL}/jobs/{job_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == job_id
    
    # Prove it's missing from the database
    get_response = httpx.get(f"{BASE_URL}/jobs/{job_id}")
    assert get_response.status_code == 404


def test_dashboard_data_integration(mock_job_id):
    """GET /api/dashboard_data -> Verify JSON formatting for the dashboard logic."""
    response = httpx.get(f"{BASE_URL}/api/dashboard_data")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
    
    # Find our mock job in the formatted list and check the keys
    mock_job_entry = next((j for j in data if j["id"] == mock_job_id), None)
    assert mock_job_entry is not None
    assert mock_job_entry["status"] == "QUEUED"
    assert "last_loss" in mock_job_entry
    assert "epoch" in mock_job_entry


def test_dashboard_html_integration():
    """GET /dashboard -> Verify the HTML page renders without server 500s."""
    response = httpx.get(f"{BASE_URL}/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>Dashboard</title>" in response.text
