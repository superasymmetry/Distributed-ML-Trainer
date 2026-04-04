import pytest
import sqlite3
import sys
import os

# Add the root directory to sys.path so Python can find main.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

class MockConnection:
    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return self._conn.cursor()

    def commit(self):
        self._conn.commit()

    def close(self):
        pass

@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Overrides the database connection to use an in-memory database for testing."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, status TEXT DEFAULT 'queued',
            config TEXT, metrics TEXT DEFAULT '{}', retries INTEGER DEFAULT 0,
            pod_name TEXT, created_at TEXT, updated_at TEXT
        )
    """)
    conn.commit()

    def mock_connect(*args, **kwargs):
        return MockConnection(conn), conn.cursor()

    # The mock needs to patch where it's used
    monkeypatch.setattr("main.connect_db", mock_connect)
    yield
    conn.close()

@pytest.mark.parametrize("path", ["/", "/dashboard", "/submit", "/manage"])
def test_html_pages(path):
    """Verify all frontend HTML routes render successfully."""
    response = client.get(path)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_job_lifecycle():
    """End-to-end test of job creation, retrieval, listing, and deletion."""
    
    # 1. Submit a new job (includes dummy 'code' to avoid FileNotFoundError reading train_code.txt in test)
    payload = {"model": "resnet34", "dataset": "mnist", "epochs": 5, "lr": 0.01, "code": "pass"}
    res_post = client.post("/jobs", json=payload)
    assert res_post.status_code == 200
    job_id = res_post.json()["job_id"]
    assert len(job_id) == 8 

    # 2. Get the specific job
    res_get = client.get(f"/jobs/{job_id}")
    assert res_get.status_code == 200
    assert res_get.json()[0] == job_id      # API currently returns tuples from sqlite
    assert res_get.json()[1] == "queued"

    # 3. List all jobs
    res_list = client.get("/jobs")
    assert res_list.status_code == 200
    assert len(res_list.json()) == 1

    # 4. Check dashboard formatted data
    res_dash = client.get("/api/dashboard_data")
    assert res_dash.status_code == 200
    assert len(res_dash.json()) == 1
    assert res_dash.json()[0]["id"] == job_id
    assert res_dash.json()[0]["status"] == "QUEUED"

    # 5. Delete the job
    res_del = client.delete(f"/jobs/{job_id}")
    assert res_del.status_code == 200

    # 6. Verify job was deleted (Expect 404)
    res_get_deleted = client.get(f"/jobs/{job_id}")
    assert res_get_deleted.status_code == 404