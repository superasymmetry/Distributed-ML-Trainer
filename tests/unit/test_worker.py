import pytest
import sqlite3
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import worker.worker

@pytest.fixture
def mock_db(monkeypatch, tmp_path):
    """Create a temporary SQLite database for testing."""
    db_path = tmp_path / "test_jobs.db"
    
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, status TEXT DEFAULT 'queued',
            config TEXT, metrics TEXT DEFAULT '{}', retries INTEGER DEFAULT 0,
            pod_name TEXT, created_at TEXT, updated_at TEXT
        )
    """)
    conn.execute("INSERT INTO jobs (id, status) VALUES ('test-job-123', 'running')")
    conn.commit()
    conn.close()

    # Crucial: Override the module-level DB variable in worker.py directly
    monkeypatch.setattr(worker.worker, "DB", str(db_path))
    
    return str(db_path)

def test_write_metrics(mock_db):
    metrics_payload = {"loss": 0.5, "epochs": 5, "status": "running"}
    
    # Correct argument order: (dict, id)
    worker.worker.write_metrics(metrics_payload, "test-job-123")
    
    conn = sqlite3.connect(mock_db)
    result = conn.execute("SELECT metrics FROM jobs WHERE id='test-job-123'").fetchone()
    conn.close()
    
    assert json.loads(result[0]) == metrics_payload

def test_basic_train(mock_db):
    worker.worker.train("test-job-123", "test_convnext.r160_in1k", 2, 0.01, "mnist")
    
    conn = sqlite3.connect(mock_db)
    result = conn.execute("SELECT metrics FROM jobs WHERE id='test-job-123'").fetchone()
    conn.close()
    
    stored = json.loads(result[0])
    assert "loss_history" in stored
    assert len(stored["loss_history"]) == 3