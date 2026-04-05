import json
import sqlite3
from datetime import datetime

import pytest


class FakeTensor:
    def to(self, _device):
        return self


class FakeLoss:
    def __init__(self, value: float):
        self._value = value

    def backward(self):
        return None

    def item(self):
        return self._value


class FakeOptimizer:
    def __init__(self):
        self.loaded_state = None

    def zero_grad(self):
        return None

    def step(self):
        return None

    def state_dict(self):
        return {"optim": 1}

    def load_state_dict(self, state):
        self.loaded_state = state


class FakeModel:
    def __init__(self):
        self.loaded_state = None

    def to(self, _device):
        return self

    def parameters(self):
        return []

    def train(self):
        return None

    def __call__(self, _batch):
        return object()

    def state_dict(self):
        return {"weights": 1}

    def load_state_dict(self, state):
        self.loaded_state = state


def test_write_metrics_updates_row(worker_module, temp_jobs_db):
    conn = sqlite3.connect(temp_jobs_db)
    conn.execute("INSERT INTO jobs (id, status) VALUES (?, ?)", ("job-1", "running"))
    conn.commit()
    conn.close()

    payload = {"loss": 0.123, "status": "running"}
    worker_module.write_metrics(payload, "job-1")

    conn = sqlite3.connect(temp_jobs_db)
    row = conn.execute("SELECT metrics, updated_at FROM jobs WHERE id = ?", ("job-1",)).fetchone()
    conn.close()

    assert row is not None
    assert json.loads(row[0]) == payload
    assert row[1] is not None
    datetime.fromisoformat(row[1])


def test_save_checkpoint_writes_expected_payload(worker_module, monkeypatch, tmp_path):
    worker_module.CHECKPOINT_DIR = tmp_path / "checkpoints"

    captured = {}

    def fake_save(payload, path):
        captured["payload"] = payload
        captured["path"] = path

    monkeypatch.setattr(worker_module.torch, "save", fake_save)

    worker_module.save_checkpoint("job-save", 2, FakeModel(), FakeOptimizer(), {"epoch": 2})

    assert captured["payload"]["epoch"] == 2
    assert captured["path"] == worker_module.CHECKPOINT_DIR / "job-job-save.pt"
    assert worker_module.CHECKPOINT_DIR.exists()


def test_load_checkpoint_returns_defaults_when_missing(worker_module, tmp_path):
    worker_module.CHECKPOINT_DIR = tmp_path / "checkpoints"

    epoch, metrics = worker_module.load_checkpoint("job-missing", FakeModel(), FakeOptimizer())

    assert epoch == 0
    assert metrics == {"loss_history": [], "epoch": 0, "status": "training"}


def test_load_checkpoint_restores_model_and_optimizer(worker_module, monkeypatch, tmp_path):
    checkpoint = {
        "epoch": 3,
        "metrics": {"loss_history": [0.8, 0.6, 0.5], "epoch": 3, "status": "training"},
        "model_state_dict": {"weights": 7},
        "optimizer_state_dict": {"optim": 9},
    }

    worker_module.CHECKPOINT_DIR = tmp_path
    checkpoint_file = tmp_path / "job-job-restore.pt"
    checkpoint_file.write_bytes(b"stub")

    captured = {}

    def fake_load(path, map_location=None):
        captured["path"] = path
        captured["map_location"] = map_location
        return checkpoint

    monkeypatch.setattr(worker_module.torch, "load", fake_load)

    model = FakeModel()
    optimizer = FakeOptimizer()
    epoch, metrics = worker_module.load_checkpoint("job-restore", model, optimizer)

    assert epoch == 3
    assert metrics == checkpoint["metrics"]
    assert model.loaded_state == checkpoint["model_state_dict"]
    assert optimizer.loaded_state == checkpoint["optimizer_state_dict"]
    assert captured["path"] == checkpoint_file
    assert captured["map_location"] == "cpu"


def test_load_checkpoint_falls_back_to_defaults_on_error(worker_module, monkeypatch, tmp_path):
    worker_module.CHECKPOINT_DIR = tmp_path
    (tmp_path / "job-job-bad.pt").write_bytes(b"stub")

    def failing_load(_path, map_location=None):
        _ = map_location
        raise RuntimeError("corrupt checkpoint")

    monkeypatch.setattr(worker_module.torch, "load", failing_load)

    epoch, metrics = worker_module.load_checkpoint("job-bad", FakeModel(), FakeOptimizer())

    assert epoch == 0
    assert metrics == {"loss_history": [], "epoch": 0, "status": "training"}


def test_train_uses_timm_model_when_path_is_missing(worker_module, monkeypatch):
    captured = {"called": False}

    def fake_create_model(_name, pretrained=True):
        captured["called"] = pretrained
        return FakeModel()

    monkeypatch.setattr(worker_module.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(worker_module.timm, "create_model", fake_create_model)
    monkeypatch.setattr(worker_module, "load_checkpoint", lambda *_args, **_kwargs: (0, {}))
    monkeypatch.setattr(worker_module, "save_checkpoint", lambda *_args, **_kwargs: None)

    final_metrics = {}

    def capture_metrics(metrics, _job_id):
        final_metrics.update(metrics)

    monkeypatch.setattr(worker_module, "write_metrics", capture_metrics)
    monkeypatch.setattr(worker_module.datasets, "FakeData", lambda **_kwargs: object())
    monkeypatch.setattr(worker_module, "DataLoader", lambda *_args, **_kwargs: [(FakeTensor(), FakeTensor())])
    monkeypatch.setattr(worker_module.optim, "Adam", lambda *_args, **_kwargs: FakeOptimizer())
    monkeypatch.setattr(worker_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        worker_module.torch.nn.functional,
        "cross_entropy",
        lambda *_args, **_kwargs: FakeLoss(0.42),
    )

    worker_module.train("job-train", "missing-model", 0, 0.01, "mnist")

    assert captured["called"] is True
    assert final_metrics["status"] == "complete"
    assert final_metrics["epoch"] == 0
    assert final_metrics["loss_history"] == [0.42]


def test_train_loads_local_model_path_with_cpu_map_location(worker_module, monkeypatch):
    model = FakeModel()
    captured = {}

    monkeypatch.setattr(worker_module.os.path, "exists", lambda path: path == "local-model.pt")

    def fake_torch_load(path, map_location=None):
        captured["path"] = path
        captured["map_location"] = map_location
        return model

    monkeypatch.setattr(worker_module.torch, "load", fake_torch_load)
    monkeypatch.setattr(worker_module, "load_checkpoint", lambda *_args, **_kwargs: (0, {}))
    monkeypatch.setattr(worker_module, "save_checkpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_module, "write_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_module.datasets, "FakeData", lambda **_kwargs: object())
    monkeypatch.setattr(worker_module, "DataLoader", lambda *_args, **_kwargs: [(FakeTensor(), FakeTensor())])
    monkeypatch.setattr(worker_module.optim, "Adam", lambda *_args, **_kwargs: FakeOptimizer())
    monkeypatch.setattr(worker_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        worker_module.torch.nn.functional,
        "cross_entropy",
        lambda *_args, **_kwargs: FakeLoss(0.9),
    )

    worker_module.train("job-local", "local-model.pt", 0, 0.01, "mnist")

    assert captured["path"] == "local-model.pt"
    assert captured["map_location"] == "cpu"


def test_train_marks_failed_and_reraises_on_training_exception(worker_module, monkeypatch):
    monkeypatch.setattr(worker_module.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(worker_module.timm, "create_model", lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(worker_module, "load_checkpoint", lambda *_args, **_kwargs: (0, {"loss_history": [], "epoch": 0, "status": "training"}))
    monkeypatch.setattr(worker_module, "save_checkpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker_module.datasets, "FakeData", lambda **_kwargs: object())
    monkeypatch.setattr(worker_module, "DataLoader", lambda *_args, **_kwargs: [(FakeTensor(), FakeTensor())])
    monkeypatch.setattr(worker_module.optim, "Adam", lambda *_args, **_kwargs: FakeOptimizer())
    monkeypatch.setattr(worker_module.torch.cuda, "is_available", lambda: False)

    def failing_loss(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(worker_module.torch.nn.functional, "cross_entropy", failing_loss)

    captured = {}

    def capture_metrics(metrics, _job_id):
        captured.update(metrics)

    monkeypatch.setattr(worker_module, "write_metrics", capture_metrics)

    with pytest.raises(RuntimeError, match="boom"):
        worker_module.train("job-fail", "missing-model", 0, 0.01, "mnist")

    assert captured["status"] == "failed"
    assert "boom" in captured["error"]
