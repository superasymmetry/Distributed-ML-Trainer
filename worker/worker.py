import json
import os
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path

import timm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from logging_utils import configure_json_logging

DB = os.getenv("DB_PATH", "jobs.db")
MODEL = os.getenv("MODEL", "linear")
EPOCHS = int(os.getenv("EPOCHS", "10"))
LR = float(os.getenv("LR", "0.01"))
DATASET = os.getenv("DATASET", "mnist")
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "/data/checkpoints"))

configure_json_logging(service="worker")
log = logging.getLogger("trainctl.worker")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_metrics(epoch: int = 0) -> dict:
    return {"loss_history": [], "epoch": epoch, "status": "training"}


def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def write_metrics(metrics: dict, job_id: str) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE jobs SET metrics=?, updated_at=? WHERE id=?",
            (json.dumps(metrics), utc_now_iso(), job_id),
        )


def checkpoint_path(job_id: str) -> Path:
    return CHECKPOINT_DIR / f"job-{job_id}.pt"


def save_checkpoint(job_id, epoch, model, optimizer, metrics):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(job_id)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    log.info(
        "Checkpoint saved",
        extra={"event": "checkpoint_saved", "job_id": job_id},
    )


def load_checkpoint(job_id, model, optimizer):
    path = checkpoint_path(job_id)
    if not path.exists():
        return 0, default_metrics()

    try:
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        epoch = int(ckpt.get("epoch", 0))
        metrics = ckpt.get("metrics")
        if not isinstance(metrics, dict):
            metrics = default_metrics(epoch)

        metrics.setdefault("loss_history", [])
        metrics.setdefault("epoch", epoch)
        metrics.setdefault("status", "training")

        log.info(
            "Resumed from checkpoint",
            extra={"event": "checkpoint_resumed", "job_id": job_id},
        )
        return epoch, metrics
    except Exception as exc:
        log.warning(
            "Failed to load checkpoint; using defaults",
            extra={"event": "checkpoint_load_failed", "job_id": job_id},
        )
        return 0, default_metrics()


def train(job_id, model_name, epochs, lr, dataset_name):
    if os.path.exists(model_name):
        log.info(
            "Loading existing model path",
            extra={"event": "model_load_local", "job_id": job_id},
        )
        model = torch.load(model_name, map_location="cpu")
    else:
        log.info(
            "Loading model from timm",
            extra={"event": "model_load_timm", "job_id": job_id},
        )
        model = timm.create_model(model_name, pretrained=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_epoch, metrics = load_checkpoint(job_id, model, optimizer)
    if not isinstance(metrics, dict):
        metrics = default_metrics(start_epoch)

    metrics.setdefault("loss_history", [])
    metrics.setdefault("epoch", start_epoch)
    metrics["status"] = "training"

    log.info(
        "Training started",
        extra={"event": "training_started", "job_id": job_id},
    )

    train_dataset = datasets.FakeData(
        size=128,
        image_size=(3, 32, 32),
        num_classes=10,
        transform=transforms.ToTensor(),
    )

    try:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            total_loss = 0.0

            for batch_idx, (data, target) in enumerate(
                DataLoader(train_dataset, batch_size=8, num_workers=0, shuffle=True)
            ):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / (batch_idx + 1)
            metrics["loss_history"].append(round(avg_loss, 4))
            metrics["epoch"] = epoch

            save_checkpoint(job_id, epoch, model, optimizer, metrics)
            log.info(
                "Epoch complete",
                extra={"event": "epoch_complete", "job_id": job_id},
            )

        metrics["status"] = "complete"
        write_metrics(metrics, job_id)
        log.info("Job complete", extra={"event": "job_complete", "job_id": job_id})
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["error"] = str(exc)
        write_metrics(metrics, job_id)
        log.error("Training failed", extra={"event": "job_failed", "job_id": job_id})
        raise


if __name__ == "__main__":
    job_id = os.getenv("JOB_ID")
    if not job_id:
        log.error("No JOB_ID provided", extra={"event": "missing_job_id"})
        raise SystemExit(1)

    try:
        train(job_id, MODEL, EPOCHS, LR, DATASET)
    except Exception as exc:
        log.error("Worker exiting after failure", extra={"event": "worker_exit_failed", "job_id": job_id})
        raise SystemExit(1)
