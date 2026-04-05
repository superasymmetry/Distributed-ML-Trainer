import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import timm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DB = os.getenv("DB_PATH", "jobs.db")
MODEL = os.getenv("MODEL", "linear")
EPOCHS = int(os.getenv("EPOCHS", "10"))
LR = float(os.getenv("LR", "0.01"))
DATASET = os.getenv("DATASET", "mnist")
CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "/data/checkpoints"))


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
    print(f"checkpoint saved at epoch {epoch}", flush=True)


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

        print(f"resumed from checkpoint at epoch {epoch}", flush=True)
        return epoch, metrics
    except Exception as exc:
        print(f"failed to load checkpoint from {path}: {exc}", flush=True)
        return 0, default_metrics()


def train(job_id, model_name, epochs, lr, dataset_name):
    if os.path.exists(model_name):
        print(f"loading existing model from {model_name}", flush=True)
        model = torch.load(model_name, map_location="cpu")
    else:
        print(f"using model from timm library {model_name}", flush=True)
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

    print(f"starting job {job_id} model={model_name} epochs={epochs} lr={lr}", flush=True)
    print(f"dataset={dataset_name}", flush=True)

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
                print(" batch", batch_idx, "loss", loss.item(), flush=True)

            avg_loss = total_loss / (batch_idx + 1)
            metrics["loss_history"].append(round(avg_loss, 4))
            metrics["epoch"] = epoch

            print("metrics at epoch", epoch, metrics, flush=True)
            save_checkpoint(job_id, epoch, model, optimizer, metrics)
            print("saved checkpoint for epoch", epoch, flush=True)
            print(f"epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

        metrics["status"] = "complete"
        write_metrics(metrics, job_id)
        print(f"job {job_id} complete")
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["error"] = str(exc)
        write_metrics(metrics, job_id)
        raise


if __name__ == "__main__":
    job_id = os.getenv("JOB_ID")
    if not job_id:
        print("No JOB_ID provided, exiting.")
        raise SystemExit(1)

    try:
        train(job_id, MODEL, EPOCHS, LR, DATASET)
    except Exception as exc:
        print(f"job {job_id} failed: {exc}", flush=True)
        raise SystemExit(1)
