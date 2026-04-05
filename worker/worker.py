import os, sqlite3, json, time, random, math
from datetime import datetime

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

DB = os.getenv("DB_PATH", "jobs.db")
MODEL = os.getenv("MODEL", "linear")
EPOCHS = int(os.getenv("EPOCHS", "10"))
LR = float(os.getenv("LR", "0.01"))
DATASET = os.getenv("DATASET", "mnist")

def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn
 
def write_metrics(metrics: dict, JOB_ID):
    with db() as conn:
        conn.execute(
            "UPDATE jobs SET metrics=?, updated_at=? WHERE id=?",
            (json.dumps(metrics), datetime.utcnow().isoformat(), JOB_ID))
        
def save_checkpoint(job_id, epoch, model, optimizer, metrics):
    os.makedirs("/data/checkpoints", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, f"/data/checkpoints/job-{job_id}.pt")
    print(f"checkpoint saved at epoch {epoch}", flush=True)

def load_checkpoint(job_id, model, optimizer):
    path = f"/data/checkpoints/job-{job_id}.pt"
    if not os.path.exists(path):
        return 0, {"loss_history": [], "epoch": 0, "status": "training"}
    
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"resumed from checkpoint at epoch {ckpt['epoch']}", flush=True)
    return ckpt["epoch"], ckpt["metrics"]

def train(JOB_ID, MODEL, EPOCHS, LR, DATASET):
    if os.path.exists(MODEL):
        print(f"loading existing model from {MODEL}",flush=True)
        model = torch.load(MODEL)
    else:
        print(f"using model from timm library {MODEL}",flush=True)
        model = timm.create_model(MODEL, pretrained=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    start_epoch, metrics = load_checkpoint(JOB_ID, model, optimizer)

    metrics = {"loss_history": [], "epoch": 0, "status": "training"}

    print(f"starting job {JOB_ID} model={MODEL} epochs={EPOCHS} lr={LR}", flush=True)

    train_dataset = datasets.FakeData(size=128, image_size=(3, 32, 32), num_classes=10, transform=transforms.ToTensor())

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(DataLoader(train_dataset, batch_size=8, num_workers=0, shuffle=True)):
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
        save_checkpoint(JOB_ID, epoch, model, optimizer, metrics)
        # write_metrics(metrics, JOB_ID)
        print("saved checkpoint for epoch", epoch, flush=True)
        
        print(f"epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}")

    # try:
    #     with db() as conn:
    #         job = conn.execute("SELECT config FROM jobs WHERE id=?", (JOB_ID,)).fetchone()
    #         job_config = json.loads(job["config"])
    #         custom_train_code = job_config.get("code", "")
    #     print("custom train code is:", custom_train_code)
            
    #     if not custom_train_code:
    #         print("Error: No custom code found in database for this job!")
    #         metrics["status"] = "failed"
    #         write_metrics(metrics, JOB_ID)
    #         return

    #     exec(custom_train_code, globals(), locals())

    # except Exception as e:
    #     import traceback
    #     traceback.print_exc()
    #     metrics["status"] = "failed"
    #     write_metrics(metrics, JOB_ID)
    #     return
 
    metrics["status"] = "complete"
    write_metrics(metrics, JOB_ID)
    print(f"job {JOB_ID} complete")

if __name__ == "__main__":
    job_id = os.getenv("JOB_ID")
    if not job_id:
        print("No JOB_ID provided, exiting.")
        exit(1)
        
    train(job_id, MODEL, EPOCHS, LR, DATASET)

    # train("42fb5cdf", "test_efficientnet.r160_in1k", 10, 0.01, "mnist")