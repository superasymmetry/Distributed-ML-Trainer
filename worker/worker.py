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
 
def fake_loss(epoch: int, lr: float) -> float:
    """Simulate a realistic loss curve with a bit of noise."""
    base = 1.0 * math.exp(-lr * epoch * 3)
    noise = random.gauss(0, 0.01)
    return max(0.01, base + noise)
 
def train(JOB_ID, MODEL, EPOCHS, LR, DATASET):
    if os.path.exists(MODEL):
        print(f"loading existing model from {MODEL}",flush=True)
        model = torch.load(MODEL)
    else:
        print(f"using model from timm library {MODEL}",flush=True)
        model = timm.create_model(MODEL, pretrained=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    metrics = {"loss_history": [], "epoch": 0, "status": "training"}

    optimizer = optim.Adam(model.parameters(), lr=LR)
    print(f"starting job {JOB_ID} model={MODEL} epochs={EPOCHS} lr={LR}", flush=True)
    
    for epoch in range(metrics["epoch"], EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(DataLoader(datasets.FakeData(transform=transforms.ToTensor()), batch_size=32)):
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
        
        # REMOVED: Database UPDATE call that crashes the pod
        # write_metrics(metrics, JOB_ID)
        
        # This print statement is what you want to see!
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