## A Machine Learning Engineer's Dream! (AMLED!)

A Machine Learning Engineer's Dream! (AMLED!) is a Kubernetes-backed ML training job scheduler. You submit training jobs through a web UI or REST API, and a controller process picks them up, launches a Kubernetes pod for each job, monitors it, and handles failures. It also automatically self-heals via an LLM-assisted code fixer.

## How it works

There are three main components:

- **API** (`main.py`) — a FastAPI server that takes job submissions, validates them, and writes them to a SQLite database. Also serves the web UI.
- **Controller** (`controller/controller.py`) — a loop that runs every 5 seconds. It reads the database, launches Kubernetes pods for queued jobs, tails their logs, and updates job status as pods finish.
- **Worker** (`worker/worker.py`) — the actual training code that runs inside the pod. Loads a model from `timm`, trains on MNIST, checkpoints after every epoch, and writes metrics back to the database.

When a pod fails, the controller calls `fixworker.py`, which sends the error logs and broken code to a Groq LLM to get a suggested fix, then requeues the job.

---

## Setup

### Prerequisites

- Docker Desktop with Kubernetes enabled (or any Kubernetes cluster)
- Python 3.11+
- `kubectl` configured and pointing at your cluster
- A Groq API key (for the self-healing feature)

### 1. Clone and install dependencies

```bash
cd Distributed-trainer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment variables

In the root folder, make a .env file and get a Groq API key. Paste in the API key.

```
GROQ_API_KEY=api_key
```

See the [Environment Variables](#environment-variables) section below for what each one does.

### 3. Set up Kubernetes storage

The controller and worker share a PersistentVolume for the SQLite database, model checkpoints, and downloaded datasets. Apply the manifests:

```bash
kubectl apply -f k8s/pvc.yaml
```

If you don't have a `k8s/pvc.yaml` yet, here's a minimal one that works with Docker Desktop:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: trainctl-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
```

### 4. Build the worker image

The controller launches pods using a local Docker image. Build it with:

```bash
docker build -t distributed-trainer-worker:latest .
```

### 5. Start the API

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start the controller

In a separate terminal:

```bash
python -m controller.controller
```

The controller will pull from the job queue every 5 seconds and schedule jobs on the Kubernetes cluster accordingly.

---

## Environment Variables

These go in your `.env` file in the project root. The API and controller load them automatically via `python-dotenv`.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes (for self-healing) | — | API key for Groq. Used by `fixworker.py` to suggest fixes when a training pod fails. Get one at console.groq.com. |
| `DB_PATH` | No | `jobs.db` | Path to the SQLite database. Inside pods this should be `/data/jobs.db` (on the PVC). Locally it defaults to the project root. |

Worker pods get their configuration injected as environment variables by the controller. You don't set these yourself — they come from the job config you submit:

| Variable | Description |
|---|---|
| `JOB_ID` | The job's UUID (first 8 chars). Used to name the checkpoint file and update the right DB row. |
| `DB_PATH` | Set to `/data/jobs.db` inside pods so the worker writes to the shared PVC. |
| `MODEL` | A model name from the `timm` library, e.g. `resnet34` or `test_efficientnet.r160_in1k`. |
| `EPOCHS` | Number of training epochs. |
| `LR` | Learning rate. |
| `DATASET` | Dataset name. Currently `mnist` is supported. |

---

## API Reference

The base URL when running locally is `http://localhost:8000`.

---

### `POST /jobs`

Submit a new training job. The API validates your config before accepting it. If anything is wrong, you'll get popups of errors and the pods will not be launched.

**Request body:**

```json
{
  "model": "resnet34",
  "dataset": "mnist",
  "epochs": 10,
  "lr": 0.01
}
```

| Field | Type | Description |
|---|---|---|
| `model` | string | Any model name valid in `timm`. Run `timm.list_models()` to browse options. |
| `dataset` | string | Dataset to train on. Supported values: `mnist`. |
| `epochs` | integer | Must be a positive integer. |
| `lr` | float | Learning rate. Must be a positive number. |

**Success response (`200`):**

```json
{ "job_id": "a3f92c10" }
```

**Validation error (`400`):**

```json
{
  "detail": [
    "Model 'resnet999' not found in timm library.",
    "Epochs must be a positive integer"
  ]
}
```

Errors come back as a list so you get all of them at once, not just the first one.

---

### `GET /jobs`

Returns all jobs as a list of raw database rows.

---

### `GET /jobs/{job_id}`

Gets the status JSON of a single job by ID. **404** if the job doesn't exist.

---

### `DELETE /jobs/{job_id}`

Deletes the job from the database. Does not delete the Kubernetes pod if it's still running — you'd need to do that manually with `kubectl delete pod job-{job_id}`.

---

### `GET /health`

Returns a health check for the API, database, and Kubernetes connectivity.

```json
{
  "database": "ok",
  "api": "ok",
  "kubernetes": {
    "livez": 200,
    "readyz": 200
  }
}
```

---

### Web UI pages

| Path | Purpose |
|---|---|
| `/` | Home page with running/total job counts |
| `/dashboard` | Live job table with status, epoch, and loss. Polls every 2 seconds. |
| `/submit` | Form to submit a new job |
| `/manage` | Here is the main UI for submitting, viewing, and deleting jobs. |

---

## Checkpointing and crash recovery

The worker saves a checkpoint after every epoch to `/data/checkpoints/job-{job_id}.pt` on the shared PVC. The checkpoint includes the model weights, optimizer state, and metrics history.

When a pod crashes, the controller detects the `Failed` pod phase, deletes the pod, and requeues the job (sets status back to `queued`). On the next reconciliation loop, a new pod is launched. Because the checkpoint is on the shared PVC, the new pod finds it and resumes training from the last completed epoch.

This means a crash mid-epoch loses at most one epoch of progress, not the entire run.

---

## Monitoring a running job

While a job is running, the controller tails the pod logs every 5 seconds and writes them to `logs/job-{job_id}.log`. You can also check the status at the `jobs/{job-id}` endpoint in the browser.
