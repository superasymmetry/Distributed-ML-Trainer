import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from kubernetes import client, config
import sqlite3
import json

import timm

app = FastAPI()
DEFAULT_TRAINING_CODE_PATH = Path("worker") / "train_code.txt"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
SAFE_UPLOAD_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def connect_db(PATH="jobs.db"):
    conn = sqlite3.connect(PATH)
    cursor = conn.cursor()
    if conn is None or cursor is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    return conn, cursor

def init_db():
    print("Initializing database...")
    conn, cursor = connect_db()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'queued',
            config TEXT,
            metrics TEXT DEFAULT '{}',
            retries INTEGER DEFAULT 0,
            pod_name TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_training_code(code: str | None) -> str:
    if code and code.strip():
        return code

    if not DEFAULT_TRAINING_CODE_PATH.exists():
        raise HTTPException(status_code=500, detail="Default training code file is missing")

    try:
        return DEFAULT_TRAINING_CODE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read default training code: {exc}",
        ) from exc


class JobConfig(BaseModel):
    model: str
    dataset: str
    epochs: int
    lr: float
    code: str | None = None

def validate_config(job_config: JobConfig):
    errors = []
    model_name = job_config.model.strip() if isinstance(job_config.model, str) else ""
    dataset_name = job_config.dataset.strip() if isinstance(job_config.dataset, str) else ""

    if not model_name:
        errors.append("Model name must be a non-empty string")
    elif not timm.is_model(model_name):
        suggestions = []
        if hasattr(timm, "list_models"):
            suggestions = timm.list_models(model_name + "*")[:3]
        hint = f" Did you mean: {suggestions}?" if suggestions else ""
        errors.append(f"Model '{model_name}' not found in timm library.{hint}")

    if not isinstance(job_config.epochs, int) or job_config.epochs <= 0:
        errors.append("Epochs must be a positive integer")

    if not isinstance(job_config.lr, (int, float)) or job_config.lr <= 0:
        errors.append("Learning rate must be a positive number")

    if not dataset_name:
        errors.append("Dataset name must be a non-empty string")

    return errors  # empty list = valid

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <title>TrainCTL</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script type="module">
        import { createApp } from 'https://unpkg.com/petite-vue?module'
        createApp({
            total: 0, running: 0,
            async poll() {
                const req = await fetch('/api/dashboard_data');
                const jobs = await req.json();
                this.total = jobs.length;
                this.running = jobs.filter(j => j.status_raw === 'running').length;
            }
        }).mount()
    </script>
    
    <body v-scope @vue:mounted="poll(); setInterval(poll, 2000)" class="flex flex-col items-center justify-center h-screen bg-gray-50 font-sans space-y-8">
        <h1 class="text-6xl font-black text-blue-600 tracking-tight">TrainCTL</h1>
        
        <div class="flex gap-6 text-center">
            <div class="p-6 bg-white shadow border rounded-2xl w-40">
                <div class="text-4xl font-bold text-blue-500">{{ running }}</div>
                <div class="text-gray-400 text-xs mt-2 uppercase tracking-widest font-bold">Running</div>
            </div>
            <div class="p-6 bg-white shadow border rounded-2xl w-40">
                <div class="text-4xl font-bold text-gray-700">{{ total }}</div>
                <div class="text-gray-400 text-xs mt-2 uppercase tracking-widest font-bold">Total Jobs</div>
            </div>
        </div>

        <div class="flex gap-4">
            <a href="/dashboard" class="px-8 py-3 bg-gray-800 text-white font-bold rounded-lg shadow hover:bg-gray-900 transition">Dashboard</a>
            <a href="/submit" class="px-8 py-3 bg-blue-600 text-white font-bold rounded-lg shadow hover:bg-blue-700 transition">New Job</a>
            <a href="/manage" class="px-8 py-3 bg-white border text-gray-700 font-bold rounded-lg shadow hover:bg-gray-50 transition">Manage</a>
        </div>
    </body>
    """


@app.post("/jobs")
def submit_job(config: JobConfig):
    errors = validate_config(config)
    if errors:
        raise HTTPException(status_code=400, detail=errors)

    job_id = str(uuid.uuid4())[:8]
    code = resolve_training_code(config.code)
    config_payload = config.model_dump()
    config_payload["model"] = config.model.strip()
    config_payload["dataset"] = config.dataset.strip()
    config_payload["code"] = code
    now = utc_now_iso()

    conn, cursor = connect_db()
    cursor.execute(
        """
        INSERT INTO jobs (id, status, config, metrics, created_at, updated_at)
        VALUES (?, 'queued', ?, '{}', ?, ?)
        """,
        (job_id, json.dumps(config_payload), now, now),
    )
    conn.commit()
    conn.close()
    return {"job_id": job_id}

@app.get("/jobs")
def list_jobs():
    conn, cursor = connect_db()
    cursor.execute("SELECT * FROM jobs")
    jobs = cursor.fetchall()
    conn.close()
    return jobs


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    conn, cursor = connect_db()
    cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    job = cursor.fetchone()
    conn.close()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str = None):
    conn, cursor = connect_db()
    cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()
    return job_id

@app.get("/health")
def health():
    health_status = {"database": "ok", "api": "ok", "kubernetes": {}}
    try:
        conn, cursor = connect_db()
        cursor.execute("SELECT 1")
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
    
    try:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        api_client = client.ApiClient()
        livez_res = api_client.call_api(
            "/livez", "GET", _preload_content=False, _request_timeout=1
        )
        readyz_res = api_client.call_api(
            "/readyz", "GET", _preload_content=False, _request_timeout=1
        )
        health_status["kubernetes"] = {
            "livez": livez_res[1],
            "readyz": readyz_res[1]
        }
    except Exception as e:
        health_status["kubernetes"] = f"error: {str(e)}"

    return health_status

@app.get("/api/dashboard_data")
def dashboard_data():
    conn, cursor = connect_db()
    cursor.execute("SELECT id, status, metrics, created_at FROM jobs ORDER BY created_at DESC")
    jobs = cursor.fetchall()
    conn.close()

    result = []
    for job in jobs:
        try:
            metrics = json.loads(job[2]) if job[2] else {}
        except:
            metrics = {}
            
        loss = metrics.get("loss_history", [])
        last_loss = f"{loss[-1]:.4f}" if loss else "-"
        
        result.append({
            "id": job[0],
            "status": job[1].upper(),
            "status_raw": job[1],
            "epoch": metrics.get("epoch", "-"),
            "last_loss": last_loss,
            "created_at": job[3] or "-"
        })
    return result


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    raw_filename = (file.filename or "").strip()
    filename = os.path.basename(raw_filename)
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")
    if raw_filename != filename or "/" in raw_filename or "\\" in raw_filename:
        raise HTTPException(status_code=400, detail="Filename must not include path separators")
    if filename in {".", ".."} or not SAFE_UPLOAD_FILENAME.fullmatch(filename):
        raise HTTPException(
            status_code=400,
            detail="Filename may contain only letters, numbers, dots, underscores, and hyphens",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / filename

    try:
        content = await file.read()
    finally:
        await file.close()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        destination.write_bytes(content)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store upload: {exc}") from exc

    return {
        "filename": filename,
        "path": str(destination),
        "size_bytes": len(content),
    }
    
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <title>Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script type="module">
        import { createApp } from 'https://unpkg.com/petite-vue?module'
        createApp({
            jobs: [],
            async poll() {
                const req = await fetch('/api/dashboard_data');
                this.jobs = await req.json();
            }
        }).mount()
    </script>
    <body v-scope @vue:mounted="poll(); setInterval(poll, 2000)" class="bg-gray-50 flex justify-center p-10 font-sans min-h-screen">
        <div class="w-full max-w-4xl bg-white p-8 rounded-2xl shadow border border-gray-100 h-fit">
            <div class="flex justify-between items-center mb-6">
                <h1 class="text-3xl font-black text-gray-800">Dashboard</h1>
                <a href="/" class="text-blue-500 font-bold hover:underline">← Home</a>
            </div>
            <table class="w-full text-left border-collapse">
                <tr class="border-b-2 text-gray-500 uppercase text-xs tracking-wider">
                    <th class="py-3 px-2">Job ID</th><th class="py-3 px-2">Status</th><th class="py-3 px-2">Epoch</th><th class="py-3 px-2">Loss</th>
                </tr>
                <tr v-for="job in jobs" class="border-b transition hover:bg-gray-50">
                    <td class="py-3 px-2 font-mono text-sm">{{ job.id }}</td>
                    <td class="py-3 px-2 font-bold text-xs" :class="job.status_raw === 'running' ? 'text-blue-600' : job.status_raw === 'failed' ? 'text-red-500' : 'text-gray-500'">{{ job.status }}</td>
                    <td class="py-3 px-2 text-sm">{{ job.epoch }}</td>
                    <td class="py-3 px-2 font-mono text-sm">{{ job.last_loss }}</td>
                </tr>
            </table>
            <div v-if="jobs.length === 0" class="text-center py-8 text-gray-400 text-sm font-bold">No jobs found</div>
        </div>
    </body>
    """

@app.get("/submit", response_class=HTMLResponse)
def submit_page():
    return """
    <title>New Job</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes slide-in {
            from { transform: translateX(120%); opacity: 0; }
            to   { transform: translateX(0);   opacity: 1; }
        }
        @keyframes slide-out {
            from { transform: translateX(0);   opacity: 1; }
            to   { transform: translateX(120%); opacity: 0; }
        }
        .toast-enter { animation: slide-in 0.3s ease forwards; }
        .toast-exit  { animation: slide-out 0.3s ease forwards; }
    </style>
    <script type="module">
        import { createApp } from 'https://unpkg.com/petite-vue?module'
        createApp({
            model: 'test_efficientnet.r160_in1k', dataset: 'mnist', epochs: 10, lr: 0.01,
            loading: false, toasts: [],
            showToast(msg, isError) {
                const id = Date.now() + Math.random();
                this.toasts.push({ id, msg, isError, exiting: false });
                setTimeout(() => {
                    const t = this.toasts.find(t => t.id === id);
                    if (t) t.exiting = true;
                    setTimeout(() => { this.toasts = this.toasts.filter(t => t.id !== id); }, 300);
                }, isError ? 5000 : 2000);
            },
            showToasts(msgs, isError) {
                msgs.forEach((msg, i) => setTimeout(() => this.showToast(msg, isError), i * 150));
            },
            async submit() {
                this.loading = true;
                try {
                    const req = await fetch('/jobs', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ model: this.model, dataset: this.dataset, epochs: this.epochs, lr: this.lr, code: '' })
                    });
                    if (req.ok) {
                        this.showToast('Job queued successfully!', false);
                        setTimeout(() => window.location.href = '/dashboard', 1200);
                    } else {
                        const data = await req.json().catch(() => ({}));
                        const detail = data?.detail;
                        const msgs = Array.isArray(detail) ? detail : [detail ?? 'Error submitting job.'];
                        this.showToasts(msgs, true);
                    }
                } catch (e) {
                    this.showToast('Network error. Is the API running?', true);
                }
                this.loading = false;
            }
        }).mount()
    </script>

    <body v-scope class="bg-gray-50 flex items-center justify-center h-screen font-sans">

        <!-- Toasts -->
        <div class="fixed top-6 right-6 z-50 flex flex-col gap-2">
            <div v-for="t in toasts" :key="t.id"
                 :class="[t.isError ? 'bg-red-500' : 'bg-green-500', t.exiting ? 'toast-exit' : 'toast-enter']"
                 class="text-white px-5 py-4 rounded-xl shadow-2xl font-bold text-sm max-w-xs flex items-start gap-3">
                <span class="text-lg leading-none">{{ t.isError ? '✕' : '✓' }}</span>
                <span>{{ t.msg }}</span>
            </div>
        </div>

        <div class="bg-white p-8 rounded-2xl shadow border border-gray-100 max-w-sm w-full">
            <div class="flex justify-between items-center mb-6">
                <h1 class="text-2xl font-black text-gray-800">New Job</h1>
                <a href="/" class="text-blue-500 font-bold hover:underline text-sm">← Home</a>
            </div>
            <div class="space-y-4 text-sm font-bold text-gray-600">
                <label class="block">Model <input v-model="model" class="font-normal mt-1 w-full bg-gray-50 border border-gray-200 rounded p-2 focus:ring-2 focus:ring-blue-500 outline-none"></label>
                <label class="block">Dataset <input v-model="dataset" class="font-normal mt-1 w-full bg-gray-50 border border-gray-200 rounded p-2 focus:ring-2 focus:ring-blue-500 outline-none"></label>
                <div class="flex gap-4">
                    <label class="block w-1/2">Epochs <input type="number" v-model="epochs" class="font-normal mt-1 w-full bg-gray-50 border border-gray-200 rounded p-2 focus:ring-2 focus:ring-blue-500 outline-none"></label>
                    <label class="block w-1/2">LR <input type="number" step="0.001" v-model="lr" class="font-normal mt-1 w-full bg-gray-50 border border-gray-200 rounded p-2 focus:ring-2 focus:ring-blue-500 outline-none"></label>
                </div>
                <button @click="submit" :disabled="loading" class="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition mt-4 font-bold">
                    {{ loading ? 'Deploying...' : 'Launch Pod' }}
                </button>
            </div>
        </div>
    </body>
    """

@app.get("/manage", response_class=HTMLResponse)
def manage_page():
    return """
    <title>Manage Jobs & Data</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes slide-in {
            from { transform: translateX(120%); opacity: 0; }
            to   { transform: translateX(0);   opacity: 1; }
        }
        @keyframes slide-out {
            from { transform: translateX(0);   opacity: 1; }
            to   { transform: translateX(120%); opacity: 0; }
        }
        .toast-enter { animation: slide-in 0.3s ease forwards; }
        .toast-exit  { animation: slide-out 0.3s ease forwards; }
    </style>
    <script type="module">
        import { createApp } from 'https://unpkg.com/petite-vue?module'
        
        function FileManager() {
            return {
                uploadStatus: '',
                uploadSelectedFile() {
                    const fileInput = document.getElementById('datasetUpload');
                    const file = fileInput.files[0];
                    if (!file) {
                        this.uploadStatus = '<span class="text-red-500 font-bold text-sm">Please select a file first.</span>';
                        return;
                    }
                    
                    this.uploadStatus = '<span class="text-blue-500 font-bold text-sm">Uploading...</span>';
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    })
                    .then(async response => {
                        const data = await response.json().catch(() => ({}));
                        if (!response.ok) {
                            const detail = data?.detail;
                            const message = Array.isArray(detail) ? detail.join(', ') : (detail || 'Upload failed.');
                            throw new Error(message);
                        }
                        return data;
                    })
                    .then(data => {
                        this.uploadStatus = `<span class="text-green-600 font-bold text-sm">Successfully uploaded ${data.filename}!</span>`;
                        fileInput.value = ''; 
                    })
                    .catch(error => {
                        this.uploadStatus = `<span class="text-red-500 font-bold text-sm">Upload failed: ${error.message}</span>`;
                    });
                }
            }
        }

        createApp({
            FileManager,
            jobs: [],
            newJob: { model: 'test_efficientnet.r160_in1k', dataset: 'mnist', epochs: 10, lr: 0.01 },
            adding: false,
            toasts: [],
            showToast(msg, isError) {
                const id = Date.now() + Math.random();
                this.toasts.push({ id, msg, isError, exiting: false });
                setTimeout(() => {
                    const t = this.toasts.find(t => t.id === id);
                    if (t) t.exiting = true;
                    setTimeout(() => { this.toasts = this.toasts.filter(t => t.id !== id); }, 300);
                }, isError ? 5000 : 2000);
            },
            showToasts(msgs, isError) {
                msgs.forEach((msg, i) => setTimeout(() => this.showToast(msg, isError), i * 150));
            },
            async poll() {
                const req = await fetch('/api/dashboard_data');
                this.jobs = await req.json();
            },
            async submitQuickJob() {
                this.adding = true;
                try {
                    const req = await fetch('/jobs', { 
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model: this.newJob.model, dataset: this.newJob.dataset, 
                            epochs: parseInt(this.newJob.epochs), lr: parseFloat(this.newJob.lr),
                            code: ''
                        })
                    });
                    if (req.ok) {
                        this.showToast('Job queued!', false);
                        this.poll();
                    } else {
                        const data = await req.json().catch(() => ({}));
                        const detail = data?.detail;
                        const msgs = Array.isArray(detail) ? detail : [detail ?? 'Error submitting job.'];
                        this.showToasts(msgs, true);
                    }
                } catch (e) {
                    this.showToast('Network error. Is the API running?', true);
                }
                this.adding = false;
            },
            async deleteJob(jobId) {
                if(!confirm(`Delete job ${jobId}?`)) return;
                await fetch(`/jobs/${jobId}`, { method: 'DELETE' });
                this.poll();
            }
        }).mount()
    </script>
    <body v-scope @vue:mounted="poll()" class="bg-gray-50 flex justify-center p-10 font-sans min-h-screen">

        <!-- Toasts -->
        <div class="fixed top-6 right-6 z-50 flex flex-col gap-2">
            <div v-for="t in toasts" :key="t.id"
                 :class="[t.isError ? 'bg-red-500' : 'bg-green-500', t.exiting ? 'toast-exit' : 'toast-enter']"
                 class="text-white px-5 py-4 rounded-xl shadow-2xl font-bold text-sm max-w-xs flex items-start gap-3">
                <span class="text-lg leading-none">{{ t.isError ? '✕' : '✓' }}</span>
                <span>{{ t.msg }}</span>
            </div>
        </div>

        <div class="w-full max-w-4xl bg-white p-8 rounded-2xl shadow border border-gray-100 h-fit">
            <div class="flex justify-between items-center mb-6">
                <h1 class="text-3xl font-black text-gray-800">Manage Jobs & Data</h1>
                <a href="/" class="text-blue-500 font-bold hover:underline">← Home</a>
            </div>
            
            <!-- Dataset Upload Section -->
            <div class="mb-8 p-4 bg-blue-50 border border-blue-100 rounded-lg" v-scope="FileManager()">
                <h2 class="text-lg font-bold text-blue-900 mb-2">Upload Custom Dataset</h2>
                <div class="flex items-center gap-4">
                    <input type="file" id="datasetUpload" class="block w-full text-sm text-gray-500
                        file:mr-4 file:py-2 file:px-4 file:rounded file:border-0
                        file:text-sm file:font-semibold file:bg-white file:text-blue-700
                        hover:file:bg-gray-50 cursor-pointer bg-transparent border border-blue-200 rounded p-1" />
                    <button @click="uploadSelectedFile" class="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded font-bold shadow transition">
                        Upload
                    </button>
                </div>
                <div class="mt-2" v-html="uploadStatus"></div>
            </div>

            <!-- Job Submission Row -->
            <div class="flex gap-2 mb-6 bg-gray-50 p-3 rounded-lg border text-sm items-center">
                <input v-model="newJob.model" class="border p-2 rounded flex-1 outline-none" placeholder="Model">
                <input v-model="newJob.dataset" class="border p-2 rounded flex-1 outline-none" placeholder="Dataset">
                <input v-model="newJob.epochs" type="number" class="border p-2 rounded w-20 outline-none" placeholder="Epochs">
                <input v-model="newJob.lr" type="number" step="0.001" class="border p-2 rounded w-24 outline-none" placeholder="LR">
                <button @click="submitQuickJob" :disabled="adding" class="bg-gray-800 hover:bg-gray-900 disabled:opacity-50 text-white font-bold py-2 px-4 rounded transition">
                    {{ adding ? '...' : '+ Add Job' }}
                </button>
            </div>

            <!-- Jobs Table -->
            <table class="w-full text-left border-collapse">
                <tr class="border-b-2 text-gray-500 uppercase text-xs tracking-wider">
                    <th class="py-3 px-2">Job ID</th><th class="py-3 px-2">Status</th><th class="py-3 px-2 text-right">Action</th>
                </tr>
                <tr v-for="job in jobs" class="border-b transition hover:bg-gray-50">
                    <td class="py-3 px-2 font-mono text-sm">{{ job.id }}</td>
                    <td class="py-3 px-2 font-bold text-xs" :class="job.status_raw === 'running' ? 'text-blue-600' : job.status_raw === 'failed' ? 'text-red-500' : 'text-gray-500'">{{ job.status }}</td>
                    <td class="py-3 px-2 text-right">
                        <button @click="deleteJob(job.id)" class="text-red-500 hover:text-white font-bold text-xs px-3 py-1 rounded border border-red-200 hover:bg-red-500 transition shadow-sm">Delete</button>
                    </td>
                </tr>
            </table>
            <div v-if="jobs.length === 0" class="text-center py-8 text-gray-400 text-sm font-bold">No jobs found</div>
        </div>
    </body>
    """
