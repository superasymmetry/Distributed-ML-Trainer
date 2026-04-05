from datetime import datetime, timezone
import json
import os
from dotenv import load_dotenv
from groq import Groq
import sqlite3

load_dotenv()

TRAIN_CODE_PATH = "worker/train_code.txt"
DB = os.getenv("DB_PATH", "jobs.db")


def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def extract_error(job_id):
    with open(f"logs/job-{job_id}.log") as f:
        logs = f.read()
    return logs

def fix(job_id, old_code):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    error_section = extract_error(job_id)
    prompt = f"""You are an expert Python SRE engineer. A distributed training worker script has \
    failed in production. Your job is to fix it and return ONLY the corrected, complete Python file.
    
    {error_section}
    BROKEN CODE (`{os.path.basename(TRAIN_CODE_PATH)}`):
    ```python
    {old_code}
    ```
    
    INSTRUCTIONS:
    1. Identify the root cause of the failure. If a runtime error is provided, use it directly. \
    If not, reason carefully about what could cause a worker pod to crash or produce bad output.
    2. Fix ALL bugs you find — do not stop at the first one.
    3. Do not change the training logic, hyperparameters, or DB schema.
    4. Do not add new dependencies that are not already imported.
    5. Preserve every existing comment and docstring.
    6. Your response must contain exactly one fenced code block with the complete fixed file. \
    No explanations before or after — just the code block.
    
    COMMON FAILURE PATTERNS TO CHECK:
    - DB writes inside the training loop without exception handling (pod crash = lost progress)
    - Missing `conn.close()` causing SQLite lock contention across pods
    - `torch.load()` called without `map_location` (GPU model loaded on CPU-only pod)
    - Environment variables read outside `__main__` guard (imported as module = wrong values)
    - Uncaught exceptions in the train loop leaving job status stuck as 'running' in DB
    - Integer/float env vars not cast before arithmetic
    
    RESPONSE FORMAT (strictly):
    ```python
    <complete fixed file here>
    ```"""
    completion = client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
    )
    corrections = completion.choices[0].message.content

    with db() as conn:
        conn.execute(
            "UPDATE jobs SET config=?, updated_at=? WHERE id=?",
            (json.dumps({"code": corrections}), datetime.now(timezone.utc).isoformat(), job_id))        