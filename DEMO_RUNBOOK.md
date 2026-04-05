## Demo Runbook

This is the script for demonstrating AMLED! end to end, including the error handling and self-healing features.

### Before you start

Make sure all of this is true:
- Docker Desktop is running with Kubernetes enabled
- `kubectl get nodes` shows a Ready node
- The worker image is built
- The PVC exists: `kubectl get pvc trainctl-data`
- The API is running on port 8000
- The controller is running in a separate terminal with `python -m controller.controller`

Open `http://localhost:8000` in a browser.

For more information on how to setup the project, read [the readme](https://github.com/superasymmetry/Distributed-ML-Trainer/blob/main/README.md)
---

### Submit a valid job

1. Click **New Job** from the home page
2. Leave the defaults (`test_efficientnet.r160_in1k`, `mnist`, 10 epochs, lr 0.01) and click **Launch Pod**
3. You'll see a green popup of *"Job queued successfully!"* and get redirected to the dashboard
4. Watch the dashboard — within 5 seconds the status changes from `QUEUED` to `RUNNING`
5. After the first epoch completes, the epoch counter and loss column start updating

---

### Failure demos

Now we will demonstrate how AMLED! recovers upon failure.

Go to `/submit` and try each of these:

**Bad model name:**
```
model: resnet999
```
Expected: red toast saying the model wasn't found, with suggestions if any close matches exist.

**Negative learning rate:**
```
lr: -0.5
```
Expected: red toast saying LR must be positive.

**Multiple errors at once:**
```
model: fakemodel123
epochs: -5
lr: 0
```
Expected: three separate red toasts sliding in one after another, one per error. No pod is ever launched.

---

**Crash recovery**

While a job is running, kill its pod to simulate a crash. For example (I have a pod, api-1, in the distributed-trainer container):

```bash
docker exec distributed-trainer-api-1 kill 1 
```
<img width="1593" height="901" alt="image" src="https://github.com/user-attachments/assets/11f5e387-7988-4cef-bcb7-0484e2b8bd59" />

1. The pod disappears
2. The controller detects the `Failed`/missing pod
3. The job status resets to `QUEUED`
4. The container restarts a pod
5. The worker logs show `"resumed from checkpoint at epoch N"` — it picks up where it left off

This demonstrates that the PVC-backed checkpointing is working correctly.

---

