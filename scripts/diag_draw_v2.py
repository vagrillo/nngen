"""Diagnostic: one compiled TaskTrainer batch per draw shape, metrics printed.

Run detached:  nohup python scripts/diag_draw_v2.py > /tmp/diag_draw.log 2>&1 &
"""
import sys
import time

import torch

torch.set_num_threads(6)
sys.path.insert(0, "/home/ubuntu/src/AI/diffusionNN")

from dnn.corpus.train_controllers import TaskTrainer
from dnn.tasks import get_task

SHAPES = ["draw_circle", "draw_triangle", "draw_star", "draw_star3", "draw_hexagon"]
for name in SHAPES:
    task = get_task(name)
    tr = TaskTrainer(task, 16, use_compile=True)
    t0 = time.time()
    flat, scores, success = tr.train_batch(42, 3e-3, 400)
    rep = task.report(task.rollout(tr.net, 16, 10042))
    print(f"{name}: success {success.float().mean():.0%} "
          + " ".join(f"{k}={v:.3f}" for k, v in rep.items())
          + f" ({time.time()-t0:.0f}s)", flush=True)
print("DIAG_DONE", flush=True)
