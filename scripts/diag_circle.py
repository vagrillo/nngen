"""Quick diagnostic: does the coverage criterion + current recipe admit
successful circle tracers? Prints per-net metric breakdown."""
import sys
import time

import torch

torch.set_num_threads(8)
sys.path.insert(0, "/home/ubuntu/src/AI/diffusionNN")

from dnn.controller import BatchedController
from dnn.tasks import get_task

task = get_task("draw_circle")
net = BatchedController(16, seed=1)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
gen = torch.Generator().manual_seed(42)
t0 = time.time()
for step in range(400):
    state = task.reset(16, gen)
    states = [state]
    for _ in range(30 if step < 120 else (60 if step < 240 else 120)):
        state = task.dynamics(state, net(task.observe(state)))
        states.append(state)
    loss = task.train_loss(states)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"step {step}: loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

with torch.no_grad():
    st = task.rollout(net, 16, 10042)
    rep = task.report(st)
    print(f"circle: success {task.success(st).float().mean():.0%} {rep} ({time.time()-t0:.0f}s)", flush=True)
    cov = task._coverage(st)
    d = task._curve_dist(st)[1:].mean(dim=0)
    lr = task._length_ratio(st)
    clo = task._closure(st)
    for i in range(8):
        print(f"  net{i}: d={d[i]:.3f} cov={cov[i]:.3f} lr={lr[i]:.2f} closure={clo[i]:.3f}", flush=True)
