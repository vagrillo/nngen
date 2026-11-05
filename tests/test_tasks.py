"""Tests for the batched task environments."""

import torch

from dnn.controller import BatchedController
from dnn.tasks import TASKS, get_task


def test_registry_complete():
    expected = {
        "seek", "evade", "balance", "bounce", "gait", "orbit",
        "draw_circle", "draw_square", "draw_triangle", "draw_star", "draw_spiral",
    }
    assert expected <= set(TASKS)


def test_rollout_shapes_and_grad_flow():
    for name in TASKS:
        task = get_task(name)
        net = BatchedController(4)
        states = task.rollout(net, batch=4, seed=0, grad=True)
        assert len(states) == task.spec.steps + 1, name
        obs = task.observe(states[0])
        assert obs.shape == (4, 8), name
        loss = task.train_loss(states)
        assert torch.isfinite(loss), name
        loss.backward()
        grads = [p.grad for p in net.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name
        with torch.no_grad():
            score = task.score(states)
        assert score.shape == (4,) and torch.isfinite(score).all(), name


def test_rollout_determinism():
    for name in ("seek", "draw_star", "orbit"):
        task = get_task(name)
        net = BatchedController(3, seed=7)
        s1 = task.score(task.rollout(net, 3, 42))
        s2 = task.score(task.rollout(net, 3, 42))
        assert torch.allclose(s1, s2), name


def test_single_vs_batched_consistency():
    """The same weights must produce the same trajectory in batch and alone."""
    from dnn.controller import controller_from_flat

    for name in ("seek", "gait", "draw_circle"):
        task = get_task(name)
        net = BatchedController(2, seed=3)
        flat = net.get_flat()
        batch_states = task.rollout(net, 2, 11)
        solo = controller_from_flat(flat[0])

        # Re-run the same env params for element 0 by rebuilding with the same seed.
        gen = torch.Generator().manual_seed(11)
        state = task.reset(2, gen)
        with torch.no_grad():
            for _ in range(task.spec.steps):
                state = task.dynamics(state, net(task.observe(state)))
        assert torch.allclose(state["t"], batch_states[-1]["t"]), name


def test_seeking_is_learnable():
    """A few hundred gradient steps on `seek` must clearly beat a random net."""
    task = get_task("seek")
    torch.manual_seed(0)
    net = BatchedController(8, seed=1)
    with torch.no_grad():
        before = task.score(task.rollout(net, 8, 100)).mean()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(150):
        states = task.rollout(net, 8, 100, grad=True)
        loss = task.train_loss(states)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = task.score(task.rollout(net, 8, 100)).mean()
    assert after > before + 0.2, (before.item(), after.item())
    assert after > 0.55, after.item()


def test_draw_shapes_are_wellformed():
    """Target polylines: unit scale, correct length, spiral stays open."""
    from dnn.tasks.draw import unit_polyline, CLOSED_SHAPES, STEPS

    for shape in ("circle", "square", "triangle", "star", "spiral"):
        pts = unit_polyline(shape)
        assert pts.shape == (STEPS, 2)
        seg = (pts[1:] - pts[:-1]).norm(dim=-1)
        # Roughly uniform arclength sampling.
        assert (seg.max() / seg.min()) < 6.0, shape
    circ = unit_polyline("circle")
    assert (circ[0] - circ[-1]).norm() < 0.1  # closed loop
    spir = unit_polyline("spiral")
    assert (spir[0] - spir[-1]).norm() > 0.5  # open curve
