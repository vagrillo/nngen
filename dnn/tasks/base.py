"""Common task interface: batched, fully differentiable mini-environments.

Every task drives the same canonical controller (obs 8 -> action 8) and uses
only the first `obs_used` observation dims and `act_used` action dims. States
are dicts of batched tensors; dynamics are torch ops so gradients flow from
loss back through the whole trajectory (used to mass-train corpus networks).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

DT = 1.0 / 30.0
EPISODE_STEPS = 120


@dataclass(frozen=True)
class TaskSpec:
    name: str
    obs_used: int
    act_used: int
    steps: int = EPISODE_STEPS
    threshold: float = 0.0  # `success` semantics are task-specific, this is advisory
    io_desc: str = ""       # short "Inputs: ... Output: ..." fragment for captions


class BaseTask(ABC):
    """A batched mini-environment. All tensors are float32, CPU or GPU."""

    spec: TaskSpec

    @abstractmethod
    def reset(self, batch: int, gen: torch.Generator) -> dict:
        """Initial state dict; all randomness comes from `gen`."""

    @abstractmethod
    def dynamics(self, state: dict, action: torch.Tensor) -> dict:
        """Advance one step. `action` is the controller output (B, OUT_DIM);
        implementations use action[:, :act_used]. Differentiable in state/action.
        The returned dict must carry a 't' float (B,1) incremented by DT."""

    @abstractmethod
    def observe(self, state: dict) -> torch.Tensor:
        """(B, IN_DIM) observation; unused trailing dims are zero."""

    @abstractmethod
    def train_loss(self, states: list[dict], w_pointwise: float = 0.0) -> torch.Tensor:
        """Differentiable surrogate loss over the trajectory (states[0] = init).

        `w_pointwise` is only used by tasks that anneal between a phase-locked
        point-matching objective and their final objective (see draw.py);
        other tasks ignore it.
        """

    @abstractmethod
    def score(self, states: list[dict]) -> torch.Tensor:
        """No-grad scalar score per element (higher = better)."""

    def success(self, states: list[dict]) -> torch.Tensor:
        """No-grad boolean success per element."""
        raise NotImplementedError

    def report(self, states: list[dict]) -> dict[str, float]:
        """Aggregate no-grad metrics for eval tables (means over the batch)."""
        return {}

    def set_difficulty(self, frac: float) -> None:
        """Curriculum hook: frac in [0,1] scales task hardness (e.g. shape size,
        ball speed). Default: constant difficulty. Called by the corpus trainer
        OUTSIDE the compiled rollout (plain Python attribute writes)."""
        return None

    # -- rollout helpers ----------------------------------------------------
    def rollout(self, net, batch: int, seed: int, grad: bool = False) -> list[dict]:
        """Simulate `steps` steps calling `net(obs) -> action`.

        With grad=True the graph is kept (training); otherwise run under
        torch.no_grad (evaluation).
        """
        gen = torch.Generator().manual_seed(seed)
        state = self.reset(batch, gen)
        states = [state]
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            for _ in range(self.spec.steps):
                obs = self.observe(state)
                action = net(obs)
                states.append({**state, "act": action})  # action chosen IN this state
                state = self.dynamics(state, action)
            states.append(state)  # terminal state, no action
        return states[1:]  # drop the bare init duplicate; states[t] = state t with act


def dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance on the last dim: (..., 2) -> (...)."""
    return (a - b).norm(dim=-1)


def u(gen: torch.Generator, *shape: int, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.empty(*shape).uniform_(lo, hi, generator=gen)
