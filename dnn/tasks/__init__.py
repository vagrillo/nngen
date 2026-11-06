"""Task registry: every trainable controller task, keyed by name."""

from __future__ import annotations

from .balance import BalanceTask
from .base import BaseTask, TaskSpec
from .bounce import BounceTask
from .draw import DrawTask, HELD_OUT_SHAPES, SHAPE_NAMES
from .evade import EvadeTask
from .gait import GaitTask
from .orbit import OrbitTask
from .seek import SeekTask

_TASK_CLASSES = {
    "seek": SeekTask,
    "evade": EvadeTask,
    "balance": BalanceTask,
    "bounce": BounceTask,
    "gait": GaitTask,
    "orbit": OrbitTask,
}
_DRAW_SHAPES = SHAPE_NAMES
_DRAW_HELD_OUT = HELD_OUT_SHAPES


def build_registry() -> dict[str, BaseTask]:
    reg: dict[str, BaseTask] = {name: cls() for name, cls in _TASK_CLASSES.items()}
    for shape in _DRAW_SHAPES + _DRAW_HELD_OUT:
        task = DrawTask(shape)
        reg[task.spec.name] = task
    return reg


TASKS: dict[str, BaseTask] = build_registry()


def get_task(name: str) -> BaseTask:
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {task_names()}")
    return TASKS[name]


def task_names() -> list[str]:
    return list(TASKS)


__all__ = ["BaseTask", "TaskSpec", "TASKS", "get_task", "task_names"]
