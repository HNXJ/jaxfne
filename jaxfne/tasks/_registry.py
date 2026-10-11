"""Private task registry for the learning lane (not exported from ``jaxfne``).

A task is an environment with ``reset(rng) -> state`` and
``step(state, action) -> (state, reward, done)``, plus an optional
``q_star() -> ndarray[n_states, n_actions]``. The pattern follows
``jaxfne/hdp_rule.py``: a frozen descriptor, a factory, ``register_task``
that validates at registration, and ``get_task`` that refuses an unknown name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

TaskFactory = Callable[[], Any]


@dataclass(frozen=True)
class TaskDescriptor:
    """Declarative metadata for a registrable task."""

    name: str
    n_states: int
    n_actions: int
    gamma: float
    has_q_star: bool


_REGISTRY: dict[str, tuple[TaskDescriptor, TaskFactory]] = {}


def register_task(descriptor: TaskDescriptor, factory: TaskFactory) -> None:
    """Register a task factory by descriptor name; validate at registration."""
    if descriptor.name in _REGISTRY:
        raise ValueError(f"task {descriptor.name!r} is already registered")
    if int(descriptor.n_states) < 1:
        raise ValueError(
            f"task {descriptor.name!r} declares n_states {descriptor.n_states!r}; expected >= 1"
        )
    if int(descriptor.n_actions) < 1:
        raise ValueError(
            f"task {descriptor.name!r} declares n_actions {descriptor.n_actions!r}; expected >= 1"
        )
    if not (0.0 <= float(descriptor.gamma) < 1.0):
        raise ValueError(
            f"task {descriptor.name!r} declares gamma {descriptor.gamma!r}; expected 0 <= gamma < 1"
        )
    _REGISTRY[descriptor.name] = (descriptor, factory)


def get_task(name: str) -> tuple[TaskDescriptor, Any]:
    """Return the descriptor and a fresh task instance for a registered name."""
    try:
        descriptor, factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown registered task {name!r}; known: {sorted(_REGISTRY)}"
        ) from None
    return descriptor, factory()


def list_tasks() -> tuple[str, ...]:
    """Sorted names of the registered tasks."""
    return tuple(sorted(_REGISTRY))
