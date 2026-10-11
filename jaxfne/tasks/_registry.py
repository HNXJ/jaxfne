"""Private task registry for the learning lane (not exported from ``jaxfne``).

A task is an environment with ``reset(rng) -> state`` and
``step(state, action) -> (state, reward, done)``, plus an optional
``q_star() -> ndarray[n_states, n_actions]``. The pattern follows
``jaxfne/hdp_rule.py``: a frozen descriptor, a factory, ``register_task``
that validates at registration, and ``get_task`` that refuses an unknown name.
"""

from __future__ import annotations

import numbers
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


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def register_task(descriptor: TaskDescriptor, factory: TaskFactory) -> None:
    """Register a task factory by descriptor name; validate at registration."""
    if not isinstance(descriptor.name, str):
        raise ValueError(f"task name must be a str, got {descriptor.name!r}")
    if descriptor.name in _REGISTRY:
        raise ValueError(f"task {descriptor.name!r} is already registered")
    for field in ("n_states", "n_actions"):
        value = getattr(descriptor, field)
        if not _is_int(value) or value < 1:
            raise ValueError(
                f"task {descriptor.name!r} declares {field} {value!r}; expected an int >= 1"
            )
    gamma = descriptor.gamma
    if not _is_real(gamma) or not (0.0 <= float(gamma) < 1.0):
        raise ValueError(
            f"task {descriptor.name!r} declares gamma {gamma!r}; expected a real 0 <= gamma < 1"
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
