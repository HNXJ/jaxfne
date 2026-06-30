"""Internal pure-function layer for the Configuration/Tensor/Model/Signals
pipeline (Phase 0 of the planned 4-object public API consolidation).

Not part of the public API yet -- not exported from ``jaxfne/__init__.py``.
Each function here is a thin, verified wrapper over existing, already-tested
logic (``construct()``, ``simulate()``, ``neuronal_tensor_to_configuration()``,
``neuronal_tensor.load``/``save_neuronal_tensor``). No new simulation, schema,
or compilation logic is introduced in this pass -- the goal is to give the
eventual ``NeuralNetwork``/``NeuralSignal`` object methods a stable internal
call surface to delegate to, without renaming or removing anything public.

Deliberately NOT implemented here (would require fabricating unverified
logic against the existing ``Model``/``EIGNetwork`` internals):
    - configuration_to_tensor / tensor_to_graph / compile_step_fn /
      scan_network / checkpoint_state / restore_state / select_signal
These stay open for a follow-up pass once ``Model``'s internal state
layout (``params`` vs ``static``) has been read in full.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import Configuration, Model, Signals
from .neuronal_tensor import (
    NeuronalTensor,
    RuntimeConfiguration,
    load as _load_tensor,
    neuronal_tensor_to_configuration,
    save_neuronal_tensor,
)


def load_tensor(path: str | Path) -> NeuronalTensor:
    """Load a NeuronalTensor JSON config. Pure wrapper over
    ``jaxfne.neuronal_tensor.load``."""
    return _load_tensor(path)


def save_tensor(tensor: NeuronalTensor, path: str | Path) -> str:
    """Save a NeuronalTensor JSON config. Pure wrapper over
    ``jaxfne.neuronal_tensor.save_neuronal_tensor``."""
    return save_neuronal_tensor(tensor, path)


def tensor_to_configuration(
    tensor: NeuronalTensor,
    *,
    seed: int = 0,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
    emitter: str = "izhikevich",
) -> Configuration:
    """Bridge a NeuronalTensor into a Configuration. Pure wrapper over
    ``jaxfne.neuronal_tensor.neuronal_tensor_to_configuration``."""
    return neuronal_tensor_to_configuration(
        tensor, seed=seed, duration_ms=duration_ms, dt_ms=dt_ms, emitter=emitter,
    )


def build_network(
    cfg_or_tensor: "Configuration | NeuronalTensor",
    runtime: "RuntimeConfiguration | None" = None,
    **construct_kwargs: Any,
) -> Model:
    """Compile a Configuration or NeuronalTensor into a runnable Model.
    Pure wrapper over ``jaxfne.core.construct`` -- both call forms
    (Configuration-only, or NeuronalTensor+RuntimeConfiguration) pass
    through unchanged."""
    from .core import construct

    if runtime is None:
        return construct(cfg_or_tensor, **construct_kwargs)
    return construct(cfg_or_tensor, runtime, **construct_kwargs)


def run_network(model: Model, **simulate_kwargs: Any) -> Signals:
    """Run a built Model and return its recorded Signals. Pure wrapper over
    ``jaxfne.core.simulate``."""
    from .core import simulate

    return simulate(model, **simulate_kwargs)
