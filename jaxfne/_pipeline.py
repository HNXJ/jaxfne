"""Internal pure-function layer for the Configuration/Tensor/Model/Signals
pipeline (Phase 0 of the planned 4-object public API consolidation).

Not part of the public API yet -- not exported from ``jaxfne/__init__.py``.
Each function here is a thin, verified wrapper over existing, already-tested
logic (``construct()``, ``simulate()``, ``neuronal_tensor_to_configuration()``,
``neuronal_tensor.load``/``save_neuronal_tensor``). No new simulation, schema,
or compilation logic is introduced in this pass -- the goal is to give the
eventual ``NeuralNetwork``/``NeuralSignal`` object methods a stable internal
call surface to delegate to, without renaming or removing anything public.

Deliberately NOT implemented here:
    - configuration_to_tensor / tensor_to_graph / select_signal -- would
      require fabricating unverified logic against Model/EIGNetwork internals
      not yet read in full.
    - compile_step_fn / scan_network -- audited and confirmed NOT a clean
      extraction. ``Model._simulate_arrays`` is a five-way Python-side
      dispatcher (homeostasis / HDP / edge_list / dense, each further split
      by ablation_mode) that builds a per-branch closure *then* JIT-wraps it;
      ``jax.lax.scan`` itself lives one level deeper inside five different
      ``emitters.py`` kernel functions with different carry shapes (HDP
      carries H_final/w_final/syn_state; homeostasis carries g_bias/r_trace;
      dense/edge_list carry neither). A generic compile_step_fn/scan_network
      pair cannot exist without either picking one canonical execution mode
      or re-implementing this same five-way dispatch -- deferred to Phase 2
      pending that explicit mode decision.

# DEFERRED: JaxFNEConfig deletion (case-2 live-in-tests-only format).
# 21 tests in test_config_schema_v015.py, test_config_runtime_hardening_v028.py,
# test_v021_config_runtime_source_fidelity.py must be migrated to NeuronalTensor
# load path before JaxFNEConfig / config_to_configuration / config_to_simulation /
# config_to_geometry / load_config / validate_config / config_truth_boundary can
# be deleted. Do not touch those tests in Phase 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .core import Configuration, Model, Signals
from .io import json_safe
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


def initialize_dynamic_state(model: Model) -> dict:
    """Return an independent copy of the time-evolving pytree in
    ``model.params`` (confirmed keys: ``emitter`` (IzhikevichParams),
    ``edge_list`` (EdgeList, sparse backend only), ``positions``,
    ``hdp_initial_H``, ``hdp_initial_w``).

    Uses ``tree_map(identity)`` rather than ``dict.copy()``: ``params``
    values are registered JAX pytree dataclasses (e.g. IzhikevichParams), so
    a shallow dict copy would alias the underlying arrays instead of
    producing independent leaves. Does NOT use ``dataclasses.asdict`` --
    that would recursively flatten nested dataclasses into plain dicts and
    break the type contract (``params["emitter"]`` must stay an
    IzhikevichParams instance) for downstream callers like ``construct()``.
    """
    return jax.tree_util.tree_map(lambda x: x, model.params)


def initialize_static_state(model: Model) -> dict:
    """Return an independent copy of the non-evolving metadata in
    ``model.static`` (confirmed keys: ``n_contacts``, ``neuron_metadata``,
    ``neuron_metadata_summary``, ``geometry``, ``recurrent_coupling``).

    A shallow ``dict()`` copy is correct here -- static values are Python
    scalars, numpy arrays, or metadata dicts, never JAX-traced leaves, so
    ``tree_map`` is unnecessary.
    """
    return dict(model.static)


def checkpoint_state(model: Model, path: str | Path) -> Path:
    """Serialize ``model.params`` (dynamic) and ``model.static`` to disk as
    a ``.npz`` array archive + a JSON-safe metadata sidecar.

    Uses ``tree_flatten`` leaves, not a reconstructable treedef -- JAX
    treedef ``str()`` serialization is not reversible. The pytree structure
    is instead recovered by the caller from a freshly built ``Model`` (see
    :func:`restore_state`), so no reconstruction logic lives on disk.
    """
    path = Path(path)
    leaves, treedef = jax.tree_util.tree_flatten(model.params)
    arrays = {str(i): np.asarray(leaf) for i, leaf in enumerate(leaves)}
    np.savez(path.with_suffix(".npz"), **arrays)
    meta = {
        "treedef": str(treedef),  # human-readable only; not used for restore
        "n_leaves": len(leaves),
        "static": json_safe(model.static),
        "schema": "checkpoint_v1",
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def restore_state(path: str | Path) -> tuple[list, dict]:
    """Inverse of :func:`checkpoint_state`. Returns ``(leaves, static)`` --
    raw flattened param leaves and the JSON-safe static dict, NOT a
    reassembled ``Model``.

    The caller is responsible for reassembly via
    ``jax.tree_util.tree_unflatten(treedef, leaves)``, where ``treedef``
    comes from a freshly constructed ``Model`` with matching structure (e.g.
    ``jax.tree_util.tree_structure(fresh_model.params)``) -- never from the
    disk-serialized ``treedef`` string, which is not parseable back into a
    real treedef.
    """
    path = Path(path)
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays_npz:
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["schema"] != "checkpoint_v1":
            raise ValueError(f"Unknown checkpoint schema: {meta['schema']!r}")
        leaves = [jnp.array(arrays_npz[str(i)]) for i in range(meta["n_leaves"])]
    static = meta["static"]
    return leaves, static
