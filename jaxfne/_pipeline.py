"""Internal pure-function layer for the Configuration/Tensor/Model/Signals
pipeline (Phase 0 of the planned 4-object public API consolidation).

The public ``ContinuationState`` owner and its pure scan helpers are exported
through ``jaxfne``; the remaining functions are internal wrappers over the
existing object model and tensor helpers. Continuation uses the existing
emitter kernels one step at a time rather than creating a second simulation
engine.

Deliberately NOT implemented here:
    - configuration_to_tensor / tensor_to_graph / select_signal -- would
      require fabricating unverified logic against Model/EIGNetwork internals
      not yet read in full.

Phase 2 (``compile_step_fn`` / ``scan_network``) wraps the canonical HDP
edge-list kernel and the ordinary recurrent edge-list kernel. Homeostasis,
dense, and the Model._simulate_arrays dispatcher remain un-wrapped for
continuation; calling the continuation API on those modes raises explicitly.

# RESOLVED 2026-06-30: JaxFNEConfig (case-2 live-in-tests-only format) and its
# 21 dependent tests (test_config_schema_v015.py, test_config_runtime_hardening_v028.py,
# test_v021_config_runtime_source_fidelity.py) were deleted -- JaxFNEConfig /
# config_to_configuration / config_to_simulation / config_to_geometry / load_config /
# validate_config / config_truth_boundary / config_to_trial_batch no longer exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

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


class DynamicState(NamedTuple):
    """Canonical recurrent edge-list carry tuple.

    The selected kernel owns the meaning of each array. HDP uses all six
    fields dynamically; the baseline recurrent kernel carries the H and w
    fields unchanged. This is verified against the emitter step call sites,
    not inferred from diagnostic output keys.

    Slot order matches the kernel's ``init``/in-carry unpack exactly:
    ``v, u, prev_spikes, syn_state, H, w``. Slot 2 (``prev_spikes``) holds
    spikes from the prior step on entry but is overwritten with *this*
    step's spikes on exit -- the carry is intentionally NOT positionally
    symmetric in/out; that asymmetry is how "previous" rolls forward each
    step, not a bug to fix.
    """

    v: jax.Array            # (n_neurons,)  membrane voltage
    u: jax.Array            # (n_neurons,)  recovery variable
    prev_spikes: jax.Array  # (n_neurons,)  spikes from t-1 on entry
    syn_state: jax.Array    # (n_edges,)    synaptic gating variable
    H: jax.Array            # (n_neurons,) scalar or (n_neurons, d_H) vector H
    w: jax.Array            # (n_edges,)    synaptic weights


class ContinuationState(NamedTuple):
    """Runtime continuation state for recurrent edge-list simulation.

    ``dynamic`` is the complete kernel carry. ``prng_key`` is the next key in
    a deterministic per-step split sequence; it is deliberately separate from
    ``dynamic`` so the existing six-field ``DynamicState`` contract remains
    unchanged. The carrier preserves the H-state array as an opaque JAX leaf;
    the current scalar kernel is a ``d_H=1`` special case, not a continuation
    shape restriction. ``step_index`` is bookkeeping for the caller and is
    not passed into the numerical transition.
    """

    dynamic: DynamicState
    prng_key: jax.Array
    step_index: int = 0


def continuation_noise_schedule(
    key: jax.Array,
    n_steps: int,
    n_neurons: int,
    dtype: jnp.dtype,
) -> jax.Array:
    """Generate the Gaussian draws used by the continuation PRNG contract."""
    _, step_keys = _advance_prng_key(key, n_steps)
    noise_keys = jax.vmap(lambda step_key: jax.random.split(step_key)[1])(step_keys)
    return jax.vmap(
        lambda noise_key: jax.random.normal(
            noise_key, shape=(int(n_neurons),), dtype=dtype
        )
    )(noise_keys)


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


def select_signal(signal: Signals, key: str, **get_kwargs: Any) -> Any:
    """Return a named, optionally-filtered signal array. Pure wrapper over
    ``Signals.get`` -- see that method's docstring for ``key`` aliases and
    the ``selector``/``area``/``layer``/``cell_type``/``ids`` filtering
    fields (``**get_kwargs``)."""
    return signal.get(key, **get_kwargs)


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

    bfloat16 leaves are upcast to float32 before ``np.savez`` and the
    original dtype name is recorded in the JSON sidecar for
    :func:`restore_state` to cast back down. This is required, not a
    defensive nicety -- confirmed 2026-07-01 that plain
    ``np.savez``/``np.load`` silently mangles ml_dtypes' bfloat16 arrays
    into raw void bytes (dtype ``|V2``) on read-back, with no error. The
    float32 upcast is exact (bfloat16 occupies the top 16 bits of float32),
    so this loses no precision.
    """
    path = Path(path)
    leaves, treedef = jax.tree_util.tree_flatten(model.params)
    dtype_names = [np.asarray(leaf).dtype.name for leaf in leaves]
    arrays = {
        str(i): (np.asarray(leaf).astype(np.float32) if name == "bfloat16" else np.asarray(leaf))
        for i, (leaf, name) in enumerate(zip(leaves, dtype_names))
    }
    np.savez(path.with_suffix(".npz"), **arrays)
    meta = {
        "treedef": str(treedef),  # human-readable only; not used for restore
        "n_leaves": len(leaves),
        "leaf_dtypes": dtype_names,
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
        dtype_names = meta.get("leaf_dtypes")
        leaves = []
        for i in range(meta["n_leaves"]):
            raw = arrays_npz[str(i)]
            target_dtype = dtype_names[i] if dtype_names is not None else raw.dtype
            leaves.append(jnp.array(raw, dtype=target_dtype))
    static = meta["static"]
    return leaves, static


def dynamic_state_from_model(
    model: Model,
    *,
    h_state_dim: int = 1,
    h_state_locality: str | None = None,
) -> DynamicState:
    """Build a cold-start :class:`DynamicState` from ``model.params``.

    Mirrors the ``init_state=None`` branch of
    ``simulate_edge_recurrent_izhikevich_hdp`` exactly (verified at the scan
    call site, not inferred): ``H_i(0)=1.0``, ``w(0)`` = native edge weight,
    ``prev_spikes``/``syn_state`` start at zero. Requires
    ``model.params["edge_list"]`` -- the canonical Phase 2 execution mode is
    HDP edge-list; a model built without an edge list (dense-only) cannot
    produce a valid DynamicState.
    """
    from ._hdp_adaptive import expected_h_shape as compute_expected_h_shape, reject_population_continuation, resolve_h_state_locality

    locality = resolve_h_state_locality(
        {"h_state_locality": h_state_locality, "h_state_dim": h_state_dim}
    )
    reject_population_continuation(locality, context="dynamic_state_from_model")
    if isinstance(h_state_dim, bool) or not isinstance(h_state_dim, int) or h_state_dim < 1:
        raise ValueError("h_state_dim must be a positive integer")
    emitter = model.params["emitter"]
    if "edge_list" not in model.params:
        raise ValueError(
            "dynamic_state_from_model requires model.params['edge_list'] -- "
            "the canonical Phase 2 execution mode is HDP edge-list; this "
            "model was not built with a sparse edge list."
        )
    edges = model.params["edge_list"]
    n_neurons = emitter.n_neurons
    n_edges = edges.n_edges
    dtype = emitter.v0.dtype
    expected_h_shape = compute_expected_h_shape(
        locality="node", n_neurons=n_neurons, h_state_dim=h_state_dim
    )
    H0 = model.params.get("hdp_initial_H")
    H0 = (
        jnp.asarray(H0, dtype=dtype)
        if H0 is not None
        else jnp.ones(expected_h_shape, dtype=dtype)
    )
    if H0.shape != expected_h_shape:
        raise ValueError(
            "hdp_initial_H must have shape "
            f"{expected_h_shape} for h_state_dim={h_state_dim}, got {H0.shape}"
        )
    w0 = model.params.get("hdp_initial_w")
    w0 = jnp.asarray(w0, dtype=dtype) if w0 is not None else edges.weight.astype(dtype)
    return DynamicState(
        v=emitter.v0.astype(dtype),
        u=emitter.u0.astype(dtype),
        prev_spikes=jnp.zeros((n_neurons,), dtype=dtype),
        syn_state=jnp.zeros((n_edges,), dtype=dtype),
        H=H0,
        w=w0,
    )


def compile_step_fn(
    model: Model,
    *,
    dt_ms: float,
    kernel: str = "hdp",
    record_dH_components: bool = False,
    record_edge_current: bool = False,
    record_weight_trace: bool = True,
    **hdp_kwargs: Any,
) -> "tuple[callable, DynamicState]":
    """Build a JIT-compiled single-step function over an edge-list carry.

    ``kernel="hdp"`` preserves the original canonical HDP behavior. The
    additive ``kernel="baseline"`` mode uses the ordinary recurrent
    Izhikevich emitter while retaining the same six-field carrier shape; its
    H and w slots are carried unchanged because that kernel has no dynamic HDP
    controller.

    ``record_weight_trace`` (default True, matching prior behavior exactly):
    when False, ``step_fn``'s per-step ``outputs`` tuple drops the per-edge
    weight snapshot (arity 4 instead of 5: v, spikes, sources, H_trace).
    Matters specifically because ``scan_network`` stacks ``outputs`` over
    every OUTER step it's driven with -- with the weight slot present, that
    stack is ``(n_outer_steps, n_edges)``, the same memory-scaling hazard as
    ``simulate_edge_recurrent_izhikevich_hdp``'s own ``w_trace`` (see that
    function's docstring: 10,000 steps x 2,000,000 edges x 4 bytes = 80GB, a
    real reproduced OOM). The inner per-call kernel here always runs at
    n_steps=1, so its own trace is negligible regardless -- this flag only
    controls whether the OUTER ``scan_network`` accumulates a weight history
    across repeated calls. ``carry.w`` (the actual weight state driving HDP's
    plasticity) is unaffected either way -- disabling the trace never
    disables HDP itself, only the optional per-outer-step weight diagnostic.

    DEVIATION FROM SPEC, surfaced explicitly rather than papered over:
    ``simulate_edge_recurrent_izhikevich_hdp``'s inner ``step`` closure
    (emitters.py line ~1298) is a local closure, not exported -- confirmed
    via ``grep -n 'def step' jaxfne/emitters.py``, which shows it nested
    inside the kernel function, not module-level. Per instruction, that
    closure is NOT extracted by monkey-patching (Option A rejected).
    Instead (Option B) this wraps the full kernel with ``n_steps=1`` per
    call and unwraps the length-1 trace.

    A consequence of Option B: the kernel generates its Gaussian noise
    INTERNALLY from a ``jax.Array`` PRNG ``key`` (via
    ``jax.random.split``/``normal`` inside the kernel) -- it has no
    parameter that accepts a pre-generated noise array. So the returned
    ``step_fn``'s second ``xs_t`` element is a **per-step PRNGKey**, not a
    pre-sampled noise array: ``xs_t = (sched_t, key_t)``, not
    ``(sched_t, noise_t)``. ``scan_network`` must be driven with a
    ``(n_steps, ...)`` array of keys (e.g. ``jax.random.split(key, n_steps)``),
    not a noise array -- documented at the call site there too.

    All HDP hyperparameters (``H_min``, ``K_HDP``, ``rho_passive``, etc, via
    ``**hdp_kwargs``) and the two record flags are captured by Python
    closure, not passed through ``lax.scan``'s carry/xs -- they are static
    for the lifetime of this compiled ``step_fn``; changing them requires
    calling ``compile_step_fn`` again (a new compile), exactly like
    ``Model._simulate_arrays``'s own recompilation-on-static-change behavior.
    """
    if kernel not in {"hdp", "baseline"}:
        raise ValueError("kernel must be 'hdp' or 'baseline'")

    emitter = model.params["emitter"]
    edges = model.params["edge_list"]
    n_neurons = emitter.n_neurons
    silence_mask = jnp.ones((n_neurons,), dtype=emitter.v0.dtype)

    from .emitters import (
        simulate_edge_recurrent_izhikevich,
        simulate_edge_recurrent_izhikevich_hdp,
    )

    def step_fn(carry: DynamicState, xs_t: tuple) -> "tuple[DynamicState, tuple]":
        sched_t, key_t = xs_t
        init_state = {
            "v": carry.v, "u": carry.u,
            "prev_spikes": carry.prev_spikes, "syn_state": carry.syn_state,
            "H_final": carry.H, "w_final": carry.w,
        }
        if kernel == "hdp":
            _, _, sources, diag = simulate_edge_recurrent_izhikevich_hdp(
                emitter, edges, n_steps=1, dt_ms=dt_ms, key=key_t,
                dtype=str(emitter.v0.dtype),
                drive_schedule=sched_t[None, :],
                silence_mask=silence_mask,
                init_state={
                    **init_state,
                    "H_final": carry.H,
                    "w_final": carry.w,
                },
                record_dH_components=record_dH_components,
                record_edge_current=record_edge_current,
                **hdp_kwargs,
            )
        else:
            _, _, sources, diag = simulate_edge_recurrent_izhikevich(
                emitter, edges, n_steps=1, dt_ms=dt_ms, key=key_t,
                dtype=str(emitter.v0.dtype),
                drive_schedule=sched_t[None, :],
                silence_mask=silence_mask,
                init_state=init_state,
                **hdp_kwargs,
            )
        new_carry = DynamicState(
            v=diag["v"], u=diag["u"],
            prev_spikes=diag["prev_spikes"], syn_state=diag["syn_state"],
            H=diag.get("H_final", carry.H),
            w=diag.get("w_final", carry.w),
        )
        # Matches the verified per-step output tuple at the real scan call
        # site (emitters.py line ~1368): (v_reset, spikes, source_proxy,
        # H_final, w_next) -- w_next slot dropped when record_weight_trace=False
        # (see this function's docstring: avoids scan_network stacking a
        # (n_outer_steps, n_edges) weight history by default at scale).
        if kernel == "hdp":
            H_trace_t = diag["H_trace"][0]
            w_trace_t = diag["w_trace"][0]
        else:
            H_trace_t = carry.H
            w_trace_t = carry.w
        if record_weight_trace:
            outputs = (
                diag["v"], diag["prev_spikes"], sources[0],
                H_trace_t, w_trace_t,
            )
        else:
            outputs = (diag["v"], diag["prev_spikes"], sources[0], H_trace_t)
        if record_dH_components and kernel == "hdp":
            outputs = outputs + (
                diag["dH_income_trace"][0], diag["dH_rate_trace"][0],
                diag["dH_weight_trace"][0], diag["dH_passive_trace"][0],
                diag["dH_barrier_trace"][0],
            )
        if record_edge_current and kernel == "hdp":
            outputs = outputs + (diag["edge_current_trace"][0],)
        return new_carry, outputs

    init = dynamic_state_from_model(
        model,
        h_state_dim=int(hdp_kwargs.get("h_state_dim", 1)),
    )
    return jax.jit(step_fn), init


def scan_network(
    step_fn: "callable",
    init: DynamicState,
    drive_schedule: jax.Array,
    keys: jax.Array,
) -> "tuple[DynamicState, tuple]":
    """Thin, pure wrapper over ``jax.lax.scan`` -- no branching, no
    Python-side dispatch, no kwargs. All static config was captured at
    ``compile_step_fn`` time.

    ``keys`` is a ``(n_steps, 2)`` array of per-step PRNGKeys (e.g.
    ``jax.random.split(key, n_steps)``), NOT a pre-sampled noise array --
    see :func:`compile_step_fn`'s docstring for why: the wrapped kernel
    generates its own Gaussian noise internally from a key per call.
    """
    xs = (drive_schedule, keys)
    final_carry, outputs = jax.lax.scan(step_fn, init, xs)
    return DynamicState(*final_carry), outputs


def _advance_prng_key(
    key: jax.Array,
    n_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """Generate stable per-step keys and return the next continuation key."""
    def split_once(carry, _):
        next_key, step_key = jax.random.split(carry)
        return next_key, step_key

    return jax.lax.scan(
        split_once,
        key,
        jnp.zeros((int(n_steps),), dtype=jnp.int32),
    )


def continuation_state_from_model(
    model: Model,
    *,
    seed: int = 0,
    step_index: int = 0,
    h_state_dim: int = 1,
) -> ContinuationState:
    """Create a cold-start continuation state without running a simulation."""
    return ContinuationState(
        dynamic=dynamic_state_from_model(model, h_state_dim=h_state_dim),
        prng_key=jax.random.PRNGKey(int(seed)),
        step_index=int(step_index),
    )


def run_continuation(
    step_fn: "callable",
    state: ContinuationState,
    drive_schedule: jax.Array,
) -> "tuple[ContinuationState, tuple]":
    """Run a segment using the carried per-step PRNG sequence."""
    schedule = jnp.asarray(drive_schedule)
    if schedule.ndim != 2:
        raise ValueError(
            "drive_schedule must have shape (n_steps, n_neurons)"
        )
    next_key, keys = _advance_prng_key(state.prng_key, schedule.shape[0])
    dynamic, outputs = scan_network(step_fn, state.dynamic, schedule, keys)
    return (
        ContinuationState(
            dynamic=dynamic,
            prng_key=next_key,
            step_index=state.step_index + int(schedule.shape[0]),
        ),
        outputs,
    )
