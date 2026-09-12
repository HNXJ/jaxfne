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
from typing import Any, Mapping, NamedTuple

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
    theta_S: jax.Array      # (n_theta,) population controller coordinates
    aux: jax.Array          # (n_aux,) registered-rule auxiliary coordinates


class ContinuationState(NamedTuple):
    """Runtime continuation state for recurrent edge-list simulation.

    ``dynamic`` is the complete kernel carry. ``prng_key`` is the next key in
    a deterministic per-step split sequence; it is deliberately separate from
    ``dynamic`` so the existing six-field ``DynamicState`` contract remains
    unchanged. The carrier preserves the H-state array as an opaque JAX leaf;
    the current scalar kernel is a ``d_H=1`` special case, not a continuation
    shape restriction. ``step_index`` is the global simulation step offset at
    the start of the next segment (alias ``continuation_step_offset``).

    ``delay_state`` is the canonical public name for the finite-delay ring
  buffer :math:`\\mathcal B_t` (legacy alias ``spike_history``). It is
    ``None`` when all edge delays are zero so legacy callers pay no buffer
    cost unless delayed continuation is active.
    """

    dynamic: DynamicState
    prng_key: jax.Array
    step_index: int = 0
    delay_state: jax.Array | None = None


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
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
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
    hdp_params: Mapping[str, Any] | None = None,
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
    from ._hdp_adaptive import expected_h_shape as compute_expected_h_shape, resolve_h_state_locality

    hp = dict(hdp_params or {})
    locality = resolve_h_state_locality(
        {"h_state_locality": h_state_locality, "h_state_dim": h_state_dim, **hp}
    )
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
        locality=locality, n_neurons=n_neurons, h_state_dim=h_state_dim
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
    aux0 = jnp.zeros((0,), dtype=dtype)
    if locality != "population":
        from .hdp_rule import expected_aux_shape, get_hdp_rule, is_registered_hdp_rule

        if is_registered_hdp_rule(hp.get("hdp_rule")):
            descriptor, _ = get_hdp_rule(str(hp["hdp_rule"]))
            aux0 = jnp.zeros(
                expected_aux_shape(
                    descriptor, n_neurons=int(n_neurons), n_edges=int(n_edges)
                ),
                dtype=dtype,
            )
    theta0 = jnp.zeros((0,), dtype=dtype)
    if locality == "population":
        init_theta = hp.get("controller_theta_S_init")
        if init_theta is not None:
            theta0 = jnp.asarray(init_theta, dtype=dtype)
        else:
            theta0 = jnp.ones((2,), dtype=dtype)
    return DynamicState(
        v=emitter.v0.astype(dtype),
        u=emitter.u0.astype(dtype),
        prev_spikes=jnp.zeros((n_neurons,), dtype=dtype),
        syn_state=jnp.zeros((n_edges,), dtype=dtype),
        H=H0,
        w=w0,
        theta_S=theta0,
        aux=aux0,
    )


def _model_edge_delay_host(model: Model) -> np.ndarray:
    from .emitters import _edge_delay_steps_host

    return _edge_delay_steps_host(model.params["edge_list"])


def model_requires_delay_state(model: Model) -> bool:
    """True when any edge carries a positive integer delay."""
    return bool(np.any(_model_edge_delay_host(model) > 0))


def _delay_buffer_shape(model: Model) -> tuple[int, int]:
    delay_host = _model_edge_delay_host(model)
    max_delay = int(np.max(delay_host)) if delay_host.size else 0
    n_neurons = model.params["emitter"].n_neurons
    return max_delay + 1, n_neurons


def validate_continuation_delay_state(
    model: Model,
    state: ContinuationState,
    *,
    continuing: bool,
) -> None:
    """Fail explicitly when delayed continuation lacks ``delay_state``."""
    if not model_requires_delay_state(model):
        return
    if continuing and state.delay_state is None:
        raise ValueError(
            "delayed continuation requires delay_state (canonical B_t; legacy "
            "alias spike_history)"
        )
    if state.delay_state is None:
        return
    bufsize, n_neurons = _delay_buffer_shape(model)
    ds_host = np.asarray(state.delay_state)
    if ds_host.shape != (bufsize, n_neurons):
        raise ValueError(
            f"delay_state must have shape ({bufsize}, {n_neurons}), got "
            f"{ds_host.shape}"
        )


def _continuation_init_dict_from_state(
    state: ContinuationState,
    *,
    include_step_offset: bool = True,
) -> dict[str, jax.Array]:
    init_state = {
        "v": state.dynamic.v,
        "u": state.dynamic.u,
        "prev_spikes": state.dynamic.prev_spikes,
        "syn_state": state.dynamic.syn_state,
        "H_final": state.dynamic.H,
        "w_final": state.dynamic.w,
    }
    if int(state.dynamic.theta_S.shape[0]):
        init_state["theta_S_final"] = state.dynamic.theta_S
    if int(state.dynamic.aux.shape[0]):
        init_state["aux_final"] = state.dynamic.aux
    if include_step_offset:
        init_state["continuation_step_offset"] = jnp.asarray(
            state.step_index, dtype=jnp.int32
        )
    if state.delay_state is not None:
        init_state["delay_state"] = state.delay_state
    return init_state


def compile_step_fn(
    model: Model,
    *,
    dt_ms: float,
    kernel: str = "hdp",
    record_dH_components: bool = False,
    record_edge_current: bool = False,
    record_current_trace: bool = False,
    record_u_trace: bool = False,
    record_weight_trace: bool = True,
    **hdp_kwargs: Any,
) -> "tuple[callable, ContinuationState]":
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

    use_delays = model_requires_delay_state(model)

    def step_fn(state: ContinuationState, xs_t: tuple) -> "tuple[ContinuationState, tuple]":
        sched_t, key_t, t_idx = xs_t
        init_state = _continuation_init_dict_from_state(
            state, include_step_offset=not use_delays
        )
        kernel_kw = dict(hdp_kwargs)
        if use_delays:
            kernel_kw["step_indices"] = jnp.reshape(t_idx, (1,))
        # Single owner for the weight-trace toggle on this path: hdp_kwargs
        # wins when present, else the named default. The same value drives
        # the kernel call and the output arity, so record_weight_trace=False
        # neither crashes (registered diag carries w_trace=None) nor stacks
        # silently (legacy kernel default would otherwise stay True).
        rwt = bool(kernel_kw.get("record_weight_trace", record_weight_trace))
        if kernel == "hdp":
            from .hdp_rule import is_registered_hdp_rule

            merged_init = {
                **init_state,
                "H_final": state.dynamic.H,
                "w_final": state.dynamic.w,
            }
            if is_registered_hdp_rule(kernel_kw.get("hdp_rule")):
                from ._hdp_registrable_kernel import (
                    simulate_edge_recurrent_izhikevich_hdp_registered,
                )

                _, _, sources, diag = simulate_edge_recurrent_izhikevich_hdp_registered(
                    emitter,
                    edges,
                    n_steps=1,
                    dt_ms=dt_ms,
                    key=key_t,
                    dtype=str(emitter.v0.dtype),
                    drive_schedule=sched_t[None, :],
                    silence_mask=silence_mask,
                    init_state=merged_init,
                    hdp_rule=str(kernel_kw["hdp_rule"]),
                    hdp_rule_params=kernel_kw.get("hdp_rule_params", {}),
                    record_weight_trace=rwt,
                    step_indices=kernel_kw.get("step_indices"),
                )
            else:
                _, _, sources, diag = simulate_edge_recurrent_izhikevich_hdp(
                    emitter, edges, n_steps=1, dt_ms=dt_ms, key=key_t,
                    dtype=str(emitter.v0.dtype),
                    drive_schedule=sched_t[None, :],
                    silence_mask=silence_mask,
                    init_state=merged_init,
                    record_dH_components=record_dH_components,
                    record_edge_current=record_edge_current,
                    record_weight_trace=rwt,
                    **kernel_kw,
                )
        else:
            _, _, sources, diag = simulate_edge_recurrent_izhikevich(
                emitter, edges, n_steps=1, dt_ms=dt_ms, key=key_t,
                dtype=str(emitter.v0.dtype),
                drive_schedule=sched_t[None, :],
                silence_mask=silence_mask,
                init_state=init_state,
                record_edge_current=record_edge_current,
                record_current_trace=record_current_trace,
                record_u_trace=record_u_trace,
                **kernel_kw,
            )
        new_dynamic = DynamicState(
            v=diag["v"], u=diag["u"],
            prev_spikes=diag["prev_spikes"], syn_state=diag["syn_state"],
            H=diag.get("H_final", state.dynamic.H),
            w=diag.get("w_final", state.dynamic.w),
            theta_S=diag.get("theta_S_final", state.dynamic.theta_S),
            aux=diag.get("aux_final", state.dynamic.aux),
        )
        delay_out = state.delay_state
        if use_delays:
            delay_out = diag.get("delay_state", diag.get("spike_history"))
        step_offset = state.step_index + 1
        if not use_delays:
            pass
        elif "continuation_step_offset" in diag:
            step_offset = diag["continuation_step_offset"]
        new_state = state._replace(
            dynamic=new_dynamic,
            step_index=step_offset,
            delay_state=delay_out,
        )
        # Matches the verified per-step output tuple at the real scan call
        # site (emitters.py line ~1368): (v_reset, spikes, source_proxy,
        # H_final, w_next) -- w_next slot dropped when record_weight_trace=False
        # (see this function's docstring: avoids scan_network stacking a
        # (n_outer_steps, n_edges) weight history by default at scale).
        if kernel == "hdp":
            H_trace_t = diag["H_trace"][0]
            w_trace_t = diag["w_trace"][0] if rwt else state.dynamic.w
        else:
            H_trace_t = state.dynamic.H
            w_trace_t = state.dynamic.w
        if rwt:
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
        if record_edge_current and "edge_current_trace" in diag:
            outputs = outputs + (diag["edge_current_trace"][0],)
        if record_current_trace and "current_trace" in diag:
            outputs = outputs + (diag["current_trace"][0],)
        if record_u_trace and "u_trace" in diag:
            outputs = outputs + (diag["u_trace"][0],)
        return new_state, outputs

    init = continuation_state_from_model(
        model,
        h_state_dim=int(hdp_kwargs.get("h_state_dim", 1)),
        hdp_params=hdp_kwargs,
        h_state_locality=hdp_kwargs.get("h_state_locality"),
    )
    return jax.jit(step_fn), init


def scan_network(
    step_fn: "callable",
    init: "DynamicState | ContinuationState",
    drive_schedule: jax.Array,
    keys: jax.Array,
) -> "tuple[DynamicState | ContinuationState, tuple]":
    """Thin, pure wrapper over ``jax.lax.scan`` -- no branching, no
    Python-side dispatch, no kwargs. All static config was captured at
    ``compile_step_fn`` time.

    ``keys`` is a ``(n_steps, 2)`` array of per-step PRNGKeys (e.g.
    ``jax.random.split(key, n_steps)``), NOT a pre-sampled noise array --
    see :func:`compile_step_fn`'s docstring for why: the wrapped kernel
    generates its own Gaussian noise internally from a key per call.

    Accepts either a legacy :class:`DynamicState` (HDP-only callers) or a
    full :class:`ContinuationState` (canonical runtime carry).
    """
    if isinstance(init, ContinuationState):
        state0 = init
        start = state0.step_index
    else:
        state0 = ContinuationState(dynamic=init, prng_key=keys[0])
        start = 0
    n_steps = int(drive_schedule.shape[0])
    t_indices = jnp.arange(n_steps, dtype=jnp.int32) + jnp.asarray(
        start, dtype=jnp.int32
    )
    xs = (drive_schedule, keys, t_indices)
    final_state, outputs = jax.lax.scan(step_fn, state0, xs)
    if isinstance(init, ContinuationState):
        return final_state, outputs
    return final_state.dynamic, outputs


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
    hdp_params: Mapping[str, Any] | None = None,
    h_state_locality: str | None = None,
) -> ContinuationState:
    """Create a cold-start continuation state without running a simulation."""
    dynamic = dynamic_state_from_model(
        model,
        h_state_dim=h_state_dim,
        h_state_locality=h_state_locality,
        hdp_params=hdp_params,
    )
    delay_state = None
    if model_requires_delay_state(model):
        bufsize, n_neurons = _delay_buffer_shape(model)
        delay_state = jnp.zeros(
            (bufsize, n_neurons), dtype=model.params["emitter"].v0.dtype
        )
    return ContinuationState(
        dynamic=dynamic,
        prng_key=jax.random.PRNGKey(int(seed)),
        step_index=int(step_index),
        delay_state=delay_state,
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
    start = state.step_index
    final_state, outputs = scan_network(step_fn, state, schedule, keys)
    if isinstance(start, int):
        next_index = start + int(schedule.shape[0])
    else:
        next_index = int(np.asarray(jax.device_get(start))) + int(schedule.shape[0])
    return (
        final_state._replace(
            prng_key=next_key,
            step_index=next_index,
        ),
        outputs,
    )


def run_continuation_strided(
    step_fn: "callable",
    state: ContinuationState,
    drive_schedule: jax.Array,
    *,
    stride: int,
) -> "tuple[ContinuationState, tuple, jax.Array]":
    """Bounded-memory decimated capture over the continuation path (23-REC-01).

    Runs the schedule in segments of ``stride`` steps through
    :func:`run_continuation` (same ``step_fn`` and carried per-step PRNG
    sequence, so draws are identical to an uninterrupted run) and keeps the
    final frame of each segment. Transient memory is O(stride) per segment
    instead of O(T) for one scan.

    Returns ``(final_state, kept_outputs, kept_indices)`` where
    ``kept_outputs[j]`` equals the uninterrupted run's frame at global step
    ``kept_indices[j]`` exactly, and ``final_state`` equals the
    uninterrupted final state (dynamic leaves, ``prng_key``,
    ``step_index``, ``delay_state``).

    ``stride=1`` reproduces :func:`run_continuation` frames exactly but with
    per-segment launch overhead — prefer :func:`run_continuation` then.
    """
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"stride must be a positive integer; got {stride!r}")
    schedule = jnp.asarray(drive_schedule)
    if schedule.ndim != 2:
        raise ValueError(
            "drive_schedule must have shape (n_steps, n_neurons)"
        )
    total = int(schedule.shape[0])
    if total == 0:
        raise ValueError("drive_schedule must have n_steps > 0")
    kept_parts: list[tuple] | None = None
    indices: list[int] = []
    cur = state
    for start in range(0, total, stride):
        seg = schedule[start:start + stride]
        cur, seg_out = run_continuation(step_fn, cur, seg)
        frame = jax.tree_util.tree_map(lambda o: o[-1:], seg_out)
        kept_parts = frame if kept_parts is None else jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=0), kept_parts, frame
        )
        indices.append(start + int(seg.shape[0]) - 1)
    assert kept_parts is not None
    return cur, kept_parts, jnp.asarray(indices, dtype=jnp.int32)


def memory_report(
    model: Model,
    simulation: Any | None = None,
    recorder: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Component memory preflight for one simulation run (23-REC-01, W17.2).

    Pure shape/dtype arithmetic — executes nothing. Estimates persistent
    (``model.params``), dynamic carry, delay ring, recording stacks, and
    transient draws for ``simulation`` (defaults like :class:`Simulation`
    when omitted) with decimation ``stride`` from ``recorder``
    (``{"stride": k}``, default 1).

    Recording model: per kept frame one T×N each of V/spikes/sources
    (sources iff ``record_sources`` or ``record_fields``, since field
    projection consumes kernel sources), plus H/w/aux traces iff HDP is
    engaged (``w`` iff ``record_weight_trace``). ``"advice"`` names the
    dominant component and the cheapest flag/stride that shrinks it.
    Byte counts are host-side ``nbytes`` of the stacked arrays.
    """
    from .hdp_rule import hdp_params_are_identity

    rec = dict(recorder or {})
    stride = rec.get("stride", 1)
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"recorder stride must be a positive integer; got {stride!r}")
    if simulation is None:
        from ._signals import Simulation as _Simulation

        simulation = _Simulation()
    n_steps = int(simulation.n_steps)
    runtime_cfg = simulation.resolved_runtime
    try:
        itemsize = int(jnp.dtype(runtime_cfg.jnp_dtype).itemsize)
    except Exception:  # noqa: BLE001 - unknown dtype policy falls back to float32
        itemsize = 4

    emitter = model.params["emitter"]
    n_neurons = int(emitter.n_neurons)
    edges = model.params.get("edge_list")
    n_edges = int(edges.n_edges) if edges is not None else 0

    def _leaves_bytes(tree: Any) -> int:
        total = 0
        for leaf in jax.tree_util.tree_leaves(tree):
            try:
                arr = np.asarray(leaf)
                total += int(arr.nbytes)
            except Exception:  # noqa: BLE001 - non-array metadata contributes 0
                pass
        return total

    persistent = _leaves_bytes(model.params)
    dynamic = _leaves_bytes(
        dynamic_state_from_model(
            model,
            h_state_dim=int((runtime_cfg.hdp_params or {}).get("h_state_dim", 1)),
            h_state_locality=(runtime_cfg.hdp_params or {}).get("h_state_locality"),
            hdp_params=dict(runtime_cfg.hdp_params or {}),
        )
    )
    delay = 0
    if edges is not None and model_requires_delay_state(model):
        _, buf_n = _delay_buffer_shape(model)
        delay = int(_model_edge_delay_host(model).max() + 1) * int(buf_n) * itemsize

    kept = (n_steps + stride - 1) // stride
    hp = dict(runtime_cfg.hdp_params or {})
    use_hdp = bool(getattr(runtime_cfg, "enable_hdp", False)) and not hdp_params_are_identity(hp)
    recording: dict[str, int] = {
        "V_m": kept * n_neurons * itemsize,
        "spikes": kept * n_neurons * itemsize,
    }
    if bool(getattr(simulation, "record_sources", True)) or bool(
        getattr(simulation, "record_fields", True)
    ):
        recording["sources"] = kept * n_neurons * itemsize
    if use_hdp:
        h_dim = int(hp.get("h_state_dim", 1))
        recording["H_trace"] = kept * n_neurons * max(h_dim, 1) * itemsize
        if bool(hp.get("record_weight_trace", True)):
            recording["w_trace"] = kept * n_edges * itemsize
    transient = {
        "bulk_noise": n_steps * n_neurons * itemsize,
        "drive_schedule": n_steps * n_neurons * itemsize,
    }
    components = {
        "persistent": persistent,
        "dynamic": dynamic,
        "delay": delay,
        **{f"recording.{k}": v for k, v in recording.items()},
        **{f"transient.{k}": v for k, v in transient.items()},
    }
    recording_total = int(sum(recording.values()))
    total = int(persistent + dynamic + delay + recording_total)
    dominant = max(
        (("persistent", persistent), ("dynamic", dynamic), ("delay", delay),
         ("recording", recording_total)),
        key=lambda kv: kv[1],
    )[0]
    advice = []
    if recording_total and dominant == "recording":
        if recording.get("w_trace"):
            advice.append("disable record_weight_trace (largest T×E term)")
        advice.append(f"raise recorder stride above {stride}")
        if recording.get("sources") and not bool(getattr(simulation, "record_fields", True)):
            advice.append("set record_sources=False when no field/readout needs sources")
    if delay and dominant == "delay":
        advice.append("shorten max edge delay or split the run into continued segments")
    return {
        "n_steps": n_steps,
        "kept_frames": kept,
        "stride": stride,
        "dtype_itemsize": itemsize,
        "n_neurons": n_neurons,
        "n_edges": n_edges,
        "components": components,
        "recording_total": recording_total,
        "total": total,
        "dominant": dominant,
        "advice": advice,
    }
