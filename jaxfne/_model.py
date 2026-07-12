"""Model: the immutable, runnable model built from a validated Configuration.

Split out of ``jaxfne/core.py`` (slice 4 of the core.py monolith split, see
``docs/v047_refactor_audit.md``). ``jaxfne/core.py`` re-exports every symbol
here for backward compatibility -- import from ``jaxfne.core``, not this
module, unless you are working on core.py itself.

This is the biggest single integration surface in the split: Model reaches
into RuntimeConfig (jaxfne/_runtime_config.py) and Signals/Objective/metric
helpers (jaxfne/_signals.py) pervasively, and is itself the target of
``construct()`` (still in core.py, the genuine hub that pulls Configuration
+ Model together -- extracted last, after this module exists). A handful of
manifest/schema constants (``_JAXFNE_VERSION``, ``_RECEIPT_SCHEMA_VERSION``,
``_MANIFEST_SCHEMA_VERSION``, ``_SOURCE_PROXY_METADATA``,
``_KNOWN_READOUT_METRICS``) moved here too since Model is their heaviest
consumer; core.py imports them back for its own manifest()-area usage --
one-directional (core.py -> _model.py), not a cycle.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp

from .emitters import (
    EdgeList,
    IzhikevichParams,
    make_edge_list_from_dense,
    simulate_edge_recurrent_izhikevich,
    simulate_edge_recurrent_izhikevich_homeostatic,
    simulate_edge_recurrent_izhikevich_hdp,
    simulate_eig_izhikevich,
    simulate_receptor_exponential_izhikevich,
)
from .fields import probe_laminar_modes, project_laminar_sources
from .io import config_hash, json_safe, manifest as build_manifest
from .presets import DEFAULT_SPIKE_IMPULSE_GAIN
from .experimental_hpc.contracts import SelectorSpec
from ._runtime_config import RuntimeConfig, _device_scope
from ._config import Configuration
from ._signals import (
    Signals,
    Objective,
    Simulation,
    StimulusSchedule,
    ReadoutSpec,
    ReadoutResult,
    ObjectiveReport,
    RunReceipt,
    TrialBatch,
    TrialResult,
    TrialBatchResult,
    ParadigmEvent,
    ParadigmCondition,
    _compute_all_metrics,
    _evaluate_gate_spec,
    _evaluate_loss_spec,
    _evaluate_regularizer_spec,
    _finite_or_none,
    _make_poisson_drive,
    _normalize_manifest_readout,
    _default_basis_dict,
)


@dataclass(frozen=True)
class MatrixParameterSpec:
    """Declarative specification for a tunable weight matrix parameter.

    Used in multi-parameter optimization to specify that a named parameter
    maps to a matrix (e.g., the synaptic weight matrix W) rather than a scalar.
    The mask field selects which matrix entries are subject to scaling:
    E_to_E, E_to_I, excitatory_to_all, or all.

    The name is always the dict key in the parameters argument to
    :func:; do **not** add a name field here.

    Attributes
    ----------
    mask : str
        Which matrix entries to scale: "E_to_E", "E_to_I",
        "excitatory_to_all", or "all".
    bounds : tuple[float, float]
        (lower, upper) multiplicative scaling bounds.
    init : str
        Initialization scope; "current" means start from the
        model's existing weight values.
    trainable : bool
        Whether this parameter participates in optimization.
    """

    mask: str
    bounds: tuple
    init: str = "current"
    trainable: bool = True


def matrix_parameter(
    *,
    mask: str,
    bounds: tuple,
    init: str = "current",
    trainable: bool = True,
) -> MatrixParameterSpec:
    """Create a matrix parameter specification for tuning weight matrices.

    Parameters
    ----------
    mask : str
        Which matrix entries to scale: "E_to_E", "E_to_I",
        "excitatory_to_all", or "all".
    bounds : tuple[float, float]
        (lower, upper) multiplicative scaling bounds.
    init : str
        Initialization; "current" uses existing weight values.
    trainable : bool
        Whether this parameter participates in optimization.

    Returns
    -------
    MatrixParameterSpec
        Frozen specification object.

    Examples
    --------
    >>> import jaxfne as jtfne
    >>> spec = jtfne.matrix_parameter(mask="E_to_E", bounds=(0.1, 5.0))
    """
    return MatrixParameterSpec(mask=mask, bounds=bounds, init=init, trainable=trainable)


@dataclass
class TuneResult:
    """Result object returned by Model.tune() with multi-parameter optimization.

    This is a typed container for tuning results, with JSON-safe serialization
    via to_dict() method for reporting and logging.

    Attributes
    ----------
    best_parameters : dict[str, float]
        Optimized parameter values.
    best_score : float
        Best (lowest) objective score achieved.
    history : list[dict[str, Any]]
        Per-generation records with scores and parameter values.
    summary : dict[str, Any]
        High-level tuning summary (targets vs achieved, initial vs final scores, etc).
    model : Optional[Any]
        The model object (if returned by tuning; may be None for metadata-only runs).
    """

    best_parameters: dict[str, float]
    best_score: float
    history: list[dict[str, Any]]
    summary: dict[str, Any]
    model: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-safe dictionary for serialization."""
        from .io import json_safe

        return json_safe({
            "best_parameters": self.best_parameters,
            "best_score": self.best_score,
            "history": self.history,
            "summary": self.summary,
        })

    def __iter__(self):
        """Support legacy tuple unpacking: ``model, report = tune(...)``.

        New code should use ``result.model`` and ``result.summary``.  The iterator
        remains to preserve existing notebooks and tests while surfacing a
        deprecation warning.
        """
        warnings.warn(
            "Tuple-unpacking TuneResult is deprecated; use result.model and result.summary.",
            DeprecationWarning,
            stacklevel=2,
        )
        yield self.model
        yield self.summary


@dataclass(frozen=True)
class Model:
    """Immutable, runnable model built from a validated :class:`Configuration`.

    Holds the source ``cfg``, the dynamic ``params`` pytree (arrays that may be
    tuned/traced), and ``static`` metadata (JIT-static, non-array). Construct via
    :func:`construct`; run via :func:`simulate` / :meth:`simulate`. Also exported
    as the alias ``Net``. The model is a computational scaffold — its field and
    probe outputs are proxy readouts, not calibrated physical signals.
    """

    cfg: Configuration
    params: dict[str, Any]
    static: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return compact JSON-safe model metadata for notebook display."""
        from .io import json_safe
        emitter: IzhikevichParams = self.params["emitter"]
        return json_safe({
            "config_hash": config_hash(self.cfg),
            "n_units": int(emitter.v0.shape[0]),
            "n_contacts": int(self.static.get("n_contacts", 16)),
            "claim_level": self.cfg.metadata.get("claim_level", "computational_scaffold"),
            "source_calibration_status": self.cfg.metadata.get(
                "source_calibration_status", "uncalibrated_izhikevich_native_current"
            ),
            "field_solver_status": self.cfg.metadata.get("field_solver_status", "linear_solver"),
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
        })


    def neuron_table(self) -> list[dict[str, Any]]:
        """Return declared neuron metadata rows for area/layer/cell-type grouping."""
        rows = self.static.get("neuron_metadata")
        if rows is not None:
            return [dict(row) for row in rows]
        emitter: IzhikevichParams = self.params["emitter"]
        layers = emitter.layer_labels or tuple("unspecified" for _ in emitter.labels)
        positions = self.params.get("positions")
        rows_out: list[dict[str, Any]] = []
        for idx, label in enumerate(emitter.labels):
            z_value = None
            try:
                z_value = float(positions[idx, 2]) if positions is not None else None
            except Exception:
                z_value = None
            rows_out.append({
                "neuron_id": int(idx),
                "area": "network",
                "layer": str(layers[idx]),
                "cell_type": str(label),
                "z": z_value,
            })
        return rows_out

    def compile_connections(self, *, seed: int = 0, **kwargs: Any):
        """Compile this model's declared connection rules into sparse edges.

        Thin wrapper over :func:`jaxfne.compile_connection_rules`: resolves the
        rule selectors against this model's :meth:`neuron_table` and the
        ``metadata["circuit"]`` declarations. Declaration-only inputs in →
        finite sparse edge arrays out; no simulation numerics change.
        """
        from .connectivity import compile_connection_rules

        circuit = dict(self.cfg.metadata.get("circuit", {}))
        return compile_connection_rules(
            self.neuron_table(),
            circuit.get("connections", []),
            circuit.get("mechanisms", []),
            seed=seed,
            **kwargs,
        )

    def select(
        self,
        *,
        area: Optional[Any] = None,
        area_id: Optional[Any] = None,
        layer: Optional[Any] = None,
        cell_type: Optional[Any] = None,
        ids: Optional[Sequence[int]] = None,
        allow_empty: bool = False,
    ) -> jax.Array:
        """Resolve semantic selectors to neuron row indices (does not mutate).

        Thin, non-mutating wrapper around :class:`SelectorSpec` over this model's
        :meth:`neuron_table`. Returns an int32 JAX array of row positions suitable
        for indexing the trailing (neuron) axis of V_m/spikes/sources. Empty
        matches raise ``ValueError`` unless ``allow_empty=True``. A requested
        field absent from the neuron table raises ``KeyError``.
        """
        spec = SelectorSpec(
            area=area,
            area_id=area_id,
            layer=layer,
            cell_type=cell_type,
            ids=tuple(ids) if ids is not None else None,
        )
        return spec.resolve(self.neuron_table(), allow_empty=allow_empty)

    def _simulate_arrays(
        self: "Model",
        sim: Simulation,
        key: jax.Array,
        runtime_cfg: RuntimeConfig,
        drive_schedule: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compile and execute the underlying simulation kernel.

        This method resolves the ablation mode masks, updates parameters, and
        dispatches to either the sparse/edge-list or dense JAX simulation kernels
        with compile-time caching.

        Parameters
        ----------
        sim : Simulation
            Simulation configuration.
        key : jax.Array
            JAX PRNG key.
        runtime_cfg : RuntimeConfig
            Resolved runtime config.
        drive_schedule : jax.Array, optional
            Input drive schedule array, by default None.

        Returns
        -------
        tuple[jax.Array, jax.Array, jax.Array]
            Voltages, spikes, and source currents.
        """
        from .emitters import _dtype_from_policy
        # Local import: _resolve_homeostasis_k_gain/_homeostasis_params_cache_fingerprint
        # stay in core.py (group-6 construct-pipeline territory); Model is their only
        # caller, so deferring the import here (rather than at module top) avoids a
        # circular import with core.py's own `from ._model import Model`.
        from .core import _resolve_homeostasis_k_gain, _homeostasis_params_cache_fingerprint
        emitter: IzhikevichParams = self.params["emitter"]
        sched = drive_schedule  # None or (n_steps, n_neurons) array
        
        # Build silence_mask if E_silence or I_silence is requested
        n_neurons = emitter.v0.shape[0]
        jdtype = _dtype_from_policy(runtime_cfg.actual_dtype)
        ablation_mode = getattr(sim, "ablation", None)

        # Sparse-direct models carry a placeholder dense W (edges live only in
        # params["edge_list"]); force the edge_list backend so the dense kernel is
        # never handed the empty W.
        if emitter.W.shape[0] != n_neurons and "edge_list" in self.params:
            runtime_cfg = replace(runtime_cfg, recurrent_backend="edge_list")

        if not hasattr(self, "_silence_masks"):
            object.__setattr__(self, "_silence_masks", {})

        if ablation_mode == "E_silence":
            if "E_silence" not in self._silence_masks:
                mask_list = [0.0 if lbl.startswith("E") else 1.0 for lbl in emitter.labels]
                self._silence_masks["E_silence"] = jnp.array(mask_list, dtype=jdtype)
            silence_mask = self._silence_masks["E_silence"]
        elif ablation_mode == "I_silence":
            if "I_silence" not in self._silence_masks:
                mask_list = [1.0 if lbl.startswith("E") else 0.0 for lbl in emitter.labels]
                self._silence_masks["I_silence"] = jnp.array(mask_list, dtype=jdtype)
            silence_mask = self._silence_masks["I_silence"]
        else:
            if "default" not in self._silence_masks:
                self._silence_masks["default"] = jnp.ones((n_neurons,), dtype=jdtype)
            silence_mask = self._silence_masks["default"]
            
        if ablation_mode == "disconnected_null":
            if runtime_cfg.recurrent_backend == "edge_list":
                edges: EdgeList = self.params["edge_list"]
                edges = replace(edges, weight=jnp.zeros_like(edges.weight))
            else:
                emitter = replace(emitter, W=jnp.zeros_like(emitter.W))

        # Reset per-call homeostasis/HDP diagnostics (populated only when enabled).
        object.__setattr__(self, "_last_homeostasis_diag", None)
        object.__setattr__(self, "_last_hdp_diag", None)

        if getattr(runtime_cfg, "enable_homeostasis", False):
            if runtime_cfg.synaptic_kernel == "receptor_exponential":
                raise ValueError(
                    "enable_homeostasis is not supported with "
                    "synaptic_kernel='receptor_exponential'; use the default "
                    "exponential synaptic kernel."
                )
            # Homeostasis is sparse-edge based; edge_list always exists from construct().
            edges: EdgeList = self.params["edge_list"]
            if ablation_mode == "disconnected_null":
                edges = replace(edges, weight=jnp.zeros_like(edges.weight))
            hp = dict(runtime_cfg.homeostasis_params or {})
            _plastic_active = float(hp.get("eta", 0.0) or 0.0) != 0.0

            def _homeo_packed(k, s):
                """Return (V, spikes, sources, g_bias, r_trace[, w_final, w_trace])."""
                V, S, src, diag = simulate_edge_recurrent_izhikevich_homeostatic(
                    emitter, edges, sim.n_steps, sim.dt_ms, k,
                    dtype=runtime_cfg.actual_dtype, drive_schedule=s,
                    silence_mask=silence_mask,
                    r_star=hp.get("r_star", 0.05),
                    tau_r_ms=hp.get("tau_r_ms", 300.0),
                    alpha=hp.get("alpha", 1.0),
                    k_gain=_resolve_homeostasis_k_gain(hp, emitter),
                    g_min=hp.get("g_min", -12.0),
                    g_max=hp.get("g_max", 8.0),
                    r_max=hp.get("r_max", 1.0),
                    eta=hp.get("eta", 0.0),
                    tau_x_ms=hp.get("tau_x_ms", 100.0),
                    w_min=hp.get("w_min", -10.0),
                    w_max=hp.get("w_max", 10.0),
                    v_floor=hp.get("v_floor", -150.0),
                    v_ceiling=hp.get("v_ceiling", 100.0),
                    u_abs_max=hp.get("u_abs_max", 2000.0),
                    syn_abs_max=hp.get("syn_abs_max", 1.0e4),
                )
                if _plastic_active:
                    return V, S, src, diag["g_bias"], diag["r_trace"], diag["w_final"], diag["w_trace"]
                return V, S, src, diag["g_bias"], diag["r_trace"]

            effective_jit = runtime_cfg.resolve_jit(sim.n_steps, emitter.n_neurons)
            if effective_jit:
                if not hasattr(self, "_compiled_cache"):
                    object.__setattr__(self, "_compiled_cache", {})
                from .validation import make_recompilation_guard
                B = 1
                Z = int(self.static.get("n_contacts", 16))
                C = int(emitter.n_neurons)
                T = int(sim.n_steps)
                guard_mode = getattr(runtime_cfg, "recompilation_guard", "warning")
                cache_key = ("simulate_homeostatic", B, Z, C, T, runtime_cfg.actual_dtype,
                             ablation_mode, runtime_cfg.selected_backend, _plastic_active,
                             _homeostasis_params_cache_fingerprint(hp))
                with _device_scope(runtime_cfg.selected_backend):
                    if cache_key not in self._compiled_cache:
                        import time
                        guard_name = ("simulate_homeostatic_plastic" if _plastic_active
                                      else "simulate_homeostatic")
                        target_fn = make_recompilation_guard(
                            _homeo_packed, name=guard_name,
                            recompilation_guard=guard_mode, B=B, Z=Z, C=C, T=T,
                        )
                        self._compiled_cache[cache_key] = jax.jit(target_fn)
                        t0 = time.perf_counter()
                        result = self._compiled_cache[cache_key](key, sched)
                        t1 = time.perf_counter()
                        if not hasattr(self, "_warmup_times"):
                            object.__setattr__(self, "_warmup_times", [])
                        self._warmup_times.append(t1 - t0)
                    else:
                        result = self._compiled_cache[cache_key](key, sched)
            else:
                with _device_scope(runtime_cfg.selected_backend):
                    result = _homeo_packed(key, sched)
            if _plastic_active:
                V, S, src, g_bias, r_trace, w_final, w_trace = result
                object.__setattr__(self, "_last_homeostasis_diag",
                                   {"g_bias": g_bias, "r_trace": r_trace,
                                    "w_final": w_final, "w_trace": w_trace})
            else:
                V, S, src, g_bias, r_trace = result
                object.__setattr__(self, "_last_homeostasis_diag",
                                   {"g_bias": g_bias, "r_trace": r_trace})
            return V, S, src

        if getattr(runtime_cfg, "enable_hdp", False):
            if runtime_cfg.synaptic_kernel == "receptor_exponential":
                raise ValueError(
                    "enable_hdp is not supported with "
                    "synaptic_kernel='receptor_exponential'; use the default "
                    "exponential synaptic kernel."
                )
            # HDP is sparse-edge based; edge_list always exists from construct().
            edges: EdgeList = self.params["edge_list"]
            if ablation_mode == "disconnected_null":
                edges = replace(edges, weight=jnp.zeros_like(edges.weight))
            hp = dict(runtime_cfg.hdp_params or {})

            # Optional caller-supplied initial HDP state (Model.with_hdp_initial_state).
            # Absent by default -> init_state=None, the exact prior behavior
            # (kernel's own equilibrium H=1.0, native edge weight).
            _hdp_H0 = self.params.get("hdp_initial_H")
            _hdp_w0 = self.params.get("hdp_initial_w")
            init_state = None
            if _hdp_H0 is not None or _hdp_w0 is not None:
                _idt = runtime_cfg.actual_dtype
                init_state = {
                    "v": emitter.v0.astype(_idt),
                    "u": emitter.u0.astype(_idt),
                    "prev_spikes": jnp.zeros_like(emitter.v0, dtype=_idt),
                    "syn_state": jnp.zeros_like(edges.weight, dtype=_idt),
                }
                if _hdp_H0 is not None:
                    init_state["H_final"] = jnp.asarray(_hdp_H0, dtype=_idt)
                if _hdp_w0 is not None:
                    init_state["w_final"] = jnp.asarray(_hdp_w0, dtype=_idt)

            def _hdp_packed(k, s):
                """Return (V, spikes, sources, H_final, H_trace, w_final, w_trace)."""
                V, S, src, diag = simulate_edge_recurrent_izhikevich_hdp(
                    emitter, edges, sim.n_steps, sim.dt_ms, k,
                    dtype=runtime_cfg.actual_dtype, drive_schedule=s,
                    silence_mask=silence_mask,
                    init_state=init_state,
                    H_min=hp.get("H_min", 0.1), H_max=hp.get("H_max", 10.0),
                    tau_0_ms=hp.get("tau_0_ms", 100.0),
                    alpha=hp.get("alpha", 0.0), beta=hp.get("beta", 0.0),
                    gamma=hp.get("gamma", 0.0), delta=hp.get("delta", 0.0),
                    C_spike=hp.get("C_spike", 0.0), K_HDP=hp.get("K_HDP", 1.0),
                    K_ctrl=hp.get("K_ctrl", 0.0),
                    K_w_ctrl=hp.get("K_w_ctrl", 0.0),
                    barrier_c=hp.get("barrier_c", 0.0), barrier_d=hp.get("barrier_d", 0.0),
                    barrier_eps=hp.get("barrier_eps", 1.0e-3),
                    w_floor=hp.get("w_floor", 1.0e-3), w_ceiling=hp.get("w_ceiling", 50.0),
                    v_floor=hp.get("v_floor", -150.0), v_ceiling=hp.get("v_ceiling", 100.0),
                    u_abs_max=hp.get("u_abs_max", 2000.0), syn_abs_max=hp.get("syn_abs_max", 1.0e4),
                    H_boost_gain=hp.get("H_boost_gain", 0.0),
                    size_scale_by_cell_type=hp.get("size_scale_by_cell_type"),
                    size_scale_override=hp.get("size_scale_override"),
                )
                return V, S, src, diag["H_final"], diag["H_trace"], diag["w_final"], diag["w_trace"]

            effective_jit = runtime_cfg.resolve_jit(sim.n_steps, emitter.n_neurons)
            if effective_jit:
                if not hasattr(self, "_compiled_cache"):
                    object.__setattr__(self, "_compiled_cache", {})
                from .validation import make_recompilation_guard
                B = 1
                Z = int(self.static.get("n_contacts", 16))
                C = int(emitter.n_neurons)
                T = int(sim.n_steps)
                guard_mode = getattr(runtime_cfg, "recompilation_guard", "warning")
                cache_key = ("simulate_hdp", B, Z, C, T, runtime_cfg.actual_dtype,
                             ablation_mode, runtime_cfg.selected_backend,
                             _homeostasis_params_cache_fingerprint(hp))
                with _device_scope(runtime_cfg.selected_backend):
                    if cache_key not in self._compiled_cache:
                        import time
                        target_fn = make_recompilation_guard(
                            _hdp_packed, name="simulate_hdp",
                            recompilation_guard=guard_mode, B=B, Z=Z, C=C, T=T,
                        )
                        self._compiled_cache[cache_key] = jax.jit(target_fn)
                        t0 = time.perf_counter()
                        result = self._compiled_cache[cache_key](key, sched)
                        t1 = time.perf_counter()
                        if not hasattr(self, "_warmup_times"):
                            object.__setattr__(self, "_warmup_times", [])
                        self._warmup_times.append(t1 - t0)
                    else:
                        result = self._compiled_cache[cache_key](key, sched)
            else:
                with _device_scope(runtime_cfg.selected_backend):
                    result = _hdp_packed(key, sched)
            V, S, src, H_final, H_trace, w_final, w_trace = result
            object.__setattr__(self, "_last_hdp_diag",
                               {"H_final": H_final, "H_trace": H_trace,
                                "w_final": w_final, "w_trace": w_trace})
            return V, S, src

        if runtime_cfg.recurrent_backend == "edge_list":
            edges: EdgeList = self.params["edge_list"]
            if ablation_mode == "disconnected_null":
                edges = replace(edges, weight=jnp.zeros_like(edges.weight))
            kernel_fn = (
                simulate_receptor_exponential_izhikevich
                if runtime_cfg.synaptic_kernel == "receptor_exponential"
                else simulate_edge_recurrent_izhikevich
            )
            effective_jit = runtime_cfg.resolve_jit(sim.n_steps, emitter.n_neurons)
            if effective_jit:
                if not hasattr(self, "_compiled_cache"):
                    object.__setattr__(self, "_compiled_cache", {})
                from .validation import make_recompilation_guard
                B = 1
                Z = int(self.static.get("n_contacts", 16))
                C = int(emitter.n_neurons)
                T = int(sim.n_steps)
                guard_mode = getattr(runtime_cfg, "recompilation_guard", "warning")

                cache_key = ("simulate_recurrent", B, Z, C, T, runtime_cfg.actual_dtype, runtime_cfg.synaptic_kernel, ablation_mode, runtime_cfg.selected_backend)
                with _device_scope(runtime_cfg.selected_backend):
                    if cache_key not in self._compiled_cache:
                        import time
                        target_fn = lambda k, s: kernel_fn(
                            emitter, edges, sim.n_steps, sim.dt_ms, k,
                            dtype=runtime_cfg.actual_dtype, drive_schedule=s,
                            silence_mask=silence_mask,
                        )[:3]
                        target_fn = make_recompilation_guard(
                            target_fn,
                            name="simulate",
                            recompilation_guard=guard_mode,
                            B=B, Z=Z, C=C, T=T
                        )
                        self._compiled_cache[cache_key] = jax.jit(target_fn)
                        t0 = time.perf_counter()
                        res = self._compiled_cache[cache_key](key, sched)
                        t1 = time.perf_counter()
                        if not hasattr(self, "_warmup_times"):
                            object.__setattr__(self, "_warmup_times", [])
                        self._warmup_times.append(t1 - t0)
                        return res
                    run = self._compiled_cache[cache_key]
                    return run(key, sched)
            with _device_scope(runtime_cfg.selected_backend):
                return kernel_fn(
                    emitter, edges, sim.n_steps, sim.dt_ms, key,
                    dtype=runtime_cfg.actual_dtype, drive_schedule=sched,
                    silence_mask=silence_mask,
                )[:3]
        effective_jit = runtime_cfg.resolve_jit(sim.n_steps, emitter.n_neurons)
        if effective_jit:
            if not hasattr(self, "_compiled_cache"):
                object.__setattr__(self, "_compiled_cache", {})
            from .validation import make_recompilation_guard
            B = 1
            Z = int(self.static.get("n_contacts", 16))
            C = int(emitter.n_neurons)
            T = int(sim.n_steps)
            guard_mode = getattr(runtime_cfg, "recompilation_guard", "warning")

            cache_key = ("simulate_dense", B, Z, C, T, runtime_cfg.actual_dtype, ablation_mode, runtime_cfg.selected_backend)
            with _device_scope(runtime_cfg.selected_backend):
                if cache_key not in self._compiled_cache:
                    import time
                    target_fn = lambda k, s: simulate_eig_izhikevich(
                        emitter, sim.n_steps, sim.dt_ms, k,
                        dtype=runtime_cfg.actual_dtype, drive_schedule=s,
                        silence_mask=silence_mask,
                    )
                    target_fn = make_recompilation_guard(
                        target_fn,
                        name="simulate",
                        recompilation_guard=guard_mode,
                        B=B, Z=Z, C=C, T=T
                    )
                    self._compiled_cache[cache_key] = jax.jit(target_fn)
                    t0 = time.perf_counter()
                    res = self._compiled_cache[cache_key](key, sched)
                    t1 = time.perf_counter()
                    if not hasattr(self, "_warmup_times"):
                        object.__setattr__(self, "_warmup_times", [])
                    self._warmup_times.append(t1 - t0)
                    return res
                run = self._compiled_cache[cache_key]
                return run(key, sched)
        with _device_scope(runtime_cfg.selected_backend):
            return simulate_eig_izhikevich(
                emitter, sim.n_steps, sim.dt_ms, key,
                dtype=runtime_cfg.actual_dtype, drive_schedule=sched,
                silence_mask=silence_mask,
            )

    def _resolve_stimulus_schedule(
        self,
        paradigm: Any,
        sim: Simulation,
        runtime_cfg: RuntimeConfig,
    ) -> Optional["StimulusSchedule"]:
        """Return a StimulusSchedule from paradigm arg, or None."""
        if paradigm is None:
            return None
        if isinstance(paradigm, StimulusSchedule):
            return paradigm
        if isinstance(paradigm, ParadigmCondition):
            return stimulus_schedule(
                paradigm.events,
                n_neurons=self.params["emitter"].n_neurons,
            )
        return None

    def simulate(
        self: "Model",
        sim: Simulation,
        paradigm: "Optional[Any]" = None,
    ) -> Signals:
        """Run the default EIG/Izhikevich vertical slice.

        When ``paradigm`` is None, behavior is identical to v0.0.11.
        When ``paradigm`` is a :class:`StimulusSchedule`, its drive array is
        injected as native (uncalibrated) current at each timestep.
        When ``paradigm`` is a :class:`ParadigmCondition`, its events are
        converted to a ``StimulusSchedule`` and injected.

        JIT is opt-in through ``Simulation(runtime=RuntimeConfig(jit=True))`` or
        ``runtime(jit=True)``.  The compiled path preserves the same proxy-field
        truth status as the eager path. No calibrated amplitude, PDE, or empirical
        claim is introduced by stimulus injection.
        """
        # Local import: _simulate_homeostasis_metadata/_simulate_hdp_metadata stay
        # in core.py (group-6 construct-pipeline territory); Model is their only
        # caller, so deferring the import here avoids a circular import with
        # core.py's own `from ._model import Model`.
        from .core import _simulate_homeostasis_metadata, _simulate_hdp_metadata

        runtime_cfg = sim.resolved_runtime
        key = jax.random.PRNGKey(sim.seed)

        schedule = self._resolve_stimulus_schedule(paradigm, sim, runtime_cfg)
        drive_array: Optional[Any] = None
        if schedule is not None:
            drive_array = schedule.to_array(sim.n_steps, sim.dt_ms, dtype=runtime_cfg.actual_dtype)
        if sim.poisson_drive is not None:
            _emitter: IzhikevichParams = self.params["emitter"]
            _pd = sim.poisson_drive
            _poisson_arr = _make_poisson_drive(
                n_steps=sim.n_steps,
                n_neurons=_emitter.n_neurons,
                rate_hz=float(_pd.get("rate_hz", 2.0)),
                amplitude=float(_pd.get("amplitude", 0.5)),
                dt_ms=sim.dt_ms,
                seed=int(_pd.get("seed", sim.seed + 7919)),
                target=str(_pd.get("target", "all")),
            )
            drive_array = _poisson_arr if drive_array is None else drive_array + _poisson_arr

        # shuffled_timing ablation: shuffle drive_array along time axis (axis 0) independently for each neuron
        ablation_mode = getattr(sim, "ablation", None)
        if ablation_mode == "shuffled_timing" and drive_array is not None:
            shuffle_key = jax.random.PRNGKey(sim.seed + 12345)
            n_neurons = drive_array.shape[1]
            keys = jax.random.split(shuffle_key, n_neurons)
            # Use vmap to shuffle each neuron's temporal drive independently
            shuffled = jax.vmap(lambda arr, k: jax.random.permutation(k, arr))(drive_array.T, keys)
            drive_array = shuffled.T

        voltages, spikes, sources = self._simulate_arrays(sim, key, runtime_cfg, drive_schedule=drive_array)
        time_ms = jnp.arange(sim.n_steps, dtype=runtime_cfg.jnp_dtype) * jnp.asarray(
            sim.dt_ms, dtype=runtime_cfg.jnp_dtype
        )
        positions = jnp.asarray(self.params["positions"], dtype=runtime_cfg.jnp_dtype)
        field_output = None
        if sim.record_fields:
            field_output = project_laminar_sources(
                sources=sources,
                positions=positions,
                n_contacts=self.static.get("n_contacts", 16),
                dtype=runtime_cfg.actual_dtype,
            )

        paradigm_meta: Optional[dict[str, Any]] = None
        if isinstance(paradigm, Mapping):
            paradigm_meta = dict(paradigm)
        elif hasattr(paradigm, "to_dict"):
            paradigm_meta = paradigm.to_dict()

        metadata: dict[str, Any] = {
            "config_hash": config_hash(self.cfg),
            "source_calibration_status": self.cfg.metadata.get("source_calibration_status"),
            "field_claim_level": "proxy_readout",
            "paradigm": paradigm_meta,
            "duration_ms": float(sim.duration_ms),
            "dt_ms": float(sim.dt_ms),
            "n_steps": int(sim.n_steps),
            "record_sources": bool(sim.record_sources),
            "record_fields": bool(sim.record_fields),
            "plasticity_gain": sim.plasticity,
            "runtime": runtime_cfg.runtime_report(),
            "recurrent_backend": runtime_cfg.recurrent_backend,
            "synaptic_kernel": runtime_cfg.synaptic_kernel,
            "source_model": _SOURCE_PROXY_METADATA,
            "neuron_metadata": self.static.get("neuron_metadata"),
            "neuron_metadata_summary": self.static.get("neuron_metadata_summary"),
            "ablation": ablation_mode,
        }
        # v0.2.0: Add source bookkeeping metadata for theoretical validation.
        metadata["source_bookkeeping"] = {
            "source_mode": _SOURCE_PROXY_METADATA.get("source_mode"),
            "source_projection_mode": self.cfg.metadata.get("source_projection_mode", "proxy_no_field_solve"),
            "source_decomposition": self.cfg.metadata.get("source_decomposition", "proxy_reduced_emitter"),
            "source_calibration_status": _SOURCE_PROXY_METADATA.get("source_calibration_status"),
            "synaptic_current_counting": _SOURCE_PROXY_METADATA.get("double_count_synaptic_current_guard"),
            "source_mode_exclusive": True,
            "physical_amplitude_calibrated": _SOURCE_PROXY_METADATA.get("physical_amplitude_calibrated", False),
            "double_count_guard": "passed",
            "double_count_evidence": None,
        }
        if schedule is not None:
            metadata["stimulus_injection_status"] = "native_drive_schedule_v0.0.12"
            metadata["stimulus_schedule"] = schedule.to_dict()
            if isinstance(paradigm, ParadigmCondition):
                metadata["condition_name"] = paradigm.name
                metadata["has_omission"] = paradigm.has_omission()
        if sim.poisson_drive is not None:
            metadata["poisson_drive"] = {
                "rate_hz": float(sim.poisson_drive.get("rate_hz", 2.0)),
                "amplitude": float(sim.poisson_drive.get("amplitude", 0.5)),
                "target": str(sim.poisson_drive.get("target", "all")),
                "seed": int(sim.poisson_drive.get("seed", sim.seed + 7919)),
                "status": "stochastic_drive_applied",
            }
        if getattr(runtime_cfg, "enable_homeostasis", False):
            diag = getattr(self, "_last_homeostasis_diag", None)
            metadata["homeostasis"] = _simulate_homeostasis_metadata(runtime_cfg, diag)
        if getattr(runtime_cfg, "enable_hdp", False):
            diag = getattr(self, "_last_hdp_diag", None)
            metadata["hdp"] = _simulate_hdp_metadata(runtime_cfg, diag)
        return Signals(
            time_ms=time_ms,
            V_m=voltages.astype(runtime_cfg.jnp_dtype),
            spikes=spikes,
            sources=sources.astype(runtime_cfg.jnp_dtype) if sim.record_sources else None,
            field=field_output,
            metadata=metadata,
        )

    def last_homeostasis_diagnostics(self) -> "Optional[dict[str, Any]]":
        """Return the full per-step homeostasis diagnostics from the most recent
        ``simulate(...)`` call with ``enable_homeostasis=True``.

        Returns a dict with arrays ``g_bias`` and ``r_trace`` of shape
        ``(n_steps, n_neurons)``, or ``None`` if homeostasis was not enabled on
        the last run. When ``homeostasis_params["eta"] != 0`` (homeostatic
        synaptic plasticity active), the dict also carries ``w_final``
        ``(n_edges,)`` and ``w_trace`` ``(n_steps, n_edges)`` — the plastic
        edge-weight trajectory. These are computational-control diagnostics
        (proxy), not a biological-mechanism claim.
        """
        return getattr(self, "_last_homeostasis_diag", None)

    def last_hdp_diagnostics(self) -> "Optional[dict[str, Any]]":
        """Return the full per-step HDP diagnostics from the most recent
        ``simulate(...)`` call with ``enable_hdp=True``.

        Returns a dict with ``H_final``/``H_trace`` ``(n_steps, n_neurons)``
        and ``w_final``/``w_trace`` ``(n_steps, n_edges)``, or ``None`` if HDP
        was not enabled on the last run. See
        ``jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp`` for the
        underlying kernel and ``jaxfne.hdp_network.DEFAULT_HDP`` /
        ``DEFAULT_HDP_DESYNC`` for tuned presets. Computational-control
        diagnostics (proxy), not a biological-mechanism claim.
        """
        return getattr(self, "_last_hdp_diag", None)

    def simulate_condition(
        self,
        sim: Simulation,
        condition: "ParadigmCondition",
        *,
        drive_amplitude: float = 5.0,
        event_duration_ms: float = 50.0,
    ) -> Signals:
        """Convenience wrapper: simulate one trial condition with event-aligned drive injection.

        Equivalent to ``simulate(sim, paradigm=condition)`` but allows per-call
        override of ``drive_amplitude`` and ``event_duration_ms``.
        No calibrated amplitude, PDE, or empirical claim is introduced.
        """
        schedule = stimulus_schedule(
            condition.events,
            n_neurons=self.params["emitter"].n_neurons,
            drive_amplitude=drive_amplitude,
            event_duration_ms=event_duration_ms,
        )
        signals = self.simulate(sim, paradigm=schedule)
        signals.metadata["condition_name"] = condition.name
        signals.metadata["has_omission"] = condition.has_omission()
        return signals

    def simulate_batch(self, sim: Simulation, n_seeds: int = 4, seed: int | None = None) -> dict[str, Any]:
        """Run a vectorized seed batch and return JSON-safe metadata plus arrays.

        This is a trial-replicate utility for notebook statistics.  It uses
        ``jax.vmap`` over PRNG keys and returns proxy arrays without changing the
        field-solver or calibration status.
        """
        from .io import json_safe
        # Local import: see _simulate_arrays' matching comment above.
        from .core import _resolve_homeostasis_k_gain, _homeostasis_params_cache_fingerprint
        runtime_cfg = sim.resolved_runtime
        base_seed = sim.seed if seed is None else int(seed)
        keys = jax.random.split(jax.random.PRNGKey(base_seed), int(n_seeds))
        emitter: IzhikevichParams = self.params["emitter"]

        # Sparse-direct models (placeholder dense W) must use the edge_list backend.
        if emitter.W.shape[0] != int(emitter.v0.shape[0]) and "edge_list" in self.params:
            runtime_cfg = replace(runtime_cfg, recurrent_backend="edge_list")

        homeo_on = bool(getattr(runtime_cfg, "enable_homeostasis", False))
        if homeo_on and runtime_cfg.synaptic_kernel == "receptor_exponential":
            raise ValueError(
                "enable_homeostasis is not supported with "
                "synaptic_kernel='receptor_exponential'."
            )
        edge_kernel_fn = (
            simulate_receptor_exponential_izhikevich
            if runtime_cfg.synaptic_kernel == "receptor_exponential"
            else simulate_edge_recurrent_izhikevich
        )
        _hp = dict(runtime_cfg.homeostasis_params or {})

        def one(k):
            """Documented public function `one`."""
            if homeo_on:
                # Homeostasis engages the sparse-edge homeostatic kernel; per-step
                # g_bias/r_trace diagnostics are dropped here (batch is a seed-replicate
                # statistics utility — use simulate() for full diagnostics passthrough).
                return simulate_edge_recurrent_izhikevich_homeostatic(
                    emitter, self.params["edge_list"], sim.n_steps, sim.dt_ms, k,
                    dtype=runtime_cfg.actual_dtype,
                    r_star=_hp.get("r_star", 0.05), tau_r_ms=_hp.get("tau_r_ms", 300.0),
                    alpha=_hp.get("alpha", 1.0), k_gain=_resolve_homeostasis_k_gain(_hp, emitter),
                    g_min=_hp.get("g_min", -12.0), g_max=_hp.get("g_max", 8.0),
                    r_max=_hp.get("r_max", 1.0),
                    eta=_hp.get("eta", 0.0), tau_x_ms=_hp.get("tau_x_ms", 100.0),
                    w_min=_hp.get("w_min", -10.0), w_max=_hp.get("w_max", 10.0),
                    v_floor=_hp.get("v_floor", -150.0), v_ceiling=_hp.get("v_ceiling", 100.0),
                    u_abs_max=_hp.get("u_abs_max", 2000.0), syn_abs_max=_hp.get("syn_abs_max", 1.0e4),
                )[:3]
            if runtime_cfg.recurrent_backend == "edge_list":
                return edge_kernel_fn(
                    emitter,
                    self.params["edge_list"],
                    sim.n_steps,
                    sim.dt_ms,
                    k,
                    dtype=runtime_cfg.actual_dtype,
                )[:3]
            return simulate_eig_izhikevich(
                emitter, sim.n_steps, sim.dt_ms, k, dtype=runtime_cfg.actual_dtype
            )

        # v0.0.21: honor runtime.vmap flag behaviorally.
        # vmap=True  → jax.vmap over keys (one compiled call, vectorized over batch).
        # vmap=False → Python-loop + jnp.stack (each key runs independently, no vmap).
        effective_vmap = runtime_cfg.resolve_vmap(int(n_seeds))
        if effective_vmap:
            if not hasattr(self, "_compiled_cache"):
                object.__setattr__(self, "_compiled_cache", {})
            B = int(n_seeds)
            Z = int(self.static.get("n_contacts", 16))
            C = int(emitter.n_neurons)
            T = int(sim.n_steps)
            cache_key = ("simulate_batch", B, Z, C, T, runtime_cfg.actual_dtype, runtime_cfg.synaptic_kernel, runtime_cfg.recurrent_backend, homeo_on, runtime_cfg.selected_backend,
                         _homeostasis_params_cache_fingerprint(_hp) if homeo_on else ())
            with _device_scope(runtime_cfg.selected_backend):
                effective_jit = runtime_cfg.resolve_jit(sim.n_steps, emitter.n_neurons, batch=B)
                if effective_jit:
                    if cache_key not in self._compiled_cache:
                        import time
                        from .validation import make_recompilation_guard
                        guard_mode = getattr(runtime_cfg, "recompilation_guard", "warning")
                        run_mapped = jax.vmap(one)
                        run_mapped = make_recompilation_guard(
                            run_mapped,
                            name="simulate_batch",
                            recompilation_guard=guard_mode,
                            B=B, Z=Z, C=C, T=T
                        )
                        self._compiled_cache[cache_key] = jax.jit(run_mapped)
                        t0 = time.perf_counter()
                        results = self._compiled_cache[cache_key](keys)
                        t1 = time.perf_counter()
                        if not hasattr(self, "_warmup_times"):
                            object.__setattr__(self, "_warmup_times", [])
                        self._warmup_times.append(t1 - t0)
                        voltages, spikes, sources = results
                    else:
                        run = self._compiled_cache[cache_key]
                        voltages, spikes, sources = run(keys)
                else:
                    run = jax.vmap(one)
                    voltages, spikes, sources = run(keys)
            batch_execution_mode = "jax_vmap"
        else:
            per_key = [one(k) for k in keys]
            voltages = jnp.stack([t[0] for t in per_key], axis=0)
            spikes = jnp.stack([t[1] for t in per_key], axis=0)
            sources = jnp.stack([t[2] for t in per_key], axis=0)
            batch_execution_mode = "python_loop_stack"

        if runtime_cfg.recurrent_backend == "edge_list":
            batch_status = (
                "vmap_seed_batch_v0.0.11"
                if runtime_cfg.synaptic_kernel == "receptor_exponential"
                else "vmap_seed_batch_v0.0.9"
            )
        else:
            batch_status = "vmap_seed_batch_v0.0.8"
        return {
            "V_m": voltages.astype(runtime_cfg.jnp_dtype),
            "spikes": spikes,
            "sources": sources.astype(runtime_cfg.jnp_dtype),
            "metadata": json_safe({
                "batch_status": batch_status,
                "batch_execution_mode": batch_execution_mode,
                "n_seeds": int(n_seeds),
                "seed": base_seed,
                "runtime": runtime_cfg.runtime_report(),
                "field_claim_level": "proxy_readout",
                "physical_amplitude_calibrated": False,
                "recurrent_backend": runtime_cfg.recurrent_backend,
                "synaptic_kernel": runtime_cfg.synaptic_kernel,
                "enable_homeostasis": homeo_on,
                "homeostasis_params": _hp if homeo_on else None,
                "source_model": _SOURCE_PROXY_METADATA,
            }),
        }

    def run_trials(self, batch: TrialBatch, sim: Simulation, collect_errors: bool = False) -> TrialBatchResult:
        """Execute a batch of trials sequentially.

        For each trial in the batch, this method:
        1. Replaces sim.seed with trial.seed.
        2. Calls self.simulate(sim_trial, paradigm=trial.condition).
        3. If collect_errors=False (default): raises immediately on failure.
           If collect_errors=True: records exception in TrialResult and continues.

        Returns a TrialBatchResult containing all individual TrialResults (or raises on first failure).
        """
        results: list[TrialResult] = []
        for trial in batch.trials:
            sim_trial = replace(sim, seed=trial.seed)
            try:
                signals = self.simulate(sim_trial, paradigm=trial.condition)
                results.append(
                    TrialResult(
                        trial_id=trial.trial_id,
                        condition_label=trial.condition.name if trial.condition else None,
                        signals=signals,
                        success=True,
                        metadata=trial.metadata,
                    )
                )
            except Exception as e:
                if not collect_errors:
                    raise
                results.append(
                    TrialResult(
                        trial_id=trial.trial_id,
                        condition_label=trial.condition.name if trial.condition else None,
                        signals=None,
                        success=False,
                        error_message=str(e),
                        metadata=trial.metadata,
                    )
                )
        return TrialBatchResult(batch_id=batch.batch_id, results=tuple(results), metadata=batch.metadata)

    def run_receipt(self, signals: Signals, *, tags: Optional[dict[str, Any]] = None) -> RunReceipt:
        """Build a RunReceipt capturing this run for audit and reproducibility.

        **Canonical v0.1 workflow method.**  Prefer this over :meth:`manifest`
        for recording completed simulation runs.

        Args:
            signals: Signals returned by self.simulate().
            tags: Optional user-supplied key-value metadata (condition, paper, etc.).

        Returns:
            RunReceipt with frozen truth gates and deterministic receipt_id.

        Note:
            ``receipt_id`` is deterministic for the same
            ``(config_hash, seed, _JAXFNE_VERSION)`` triple.  Upgrading the
            package version changes the ID even when config and seed are
            identical, because the computational kernel may have changed.
            IDs are audit identifiers; they are not empirical claims.
        """
        from .io import json_safe, sha256_text

        cfg_h = config_hash(self.cfg)
        # Seed is stored inside the runtime sub-dict (via RuntimeConfig.runtime_report)
        seed = int(signals.metadata.get("runtime", {}).get("seed", signals.metadata.get("seed", 0)))

        sim_meta = signals.metadata
        sim_summary: dict[str, Any] = {
            "duration_ms": sim_meta.get("duration_ms"),
            "dt_ms": sim_meta.get("dt_ms"),
            "seed": seed,
            "n_steps": int(signals.time_ms.shape[0]),
            "record_sources": sim_meta.get("record_sources"),
            "record_fields": sim_meta.get("record_fields"),
        }

        # Deterministic receipt_id based on config, version, simulation, and key runtime metadata
        receipt_payload = {
            "config_hash": cfg_h,
            "jaxfne_version": _JAXFNE_VERSION,
            "simulation": sim_summary,
            "runtime": sim_meta.get("runtime"),
            "condition_name": sim_meta.get("condition_name"),
            "stimulus_schedule": sim_meta.get("stimulus_schedule"),
            "recurrent_backend": sim_meta.get("recurrent_backend"),
            "synaptic_kernel": sim_meta.get("synaptic_kernel"),
            "source_model": sim_meta.get("source_model"),
        }
        receipt_id = sha256_text(
            json.dumps(json_safe(receipt_payload), sort_keys=True, allow_nan=False)
        )[:16]

        truth: dict[str, Any] = {
            "claim_level": "computational_scaffold",
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "field_solver_status": "linear_solver",
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
        }

        claim_labels: dict[str, Any] = {
            "receipt_status": _RECEIPT_SCHEMA_VERSION,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
            "physical_amplitude_calibrated": False,
        }

        backend: dict[str, Any] = {
            "recurrent_backend": signals.metadata.get("recurrent_backend", "dense"),
            "synaptic_kernel": signals.metadata.get("synaptic_kernel", "exponential"),
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "physical_amplitude_calibrated": False,
            "source_model": signals.metadata.get("source_model"),
            "source_bookkeeping": signals.metadata.get("source_bookkeeping"),
        }
        if "edge_list" in self.params:
            edges = self.params["edge_list"]
            backend["edge_list_n_edges"] = int(edges.n_edges)
            backend["edge_list_backend"] = "edge_list_recurrent_v0.0.9"

        return RunReceipt(
            receipt_id=receipt_id,
            jaxfne_version=_JAXFNE_VERSION,
            config_hash=cfg_h,
            simulation=sim_summary,
            signals_summary=signals.summary(),
            truth=truth,
            claim_labels=claim_labels,
            backend=backend,
            tags=dict(tags or {}),
        )


    def compute_readout(
        self,
        signals: Signals,
        specs: "Sequence[ReadoutSpec]",
    ) -> "list[ReadoutResult]":
        """Compute scalar features from Signals according to a list of ReadoutSpecs.

        **Canonical v0.1 workflow method.**  Prefer this over :meth:`probe`
        for declarative, typed feature extraction.

        Args:
            signals: Signals returned by self.simulate().
            specs: Sequence of ReadoutSpec objects declaring what to extract.

        Returns:
            List of ReadoutResult objects in the same order as specs.
            Values are None when not applicable (missing field, unknown metric).

        No physical-amplitude, empirical-validation, or mechanism claim is
        introduced.  All values are proxy/native-current scaffold outputs.
        """
        results: list[ReadoutResult] = []
        for spec in specs:
            if spec.metric not in _KNOWN_READOUT_METRICS:
                results.append(ReadoutResult(
                    spec_name=spec.name,
                    metric=spec.metric,
                    value=None,
                    status="unknown_metric",
                ))
                continue

            dt_ms = (
                float(signals.time_ms[1] - signals.time_ms[0])
                if signals.time_ms.shape[0] > 1
                else 1.0
            )

            # Time slice (optional); negative start is treated as empty window.
            if spec.time_window_ms is not None:
                start_ms, end_ms = spec.time_window_ms
                t0 = max(0, int(start_ms / dt_ms))
                t1 = min(int(signals.time_ms.shape[0]), int(end_ms / dt_ms))
                if t0 >= t1:
                    results.append(ReadoutResult(
                        spec_name=spec.name,
                        metric=spec.metric,
                        value=None,
                        status="empty_time_window",
                    ))
                    continue
                V_m_sl = signals.V_m[t0:t1]
                sp_sl = signals.spikes[t0:t1]
                src_sl = signals.sources[t0:t1] if signals.sources is not None else None
                field_t0, field_t1 = t0, t1
            else:
                V_m_sl = signals.V_m
                sp_sl = signals.spikes
                src_sl = signals.sources
                field_t0, field_t1 = 0, int(signals.time_ms.shape[0])

            if spec.metric == "spike_rate_hz":
                value = float(jnp.mean(sp_sl) * (1000.0 / dt_ms))
            elif spec.metric == "spike_count":
                value = float(jnp.sum(sp_sl))
            elif spec.metric == "mean_V_m":
                value = float(jnp.mean(V_m_sl))
            elif spec.metric == "source_abs_mean":
                if src_sl is None:
                    results.append(ReadoutResult(
                        spec_name=spec.name,
                        metric=spec.metric,
                        value=None,
                        status="missing_sources",
                    ))
                    continue
                value = float(jnp.mean(jnp.abs(src_sl)))
            elif spec.metric in ("csd_abs_mean", "lfp_abs_mean"):
                if signals.field is None:
                    results.append(ReadoutResult(
                        spec_name=spec.name,
                        metric=spec.metric,
                        value=None,
                        status="no_field",
                    ))
                    continue
                arr = signals.field.csd if spec.metric == "csd_abs_mean" else signals.field.lfp
                # Apply time-window slice first, then contact slice.
                arr = arr[field_t0:field_t1]
                if spec.n_contacts_slice is not None:
                    c0, c1 = spec.n_contacts_slice
                    arr = arr[:, c0:c1]
                value = float(jnp.mean(jnp.abs(arr)))
            else:
                value = None

            results.append(ReadoutResult(
                spec_name=spec.name,
                metric=spec.metric,
                value=value,
                status="computed",
            ))
        return results

    def probe(self, signals: Signals, modes: Sequence[str] | None = None) -> dict[str, Any]:
        """Extract named arrays from Signals by mode.

        Compatibility alias retained from v0.0.3–v0.0.14.  For typed,
        declarative feature extraction in the canonical v0.1 workflow, prefer
        :meth:`compute_readout` with :class:`ReadoutSpec` objects.
        """

        modes = list(modes or [])
        out: dict[str, Any] = {"requested_modes": modes}
        if "spikes" in modes:
            out["spikes"] = signals.spikes
        if "V_m" in modes:
            out["V_m"] = signals.V_m
        if "source" in modes or "sources" in modes:
            out["sources"] = signals.sources
        if signals.field is not None:
            out.update(probe_laminar_modes(signals.field, modes))
        return out

    def record(self, signals: Signals, modes: Sequence[str]) -> dict[str, Any]:
        """User-friendly alias for :meth:`probe`."""

        return self.probe(signals, modes)

    def evaluate(
        self,
        signals: Signals,
        objective: "Objective | str",
        readout: Optional[dict[str, Any]] = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Full objective/gate evaluation with JSON-safe report.

        Gate pass/fail is a computational diagnostic only.  It does not imply
        empirical validation, biological calibration, or mechanism proof.
        All truth gates from v0.0.4 are preserved in the report.
        """
        from .io import json_safe

        if isinstance(objective, str):
            objective = Objective(name=objective)

        cfg_meta = self.cfg.metadata
        warnings: list[str] = []

        # Special dispatch for group-rate targets objective
        if getattr(objective, "kind", "generic") == "group_rate_targets":
            return self._evaluate_group_rate_targets(signals, objective, warnings, cfg_meta)

        computed_metrics = _compute_all_metrics(signals, readout)

        loss_results = []
        total_loss = 0.0
        has_loss_value = False
        for spec in getattr(objective, "losses", []):
            r = _evaluate_loss_spec(spec, computed_metrics, warnings, strict)
            loss_results.append(r)
            if r.get("weighted_value") is not None:
                total_loss += r["weighted_value"]
                has_loss_value = True

        reg_results = []
        for spec in getattr(objective, "regularizers", []):
            r = _evaluate_regularizer_spec(spec, computed_metrics, warnings, strict)
            reg_results.append(r)
            if r.get("weighted_value") is not None:
                total_loss += r["weighted_value"]
                has_loss_value = True

        gate_results = []
        all_gates_pass = True
        for spec in getattr(objective, "gates", []):
            r = _evaluate_gate_spec(spec, computed_metrics, warnings, strict)
            gate_results.append(r)
            if not r.get("pass", True):
                all_gates_pass = False

        acceptance = "gates_pass" if all_gates_pass else "gates_fail"

        return json_safe({
            "evaluation_status": "objective_evaluate_v0.0.5",
            "objective_name": getattr(objective, "name", "spectrolaminar_objective"),
            "total_loss": _finite_or_none(total_loss) if has_loss_value else None,
            "losses": loss_results,
            "regularizers": reg_results,
            "gates": gate_results,
            "all_gates_pass": all_gates_pass,
            "acceptance_decision": acceptance,
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
            "warnings": warnings,
        })


    def evaluate_report(
        self,
        signals: Signals,
        objective: "Objective | str",
        *,
        readout_specs: "Optional[Sequence[ReadoutSpec]]" = None,
        readout: Optional[dict[str, Any]] = None,
    ) -> ObjectiveReport:
        """Evaluate an objective and return a structured, immutable ObjectiveReport.

        **Canonical v0.1 workflow method.**  Prefer this over :meth:`evaluate`
        when a typed, JSON-safe, auditable result is needed.

        Wraps :meth:`evaluate` into a frozen dataclass.  Optionally computes
        ReadoutSpecs via :meth:`compute_readout` and embeds results in the report.

        Gate pass/fail is a computational diagnostic only.  No biological
        calibration, no physical-amplitude, empirical-validation, or
        mechanism claim is introduced.

        Args:
            signals: Signals returned by self.simulate().
            objective: Objective or objective name string.
            readout_specs: Optional list of ReadoutSpec for feature extraction.
            readout: Optional readout dict (passed through to evaluate()).

        Returns:
            ObjectiveReport (frozen, JSON-safe).
        """
        eval_dict = self.evaluate(signals, objective, readout=readout)
        rr: tuple[ReadoutResult, ...] = ()
        if readout_specs:
            rr = tuple(self.compute_readout(signals, readout_specs))
        truth: dict[str, Any] = {
            "claim_level": "computational_scaffold",
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "field_solver_status": "linear_solver",
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
        }
        return ObjectiveReport(
            objective_name=eval_dict.get("objective_name", "anonymous"),
            evaluation_status="objective_report_v0.0.18",
            total_loss=eval_dict.get("total_loss"),
            all_gates_pass=bool(eval_dict.get("all_gates_pass", True)),
            losses=tuple(eval_dict.get("losses", [])),
            regularizers=tuple(eval_dict.get("regularizers", [])),
            gates=tuple(eval_dict.get("gates", [])),
            readout_results=rr,
            truth=truth,
            warnings=tuple(eval_dict.get("warnings", [])),
        )

    def _evaluate_group_rate_targets(
        self: "Model",
        signals: Signals,
        objective: "Objective",
        warnings: list[str],
        cfg_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate group-wise firing rate targets objective.

        Extracts group definitions and target rates from objective metadata,
        computes group-wise firing rates, and returns squared relative error loss.
        """
        from .io import json_safe

        # Extract metadata from gates (set by rate_targets())
        groups_dict: Optional[dict[str, Any]] = None
        targets_hz_dict: Optional[dict[str, float]] = None
        weights_dict: Optional[dict[str, float]] = None

        for gate_spec in objective.gates:
            if "metadata" in gate_spec:
                meta = gate_spec["metadata"]
                if "groups" in meta:
                    groups_dict = meta.get("groups")
                    targets_hz_dict = meta.get("targets_hz", {})
                    weights_dict = meta.get("weights", {})
                    break

        if groups_dict is None or targets_hz_dict is None:
            warnings.append("group_rate_targets_missing_metadata")
            return json_safe({
                "evaluation_status": "objective_evaluate_group_rate_targets_v0.0.1",
                "objective_name": getattr(objective, "name", "spectrolaminar_objective"),
                "total_loss": None,
                "losses": [],
                "regularizers": [],
                "gates": [],
                "all_gates_pass": False,
                "acceptance_decision": "gates_fail",
                "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
                "field_claim_level": "proxy_readout",
                "physical_amplitude_calibrated": False,
                "warnings": warnings,
            })

        if weights_dict is None:
            weights_dict = {name: 1.0 for name in groups_dict.keys()}

        # Compute dt from metadata to avoid JAX device-to-host transfer
        dt_ms = float(signals.metadata.get("dt_ms", 0.0))
        if dt_ms <= 0:
            dt_ms = float(signals.time_ms[1] - signals.time_ms[0]) if signals.time_ms.shape[0] > 1 else 0.05
        if dt_ms <= 0:
            dt_ms = 0.05

        # Compute group-wise firing rates and loss
        total_loss = 0.0
        loss_details = []
        all_gates_pass = True

        for group_name in sorted(groups_dict.keys()):
            group_indices = groups_dict[group_name]
            target_hz = float(targets_hz_dict.get(group_name, 10.0))
            weight = float(weights_dict.get(group_name, 1.0))

            # Convert group indices to list of ints
            if isinstance(group_indices, list):
                idx_list = [int(i) for i in group_indices]
            else:
                idx_list = list(group_indices)

            if not idx_list:
                warnings.append(f"group_{group_name}_empty")
                continue

            try:
                # Extract spikes for this group
                group_spikes = signals.spikes[:, idx_list]  # Shape: [n_steps, n_neurons_in_group]

                # Compute mean spike rate over time and neurons in group
                group_rate_hz = float(jnp.mean(group_spikes) * (1000.0 / dt_ms))

                # Compute squared relative error: ((rate - target) / target)^2
                if target_hz == 0:
                    if group_rate_hz == 0:
                        raw_loss = 0.0
                    else:
                        raw_loss = float("inf")
                else:
                    raw_loss = ((group_rate_hz - target_hz) / target_hz) ** 2

                weighted_loss = weight * raw_loss
                total_loss += weighted_loss

                loss_details.append({
                    "group": group_name,
                    "target_hz": float(target_hz),
                    "achieved_hz": _finite_or_none(group_rate_hz),
                    "weight": float(weight),
                    "raw_loss": _finite_or_none(raw_loss),
                    "weighted_loss": _finite_or_none(weighted_loss),
                    "status": "ok",
                })
            except Exception as e:
                warnings.append(f"group_{group_name}_evaluation_error: {str(e)}")
                loss_details.append({
                    "group": group_name,
                    "target_hz": float(target_hz),
                    "achieved_hz": None,
                    "weight": float(weight),
                    "raw_loss": None,
                    "weighted_loss": None,
                    "status": str(e),
                })
                all_gates_pass = False

        # Check if loss is finite
        has_loss_value = math.isfinite(total_loss)
        if not has_loss_value:
            all_gates_pass = False

        acceptance = "gates_pass" if (all_gates_pass and has_loss_value) else "gates_fail"

        return json_safe({
            "evaluation_status": "objective_evaluate_group_rate_targets_v0.0.1",
            "objective_name": getattr(objective, "name", "spectrolaminar_objective"),
            "total_loss": _finite_or_none(total_loss) if has_loss_value else None,
            "group_rate_losses": loss_details,
            "losses": [],
            "regularizers": [],
            "gates": [],
            "all_gates_pass": all_gates_pass,
            "acceptance_decision": acceptance,
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
            "warnings": warnings,
        })

    def tune(
        self: "Model",
        objective: Optional["Objective"] = None,
        optimizer: Any = None,
        steps: int = 0,
        seed: int = 0,
        scope: Optional[str] = None,
        strict: bool = False,
        simulation: Optional[Simulation] = None,
        parameter: Optional[str] = None,
        bounds: Optional[tuple[float, float]] = None,
        # Multi-parameter optimization path
        parameters: Optional[dict[str, tuple[float, float]]] = None,
        generations: Optional[int] = None,
        population_size: Optional[int] = None,
        # New plural form for public API
        objectives: Optional["Objective"] = None,
    ) -> "TuneResult":
        """Run black-box tuning loop (single or multi-parameter).

        Public API: tune(objectives=objectives, optimizer=optimizer, simulation=simulation)
        Returns TuneResult with best_parameters, best_score, history, and summary.

        Legacy API: tune(objective=objective, parameter=..., bounds=...) for backward compatibility.
        Also returns TuneResult (not tuple).

        This is a computational scaffold: no biological calibration, no field-solver upgrade,
        and no optimizer-selected mechanism claim are made.
        """
        from .io import json_safe
        from .optim import _resolve_optimizer, propose_blackbox_candidates, require_optax

        # Normalize objectives vs objective
        if objectives is not None:
            objective = objectives
        elif objective is None:
            raise ValueError("Either 'objective' (legacy) or 'objectives' (public) must be provided")

        cfg_meta = self.cfg.metadata
        spec = _resolve_optimizer(optimizer)
        sim = simulation or Simulation(duration_ms=10.0, dt_ms=0.1, seed=seed)

        # Detect multi-parameter path: either explicit parameters dict, or AGSDROptimizerSpec
        # If optimizer is an AGSDROptimizerSpec, extract parameters from it
        if parameters is None and hasattr(optimizer, "parameters"):
            # optimizer is likely an AGSDROptimizerSpec
            parameters = optimizer.parameters
            if generations is None and hasattr(optimizer, "generations"):
                generations = optimizer.generations
            if population_size is None and hasattr(optimizer, "population_size"):
                population_size = optimizer.population_size

        # Detect multi-parameter path
        if parameters is not None:
            # Extract seed from optimizer if not explicitly overridden via model.tune(..., seed=nonzero)
            opt_seed = getattr(optimizer, "seed", 0)
            actual_seed = seed if seed != 0 else opt_seed
            return self._tune_multiparameter(
                objective=objective,
                optimizer=optimizer,
                spec=spec,
                parameters=parameters,
                generations=generations or 8,
                population_size=population_size or 6,
                seed=int(actual_seed),
                strict=strict,
                simulation=sim,
            )

        # Single-parameter path (backward compat)
        if parameter is None:
            parameter = "source_scale"
        if bounds is None:
            bounds = (0.25, 4.0)

        n_steps = max(0, int(steps))
        base_report: dict[str, Any] = {
            "same_model_unchanged": True,
            "steps_requested": n_steps,
            "seed": int(seed),
            "scope": scope or spec.optimizer,
            "parameter": parameter,
            "bounds": [float(bounds[0]), float(bounds[1])],
            "optimizer": spec.to_dict(),
            "objective_name": getattr(objective, "name", "spectrolaminar_objective") if not isinstance(objective, str) else objective,
            "losses_declared": len(getattr(objective, "losses", [])) if not isinstance(objective, str) else 0,
            "regularizers_declared": len(getattr(objective, "regularizers", [])) if not isinstance(objective, str) else 0,
            "gates_declared": len(getattr(objective, "gates", [])) if not isinstance(objective, str) else 0,
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "source_calibration_status": cfg_meta.get(
                "source_calibration_status", "uncalibrated_izhikevich_native_current"
            ),
            "source_projection_mode": cfg_meta.get("source_projection_mode", "proxy_no_field_solve"),
            "field_solver_status": cfg_meta.get("field_solver_status", "linear_solver"),
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
            "biological_learning_claim": False,
            "amplitude_claim_allowed": False,
            "surrogate_status": spec.surrogate_status,
            "final_evaluation_basis": "total_loss",
        }

        if spec.is_differentiable_path():
            if not spec.gradient_path_safe():
                report = {
                    **base_report,
                    "tuning_status": "blocked_non_differentiable_path",
                    "acceptance_decision": "REVISE",
                    "warnings": [
                        "optax_requires_differentiable_or_declared_surrogate",
                        "spiking_reset_not_differentiable_without_surrogate",
                    ],
                }
                return TuneResult(
                    best_parameters={},
                    best_score=float("inf"),
                    history=[],
                    summary=json_safe(report),
                    model=self,
                )
            try:
                require_optax()
                optax_status = "available"
            except ImportError:
                if strict:
                    raise
                optax_status = "unavailable"
            report = {
                **base_report,
                "tuning_status": "optax_guarded_path_no_loop_v0.0.8",
                "acceptance_decision": "REVISE" if optax_status == "unavailable" else "ACCEPT_CANDIDATE",
                "optax_status": optax_status,
                "same_model_unchanged": True,
                "warnings": ["differentiable_loop_not_enabled_for_spiking_reset_without_explicit_surrogate_kernel"],
            }
            return TuneResult(
                best_parameters={},
                best_score=float("inf"),
                history=[],
                summary=json_safe(report),
                model=self,
            )

        if n_steps <= 0:
            report = {
                **base_report,
                "tuning_status": "metadata_only_no_steps_requested",
                "acceptance_decision": "REVISE",
                "candidate_history": [],
                "warnings": ["no_blackbox_steps_requested"],
            }
            return TuneResult(
                best_parameters={},
                best_score=float("inf"),
                history=[],
                summary=json_safe(report),
                model=self,
            )

        candidates = propose_blackbox_candidates(
            optimizer=spec,
            n_steps=n_steps,
            seed=int(seed),
            bounds=(float(bounds[0]), float(bounds[1])),
        )
        best_model: Model = self
        best_loss: Optional[float] = None
        best_value: Optional[float] = None
        history: list[dict[str, Any]] = []
        warnings: list[str] = []
        for idx, candidate_value in enumerate(candidates):
            candidate_model = _model_with_scalar_parameter(self, parameter, float(candidate_value))
            candidate_signals = candidate_model.simulate(replace(sim, seed=int(seed) + idx))
            candidate_report = candidate_model.evaluate(candidate_signals, objective, strict=strict)
            score = candidate_report.get("total_loss")
            gates_pass = bool(candidate_report.get("all_gates_pass", False))
            if score is None:
                score = 0.0 if gates_pass else float("inf")
            score = float(score)
            accepted = math.isfinite(score) and (best_loss is None or score < best_loss)
            if accepted:
                best_loss = score
                best_value = float(candidate_value)
                best_model = candidate_model

            reasons = []
            if not gates_pass:
                reasons.append("failed_objective_gates")
            if not math.isfinite(score):
                reasons.append("non_finite_loss")

            history.append({
                "step": idx,
                "candidate_value": float(candidate_value),
                "score": _finite_or_none(score),
                "all_gates_pass": gates_pass,
                "accepted_as_best": bool(accepted),
                "evaluation_status": candidate_report.get("evaluation_status"),
                "rejection_reasons": reasons,
            })
        if best_loss is None:
            warnings.append("no_finite_candidate_score")
            best_model = self

        # Compute candidate statistics for enhanced report
        candidate_values = [float(h["candidate_value"]) for h in history]
        candidate_scores = [h.get("score") for h in history]
        finite_scores = [s for s in candidate_scores if s is not None and math.isfinite(s)]

        score_variance = 0.0
        n_unique_scores = 0
        if len(finite_scores) > 1:
            score_variance = float(jnp.var(jnp.asarray(finite_scores)))
            n_unique_scores = len(set(finite_scores))

        report = {
            **base_report,
            "same_model_unchanged": best_model is self,
            "tuning_status": "blackbox_loop_v0.0.6",
            "acceptance_decision": "ACCEPT_CANDIDATE" if best_loss is not None else "REVISE",
            "best_parameter_value": best_value,
            "best_score": _finite_or_none(best_loss) if best_loss is not None else None,
            "candidate_values": candidate_values,
            "candidate_scores": candidate_scores,
            "score_variance": score_variance,
            "n_unique_scores": n_unique_scores,
            "tuning_path": "scalar_black_box",
            "candidate_history": history,
            "warnings": warnings + [
                "blackbox_loop_is_computational_scaffold_only",
                "optimizer_selected_candidate_is_not_biological_truth",
            ],
        }
        # Return TuneResult (new public API)
        # Note: model not included in summary (would not be JSON-safe)
        # Access tuned model separately: model_result = model.tune(...); print(model_result.summary)
        return TuneResult(
            best_parameters={"best_value": best_value} if best_value is not None else {},
            best_score=float(best_loss) if best_loss is not None else float("inf"),
            history=history,
            summary=json_safe(report),
            model=best_model,
        )

    def _tune_multiparameter(
        self,
        objective: "Objective",
        optimizer: Any,
        spec: "OptimizerSpec",
        parameters: Any,
        generations: int,
        population_size: int,
        seed: int,
        strict: bool,
        simulation: "Simulation",
    ) -> "TuneResult":
        """Run multi-parameter AGSDR optimization loop.

        This is an internal helper called by tune() when the multi-parameter
        path is requested (parameters dict provided).

        If any parameter values are :class: objects and the
        optimizer has an inner_optimizer, routes to the two-level AGSDR+Adam path.
        Otherwise uses the scalar AGSDR black-box path.
        """
        from .io import json_safe
        from .optim import (
            _run_agsdr_optimization_loop,
            _tune_matrix_agsdr_optax,
        )

        cfg_meta = self.cfg.metadata

        # Build base report
        base_report: dict[str, Any] = {
            "same_model_unchanged": True,
            "seed": int(seed),
            "scope": "agsdr_multiparameter",
            "parameters": {
                k: (
                    {"type": "MatrixParameterSpec", "mask": v.mask, "bounds": list(v.bounds)}
                    if isinstance(v, MatrixParameterSpec)
                    else [float(v[0]), float(v[1])]
                )
                for k, v in parameters.items()
            },
            "generations": int(generations),
            "population_size": int(population_size),
            "optimizer": spec.to_dict(),
            "objective_name": getattr(objective, "name", "spectrolaminar_objective") if not isinstance(objective, str) else objective,
            "losses_declared": len(getattr(objective, "losses", [])) if not isinstance(objective, str) else 0,
            "regularizers_declared": len(getattr(objective, "regularizers", [])) if not isinstance(objective, str) else 0,
            "gates_declared": len(getattr(objective, "gates", [])) if not isinstance(objective, str) else 0,
            "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
            "source_calibration_status": cfg_meta.get(
                "source_calibration_status", "uncalibrated_izhikevich_native_current"
            ),
            "source_projection_mode": cfg_meta.get("source_projection_mode", "proxy_no_field_solve"),
            "field_solver_status": cfg_meta.get("field_solver_status", "linear_solver"),
            "field_claim_level": "proxy_readout",
            "physical_amplitude_calibrated": False,
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
            "biological_learning_claim": False,
            "amplitude_claim_allowed": False,
            "surrogate_status": spec.surrogate_status,
            "final_evaluation_basis": "total_loss",
        }

        # Separate scalar bounds from MatrixParameterSpec entries
        param_specs: dict[str, Any] = {}
        scalar_bounds: dict[str, tuple] = {}
        for k, v in parameters.items():
            if isinstance(v, MatrixParameterSpec):
                param_specs[k] = v
                scalar_bounds[k] = v.bounds
            else:
                scalar_bounds[k] = tuple(v)

        # Two-level dispatch: matrix + inner_optimizer => AGSDR+Adam path
        inner_optimizer = getattr(optimizer, "inner_optimizer", None)
        inner_steps = getattr(optimizer, "inner_steps", 0)
        inner_objective = getattr(optimizer, "inner_objective", None)
        has_matrix = bool(param_specs)

        if has_matrix and inner_optimizer is not None:
            return _tune_matrix_agsdr_optax(
                model=self,
                objective=objective,
                parameters=parameters,
                param_specs=param_specs,
                scalar_bounds=scalar_bounds,
                inner_optimizer=inner_optimizer,
                inner_steps=inner_steps,
                inner_objective=inner_objective,
                spec=spec,
                generations=generations,
                population_size=population_size,
                seed=seed,
                strict=strict,
                simulation=simulation,
                base_report=base_report,
            )

        rejections_map = []

        # Define scoring function for AGSDR loop
        def evaluate_fn(candidate_params: dict[str, float]) -> float:
            """Evaluate a candidate parameter dict and return loss."""
            reasons = []
            candidate_model = _model_with_parameters(self, candidate_params, param_specs if param_specs else None)
            candidate_signals = candidate_model.simulate(simulation)
            candidate_report = candidate_model.evaluate(candidate_signals, objective, strict=strict)
            score = candidate_report.get("total_loss")
            gates_pass = bool(candidate_report.get("all_gates_pass", False))
            if score is None:
                score = 0.0 if gates_pass else float("inf")
            score = float(score)

            if not gates_pass:
                reasons.append("failed_objective_gates")
            if not math.isfinite(score):
                reasons.append("non_finite_loss")

            rejections_map.append(reasons)

            return float(score)

        # Run AGSDR optimization (scalar_bounds only, not MatrixParameterSpec objects)
        try:
            agsdr_result = _run_agsdr_optimization_loop(
                evaluate_fn=evaluate_fn,
                parameter_bounds=scalar_bounds,
                n_generations=int(generations),
                n_population=int(population_size),
                alpha=float(spec.alpha),
                exploration=float(spec.exploration),
                seed=int(seed),
                rejections_map=rejections_map,
            )

            best_parameters = agsdr_result["best_parameters"]
            best_score = agsdr_result["best_score"]
            generation_records = agsdr_result["generation_records"]

            # Apply best parameters to model
            best_model = _model_with_parameters(self, best_parameters, param_specs if param_specs else None)

            # Build detailed report
            report = {
                **base_report,
                "same_model_unchanged": False,
                "tuning_status": "multiparameter_agsdr_v0.0.7",
                "acceptance_decision": "ACCEPT_CANDIDATE" if math.isfinite(best_score) else "REVISE",
                "best_parameters": best_parameters,
                "best_score": _finite_or_none(best_score),
                "generation_records": generation_records,
                "all_scores": agsdr_result["all_scores"],
                "n_candidates_evaluated": len(agsdr_result["all_scores"]),
                "tuning_path": "multiparameter_black_box",
                "warnings": [
                    "blackbox_loop_is_computational_scaffold_only",
                    "optimizer_selected_candidate_is_not_biological_truth",
                ],
            }

            return TuneResult(
                best_parameters=best_parameters,
                best_score=float(best_score) if math.isfinite(best_score) else float("inf"),
                history=generation_records,
                summary=json_safe(report),
                model=best_model,
            )

        except Exception as e:
            report = {
                **base_report,
                "tuning_status": "multiparameter_agsdr_error",
                "acceptance_decision": "REVISE",
                "error": str(e),
                "warnings": ["multiparameter_optimization_failed"],
            }
            return TuneResult(
                best_parameters={},
                best_score=float("inf"),
                history=[],
                summary=json_safe(report),
                model=self,
            )

    def with_emitter_parameters(
        self,
        *,
        a: "float | None" = None,
        b: "float | None" = None,
        c: "float | None" = None,
        d: "float | None" = None,
        drive_scale: "float | None" = None,
        # New per-neuron overrides (v0.3.3)
        a_per_neuron: "jax.Array | None" = None,
        b_per_neuron: "jax.Array | None" = None,
        c_per_neuron: "jax.Array | None" = None,
        d_per_neuron: "jax.Array | None" = None,
        drive_per_neuron: "jax.Array | None" = None,
    ) -> "Model":
        """Return a new Model with Izhikevich parameter overrides.

        Supports both scalar (uniform) and per-neuron (array) overrides.
        Per-neuron arrays take priority over scalar values.
        Explicit None checks are used to handle zero-valued arrays correctly.

        Args:
            a: Scalar recovery time scale override (uniform to all neurons).
            b: Scalar voltage-sensitivity override (uniform).
            c: Scalar post-spike reset override (uniform).
            d: Scalar post-spike increment override (uniform).
            drive_scale: Multiplicative gain on native drive.
            a_per_neuron: Per-neuron recovery time scale (shape: [n_neurons]).
            b_per_neuron: Per-neuron voltage sensitivity (shape: [n_neurons]).
            c_per_neuron: Per-neuron reset voltage (shape: [n_neurons]).
            d_per_neuron: Per-neuron recovery increment (shape: [n_neurons]).
            drive_per_neuron: Per-neuron absolute drive (shape: [n_neurons]).
                Overrides both scalar drive_scale and emitter.drive.

        Returns:
            New Model — original is not mutated.
        """
        emitter: IzhikevichParams = self.params["emitter"]
        updates: dict[str, Any] = {}

        # a: per-neuron takes priority over scalar
        if a_per_neuron is not None:
            updates["a"] = jnp.asarray(a_per_neuron, dtype=emitter.a.dtype)
        elif a is not None:
            updates["a"] = jnp.ones_like(emitter.a) * float(a)

        # b: per-neuron takes priority over scalar
        if b_per_neuron is not None:
            updates["b"] = jnp.asarray(b_per_neuron, dtype=emitter.b.dtype)
        elif b is not None:
            updates["b"] = jnp.ones_like(emitter.b) * float(b)

        # c: per-neuron takes priority over scalar
        if c_per_neuron is not None:
            updates["c"] = jnp.asarray(c_per_neuron, dtype=emitter.c.dtype)
        elif c is not None:
            updates["c"] = jnp.ones_like(emitter.c) * float(c)

        # d: per-neuron takes priority over scalar
        if d_per_neuron is not None:
            updates["d"] = jnp.asarray(d_per_neuron, dtype=emitter.d.dtype)
        elif d is not None:
            updates["d"] = jnp.ones_like(emitter.d) * float(d)

        # drive: per-neuron absolute takes priority; scalar applies multiplicative scale
        if drive_per_neuron is not None:
            updates["drive"] = jnp.asarray(drive_per_neuron, dtype=emitter.drive.dtype)
        elif drive_scale is not None:
            updates["drive"] = emitter.drive * float(drive_scale)

        new_emitter = replace(emitter, **updates)
        new_params = dict(self.params)
        new_params["emitter"] = new_emitter
        return replace(self, params=new_params)

    def with_hdp_initial_state(
        self,
        *,
        H0: "jax.Array | None" = None,
        w0: "jax.Array | None" = None,
    ) -> "Model":
        """Return a new Model with a custom initial HDP controller state.

        Only takes effect when HDP is separately enabled via
        ``Configuration.hdp(...)`` (``RuntimeConfig.enable_hdp=True``); stored
        but inert otherwise, mirroring :meth:`with_emitter_parameters`'s
        additive-override pattern.

        Args:
            H0: Per-neuron initial homeostatic factor (shape: [n_neurons]).
                Defaults to the HDP kernel's own equilibrium value (1.0 for
                every neuron) when not provided.
            w0: Per-edge initial weight (shape: [n_edges], aligned to
                ``self.params["edge_list"]``). Defaults to the edge list's
                native ``weight`` when not provided.

        Returns:
            New Model — original is not mutated.
        """
        # Local import: _runtime_config_from_metadata stays in core.py (used by
        # both Model and the module-level simulate() function); deferring the
        # import here avoids a circular import with core.py's own
        # `from ._model import Model`.
        from .core import _runtime_config_from_metadata
        jdtype = _runtime_config_from_metadata(self.cfg.metadata).jnp_dtype
        new_params = dict(self.params)
        if H0 is not None:
            new_params["hdp_initial_H"] = jnp.asarray(H0, dtype=jdtype)
        if w0 is not None:
            new_params["hdp_initial_w"] = jnp.asarray(w0, dtype=jdtype)
        return replace(self, params=new_params)

    def with_recurrent_coupling(
        self,
        *,
        g_ei: float = 5.0,
        g_ie: float = 3.0,
        tau_syn_e_ms: float = 5.0,
        tau_syn_i_ms: float = 10.0,
    ) -> "Model":
        """Return a new Model with recurrent E/I coupling parameters stored.

        Stores coupling parameters in model.static["recurrent_coupling"] for
        use with simulate_dynamic_ei_coupling(). The original model is not mutated
        (frozen dataclass contract is preserved via replace()).

        Coupling is stored as metadata; it does not modify the emitter's W matrix.
        Use with simulate_dynamic_ei_coupling() to apply dynamic coupling at runtime.

        Args:
            g_ei: E→I excitatory coupling conductance (model units).
            g_ie: I→E inhibitory coupling magnitude (model units).
            tau_syn_e_ms: Excitatory synaptic time constant (ms).
            tau_syn_i_ms: Inhibitory synaptic time constant (ms).

        Returns:
            New Model with coupling parameters in static["recurrent_coupling"].
            Original model is not mutated.
        """
        coupling_params = {
            "g_ei": float(g_ei),
            "g_ie": float(g_ie),
            "tau_syn_e_ms": float(tau_syn_e_ms),
            "tau_syn_i_ms": float(tau_syn_i_ms),
            "source_calibration_status": "uncalibrated_izhikevich_native_current",
            "physical_amplitude_calibrated": False,
            "claim_level": "computational_scaffold",
        }
        return replace(
            self,
            static={**self.static, "recurrent_coupling": coupling_params}
        )

    def manifest(
        self,
        signals: Optional[Signals] = None,
        readout: Optional[Any] = None,
        paradigm: Optional[dict[str, Any]] = None,
        objective: Optional[dict[str, Any]] = None,
        evaluation: Optional[dict[str, Any]] = None,
        tuning: Optional[dict[str, Any]] = None,
        dataset: Optional[dict[str, Any]] = None,
        trials: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build a JSON-safe run manifest dict.

        Compatibility method retained from v0.0.4–v0.0.14.  For the canonical
        v0.1 workflow, prefer :meth:`run_receipt` (typed, immutable, with
        deterministic receipt ID) and :meth:`evaluate_report` (typed objective
        evaluation).  This method remains supported and is not scheduled for
        removal.

        The ``readout`` argument accepts any of:

        * ``None`` — no readout section included.
        * ``dict`` — passed through to the manifest as-is (legacy shape).
        * ``list`` or ``tuple`` of :class:`ReadoutResult` objects — the canonical
          output of :meth:`compute_readout`.  Converted to a JSON-safe readout
          summary dict with ``readout_results`` and ``requested_metrics`` keys.
        * ``list`` or ``tuple`` of ``dict`` — same conversion applied to each element.
        * Single :class:`ReadoutResult` — wrapped in a list and handled as above.
        """
        readout_normalized = _normalize_manifest_readout(readout)
        runtime_cfg = None
        if signals is not None and "runtime" in signals.metadata:
            runtime_cfg = _RuntimeReportAdapter(signals.metadata["runtime"])
        res = build_manifest(
            self.cfg,
            signals=signals,
            readout=readout_normalized,
            runtime_config=runtime_cfg,
            paradigm=paradigm,
            objective=objective,
            evaluation=evaluation,
            tuning=tuning,
            dataset=dataset,
        )
        if trials is not None:
            res["trials"] = trials
        # If readout was provided as ReadoutResult list (canonical v0.1 workflow),
        # surface the normalized readout summary in the manifest under "readout_results".
        # Dict-shaped readouts are already surfaced via build_manifest's field_diagnostics
        # logic; non-dict shapes are added here only.
        if readout_normalized is not None and isinstance(readout_normalized, dict):
            if "readout_results" in readout_normalized:
                res["readout_results"] = readout_normalized
        # Backend metadata: distinguish executed backend from available infrastructure.
        used_backend = "dense"
        used_kernel = "exponential"
        if signals is not None:
            used_backend = signals.metadata.get("recurrent_backend", "dense")
            used_kernel = signals.metadata.get("synaptic_kernel", "exponential")
        elif "edge_list" in self.params:
            used_backend = "unknown_not_run"
        backend_meta: dict[str, Any] = {
            "used_recurrent_backend": used_backend,
            "used_synaptic_kernel": used_kernel,
            "available_edge_list": "edge_list" in self.params,
            # Renamed 2026-07-05 from "manifest_schema_version" -- that name
            # collided (same key, different meaning/value) with the manifest
            # ROOT's own "manifest_schema_version" (Configuration's
            # _default_metadata, "0.0.4"). Both coexisted without silently
            # overwriting each other (different nesting depth), but the
            # shared name was a real trap for a future reader. This key
            # versions only this backend_metadata block, not the manifest.
            "backend_metadata_schema_version": _MANIFEST_SCHEMA_VERSION,
            "source_model": dict(_SOURCE_PROXY_METADATA),
        }
        # v0.2.0: Field admissibility metadata
        if signals is not None and signals.field is not None:
            from .validation import build_field_admissibility_report
            field_admissibility = build_field_admissibility_report(
                field_output=signals.field,
                cfg_metadata=dict(self.cfg.metadata or {}),
            )
            backend_meta["field_admissibility"] = field_admissibility
            if "field_admissibility" in signals.field.diagnostics:
                backend_meta["field_admissibility_diagnostics"] = signals.field.diagnostics.get(
                    "field_admissibility"
                )
        if "edge_list" in self.params:
            edges = self.params["edge_list"]
            backend_meta["edge_count"] = int(edges.n_edges)
            backend_meta["receptor_indexed"] = True
            backend_meta["edge_list_source_calibration_status"] = edges.source_calibration_status
            backend_meta["edge_list_physical_amplitude_calibrated"] = False
            # v0.0.21: explicitly document which tau source each kernel uses.
            # simulate_edge_recurrent_izhikevich → edges.tau_ms (per-edge field)
            # simulate_receptor_exponential_izhikevich → standard_receptor_tau_table
            #   (receptor_index → standard catalog). Current standard table agrees
            #   with make_edge_list_from_dense for receptor_index ∈ {0, 1}, so
            #   these are numerically equivalent in the default scaffold flow.
            backend_meta["receptor_tau_source"] = {
                "exponential_kernel_uses": "edges.tau_ms",
                "receptor_exponential_kernel_uses": "standard_receptor_tau_table_by_receptor_index",
                "consistent_for_receptor_index_in": [0, 1],
            }
            # v0.0.21: surface receptor spec metadata so manifest documents
            # the receptor labels/taus the kernel can index. The actual per-edge
            # tau_ms lives on EdgeList; this is the catalog.
            from .emitters import standard_receptor_specs
            backend_meta["receptor_specs"] = {
                name: {
                    "name": spec.name,
                    "receptor_index": spec.receptor_index,
                    "sign": spec.sign,
                    "tau_ms": spec.tau_ms,
                    "reversal_mV": spec.reversal_mV,
                    "source_calibration_status": spec.source_calibration_status,
                }
                for name, spec in standard_receptor_specs().items()
            }
        # v0.0.21: explicit source model in manifest.
        res["source_model"] = dict(_SOURCE_PROXY_METADATA)
        res["backend_metadata"] = backend_meta
        if "geometry" in self.static:
            res["source_geometry"] = self.static["geometry"]
        # v0.2.26: computation-basis block
        res["basis"] = _default_basis_dict()
        # v0.2.27: conservation-inspired proxy diagnostics
        if signals is not None and signals.field is not None:
            from .fields import compute_conservation_proxy_diagnostics
            _src_cal = (
                signals.metadata.get("source_calibration_status",
                                     "uncalibrated_izhikevich_native_current")
            )
            res["conservation_proxy_diagnostics"] = compute_conservation_proxy_diagnostics(
                field_solution=signals.field,
                source_calibration_status=_src_cal,
                field_solver_status="linear_solver",
                field_claim_level="proxy_readout",
            )
        return res


def _model_with_scalar_parameter(model: Model, parameter: str, value: float) -> Model:
    """Return a Model copy with one safe scalar emitter parameter changed.

    Supported parameters:
    - source_scale: multiplicative gain on all source signals
    - drive_gain: multiplicative gain on all drive signals
    - synaptic_gain: multiplicative gain on all synaptic weights
    - drive_scale_a: multiplicative gain on first-half neuron drive signals
    - drive_scale_b: multiplicative gain on second-half neuron drive signals
    - gAMPA: multiplicative gain on all excitatory (positive) synaptic weights
    """
    import numpy as np

    emitter = model.params["emitter"]
    value = float(value)

    if parameter == "source_scale":
        new_emitter = replace(emitter, source_scale=jnp.asarray(value, dtype=emitter.source_scale.dtype))
    elif parameter == "drive_gain":
        new_emitter = replace(emitter, drive=emitter.drive * jnp.asarray(value, dtype=emitter.drive.dtype))
    elif parameter == "synaptic_gain":
        new_emitter = replace(emitter, W=emitter.W * jnp.asarray(value, dtype=emitter.W.dtype))
    elif parameter in ("drive_scale_a", "drive_scale_b"):
        import numpy as _np_dsa
        base_drive = _np_dsa.asarray(emitter.drive, dtype=float).reshape(-1)
        n_units = base_drive.shape[0]
        split = n_units // 2
        drive_scale = _np_dsa.ones(n_units, dtype=float)
        if parameter == "drive_scale_a":
            drive_scale[:split] = value
        else:
            drive_scale[split:] = value
        drive_per_neuron = base_drive * drive_scale
        new_emitter = replace(emitter, drive=jnp.asarray(drive_per_neuron, dtype=emitter.drive.dtype))
    elif parameter == "gAMPA":
        import numpy as np
        W = np.asarray(emitter.W, dtype=float)
        new_W = W.copy()
        # Scale only excitatory (positive) weights
        new_W[W > 0] = W[W > 0] * value
        new_emitter = replace(emitter, W=jnp.asarray(new_W, dtype=emitter.W.dtype))
    else:
        supported = ["source_scale", "drive_gain", "synaptic_gain", "drive_scale_a", "drive_scale_b", "gAMPA"]
        raise ValueError(
            f"Unsupported tunable parameter: {parameter!r}. "
            f"Supported parameters: {supported}"
        )
    params = dict(model.params)
    params["emitter"] = new_emitter
    return Model(cfg=model.cfg, params=params, static=dict(model.static))


def _mask_for_parameter(model: "Model", parameter_name: str, mask_type: str) -> "jax.Array":
    """Return a boolean mask over the W matrix for the given mask type.

    Parameters
    ----------
    model : Model
        Model whose W matrix determines the mask shape.
    parameter_name : str
        Name of the parameter (used only for error messages).
    mask_type : str
        One of: "E_to_E", "E_to_I", "excitatory_to_all", "all".

    Returns
    -------
    jax.Array
        Boolean mask of shape (n, n) where True marks entries to scale.
    """
    import numpy as _np_mask
    emitter = model.params["emitter"]
    W = _np_mask.asarray(emitter.W, dtype=float)
    n = W.shape[0]

    if mask_type == "all":
        return jnp.ones((n, n), dtype=bool)

    if mask_type == "excitatory_to_all":
        # Scale entries in rows corresponding to excitatory neurons (positive out-degree).
        # We identify E neurons by their sign label or by majority positive outgoing weights.
        row_mask = _np_mask.zeros(n, dtype=bool)
        for i in range(n):
            # E neurons: positive outgoing (sign > 0) or labeled E
            label = emitter.labels[i] if i < len(emitter.labels) else ""
            if label.startswith("E") or _np_mask.sum(W[i, :] > 0) > _np_mask.sum(W[i, :] < 0):
                row_mask[i] = True
        return jnp.asarray(_np_mask.outer(row_mask, _np_mask.ones(n, dtype=bool)), dtype=bool)

    # E_to_E and E_to_I: identify E vs I neurons by labels
    e_mask = _np_mask.zeros(n, dtype=bool)
    i_mask = _np_mask.zeros(n, dtype=bool)
    for idx in range(n):
        label = emitter.labels[idx] if idx < len(emitter.labels) else ""
        if label.startswith("E"):
            e_mask[idx] = True
        else:
            i_mask[idx] = True

    if mask_type == "E_to_E":
        return jnp.asarray(_np_mask.outer(e_mask, e_mask), dtype=bool)
    if mask_type == "E_to_I":
        return jnp.asarray(_np_mask.outer(e_mask, i_mask), dtype=bool)

    raise ValueError(
        f"Unknown mask type for parameter {parameter_name!r}: {mask_type!r}. "
        "Supported: E_to_E, E_to_I, excitatory_to_all, all"
    )


def _model_with_matrix_parameter(
    model: "Model",
    parameter_name: str,
    spec: "MatrixParameterSpec",
    value: float,
) -> "Model":
    """Return a Model copy with a matrix parameter scaled by value.

    The value is treated as a multiplicative scale factor applied to
    the subset of W entries selected by spec.mask.  The result is then
    clipped to spec.bounds.

    Parameters
    ----------
    model : Model
        Original model (not mutated).
    parameter_name : str
        Name of the parameter (for diagnostics).
    spec : MatrixParameterSpec
        Matrix parameter specification.
    value : float
        Multiplicative scale factor (clipped to spec.bounds).

    Returns
    -------
    Model
        New model with scaled matrix entries.
    """
    import numpy as _np_matrix
    lo, hi = float(spec.bounds[0]), float(spec.bounds[1])
    value = float(_np_matrix.clip(value, lo, hi))

    emitter = model.params["emitter"]
    W = _np_matrix.asarray(emitter.W, dtype=float)
    mask = _np_matrix.asarray(_mask_for_parameter(model, parameter_name, spec.mask), dtype=bool)

    new_W = W.copy()
    new_W[mask] = W[mask] * value
    new_emitter = replace(emitter, W=jnp.asarray(new_W, dtype=emitter.W.dtype))
    params = dict(model.params)
    params["emitter"] = new_emitter
    return Model(cfg=model.cfg, params=params, static=dict(model.static))


def _model_with_parameters(
    model: "Model",
    parameters: Any,
    param_specs: Optional[Any] = None,
) -> "Model":
    """Return a Model copy with multiple emitter parameters changed.

    Dispatches each parameter to the scalar or matrix path depending on
    whether param_specs contains a :class: for that name.

    Parameters
    ----------
    model : Model
        Original model (not mutated).
    parameters : dict[str, float]
        Mapping from parameter names to float values.
    param_specs : dict[str, Any], optional
        Mapping from parameter names to spec objects (e.g. MatrixParameterSpec).
        When None, all parameters are treated as scalars.

    Returns
    -------
    Model
        New model with all parameters updated.
    """
    result = model
    for param_name, param_value in parameters.items():
        if param_specs is not None and param_name in param_specs:
            spec = param_specs[param_name]
            if isinstance(spec, MatrixParameterSpec):
                result = _model_with_matrix_parameter(result, param_name, spec, float(param_value))
                continue
        result = _model_with_scalar_parameter(result, param_name, float(param_value))
    return result


def _evaluate_soft_rate_targets(
    V_m: "jax.Array",
    groups: dict,
    targets_hz: dict,
    duration_ms: float,
    dt_ms: float,
    threshold: float = -45.0,
    temperature: float = 5.0,
) -> "jax.Array":
    """Compute a differentiable soft firing-rate MSE loss using a sigmoid spike surrogate.

    This function is used in the Adam inner loop for matrix parameter optimization.
    It provides smooth gradients through the spike threshold by approximating
    spike probability with a sigmoid function, making the loss differentiable
    with respect to V_m (and thus to the weight matrix W).

    Parameters
    ----------
    V_m : jax.Array
        Membrane voltages, shape (n_steps, n_neurons).
    groups : dict
        Mapping from group name to list of neuron indices.
    targets_hz : dict
        Mapping from group name to target firing rate in Hz.
    duration_ms : float
        Simulation duration in milliseconds.
    dt_ms : float
        Simulation time step in milliseconds.
    threshold : float
        Spike threshold in mV (default -45 mV for Izhikevich scaffold).
    temperature : float
        Sigmoid sharpness (lower = sharper threshold, default 5.0).

    Returns
    -------
    jax.Array
        Scalar MSE loss (differentiable).
    """
    duration_s = duration_ms / 1000.0

    # Soft spike approximation: sigmoid((V_m - threshold) / temperature)
    soft_spikes = jax.nn.sigmoid((V_m - threshold) / temperature)

    total_loss = jnp.zeros((), dtype=jnp.float32)
    for group_name, idx_list in groups.items():
        if not idx_list:
            continue
        idx_arr = jnp.asarray(idx_list, dtype=jnp.int32)
        group_soft = soft_spikes[:, idx_arr]  # (n_steps, n_group)
        n_neurons = group_soft.shape[1]
        # Soft rate in Hz: sum over steps / (duration_s * n_neurons)
        soft_rate_hz = jnp.sum(group_soft) / (duration_s * n_neurons)
        target_hz = float(targets_hz.get(group_name, 10.0))
        target_arr = jnp.asarray(target_hz, dtype=jnp.float32)
        # Normalized MSE
        denom = jnp.maximum(jnp.abs(target_arr), jnp.asarray(1.0, dtype=jnp.float32))
        loss_i = ((soft_rate_hz - target_arr) / denom) ** 2
        total_loss = total_loss + loss_i

    return total_loss


@dataclass(frozen=True)
class _RuntimeReportAdapter:
    report: dict[str, Any]

    def runtime_report(self) -> dict[str, Any]:
        """Documented public function `runtime_report`."""
        return self.report


def _mean_pairwise_corr_proxy(spikes: jax.Array) -> jax.Array:
    x = spikes.astype(jnp.float32)
    x = x - jnp.mean(x, axis=0, keepdims=True)
    denom = jnp.std(x, axis=0, keepdims=True) + 1e-6
    z = x / denom
    corr = (z.T @ z) / jnp.maximum(1, z.shape[0] - 1)
    n = corr.shape[0]
    mask = 1.0 - jnp.eye(n)
    return jnp.sum(jnp.abs(corr) * mask) / jnp.maximum(1.0, jnp.sum(mask))


def with_emitter_parameters(
    model: Model,
    *,
    a: "float | None" = None,
    b: "float | None" = None,
    c: "float | None" = None,
    d: "float | None" = None,
    drive_scale: "float | None" = None,
    a_per_neuron: "jax.Array | None" = None,
    b_per_neuron: "jax.Array | None" = None,
    c_per_neuron: "jax.Array | None" = None,
    d_per_neuron: "jax.Array | None" = None,
    drive_per_neuron: "jax.Array | None" = None,
) -> Model:
    """Functional wrapper for :meth:`Model.with_emitter_parameters`.

    Supports both scalar (uniform) and per-neuron (array) overrides.
    Per-neuron arrays take priority over scalars.
    Explicit None checks used — zero-valued JAX arrays handled correctly.
    """
    return model.with_emitter_parameters(
        a=a, b=b, c=c, d=d, drive_scale=drive_scale,
        a_per_neuron=a_per_neuron,
        b_per_neuron=b_per_neuron,
        c_per_neuron=c_per_neuron,
        d_per_neuron=d_per_neuron,
        drive_per_neuron=drive_per_neuron,
    )



_JAXFNE_VERSION = "0.4.6"
_RECEIPT_SCHEMA_VERSION = "run_receipt_v0.0.21"
_MANIFEST_SCHEMA_VERSION = "manifest.v0.0.21"


# v0.0.21: explicit source proxy metadata.
# Documents what the current Izhikevich scaffold computes as the "source" field.
# Reading the edge/dense kernels: source_proxy = source_scale * (current_native
# + DEFAULT_SPIKE_IMPULSE_GAIN * spikes), where current_native = drive +
# recurrent_syn + noise. The gain constant lives in presets.py and is shared by
# every kernel variant in emitters.py (simulate_edge_recurrent_izhikevich and
# the dense variants) -- they must stay in sync or the double-count guard
# below breaks. No physical-amplitude claim is made; this remains an
# uncalibrated proxy. The double-count guard records that synaptic current
# enters the source only via the single proxy expression, not as a separate
# additive term.
_SOURCE_PROXY_METADATA: dict[str, Any] = {
    "source_model": "izhikevich_native_current_plus_spike_impulse_proxy",
    "source_mode": "native_current_plus_spike_impulse_proxy",
    "includes_native_current": True,
    "includes_drive_current": True,
    "includes_recurrent_synaptic_current": True,
    "includes_noise_current": True,
    "includes_spike_impulse": True,
    "spike_impulse_gain": DEFAULT_SPIKE_IMPULSE_GAIN,
    "source_calibration_status": "uncalibrated_izhikevich_native_current",
    "physical_amplitude_calibrated": False,
    "double_count_synaptic_current_guard": (
        "single_proxy_expression_no_extra_synaptic_source"
    ),
}

_KNOWN_READOUT_METRICS = frozenset({
    "spike_rate_hz",
    "spike_count",
    "mean_V_m",
    "csd_abs_mean",
    "lfp_abs_mean",
    "source_abs_mean",
})


def stimulus_schedule(
    events: Sequence[Any],
    n_neurons: int,
    *,
    drive_amplitude: float = 5.0,
    event_duration_ms: float = 50.0,
) -> StimulusSchedule:
    """Build a :class:`StimulusSchedule` from a sequence of events.

    Each event may be a :class:`ParadigmEvent` or a dict-like with at least
    ``onset_ms``.  The ``drive_amplitude`` and ``event_duration_ms`` are the
    default values applied to all events that do not specify their own.

    Events that carry ``is_omission=True`` or an explicit ``amplitude=0`` inject
    zero drive (generic no-drive semantics, not cognitive omission logic).
    No calibrated-current or physical-amplitude claim is made.
    """
    ev_dicts: list[dict[str, Any]] = []
    for e in events:
        if isinstance(e, ParadigmEvent):
            amp = float(e.metadata.get("drive_amplitude", drive_amplitude))
            dur = float(e.metadata.get("event_duration_ms", event_duration_ms))
            is_drive = not e.is_omission and e.onset_ms is not None
            ev_dict = {
                "label": e.label,
                "onset_ms": float(e.onset_ms) if e.onset_ms is not None else 0.0,
                "duration_ms": dur,
                "amplitude": amp if is_drive else 0.0,
                "is_drive_event": is_drive,
            }
            if "target_indices" in e.metadata:
                ev_dict["target_indices"] = e.metadata["target_indices"]
            ev_dicts.append(ev_dict)
        else:
            d = dict(e)
            if "amplitude" not in d:
                d["amplitude"] = drive_amplitude
            if "duration_ms" not in d:
                d["duration_ms"] = event_duration_ms
            if "is_drive_event" not in d:
                d["is_drive_event"] = d.get("onset_ms") is not None and d["amplitude"] != 0.0
            ev_dicts.append(d)
    return StimulusSchedule(
        events=tuple(ev_dicts),
        n_neurons=int(n_neurons),
    )
