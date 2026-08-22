"""Model.simulate() and its supporting kernel-dispatch, stimulus-schedule
resolution, batch/trial/condition wrappers, and HDP diagnostics accessors.

Split out of ``jaxfne/_model.py`` (Phase 2 defragmentation, 2026-07-20, part
of the 0.4.8-0.4.48 roadmap's Defragmentation wave 1). Every function here
takes the ``Model`` instance as an explicit first argument named ``self``
(matching the original method signatures exactly, since these were methods
before the split) -- ``jaxfne/_model.py``'s ``Model`` class delegates to
these as thin wrapper methods. Import from ``jaxfne.core`` or
``jaxfne._model``, not this module, unless working on the split itself.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping, Optional

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from ._model import Model

from .emitters import (
    EdgeList,
    IzhikevichParams,
    simulate_edge_recurrent_izhikevich,
    simulate_edge_recurrent_izhikevich_homeostatic,
    simulate_edge_recurrent_izhikevich_hdp,
    simulate_eig_izhikevich,
    simulate_receptor_exponential_izhikevich,
)
from .emitters_homeostatic_ei import HomeostaticEIParams, simulate_homeostatic_ei
from .fields import project_laminar_sources
from .io import config_hash, json_safe
from ._runtime_config import RuntimeConfig, _device_scope
from ._signals import (
    Signals,
    Simulation,
    StimulusSchedule,
    TrialBatch,
    TrialResult,
    TrialBatchResult,
    ParadigmCondition,
    _make_poisson_drive,
)
from ._model import _SOURCE_PROXY_METADATA, stimulus_schedule


def _hdp_kernel_kwargs(hp: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one shared HDP parameter contract for every execution path."""
    from ._hdp_adaptive import normalize_hdp_params_boundary

    hp = normalize_hdp_params_boundary(hp)
    return {
        "H_min": hp.get("H_min", 0.1),
        "H_max": hp.get("H_max", 10.0),
        "tau_0_ms": hp.get("tau_0_ms", 100.0),
        "alpha": hp.get("alpha", 0.0),
        "beta": hp.get("beta", 0.0),
        "gamma": hp.get("gamma", 0.0),
        "delta": hp.get("delta", 0.0),
        "C_spike": hp.get("C_spike", 0.0),
        "K_HDP": hp.get("K_HDP", 1.0),
        "K_ctrl": hp.get("K_ctrl", 0.0),
        "K_w_ctrl": hp.get("K_w_ctrl", 0.0),
        "rho_passive": hp.get("rho_passive", 0.0),
        "barrier_c": hp.get("barrier_c", 0.0),
        "barrier_d": hp.get("barrier_d", 0.0),
        "barrier_eps": hp.get("barrier_eps", 1.0e-3),
        "w_floor": hp.get("w_floor", 1.0e-3),
        "w_ceiling": hp.get("w_ceiling", 50.0),
        "v_floor": hp.get("v_floor", -150.0),
        "v_ceiling": hp.get("v_ceiling", 100.0),
        "u_abs_max": hp.get("u_abs_max", 2000.0),
        "syn_abs_max": hp.get("syn_abs_max", 1.0e4),
        "H_boost_gain": hp.get("H_boost_gain", 0.0),
        "size_scale_by_cell_type": hp.get("size_scale_by_cell_type"),
        "size_scale_override": hp.get("size_scale_override"),
        "noise_scale": hp.get("noise_scale"),
        "hdp_rule": hp.get("hdp_rule", "signed_linear"),
        "h_state_dim": hp.get("h_state_dim", 1),
        "h_state_locality": hp.get("h_state_locality"),
        "h_state_readout": hp.get("h_state_readout"),
        "h_state_coupling": hp.get("h_state_coupling"),
        "controller_B": hp.get("controller_B"),
        "controller_lambda": hp.get("controller_lambda", 0.45),
        "controller_tau_H_s": hp.get("controller_tau_H_s", 0.2),
        "controller_tau_theta_s": hp.get("controller_tau_theta_s", 2.0),
        "controller_rate_setpoint_E_hz": hp.get("controller_rate_setpoint_E_hz"),
        "controller_rate_setpoint_I_hz": hp.get("controller_rate_setpoint_I_hz"),
        "controller_theta_S_init": hp.get("controller_theta_S_init"),
        "m_ei_edge_mask": hp.get("m_ei_edge_mask"),
        "e_neuron_mask": hp.get("e_neuron_mask"),
        "theta_m_EI_bounds": hp.get("theta_m_EI_bounds", (0.1, 5.0)),
        "theta_eta_a_bounds": hp.get("theta_eta_a_bounds", (0.25, 4.0)),
        "record_dH_components": bool(hp.get("record_dH_components", False)),
        "record_edge_current": bool(hp.get("record_edge_current", False)),
        "record_weight_trace": bool(hp.get("record_weight_trace", True)),
    }


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

    if "edge_list" in self.params:
        import numpy as np

        _delay_host = np.asarray(self.params["edge_list"].delay_steps)
        if _delay_host.size and int(_delay_host.min()) < 0:
            raise ValueError("edge delay_steps must be >= 0")
        if (
            runtime_cfg.recurrent_backend != "edge_list"
            and _delay_host.size
            and int(_delay_host.sum()) != 0
        ):
            raise ValueError(
                "recurrent_backend='dense' has no finite-delay path; "
                "edge delay_steps must be all zero (use "
                "recurrent_backend='edge_list' for delayed runs)"
            )

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
        # The HDP recurrent kernel does not consume per-edge finite delays
        # (it has no spike-history ring buffer). Reject nonzero delays loudly
        # instead of silently ignoring them, so an HDP run with declared
        # delays is never mistaken for a delayed simulation.
        if int(jnp.asarray(edges.delay_steps).sum()) != 0:
            raise ValueError(
                "enable_hdp does not support nonzero edge delay_steps in this "
                "release: the HDP kernel has no finite-delay path (see "
                "jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp). "
                "Use the non-HDP finite-delay recurrent kernel "
                "(recurrent_backend='edge_list', enable_hdp=False) for delayed "
                "simulations."
            )
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
            from ._pipeline import continuation_noise_schedule

            kernel_kwargs = _hdp_kernel_kwargs(hp)
            V, S, src, diag = simulate_edge_recurrent_izhikevich_hdp(
                emitter, edges, sim.n_steps, sim.dt_ms, k,
                dtype=runtime_cfg.actual_dtype, drive_schedule=s,
                silence_mask=silence_mask,
                init_state=init_state,
                noise_schedule=continuation_noise_schedule(
                    k, sim.n_steps, emitter.n_neurons, runtime_cfg.jnp_dtype
                ),
                **kernel_kwargs,
            )
            theta_trace = diag.get("theta_S_trace")
            theta_final = diag.get("theta_S_final")
            return (
                V,
                S,
                src,
                diag["H_final"],
                diag["H_trace"],
                diag["w_final"],
                diag["w_trace"],
                theta_final,
                theta_trace,
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
        V, S, src, H_final, H_trace, w_final, w_trace, theta_final, theta_trace = result
        diag_store = {
            "H_final": H_final,
            "H_trace": H_trace,
            "w_final": w_final,
            "w_trace": w_trace,
        }
        if theta_final is not None:
            diag_store["theta_S_final"] = theta_final
        if theta_trace is not None:
            diag_store["theta_S_trace"] = theta_trace
        object.__setattr__(self, "_last_hdp_diag", diag_store)
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
                    def target_fn(k, s):
                        return kernel_fn(
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
                def target_fn(k, s):
                    return simulate_eig_izhikevich(
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


def _maybe_poisson_final_step(self: "Model", sources: Any) -> "Optional[dict[str, Any]]":
    """Opt-in real 1D Poisson solve on the FINAL timestep's sources.

    Returns ``None`` unless the caller explicitly opted in via
    ``Configuration.field(solver="experimental_poisson_1d")``. This is
    ADDITIVE: ``signals.field`` remains the unchanged proxy projection over
    every timestep. The solver is single-timestep by construction (it takes a
    ``(n_neurons,)`` source vector, not the ``(n_steps, n_neurons)`` series),
    so wiring it as a drop-in replacement for ``project_laminar_sources``
    would require a per-step dense solve and a different output shape --
    deliberately not attempted here.

    Any solver failure is captured as an ``error`` entry rather than raised:
    this is an opt-in diagnostic accessory, and it must not be able to break
    an otherwise-valid ``simulate()`` call.
    """
    declared = [f for f in (self.cfg.fields or []) if f.get("solver") == "experimental_poisson_1d"]
    if not declared:
        return None

    from .fields import experimental_poisson_1d_from_neuron_table

    spec = declared[0]
    try:
        table = self.neuron_table()
        final = jnp.asarray(sources)[-1]
        _phi, _residual, manifest = experimental_poisson_1d_from_neuron_table(
            neuron_table=table,
            sources=final,
            conductivity=float(spec.get("conductivity", 1.0)),
            n_bins=int(spec.get("n_bins", self.static.get("n_contacts", 16))),
        )
        return {"applied_to": "final_timestep_only", "manifest": manifest}
    except Exception as exc:  # noqa: BLE001 - opt-in accessory must not break simulate()
        return {"applied_to": "final_timestep_only", "error": f"{type(exc).__name__}: {exc}"}


def _simulate_continuation_arrays(
    self: "Model",
    sim: Simulation,
    runtime_cfg: RuntimeConfig,
    drive_schedule: Optional[jax.Array],
    continuation: "Any | None",
):
    """Run one segment through the explicit full-state continuation path."""
    from ._pipeline import (
        ContinuationState,
        compile_step_fn,
        continuation_state_from_model,
        run_continuation,
        validate_continuation_delay_state,
    )

    if runtime_cfg.enable_homeostasis:
        raise ValueError(
            "full-state continuation currently supports recurrent and HDP "
            "edge-list kernels, not the homeostasis dispatch"
        )
    ablation_mode = getattr(sim, "ablation", None)
    if ablation_mode is not None:
        raise ValueError(
            "full-state continuation temporarily does not support "
            f"ablation={ablation_mode!r}; the continuation kernel does not "
            "propagate ablation semantics"
        )
    if runtime_cfg.recurrent_backend != "edge_list":
        raise ValueError(
            "full-state continuation requires recurrent_backend='edge_list'"
        )
    if runtime_cfg.enable_hdp and "edge_list" in self.params:
        edges_cont: EdgeList = self.params["edge_list"]
        if int(jnp.asarray(edges_cont.delay_steps).sum()) != 0:
            raise ValueError(
                "enable_hdp does not support nonzero edge delay_steps in this "
                "release: the HDP kernel has no finite-delay path (see "
                "jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp). "
                "This guard also applies to the full-state continuation path."
            )
    if runtime_cfg.synaptic_kernel != "exponential":
        raise ValueError(
            "full-state continuation temporarily supports only "
            "synaptic_kernel='exponential'; requested "
            f"{runtime_cfg.synaptic_kernel!r}"
        )
    if sim.poisson_drive is not None:
        raise ValueError(
            "full-state continuation requires an explicit drive schedule; "
            "poisson_drive is pre-generated per simulation and cannot be "
            "continued without its drive cursor"
        )

    emitter: IzhikevichParams = self.params["emitter"]
    n_neurons = emitter.n_neurons
    if drive_schedule is None:
        schedule = jnp.zeros(
            (sim.n_steps, n_neurons), dtype=runtime_cfg.jnp_dtype
        )
    else:
        schedule = jnp.asarray(drive_schedule, dtype=runtime_cfg.jnp_dtype)
        if schedule.shape != (sim.n_steps, n_neurons):
            raise ValueError(
                "drive schedule shape must be "
                f"({sim.n_steps}, {n_neurons}), got {schedule.shape}"
            )

    hp = dict(runtime_cfg.hdp_params or {}) if runtime_cfg.enable_hdp else {}
    baseline_kw = {}
    if not runtime_cfg.enable_hdp and runtime_cfg.hdp_params:
        if "noise_scale" in runtime_cfg.hdp_params:
            baseline_kw["noise_scale"] = runtime_cfg.hdp_params["noise_scale"]
    if runtime_cfg.enable_hdp:
        from ._hdp_adaptive import reject_population_continuation, resolve_h_state_locality

        reject_population_continuation(
            resolve_h_state_locality(hp),
            context="simulate_continuation",
        )
    if continuation is None:
        state = continuation_state_from_model(
            self,
            seed=sim.seed,
            h_state_dim=int(hp.get("h_state_dim", 1)),
        )
    elif isinstance(continuation, ContinuationState):
        state = continuation
        validate_continuation_delay_state(
            self, state, continuing=True
        )
    else:
        raise TypeError(
            "continuation must be a jaxfne.ContinuationState returned by "
            "simulate(..., return_state=True)"
        )

    if runtime_cfg.enable_hdp:
        hdp_kwargs = _hdp_kernel_kwargs(hp)
        step_fn, _ = compile_step_fn(
            self,
            dt_ms=sim.dt_ms,
            kernel="hdp",
            **hdp_kwargs,
        )
    else:
        step_fn, _ = compile_step_fn(
            self,
            dt_ms=sim.dt_ms,
            kernel="baseline",
            **baseline_kw,
        )

    next_state, outputs = run_continuation(step_fn, state, schedule)
    voltages, spikes, sources = outputs[:3]
    if runtime_cfg.enable_hdp:
        object.__setattr__(
            self,
            "_last_hdp_diag",
            {
                "H_final": next_state.dynamic.H,
                "H_trace": outputs[3],
                "w_final": next_state.dynamic.w,
                "w_trace": outputs[4] if len(outputs) > 4 else None,
            },
        )
    return voltages, spikes, sources, next_state


def simulate(
    self: "Model",
    sim: Simulation,
    paradigm: "Optional[Any]" = None,
    *,
    continuation: "Any | None" = None,
    return_state: bool = False,
) -> "Signals | tuple[Signals, Any]":
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

    ``return_state=True`` opts into the additive full-state continuation path.
    It returns ``(Signals, ContinuationState)``; pass that state back through
    ``continuation=`` for the next segment. ``with_hdp_initial_state`` remains
    a partial H/W initializer and is not changed by this contract.
    """
    # Local import: _simulate_homeostasis_metadata/_simulate_hdp_metadata stay
    # in core.py (group-6 construct-pipeline territory); Model is their only
    # caller, so deferring the import here avoids a circular import with
    # core.py's own `from ._model import Model`.
    from .core import _simulate_homeostasis_metadata, _simulate_hdp_metadata

    runtime_cfg = sim.resolved_runtime
    key = jax.random.PRNGKey(sim.seed)
    # Diagnostics belong to the current evaluation. Clear prior adaptive-state
    # receipts so an HDP-off control cannot inherit W/H evidence from a prior
    # HDP-on candidate on the same immutable model instance.
    object.__setattr__(self, "_last_hdp_diag", None)
    object.__setattr__(self, "_last_homeostasis_diag", None)

    if isinstance(self.params["emitter"], HomeostaticEIParams):
        if continuation is not None or return_state:
            raise ValueError(
                "full-state continuation is not available for homeostatic_ei"
            )
        return self._simulate_homeostatic_ei(sim, key, runtime_cfg)

    if (continuation is not None or return_state) and sim.poisson_drive is not None:
        raise ValueError(
            "full-state continuation does not support poisson_drive; provide "
            "an explicit schedule so its cursor is unambiguous"
        )

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

    if continuation is not None or return_state:
        voltages, spikes, sources, continuation_out = _simulate_continuation_arrays(
            self,
            sim,
            runtime_cfg,
            drive_array,
            continuation,
        )
    else:
        voltages, spikes, sources = self._simulate_arrays(
            sim, key, runtime_cfg, drive_schedule=drive_array
        )
        continuation_out = None
    time_ms = jnp.arange(sim.n_steps, dtype=runtime_cfg.jnp_dtype) * jnp.asarray(
        sim.dt_ms, dtype=runtime_cfg.jnp_dtype
    )
    positions = jnp.asarray(self.params["positions"], dtype=runtime_cfg.jnp_dtype)
    field_output = None
    poisson_final_step = None
    if sim.record_fields:
        field_output = project_laminar_sources(
            sources=sources,
            positions=positions,
            n_contacts=self.static.get("n_contacts", 16),
            dtype=runtime_cfg.actual_dtype,
        )
        poisson_final_step = _maybe_poisson_final_step(self, sources)

    paradigm_meta: Optional[dict[str, Any]] = None
    if isinstance(paradigm, Mapping):
        paradigm_meta = dict(paradigm)
    elif hasattr(paradigm, "to_dict"):
        paradigm_meta = paradigm.to_dict()

    metadata: dict[str, Any] = {
        "config_hash": config_hash(self.cfg),
        "source_calibration_status": self.cfg.metadata.get("source_calibration_status"),
        "representation": "relative",
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
        "source_mode_class": _SOURCE_PROXY_METADATA.get("source_mode_class"),
        "source_contract": _SOURCE_PROXY_METADATA.get("source_contract"),
        "source_projection_mode": self.cfg.metadata.get("source_projection_mode", "proxy_no_field_solve"),
        "source_decomposition": self.cfg.metadata.get("source_decomposition", "proxy_reduced_emitter"),
        "source_calibration_status": _SOURCE_PROXY_METADATA.get("source_calibration_status"),
        "representation": _SOURCE_PROXY_METADATA["source_contract"]["representation"],
        "calibration_transform": _SOURCE_PROXY_METADATA["source_contract"]["calibration"],
        "synaptic_current_counting": _SOURCE_PROXY_METADATA.get("double_count_synaptic_current_guard"),
        "source_mode_exclusive": True,
        "physical_amplitude_calibrated": _SOURCE_PROXY_METADATA.get("physical_amplitude_calibrated", False),
        "double_count_guard": "passed",
        "double_count_evidence": _SOURCE_PROXY_METADATA.get("double_count_evidence"),
    }
    if poisson_final_step is not None:
        metadata["poisson_field_final_step"] = poisson_final_step
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
    signals = Signals(
        time_ms=time_ms,
        V_m=voltages.astype(runtime_cfg.jnp_dtype),
        spikes=spikes,
        sources=sources.astype(runtime_cfg.jnp_dtype) if sim.record_sources else None,
        field=field_output,
        metadata=metadata,
    )
    if return_state:
        return signals, continuation_out
    return signals

def _simulate_homeostatic_ei(
    self: "Model",
    sim: Simulation,
    key: jax.Array,
    runtime_cfg: RuntimeConfig,
) -> Signals:
    """The homeostatic_ei emitter family's ``simulate()`` path -- the
    second canonical HDP sanity circuit (see
    ``jaxfne/emitters_homeostatic_ei.py``). Duplicates only the
    family-agnostic parts of :meth:`simulate` (time axis, field
    projection, metadata scaffold, ``Signals`` construction); does not
    call ``_simulate_homeostasis_metadata``/``_simulate_hdp_metadata``
    (those assume the edge-list HDP kernel's diagnostic shape) -- HDP
    metadata here is built directly from ``G_history``/``H_history``.
    """
    emitter: HomeostaticEIParams = self.params["emitter"]
    voltages, spikes, sources, G_history, H_history, diag = simulate_homeostatic_ei(
        emitter,
        n_steps=sim.n_steps,
        dt_ms=sim.dt_ms,
        key=key,
        activation_rule=emitter.activation_rule_name,
        conductance_rule=emitter.conductance_rule_name,
        homeostasis_rule=emitter.homeostasis_rule_name,
        dtype=runtime_cfg.actual_dtype,
        bound_mode=emitter.bound_mode,
    )
    time_ms = jnp.arange(sim.n_steps, dtype=runtime_cfg.jnp_dtype) * jnp.asarray(
        sim.dt_ms, dtype=runtime_cfg.jnp_dtype
    )
    source_mode = "homeostatic_ei_activity_proxy"
    source_mode_class = "specialized"
    source_decomposition = "homeostatic_ei_activity_trace"
    source_contract = {
        "native_current": "homeostatic_ei_activity_trace",
        "spike_term": "none",
        "gain_owner": "HomeostaticEIParams.source_scale",
        "sign_convention": "activity trace sign follows emitter state",
        "support": "time_by_neuron",
        "normalization": "per_neuron_source_scale",
        "representation": "relative",
        "calibration": "explicit_boundary_transform",
    }
    field_output = None
    if sim.record_fields:
        positions = jnp.asarray(self.params["positions"], dtype=runtime_cfg.jnp_dtype)
        field_output = project_laminar_sources(
            sources=sources,
            positions=positions,
            n_contacts=self.static.get("n_contacts", 16),
            dtype=runtime_cfg.actual_dtype,
        )
        field_output = replace(
            field_output,
            diagnostics={
                **field_output.diagnostics,
                "source_calibration_status": emitter.source_calibration_status,
                "source_mode": source_mode,
                "source_mode_class": source_mode_class,
                "source_decomposition": source_decomposition,
                "source_contract": source_contract,
            },
        )
    metadata: dict[str, Any] = {
        "config_hash": config_hash(self.cfg),
        "emitter_family": "homeostatic_ei",
        "source_mode": source_mode,
        "source_mode_class": source_mode_class,
        "source_decomposition": source_decomposition,
        "source_contract": source_contract,
        "representation": "relative",
        "source_calibration_status": emitter.source_calibration_status,
        "calibration_transform": "explicit_boundary_transform",
        "physical_amplitude_calibrated": False,
        "field_claim_level": "proxy_readout",
        "duration_ms": float(sim.duration_ms),
        "dt_ms": float(sim.dt_ms),
        "n_steps": int(sim.n_steps),
        "runtime": runtime_cfg.runtime_report(),
        "hdp": {
            "enabled": True,
            "rules": {
                "activation_rule": emitter.activation_rule_name,
                "conductance_rule": emitter.conductance_rule_name,
                "homeostasis_rule": emitter.homeostasis_rule_name,
                "bound_mode": emitter.bound_mode,
            },
            "H_trace": H_history,
            "G_trace": G_history,
            "error": bool(diag["error"]),
        },
    }
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

    Returns a dict with ``H_final``/``H_trace`` shaped
    ``(n_steps, n_neurons)`` for scalar H or
    ``(n_steps, n_neurons, h_state_dim)`` for vector H, plus
    ``w_final``/``w_trace`` ``(n_steps, n_edges)``, or ``None`` if HDP
    was not enabled on the last run. If ``hdp_params["record_weight_trace"]``
    was explicitly set to ``False`` (recommended when n_steps * n_edges
    would exceed device memory -- e.g. 10,000 steps x 2,000,000 edges x
    4 bytes = 80GB, a real reproduced OOM), ``w_trace`` is ``None`` while
    ``w_final`` remains the correct terminal weight state either way; the
    default (``True``) matches this method's documented contract exactly.
    See ``jaxfne.emitters.simulate_edge_recurrent_izhikevich_hdp`` for the
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
    # Local import: see _simulate_arrays' matching comment above.
    from .core import _resolve_homeostasis_k_gain, _homeostasis_params_cache_fingerprint
    runtime_cfg = sim.resolved_runtime
    base_seed = sim.seed if seed is None else int(seed)
    keys = jax.random.split(jax.random.PRNGKey(base_seed), int(n_seeds))
    emitter = self._require_izhikevich_emitter("simulate_batch")

    # Sparse-direct models (placeholder dense W) must use the edge_list backend.
    if emitter.W.shape[0] != int(emitter.v0.shape[0]) and "edge_list" in self.params:
        runtime_cfg = replace(runtime_cfg, recurrent_backend="edge_list")

    homeo_on = bool(getattr(runtime_cfg, "enable_homeostasis", False))
    hdp_on = bool(getattr(runtime_cfg, "enable_hdp", False))
    if homeo_on and runtime_cfg.synaptic_kernel == "receptor_exponential":
        raise ValueError(
            "enable_homeostasis is not supported with "
            "synaptic_kernel='receptor_exponential'."
        )
    if hdp_on and runtime_cfg.synaptic_kernel == "receptor_exponential":
        raise ValueError(
            "enable_hdp is not supported with "
            "synaptic_kernel='receptor_exponential'; use the default "
            "exponential synaptic kernel."
        )
    edge_kernel_fn = (
        simulate_receptor_exponential_izhikevich
        if runtime_cfg.synaptic_kernel == "receptor_exponential"
        else simulate_edge_recurrent_izhikevich
    )
    _hp = dict(runtime_cfg.homeostasis_params or {})
    _hdp = dict(runtime_cfg.hdp_params or {})

    def one(k):
        """Documented public function `one`."""
        if hdp_on:
            # HDP engages the sparse-edge HDP kernel; per-step H_trace/w_trace
            # diagnostics are dropped here (batch is a seed-replicate statistics
            # utility -- use simulate() for full diagnostics passthrough).
            from ._pipeline import continuation_noise_schedule

            # Same guard as Model.simulate: the HDP kernel has no finite-delay
            # path, so nonzero edge delays must be rejected loudly here too.
            edges_batch: EdgeList = self.params["edge_list"]
            if int(jnp.asarray(edges_batch.delay_steps).sum()) != 0:
                raise ValueError(
                    "enable_hdp does not support nonzero edge delay_steps in this "
                    "release: the HDP kernel has no finite-delay path. Use the "
                    "non-HDP finite-delay recurrent kernel for delayed runs."
                )
            kernel_kwargs = _hdp_kernel_kwargs(_hdp)
            kernel_kwargs["record_weight_trace"] = False
            return simulate_edge_recurrent_izhikevich_hdp(
                emitter, self.params["edge_list"], sim.n_steps, sim.dt_ms, k,
                dtype=runtime_cfg.actual_dtype,
                noise_schedule=continuation_noise_schedule(
                    k, sim.n_steps, emitter.n_neurons, runtime_cfg.jnp_dtype
                ),
                **kernel_kwargs,
            )[:3]
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
        cache_key = ("simulate_batch", B, Z, C, T, runtime_cfg.actual_dtype, runtime_cfg.synaptic_kernel, runtime_cfg.recurrent_backend, homeo_on, hdp_on, runtime_cfg.selected_backend,
                     _homeostasis_params_cache_fingerprint(_hp) if homeo_on else (),
                     _homeostasis_params_cache_fingerprint(_hdp) if hdp_on else ())
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
            "enable_hdp": hdp_on,
            "hdp_params": _hdp if hdp_on else None,
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

