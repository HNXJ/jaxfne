"""Model.tune() -- AGSDR/GSDR/Optax parameter optimization, plus
``with_emitter_parameters()``/``with_hdp_initial_state()``/
``with_recurrent_coupling()`` and the free-function parameter-editing
helpers (``_model_with_scalar_parameter``, ``_mask_for_parameter``,
``_model_with_matrix_parameter``, ``_model_with_parameters``).

Split out of ``jaxfne/_model.py`` (Phase 2 defragmentation, 2026-07-20, part
of the 0.4.8-0.4.48 roadmap's Defragmentation wave 1).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .optim import OptimizerSpec
    from ._model import Model

from .io import json_safe
from ._signals import Objective, Simulation, _finite_or_none
from ._model import MatrixParameterSpec, TuneResult


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

        # Narrow real gradient path: source_scale is a linear post-processing
        # rescale of already-simulated arrays (source_proxy = source_scale *
        # (current_native + GAIN * spikes)) -- never crosses the spike/reset
        # boundary, so this specific case is genuinely differentiable without
        # any surrogate. Only wired for a single loss on a metric linear in
        # source_scale; anything else falls through to the honest stub below.
        from .optim.core import _tune_source_scale_optax, _SOURCE_SCALE_LINEAR_METRICS
        single_loss = getattr(objective, "losses", [])
        eligible = (
            optax_status == "available"
            and (parameter is None or parameter == "source_scale")
            and n_steps > 0
            and len(single_loss) == 1
            and single_loss[0].get("metric") in _SOURCE_SCALE_LINEAR_METRICS
            and single_loss[0].get("target") is not None
        )
        if eligible:
            return _tune_source_scale_optax(
                model=self,
                simulation=sim,
                objective=objective,
                spec=spec,
                bounds=bounds,
                n_steps=n_steps,
                seed=seed,
                base_report=base_report,
            )

        # Second narrow real gradient path: drive_gain/synaptic_gain/gAMPA
        # DO cross the spike/reset boundary, but JAX already differentiates
        # through Model.simulate()'s hard jnp.where-based reset without any
        # kernel change -- verified empirically (real nonzero gradient at
        # emitters.py's existing, unmodified step dynamics). The soft-rate
        # surrogate lives only at the loss level (_evaluate_soft_rate_targets,
        # already used and tested in the AGSDR two-level inner loop), reused
        # here rather than reinvented. Only wired for a jtfne.rate_targets(...)
        # -style objective; anything else falls through to the stub below.
        from .optim.core import _tune_scalar_soft_rate_optax, _SCALAR_SOFT_RATE_PARAMETERS
        has_rate_targets_gate = any(
            isinstance(g, dict) and "groups" in g.get("metadata", {}) and "targets_hz" in g.get("metadata", {})
            for g in getattr(objective, "gates", [])
        )
        eligible_soft_rate = (
            optax_status == "available"
            and parameter in _SCALAR_SOFT_RATE_PARAMETERS
            and n_steps > 0
            and has_rate_targets_gate
        )
        if eligible_soft_rate:
            return _tune_scalar_soft_rate_optax(
                model=self,
                simulation=sim,
                objective=objective,
                spec=spec,
                parameter=parameter,
                bounds=bounds,
                n_steps=n_steps,
                seed=seed,
                base_report=base_report,
            )

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
    emitter = self._require_izhikevich_emitter("with_emitter_parameters")
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
        H0: Per-neuron initial H-state. Scalar HDP uses shape
            ``[n_neurons]``; vector-H uses ``[n_neurons, h_state_dim]``.
            Defaults to the HDP kernel's own equilibrium value (1.0 for
            every coordinate) when not provided.
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
    emitter = model.params["emitter"]

    # source_scale/drive_gain/synaptic_gain/gAMPA are pure jnp elementwise ops --
    # `value` is passed straight into jnp.asarray without a premature float()
    # concretization, so a jax.grad tracer flows through unchanged (needed for
    # the differentiable-tune path; a bare `float(value)` here would silently
    # break jax.grad's tape exactly like the _compute_all_metrics footgun).
    # drive_scale_a/drive_scale_b remain numpy-based (data-dependent array
    # construction, not part of the differentiable-tune surface) and still
    # concretize `value` locally within that branch only.
    if parameter == "source_scale":
        new_emitter = replace(emitter, source_scale=jnp.asarray(value, dtype=emitter.source_scale.dtype))
    elif parameter == "drive_gain":
        new_emitter = replace(emitter, drive=emitter.drive * jnp.asarray(value, dtype=emitter.drive.dtype))
    elif parameter == "synaptic_gain":
        new_emitter = replace(emitter, W=emitter.W * jnp.asarray(value, dtype=emitter.W.dtype))
    elif parameter in ("drive_scale_a", "drive_scale_b"):
        import numpy as _np_dsa
        value = float(value)
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
        # jnp.where (not numpy boolean-indexed assignment) -- keeps this branch
        # jax-traceable/differentiable too; bit-identical result for concrete
        # inputs (verified: scales only W > 0 entries, leaves the rest untouched).
        W = emitter.W
        scale = jnp.asarray(value, dtype=W.dtype)
        new_W = jnp.where(W > 0, W * scale, W)
        new_emitter = replace(emitter, W=new_W)
    else:
        supported = ["source_scale", "drive_gain", "synaptic_gain", "drive_scale_a", "drive_scale_b", "gAMPA"]
        raise ValueError(
            f"Unsupported tunable parameter: {parameter!r}. "
            f"Supported parameters: {supported}"
        )
    params = dict(model.params)
    params["emitter"] = new_emitter
    from ._model import Model  # deferred: Model is defined after this module is imported

    return Model(cfg=model.cfg, params=params, static=dict(model.static))


def _mask_for_parameter(
    model: "Model", parameter_name: str, mask_type: str, target: str = "W"
) -> "jax.Array":
    """Return a boolean mask over the target matrix for the given mask type.

    Parameters
    ----------
    model : Model
        Model whose target matrix determines the mask shape.
    parameter_name : str
        Name of the parameter (used only for error messages).
    mask_type : str
        One of: "E_to_E", "E_to_I", "excitatory_to_all", "all".
    target : str
        Name of the emitter dataclass field the mask applies to (default
        "W"; e.g. "G" for the homeostatic_ei emitter's conductance matrix).

    Returns
    -------
    jax.Array
        Boolean mask of shape (n, n) where True marks entries to scale.
    """
    import numpy as _np_mask
    emitter = model.params["emitter"]
    W = _np_mask.asarray(getattr(emitter, target), dtype=float)
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

    target = getattr(spec, "target", "W")
    emitter = model.params["emitter"]
    current = getattr(emitter, target)
    W = _np_matrix.asarray(current, dtype=float)
    mask = _np_matrix.asarray(_mask_for_parameter(model, parameter_name, spec.mask, target), dtype=bool)

    new_W = W.copy()
    new_W[mask] = W[mask] * value
    new_emitter = replace(emitter, **{target: jnp.asarray(new_W, dtype=current.dtype)})
    params = dict(model.params)
    params["emitter"] = new_emitter
    from ._model import Model  # deferred: Model is defined after this module is imported

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


