"""Named-handle tuning entry point (Lane 8, gap G10).

:func:`tune_handles` fits any declared core-model parameter addressed by a
:mod:`jaxfne._param_handles` path (``edges.weight[...]``,
``cells.<family>.<field>[...]``, ``drive.<name>.rate``) on either model
family. It exists because :meth:`Model.tune` is refused on the adex route;
on the izhikevich route :meth:`Model.tune` accepts the same paths through
its ``parameter`` / ``parameters`` arguments.

Each entry of ``parameters`` maps one handle path to a scalar ``(lo, hi)``
bound pair. A multi-element selection is tuned as one scalar knob: cells and
drive broadcast the scalar uniformly; ``edges.weight`` sets a
sign-preserving per-edge magnitude (same convention as
:class:`EdgeParameterSpec`). Use :func:`jaxfne.get_parameter` /
:func:`jaxfne.set_parameter` for per-element reads and writes.

Black-box optimizers (``agsdr`` and the other non-differentiable specs) run
the canonical multi-parameter AGSDR engine
(:func:`_run_agsdr_optimization_loop`); every handle kind is supported.
Differentiable-path optimizers (``optax_adam``/``optax_sgd``/``GSGD``) run a
real ``jax.grad`` loop only on traceable writes: ``cells.izhikevich``
scalar-per-neuron fields, drive, and dense-``W`` gain. Edge-list and adex
edge writes are host-side numpy (concrete-only) and are refused up front in
the gradient loop -- use the black-box loop for those. Only
``cells.izhikevich.source_scale[all]`` is exactly differentiable; every
other handle needs an explicit ``surrogate="soft_rate"`` (a sigmoid
soft-spike rate loss on ``V_m``, reused from the scalar soft-rate path),
else a ``ValueError`` naming the path. This is a computational scaffold:
no biological calibration claim is made.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from .._param_handles import (
    HANDLE_SCHEME_VERSION,
    get_parameter,
    is_differentiable_handle,
    parse_handle,
    set_parameter,
)
from .._signals import Simulation, _finite_or_none
from ..io import json_safe
from .core import (
    _CANDIDATE_FAILURES,
    _SOURCE_SCALE_LINEAR_METRICS,
    _resolve_optimizer,
    _run_agsdr_optimization_loop,
    _tune_source_scale_optax,
    require_optax,
)

_HANDLE_OPTAX_SURROGATES = frozenset({"soft_rate"})


def _normalize_handle_bounds(
    parameters: Any,
) -> dict[str, tuple[float, float]]:
    """Validate ``parameters`` into ``{handle path: (lo, hi)}``; refuse by name."""
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(
            "tune_handles() requires a non-empty dict {handle path: (lo, hi)}; "
            f"got {parameters!r}"
        )
    normalized: dict[str, tuple[float, float]] = {}
    for path, bounds in parameters.items():
        parse_handle(path)  # ValueError names the path when unknown
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError(
                f"Parameter path {path!r} needs a scalar (lo, hi) bound pair; "
                f"got {bounds!r}"
            ) from exc
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(
                f"Parameter path {path!r} has non-finite bounds {(lo, hi)!r}"
            )
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            raise ValueError(
                f"Parameter path {path!r} has degenerate bounds {(lo, hi)!r}"
            )
        normalized[str(path)] = (lo, hi)
    return normalized


def _concrete_edge_paths(model: Any, paths: list[str]) -> list[str]:
    """Handle paths whose writes are concrete-only (numpy, not jnp-traceable)."""
    from .._param_handles import _edge_store

    concrete: list[str] = []
    for path in paths:
        handle = parse_handle(path)
        if handle.kind != "edge":
            continue
        if _edge_store(model)["kind"] in ("edge_list", "adex"):
            concrete.append(path)
    return concrete


def _check_edge_backend(model: Any, paths: list[str], simulation: Any) -> None:
    """Refuse an edges.weight fit whose writes the simulated backend ignores."""
    from .._param_handles import _edge_store

    if not any(parse_handle(path).kind == "edge" for path in paths):
        return
    backend = simulation.resolved_runtime.recurrent_backend
    if _edge_store(model)["kind"] == "dense" and backend == "edge_list":
        offenders = [p for p in paths if parse_handle(p).kind == "edge"]
        raise ValueError(
            f"Parameter path(s) {offenders!r} address dense emitter.W, but the "
            f"simulation runs recurrent_backend='edge_list' (which executes "
            "params['edge_list']); the fit would be a silent no-op"
        )


def _rate_targets_groups(objective: Any) -> tuple[dict, dict]:
    """Extract ``(groups, targets_hz)`` from a rate_targets-style objective."""
    groups: dict = {}
    targets_hz: dict = {}
    for gate_spec in getattr(objective, "gates", []):
        meta = gate_spec.get("metadata", {}) if isinstance(gate_spec, dict) else {}
        if "groups" in meta and "targets_hz" in meta:
            groups = dict(meta["groups"])
            targets_hz = {k: float(v) for k, v in meta["targets_hz"].items()}
            break
    return groups, targets_hz


def _base_handles_report(
    model: Any,
    objective: Any,
    spec: Any,
    bounds: dict[str, tuple[float, float]],
    seed: int,
    simulation: Any,
) -> dict[str, Any]:
    """Shared TuneResult summary header for the handles loops."""
    cfg_meta = model.cfg.metadata
    return {
        "same_model_unchanged": True,
        "seed": int(seed),
        "parameters": {k: [float(v[0]), float(v[1])] for k, v in bounds.items()},
        "handle_scheme": HANDLE_SCHEME_VERSION,
        "simulation_seed": int(simulation.seed),
        "optimizer_seed": int(seed),
        "runtime": simulation.resolved_runtime.runtime_report(),
        "optimizer": spec.to_dict(),
        "objective_name": getattr(objective, "name", "spectrolaminar_objective")
        if not isinstance(objective, str)
        else objective,
        "losses_declared": len(getattr(objective, "losses", []))
        if not isinstance(objective, str)
        else 0,
        "regularizers_declared": len(getattr(objective, "regularizers", []))
        if not isinstance(objective, str)
        else 0,
        "gates_declared": len(getattr(objective, "gates", []))
        if not isinstance(objective, str)
        else 0,
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


def _apply_handle_values(model: Any, values: dict[str, Any]) -> Any:
    """Apply one scalar per handle path; the input model is never mutated."""
    result = model
    for path, value in values.items():
        result = set_parameter(result, path, float(value))
    return result


def _evaluate_handle_candidate(
    model: Any,
    candidate: dict[str, float],
    objective: Any,
    simulation: Any,
    strict: bool,
) -> tuple[float, dict[str, Any], Any]:
    """Simulate one handle candidate; return ``(score, report, model)``.

    Failing candidates (refused configuration, arithmetic/runtime errors)
    score ``inf`` with rejection reasons; any other exception is a bug and
    propagates -- the same contract as the multiparameter AGSDR path.
    """
    reasons = []
    candidate_model = None
    try:
        candidate_model = _apply_handle_values(model, candidate)
        candidate_signals = candidate_model.simulate(simulation)
        candidate_report = candidate_model.evaluate(
            candidate_signals, objective, strict=strict
        )
    except (ValueError, FloatingPointError, OverflowError) as exc:
        candidate_report = {
            "evaluation_status": "candidate_evaluation_rejected",
            "total_loss": None,
            "total_score": None,
            "all_gates_pass": False,
            "invalid_status": f"{type(exc).__name__}: {exc}",
            "score_status": "positive_infinity",
            "warnings": ["expected_candidate_evaluation_failure"],
        }
        reasons.append(candidate_report["invalid_status"])

    score = candidate_report.get("total_score")
    if score is None:
        score = candidate_report.get("total_loss")
    gates_pass = bool(candidate_report.get("all_gates_pass", False))
    if score is None:
        score = 0.0 if gates_pass else float("inf")
    score = float(score)

    if not gates_pass:
        reasons.append("failed_objective_gates")
    if not math.isfinite(score):
        reasons.append("non_finite_loss")
    return score, {**candidate_report, "rejection_reasons": reasons}, candidate_model


def _tune_handles_blackbox(
    model: Any,
    objective: Any,
    spec: Any,
    bounds: dict[str, tuple[float, float]],
    generations: int,
    population_size: int,
    seed: int,
    strict: bool,
    simulation: Any,
    base_report: dict[str, Any],
) -> Any:
    """Run the canonical AGSDR engine over handle knobs (all kinds)."""
    from .._model import TuneResult

    candidate_evaluations: list[dict[str, Any]] = []
    rejections: list = []

    def evaluate_fn(candidate_params: dict[str, float]) -> float:
        score, candidate_report, candidate_model = _evaluate_handle_candidate(
            model, candidate_params, objective, simulation, strict
        )
        rejections.append(candidate_report.get("rejection_reasons", []))
        candidate_evaluations.append(
            {
                "parameters": {
                    key: float(value) for key, value in candidate_params.items()
                },
                "score": _finite_or_none(score),
                "score_status": (
                    "finite"
                    if candidate_report.get("all_gates_pass", False)
                    and math.isfinite(score)
                    else "positive_infinity"
                ),
                "accepted": bool(
                    candidate_report.get("all_gates_pass", False)
                    and math.isfinite(score)
                ),
                "rejection_reasons": candidate_report.get("rejection_reasons", []),
                "objective": {
                    key: value
                    for key, value in candidate_report.items()
                    if key != "rejection_reasons"
                },
            }
        )
        return float(score)

    try:
        agsdr_result = _run_agsdr_optimization_loop(
            evaluate_fn=evaluate_fn,
            parameter_bounds=bounds,
            n_generations=int(generations),
            n_population=int(population_size),
            alpha=float(spec.alpha),
            exploration=float(spec.exploration),
            seed=int(seed),
            rejections_map=rejections,
            initial_parameters={
                name: 0.5 * (lo + hi) for name, (lo, hi) in bounds.items()
            },
        )
    except Exception as exc:
        report = {
            **base_report,
            "tuning_status": "handles_agsdr_error",
            "acceptance_decision": "REVISE",
            "error": str(exc),
            "warnings": ["handles_optimization_failed"],
        }
        return TuneResult(
            best_parameters={},
            best_score=float("inf"),
            history=[],
            summary=json_safe(report),
            model=model,
        )

    best_parameters = agsdr_result["best_parameters"]
    best_score = agsdr_result["best_score"]
    generation_records = agsdr_result["generation_records"]
    best_model = _apply_handle_values(model, best_parameters) if best_parameters else model

    report = {
        **base_report,
        "same_model_unchanged": False,
        "tuning_status": "handles_blackbox_agsdr_v0.1",
        "acceptance_decision": "ACCEPT_CANDIDATE" if math.isfinite(best_score) else "REVISE",
        "best_parameters": best_parameters,
        "theta_best": best_parameters,
        "best_score": _finite_or_none(best_score),
        "generation_records": generation_records,
        "all_scores": agsdr_result["all_scores"],
        "n_candidates_evaluated": len(agsdr_result["all_scores"]),
        "candidate_evaluations": candidate_evaluations,
        "candidate_rejection_count": sum(
            1 for item in candidate_evaluations if item["rejection_reasons"]
        ),
        "tuning_path": "handles_black_box",
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


def _tune_handles_soft_rate_optax(
    model: Any,
    simulation: Any,
    objective: Any,
    spec: Any,
    bounds: dict[str, tuple[float, float]],
    n_steps: int,
    seed: int,
    base_report: dict[str, Any],
) -> Any:
    """Real jax.grad loop over traceable handle knobs with a soft-rate surrogate.

    The surrogate lives only at the loss level (sigmoid soft spikes on
    ``V_m``, the same helper the matrix inner loop uses); no kernel carries
    a custom gradient. Edge-list and adex edge paths are refused by the
    caller before this loop (concrete-only writes cannot carry a tracer).
    """
    import jax
    import jax.numpy as jnp

    from .._model import TuneResult
    from .._model_evaluate import _evaluate_soft_rate_targets

    optax = require_optax()

    groups, targets_hz = _rate_targets_groups(objective)
    if not groups or not targets_hz:
        report = {
            **base_report,
            "tuning_status": "handles_soft_rate_unavailable",
            "acceptance_decision": "REVISE",
            "warnings": ["objective_is_not_a_rate_targets_style_group_rate_targets_spec"],
        }
        return TuneResult(
            best_parameters={}, best_score=float("inf"), history=[],
            summary=json_safe(report), model=model,
        )

    names = sorted(bounds)
    lo = jnp.asarray([bounds[name][0] for name in names], dtype=jnp.float32)
    hi = jnp.asarray([bounds[name][1] for name in names], dtype=jnp.float32)
    theta = jnp.asarray(
        [0.5 * (bounds[name][0] + bounds[name][1]) for name in names],
        dtype=jnp.float32,
    )

    def loss_fn(value: jax.Array) -> jax.Array:
        candidate = model
        for dim, path in enumerate(names):
            candidate = set_parameter(candidate, path, value[dim])
        signals = candidate.simulate(simulation)
        return _evaluate_soft_rate_targets(
            V_m=signals.V_m,
            groups=groups,
            targets_hz=targets_hz,
            duration_ms=float(simulation.duration_ms),
            dt_ms=float(simulation.dt_ms),
        )

    lr = float(spec.learning_rate) if spec.learning_rate else 1e-2
    tx = optax.adam(lr) if spec.optimizer == "optax_adam" else optax.sgd(lr)
    opt_state = tx.init(theta)

    history = []
    best_theta = [float(v) for v in theta]
    best_loss = float(loss_fn(theta))
    fallback_count = 0
    for step in range(int(n_steps)):
        try:
            loss_val, grad = jax.value_and_grad(loss_fn)(theta)
            loss_val = float(loss_val)
            grad_arr = jnp.asarray(grad, dtype=jnp.float32)
            grad_norm = float(jnp.sum(jnp.abs(grad_arr)))
        except _CANDIDATE_FAILURES:
            fallback_count += 1
            loss_val = float("nan")
            grad_norm = 0.0
            grad_arr = None
        if grad_arr is None or grad_norm == 0.0 or not math.isfinite(grad_norm):
            fallback_count += 1
        else:
            updates, opt_state = tx.update(grad_arr, opt_state)
            theta = optax.apply_updates(theta, updates)
            theta = jnp.clip(theta, lo, hi)
        if math.isfinite(loss_val) and loss_val < best_loss:
            best_loss = loss_val
            best_theta = [float(v) for v in jnp.asarray(theta).reshape(-1)]
        history.append(
            {
                "step": step,
                "parameters": {
                    name: float(jnp.asarray(theta).reshape(-1)[dim])
                    for dim, name in enumerate(names)
                },
                "loss": loss_val,
                "gradient_norm": grad_norm,
            }
        )

    best_model = _apply_handle_values(
        model, {name: value for name, value in zip(names, best_theta)}
    )
    report = {
        **base_report,
        "same_model_unchanged": False,
        "tuning_status": "handles_differentiable_soft_rate_loop_v0.1",
        "acceptance_decision": "ACCEPT_CANDIDATE" if math.isfinite(best_loss) else "REVISE",
        "best_parameter_value": dict(zip(names, best_theta)),
        "best_score": best_loss if math.isfinite(best_loss) else None,
        "tuning_path": "handles_differentiable_soft_rate_surrogate",
        "surrogate": "soft_rate",
        "gradient_history": history,
        "fallback_step_count": fallback_count,
        "warnings": [
            "handles_soft_rate_is_computational_scaffold_only",
            "loss_uses_a_soft_rate_surrogate_from_V_m_not_the_real_declared_objective_metric",
            "inner_soft_rate_surrogate_is_not_biological_truth",
        ],
    }
    return TuneResult(
        best_parameters=dict(zip(names, best_theta)),
        best_score=best_loss if math.isfinite(best_loss) else float("inf"),
        history=history,
        summary=json_safe(report),
        model=best_model,
    )


def _optax_unavailable_result(
    base_report: dict[str, Any], model: Any, strict: bool
) -> Any:
    """Mirror Model.tune's optax-missing branch for the handles loop."""
    from .._model import TuneResult

    try:
        require_optax()
        optax_status = "available"
    except ImportError:
        if strict:
            raise
        optax_status = "unavailable"
    if optax_status == "unavailable":
        report = {
            **base_report,
            "tuning_status": "optax_guarded_path_no_loop_v0.0.8",
            "acceptance_decision": "REVISE",
            "optax_status": optax_status,
            "same_model_unchanged": True,
            "warnings": [
                "differentiable_loop_not_enabled_for_spiking_reset_without_explicit_surrogate_kernel"
            ],
        }
        return TuneResult(
            best_parameters={}, best_score=float("inf"), history=[],
            summary=json_safe(report), model=model,
        )
    return None


def _tune_single_handle_optax(
    model: Any,
    simulation: Any,
    objective: Any,
    spec: Any,
    parameter: str,
    bounds: tuple[float, float],
    n_steps: int,
    seed: int,
    base_report: dict[str, Any],
    force_surrogate: bool = False,
) -> Any | None:
    """Single-handle gradient dispatch shared by :meth:`Model.tune`.

    Returns a :class:`TuneResult`, or ``None`` when no gradient loop applies
    (caller falls through to the metadata-only guard). Concrete-only
    edge-list/adex edge writes raise ``ValueError`` naming the path, as do
    non-differentiable handles: they cross the spiking dynamics and
    :meth:`Model.tune` has no ``surrogate=`` entry point (use
    :func:`tune_handles` with an explicit ``surrogate=`` instead).
    """
    from .._model import TuneResult

    concrete = _concrete_edge_paths(model, [parameter])
    if concrete:
        raise ValueError(
            f"Gradient tuning of parameter path(s) {concrete!r} is not "
            "supported: edge-list/adex edge writes are concrete-only "
            "(host-side numpy); use the black-box tune path instead"
        )
    if is_differentiable_handle(parameter) and not force_surrogate:
        single_loss = getattr(objective, "losses", [])
        if (
            int(n_steps) > 0
            and len(single_loss) == 1
            and single_loss[0].get("metric") in _SOURCE_SCALE_LINEAR_METRICS
            and single_loss[0].get("target") is not None
        ):
            narrow = _tune_source_scale_optax(
                model=model,
                simulation=simulation,
                objective=objective,
                spec=spec,
                bounds=(float(bounds[0]), float(bounds[1])),
                n_steps=int(n_steps),
                seed=seed,
                base_report={**base_report, "parameter": parameter},
            )
            remapped_history = [
                {
                    "step": entry["step"],
                    "parameter": parameter,
                    "value": entry["source_scale"],
                    "loss": entry["loss"],
                    "gradient": entry["gradient"],
                }
                for entry in narrow.history
            ]
            return TuneResult(
                best_parameters=(
                    {parameter: narrow.best_parameters["source_scale"]}
                    if "source_scale" in narrow.best_parameters
                    else {}
                ),
                best_score=narrow.best_score,
                history=remapped_history,
                summary=narrow.summary,
                model=narrow.model,
            )
        return None
    # Model.tune cannot pass a surrogate: refuse a gradient fit on a
    # spike-crossing handle with optax, naming the path. tune_handles keeps
    # its explicit surrogate= for this case.
    raise ValueError(
        f"Model.tune() cannot gradient-fit parameter path {parameter!r}: it "
        "crosses the spiking dynamics and Model.tune() takes no explicit "
        "surrogate= argument; use tune_handles(..., surrogate=\"soft_rate\") "
        "or a black-box optimizer instead"
    )


def tune_handles(
    model: Any,
    objective: Optional[Any] = None,
    parameters: Optional[dict[str, Any]] = None,
    optimizer: Any = None,
    simulation: Optional[Any] = None,
    seed: int = 0,
    generations: int = 8,
    population_size: int = 6,
    steps: int = 0,
    strict: bool = False,
    surrogate: Optional[str] = None,
    objectives: Optional[Any] = None,
) -> Any:
    """Fit named core-model parameters addressed by handle paths.

    Public API: ``tune_handles(model, objectives, parameters={...},
    optimizer=..., simulation=...)``. Returns :class:`TuneResult` with
    ``best_parameters`` keyed by handle path, ``best_score``, ``history``
    and ``summary``. Legacy alias: ``objective=`` for ``objectives=``.

    ``parameters`` maps one handle path (``edges.weight[...]``,
    ``cells.<family>.<field>[...]``, ``drive.<name>.rate``) to a scalar
    ``(lo, hi)`` bound pair; unknown paths are refused by name. Each path
    is one scalar knob (uniform broadcast / sign-preserving magnitude).

    Black-box optimizers run the canonical AGSDR engine
    (``generations`` x ``population_size`` candidate simulations). Gradient
    optimizers need ``spec.gradient_path_safe()`` and run a real
    ``jax.grad`` loop: the exactly-differentiable
    ``cells.izhikevich.source_scale[all]`` handle tunes the declared
    linear-metric loss directly, while every other handle needs
    ``surrogate="soft_rate"`` (else ``ValueError`` naming the path);
    concrete-only edge-list/adex edge writes are refused in the gradient
    loop (use the black-box loop for those).
    """
    from .._model import TuneResult

    if objectives is not None:
        objective = objectives
    elif objective is None:
        raise ValueError(
            "Either 'objective' (legacy) or 'objectives' (public) must be provided"
        )
    if optimizer is None:
        from .core import agsdr as _agsdr_factory

        optimizer = _agsdr_factory()
    spec = _resolve_optimizer(optimizer)
    sim = simulation or Simulation(duration_ms=10.0, dt_ms=0.1, seed=seed)
    bounds = _normalize_handle_bounds(parameters)
    paths = sorted(bounds)
    for path in paths:
        get_parameter(model, path)  # refuse unmatched selectors/families by name
    _check_edge_backend(model, paths, sim)
    base_report = _base_handles_report(model, objective, spec, bounds, seed, sim)

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
                best_parameters={}, best_score=float("inf"), history=[],
                summary=json_safe(report), model=model,
            )
        unavailable = _optax_unavailable_result(base_report, model, strict)
        if unavailable is not None:
            return unavailable
        n_steps = max(0, int(steps))
        if n_steps <= 0:
            report = {
                **base_report,
                "tuning_status": "metadata_only_no_steps_requested",
                "acceptance_decision": "REVISE",
                "candidate_history": [],
                "warnings": ["no_gradient_steps_requested"],
            }
            return TuneResult(
                best_parameters={}, best_score=float("inf"), history=[],
                summary=json_safe(report), model=model,
            )
        concrete = _concrete_edge_paths(model, paths)
        if concrete:
            raise ValueError(
                f"Gradient tuning of parameter path(s) {concrete!r} is not "
                "supported: edge-list/adex edge writes are concrete-only "
                "(host-side numpy); use the black-box tune_handles path instead"
            )
        needs_surrogate = [p for p in paths if not is_differentiable_handle(p)]
        if needs_surrogate and surrogate is None:
            raise ValueError(
                f"Gradient tuning of parameter path(s) {needs_surrogate!r} "
                "crosses the spiking dynamics and needs an explicit "
                "surrogate= (e.g. surrogate=\"soft_rate\"); only "
                "'cells.izhikevich.source_scale[all]' is exactly differentiable"
            )
        if surrogate is not None and surrogate not in _HANDLE_OPTAX_SURROGATES:
            raise ValueError(
                f"Unknown handles surrogate {surrogate!r} for parameter path(s) "
                f"{paths!r}; expected one of {sorted(_HANDLE_OPTAX_SURROGATES)}"
            )
        if len(paths) == 1 and not needs_surrogate and surrogate is None:
            handled = _tune_single_handle_optax(
                model=model,
                simulation=sim,
                objective=objective,
                spec=spec,
                parameter=paths[0],
                bounds=bounds[paths[0]],
                n_steps=n_steps,
                seed=seed,
                base_report=base_report,
                force_surrogate=False,
            )
            if handled is not None:
                return handled
        return _tune_handles_soft_rate_optax(
            model=model,
            simulation=sim,
            objective=objective,
            spec=spec,
            bounds=bounds,
            n_steps=n_steps,
            seed=seed,
            base_report={**base_report, "parameter": paths[0] if len(paths) == 1 else paths},
        )

    return _tune_handles_blackbox(
        model=model,
        objective=objective,
        spec=spec,
        bounds=bounds,
        generations=generations,
        population_size=population_size,
        seed=seed,
        strict=strict,
        simulation=sim,
        base_report=base_report,
    )
