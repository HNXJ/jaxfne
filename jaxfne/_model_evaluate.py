"""Model.evaluate()/evaluate_report() -- ``Objective`` scoring against
``Signals``, plus the soft (differentiable) rate-target loss used by
``tune()``.

Split out of ``jaxfne/_model.py`` (Phase 2 defragmentation, 2026-07-20, part
of the 0.4.8-0.4.48 roadmap's Defragmentation wave 1).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Sequence

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from ._model import Model

from .io import json_safe
from ._signals import (
    Objective,
    ObjectiveReport,
    ReadoutResult,
    ReadoutSpec,
    Signals,
    _compute_all_metrics,
    _evaluate_gate_spec,
    _evaluate_loss_spec,
    _evaluate_regularizer_spec,
    _finite_or_none,
)


def evaluate(
    self,
    signals: Signals,
    objective: "Objective | str",
    readout: Optional[dict[str, Any]] = None,
    strict: bool = False,
    state_diagnostics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Full objective/gate evaluation with JSON-safe report.

    Gate pass/fail is a computational diagnostic only.  It does not imply
    empirical validation, biological calibration, or mechanism proof.
    All truth gates from v0.0.4 are preserved in the report.
    """

    if isinstance(objective, str):
        objective = Objective(name=objective)

    cfg_meta = self.cfg.metadata
    warnings: list[str] = []

    # Special dispatch for group-rate targets objective
    if getattr(objective, "kind", "generic") == "group_rate_targets":
        return self._evaluate_group_rate_targets(
            signals,
            objective,
            warnings,
            cfg_meta,
            state_diagnostics=state_diagnostics,
        )

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
    state_diagnostics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate group-wise firing rate targets objective.

    Extracts group definitions and target rates from objective metadata,
    computes group-wise firing rates, and returns squared relative error loss.
    """

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

    rate_meta = next(
        (
            gate.get("metadata", {})
            for gate in objective.gates
            if isinstance(gate, dict)
            and "metadata" in gate
            and "groups" in gate.get("metadata", {})
        ),
        {},
    )
    burn_in_ms = float(rate_meta.get("burn_in_ms", 0.0))
    window_end_ms = rate_meta.get("window_end_ms")
    epsilon = float(rate_meta.get("epsilon", 1e-6))
    n_steps = int(signals.spikes.shape[0])
    start_idx = int(math.ceil(burn_in_ms / dt_ms))
    end_idx = (
        n_steps
        if window_end_ms is None
        else min(n_steps, int(math.ceil(float(window_end_ms) / dt_ms)))
    )
    invalid_status = None
    if start_idx < 0 or start_idx >= n_steps or end_idx <= start_idx:
        invalid_status = "empty_rate_window"

    state_validity = _state_validity_report(signals, state_diagnostics)
    if state_validity["status"] != "valid":
        invalid_status = state_validity["status"]

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
            all_gates_pass = False
            continue

        try:
            # Extract spikes for this group
            if invalid_status is not None:
                raise ValueError(invalid_status)
            group_spikes = signals.spikes[start_idx:end_idx, idx_list]

            # Compute mean spike rate over time and neurons in group
            group_rate_hz = float(jnp.mean(group_spikes) * (1000.0 / dt_ms))

            # Compute squared relative error over the declared measurement window.
            raw_loss = ((group_rate_hz - target_hz) / max(abs(target_hz), epsilon)) ** 2

            weighted_loss = weight * raw_loss
            total_loss += weighted_loss

            loss_details.append({
                "group": group_name,
                "target_hz": float(target_hz),
                "achieved_hz": _finite_or_none(group_rate_hz),
                "weight": float(weight),
                "raw_loss": _finite_or_none(raw_loss),
                "weighted_loss": _finite_or_none(weighted_loss),
                "window_start_ms": float(start_idx * dt_ms),
                "window_end_ms": float(end_idx * dt_ms),
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

    state_metrics = {
        "hdp_H_abs_max": state_validity.get("H_abs_max"),
        "hdp_W_abs_max": state_validity.get("W_abs_max"),
    }
    regularizer_results = []
    regularizer_total = 0.0
    for spec in getattr(objective, "regularizers", []):
        result = _evaluate_regularizer_spec(spec, state_metrics, warnings, strict=False)
        regularizer_results.append(result)
        weighted = result.get("weighted_value")
        if weighted is None or not math.isfinite(float(weighted)):
            all_gates_pass = False
        else:
            regularizer_total += float(weighted)

    rate_loss = float(total_loss)
    total_score = rate_loss + regularizer_total
    # Check if loss and state are finite.
    has_loss_value = math.isfinite(total_score) and invalid_status is None
    if not has_loss_value:
        all_gates_pass = False

    acceptance = "gates_pass" if (all_gates_pass and has_loss_value) else "gates_fail"
    single_group = len(loss_details) == 1
    single_rate = loss_details[0]["achieved_hz"] if single_group else None
    single_target = loss_details[0]["target_hz"] if single_group else None

    return json_safe({
        "evaluation_status": "objective_evaluate_group_rate_targets_v0.0.1",
        "objective_name": getattr(objective, "name", "spectrolaminar_objective"),
        "total_loss": _finite_or_none(total_score) if has_loss_value else None,
        "total_score": _finite_or_none(total_score) if has_loss_value else None,
        "rate": single_rate,
        "target_rate": single_target,
        "rate_loss": _finite_or_none(rate_loss),
        "weight_regularizer": _finite_or_none(
            sum(
                float(r.get("weighted_value"))
                for r in regularizer_results
                if "W" in str(r.get("metric", "")).upper()
                and r.get("weighted_value") is not None
            )
        ),
        "H_regularizer": _finite_or_none(
            sum(
                float(r.get("weighted_value"))
                for r in regularizer_results
                if "H" in str(r.get("metric", ""))
                and r.get("weighted_value") is not None
            )
        ),
        "invalid_status": invalid_status,
        "state_validity": state_validity,
        "group_rate_losses": loss_details,
        "losses": [],
        "regularizers": regularizer_results,
        "gates": [],
        "all_gates_pass": all_gates_pass,
        "acceptance_decision": acceptance,
        "claim_level": cfg_meta.get("claim_level", "computational_scaffold"),
        "field_claim_level": "proxy_readout",
        "physical_amplitude_calibrated": False,
        "warnings": warnings,
    })


def _state_validity_report(
    signals: Signals,
    state_diagnostics: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Check finite neural/HDP state without imposing optimizer bounds on HDP."""
    arrays: dict[str, Any] = {
        "X": signals.V_m,
        "spikes": signals.spikes,
        "sources": signals.sources,
    }
    if state_diagnostics:
        arrays.update(
            {
                key: state_diagnostics.get(key)
                for key in ("H_trace", "H_final", "w_trace", "w_final")
                if state_diagnostics.get(key) is not None
            }
        )
    finite = {
        name: bool(jnp.all(jnp.isfinite(value)))
        for name, value in arrays.items()
        if value is not None
    }
    if not all(finite.values()):
        return {
            "status": "nonfinite_state",
            "finite": finite,
            "H_abs_max": None,
            "W_abs_max": None,
            "bounds": {},
        }
    H_values = [value for name, value in arrays.items() if name.startswith("H_")]
    W_values = [value for name, value in arrays.items() if name.startswith("w_")]
    hdp_params = signals.metadata.get("hdp", {}).get("params", {})
    bounds: dict[str, Any] = {}
    violations: list[str] = []
    if "H_min" in hdp_params and "H_max" in hdp_params:
        bounds["H"] = [float(hdp_params["H_min"]), float(hdp_params["H_max"])]
        if any(
            bool(jnp.any(value < bounds["H"][0])) or
            bool(jnp.any(value > bounds["H"][1]))
            for value in H_values
        ):
            violations.append("H_bounds")
    if "w_min" in hdp_params and "w_max" in hdp_params:
        bounds["W"] = [float(hdp_params["w_min"]), float(hdp_params["w_max"])]
        if any(
            bool(jnp.any(value < bounds["W"][0])) or
            bool(jnp.any(value > bounds["W"][1]))
            for value in W_values
        ):
            violations.append("W_bounds")
    return {
        "status": "state_bounds_violation" if violations else "valid",
        "finite": finite,
        "H_abs_max": max(
            (float(jnp.max(jnp.abs(value))) for value in H_values),
            default=None,
        ),
        "W_abs_max": max(
            (float(jnp.max(jnp.abs(value))) for value in W_values),
            default=None,
        ),
        "bounds": bounds,
        "bound_violations": violations,
    }


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


