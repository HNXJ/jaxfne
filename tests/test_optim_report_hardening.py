import math
import json
import jax
import jax.numpy as jnp
import pytest
import unittest.mock
import jaxfne as jtfne
from jaxfne.fields import spectrolaminar_objective


def _cfg(n=8):
    return (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


def test_optimization_determinism_and_stochastic_variance():
    """Verify that same seed produces identical candidates, and different seed changes sequence."""
    model = jtfne.construct(_cfg(8))
    sim = jtfne.simulation(duration_ms=4.0, dt_ms=0.1, seed=0)
    obj = spectrolaminar_objective(target_profiles={})

    # Same seed = 42 -> Should be perfectly identical
    opt_a1 = jtfne.agsdr(
        parameters={"source_scale": (0.5, 3.0)},
        generations=2,
        population_size=3,
        seed=42,
    )
    result_a1 = model.tune(objectives=obj, optimizer=opt_a1, simulation=sim)

    opt_a2 = jtfne.agsdr(
        parameters={"source_scale": (0.5, 3.0)},
        generations=2,
        population_size=3,
        seed=42,
    )
    result_a2 = model.tune(objectives=obj, optimizer=opt_a2, simulation=sim)

    assert result_a1.best_parameters == result_a2.best_parameters
    assert result_a1.best_score == result_a2.best_score
    for h1, h2 in zip(result_a1.history, result_a2.history):
        assert h1["evaluated_candidates"] == h2["evaluated_candidates"]

    # Different seed = 99 -> Candidate values must be different
    opt_b = jtfne.agsdr(
        parameters={"source_scale": (0.5, 3.0)},
        generations=2,
        population_size=3,
        seed=99,
    )
    result_b = model.tune(objectives=obj, optimizer=opt_b, simulation=sim)

    # Confirm stochastic variance across different seeds
    diff_detected = False
    for h1, hb in zip(result_a1.history, result_b.history):
        if h1["evaluated_candidates"] != hb["evaluated_candidates"]:
            diff_detected = True
            break
    assert diff_detected, "Different seeds must produce different stochastic candidates"


def test_optimization_report_schema_and_json_strictness():
    """Verify JSON strictness and that all canonical metadata keys exist and are correct."""
    model = jtfne.construct(_cfg(8))
    sim = jtfne.simulation(duration_ms=4.0, dt_ms=0.1, seed=0)
    obj = spectrolaminar_objective(target_profiles={})

    opt = jtfne.agsdr(
        parameters={"source_scale": (0.5, 3.0)},
        generations=1,
        population_size=2,
        seed=1,
    )
    result = model.tune(objectives=obj, optimizer=opt, simulation=sim)
    summary = result.summary

    # Assert canonical keys
    assert summary["physical_amplitude_calibrated"] is False
    assert summary["biological_learning_claim"] is False
    assert summary["amplitude_claim_allowed"] is False
    assert summary["surrogate_status"] in {"none", "not_applicable", "declared", "required_but_missing"}
    assert summary["final_evaluation_basis"] == "total_loss"

    # Strict JSON serialization test
    json_summary = json.dumps(summary, allow_nan=False)
    json_history = json.dumps(result.history, allow_nan=False)
    assert isinstance(json_summary, str)
    assert isinstance(json_history, str)


def test_getattr_compatibility_boundary():
    """Verify evaluate and tune behavior under standard, legacy, and custom minimal objectives."""
    model = jtfne.construct(_cfg(8))
    sim = jtfne.simulation(duration_ms=4.0, dt_ms=0.1, seed=0)

    # 1. Standard Objective
    std_obj = jtfne.Objective(name="standard_objective")
    opt = jtfne.agsdr(parameters={"source_scale": (0.5, 3.0)}, generations=1, population_size=2, seed=1)
    res_std = model.tune(objectives=std_obj, optimizer=opt, simulation=sim)
    assert res_std.summary["objective_name"] == "standard_objective"

    # 2. Legacy spectrolaminar_objective
    legacy_obj = spectrolaminar_objective(target_profiles={})
    res_legacy = model.tune(objectives=legacy_obj, optimizer=opt, simulation=sim)
    assert res_legacy.summary["objective_name"] == "spectrolaminar_objective"

    # 3. Custom Minimal Objective (no standard properties)
    class CustomMinimalObjective:
        pass

    custom_obj = CustomMinimalObjective()
    res_custom = model.tune(objectives=custom_obj, optimizer=opt, simulation=sim)
    assert res_custom.summary["objective_name"] == "spectrolaminar_objective"


def test_rejection_reasons_and_finite_gates():
    """Verify that gate failures and non-finite results trigger correct rejection reasons."""
    model = jtfne.construct(_cfg(8))
    sim = jtfne.simulation(duration_ms=4.0, dt_ms=0.1, seed=0)

    # Mock evaluate to return failing/nan values
    def failing_evaluate(self, signals, objective, strict=False):
        return {
            "total_loss": float("nan"),
            "all_gates_pass": False,
            "evaluation_status": "rejected",
        }

    obj = spectrolaminar_objective(target_profiles={})
    opt = jtfne.agsdr(
        parameters={"source_scale": (0.5, 3.0)},
        generations=1,
        population_size=2,
        seed=1,
    )

    with unittest.mock.patch.object(jtfne.core.Model, "evaluate", failing_evaluate):
        result = model.tune(objectives=obj, optimizer=opt, simulation=sim)
    
    assert len(result.history) == 1
    eval_cands = result.history[0]["evaluated_candidates"]
    assert len(eval_cands) == 2
    for cand in eval_cands:
        reasons = cand["rejection_reasons"]
        assert "failed_objective_gates" in reasons
        assert "non_finite_loss" in reasons
        assert cand["gates_pass"] is False
        assert cand["score"] is None
