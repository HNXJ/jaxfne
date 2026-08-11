"""Bounded 0.4.12 edge-native HDP optimization MCC-3."""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._model_tune import (
    _candidate_state_evidence,
    _edge_parameter_mask,
    _model_with_parameters,
)
from jaxfne.hdp_network import DEFAULT_HDP
from jaxfne.io import json_safe


def _mcc3_config():
    return (
        jtfne.configuration()
        .runtime(seed=0, recurrent_backend="edge_list")
        .network(
            name="V1",
            kind="cortical_column",
            n=10,
            cell_types={"E": 0.5, "PV": 0.5},
        )
        .cell_type_drives({"E": 8.0, "PV": 8.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(
            domain="laminar_column",
            conductivity="proxy",
            boundary="mean_zero_neumann",
            gauge="mean_zero",
        )
        .probe(name="probe", modes=["spikes", "V_m"])
    )


def _mcc3_specs():
    return {
        name: jtfne.edge_parameter(
            pre={"cell_type": pre},
            post={"cell_type": post},
            bounds=(0.1, 5.0),
        )
        for name, pre, post in (
            ("m_EE", "E", "E"),
            ("m_EI", "E", "PV"),
            ("m_IE", "PV", "E"),
            ("m_II", "PV", "PV"),
        )
    }


def _mcc3_runtime(enabled: bool) -> jtfne.RuntimeConfig:
    params = dict(DEFAULT_HDP)
    params.update({"H_min": 0.1, "H_max": 10.0, "w_min": -10.0, "w_max": 10.0})
    return jtfne.RuntimeConfig(
        enable_hdp=enabled,
        recurrent_backend="edge_list",
        jit=False,
        hdp_params=params if enabled else {},
    )


@pytest.fixture(scope="module")
def mcc3_bundle():
    model = jtfne.construct(_mcc3_config())
    sim_on = jtfne.simulation(
        duration_ms=100.0,
        dt_ms=0.1,
        seed=17,
        runtime=_mcc3_runtime(True),
    )
    objective = jtfne.rate_targets(
        groups={"all": list(range(10))},
        targets_hz={"all": 20.0},
        burn_in_ms=20.0,
    )
    specs = _mcc3_specs()
    optimizer = jtfne.agsdr(
        parameters=specs,
        generations=2,
        population_size=4,
        seed=42,
    )
    result = model.tune(
        objectives=objective,
        optimizer=optimizer,
        simulation=sim_on,
    )
    repeat = model.tune(
        objectives=objective,
        optimizer=jtfne.agsdr(
            parameters=_mcc3_specs(),
            generations=2,
            population_size=4,
            seed=42,
        ),
        simulation=sim_on,
    )

    def evaluate_condition(candidate_model, simulation):
        signals = candidate_model.simulate(simulation)
        report = candidate_model.evaluate(
            signals,
            objective,
            state_diagnostics=candidate_model.last_hdp_diagnostics(),
        )
        return {
            "signals": signals,
            "report": report,
            "state_evidence": _candidate_state_evidence(
                candidate_model,
                candidate_model.last_hdp_diagnostics(),
            ),
        }

    condition_a = evaluate_condition(model, sim_on)
    condition_b = evaluate_condition(result.model, sim_on)
    sim_off = jtfne.simulation(
        duration_ms=100.0,
        dt_ms=0.1,
        seed=17,
        runtime=_mcc3_runtime(False),
    )
    condition_c = evaluate_condition(result.model, sim_off)
    manifest = model.manifest(
        signals=condition_b["signals"],
        evaluation=condition_b["report"],
        tuning={
            **result.summary,
            "mcc3_conditions": {
                "A": condition_a["report"],
                "B": condition_b["report"],
                "C": condition_c["report"],
            },
        },
    )
    return {
        "model": model,
        "objective": objective,
        "specs": specs,
        "result": result,
        "repeat": repeat,
        "A": condition_a,
        "B": condition_b,
        "C": condition_c,
        "manifest": manifest,
    }


def test_edge_group_selection_and_sign_preservation(mcc3_bundle) -> None:
    model = mcc3_bundle["model"]
    specs = mcc3_bundle["specs"]
    masks = {name: _edge_parameter_mask(model, name, spec) for name, spec in specs.items()}

    assert {int(mask.sum()) for mask in masks.values()} == {20, 25}
    assert sum(int(mask.sum()) for mask in masks.values()) == 90

    baseline = np.asarray(model.params["edge_list"].weight)
    candidate = _model_with_parameters(
        model,
        {name: 1.5 for name in specs},
        specs,
    )
    updated = np.asarray(candidate.params["edge_list"].weight)
    assert np.all(np.sign(updated) == np.sign(baseline))
    assert np.array_equal(
        np.asarray(candidate.params["emitter"].W),
        np.asarray(model.params["emitter"].W),
    )
    for mask in masks.values():
        assert np.allclose(np.abs(updated[mask]), 1.5)


def test_edge_candidate_changes_continuous_dynamics(mcc3_bundle) -> None:
    model = mcc3_bundle["model"]
    specs = mcc3_bundle["specs"]
    low = _model_with_parameters(model, {name: 0.1 for name in specs}, specs)
    high = _model_with_parameters(model, {name: 5.0 for name in specs}, specs)
    sim = jtfne.simulation(
        duration_ms=20.0,
        dt_ms=0.1,
        seed=19,
        runtime=_mcc3_runtime(False),
    )
    low_signals = low.simulate(sim)
    high_signals = high.simulate(sim)
    assert not np.allclose(np.asarray(low.params["edge_list"].weight), np.asarray(high.params["edge_list"].weight))
    assert not np.allclose(np.asarray(low_signals.V_m), np.asarray(high_signals.V_m))


def test_matrix_parameter_is_rejected_for_edge_backend(mcc3_bundle) -> None:
    with pytest.raises(ValueError, match="edge_list.weight"):
        mcc3_bundle["model"].tune(
            objectives=mcc3_bundle["objective"],
            optimizer=jtfne.agsdr(
                parameters={"all_w": jtfne.matrix_parameter(mask="all", bounds=(0.1, 5.0))},
                generations=1,
                population_size=2,
                seed=42,
            ),
            simulation=jtfne.simulation(
                duration_ms=10.0,
                dt_ms=0.1,
                seed=17,
                runtime=_mcc3_runtime(True),
            ),
        )


def test_rate_objective_uses_post_burn_in_window(mcc3_bundle) -> None:
    model = mcc3_bundle["model"]
    spikes = jnp.zeros((10, 2), dtype=jnp.float32).at[5:, 0].set(1.0)
    signals = jtfne.Signals(
        time_ms=jnp.arange(10, dtype=jnp.float32),
        V_m=jnp.zeros((10, 2), dtype=jnp.float32),
        spikes=spikes,
        sources=jnp.zeros((10, 2), dtype=jnp.float32),
        field=None,
        metadata={"dt_ms": 1.0},
    )
    objective = jtfne.rate_targets(
        groups={"all": [0, 1]},
        targets_hz={"all": 500.0},
        burn_in_ms=5.0,
    )
    report = model.evaluate(signals, objective)
    assert report["rate"] == 500.0
    assert report["target_rate"] == 500.0
    assert report["rate_loss"] == 0.0
    assert report["group_rate_losses"][0]["window_start_ms"] == 5.0


def test_invalid_rate_window_is_an_explicit_rejection(mcc3_bundle) -> None:
    model = mcc3_bundle["model"]
    signals = mcc3_bundle["A"]["signals"]
    objective = jtfne.rate_targets(
        groups={"all": list(range(10))},
        targets_hz={"all": 20.0},
        burn_in_ms=signals.metadata["duration_ms"] + 1.0,
    )
    report = model.evaluate(signals, objective)
    assert report["invalid_status"] == "empty_rate_window"
    assert report["total_score"] is None
    assert report["all_gates_pass"] is False


def test_candidate_exception_becomes_rejected_score(mcc3_bundle, monkeypatch) -> None:
    import jaxfne._model_tune as model_tune

    def reject_candidate(*args, **kwargs):
        raise ValueError("candidate state invalid")

    monkeypatch.setattr(model_tune, "_model_with_parameters", reject_candidate)
    result = mcc3_bundle["model"].tune(
        objectives=mcc3_bundle["objective"],
        optimizer=jtfne.agsdr(
            parameters=_mcc3_specs(),
            generations=1,
            population_size=2,
            seed=42,
        ),
        simulation=jtfne.simulation(
            duration_ms=20.0,
            dt_ms=0.1,
            seed=17,
            runtime=_mcc3_runtime(True),
        ),
    )
    assert result.summary["tuning_status"] == "multiparameter_agsdr_v0.0.7"
    assert result.summary["candidate_rejection_count"] == 2
    assert result.best_score == float("inf")
    assert all(
        item["objective"]["invalid_status"].startswith("ValueError:")
        for item in result.summary["candidate_evaluations"]
    )
    assert all(
        item["score_status"] == "positive_infinity"
        for item in result.summary["candidate_evaluations"]
    )


def test_mcc3_shared_execution_closes_ab(mcc3_bundle) -> None:
    A = mcc3_bundle["A"]
    B = mcc3_bundle["B"]
    C = mcc3_bundle["C"]
    result = mcc3_bundle["result"]
    repeat = mcc3_bundle["repeat"]

    assert set(result.best_parameters) == {"m_EE", "m_EI", "m_IE", "m_II"}
    assert all(0.1 <= value <= 5.0 for value in result.best_parameters.values())
    assert A["signals"].metadata["hdp"]["enabled"] is True
    assert B["signals"].metadata["hdp"]["enabled"] is True
    assert "hdp" not in C["signals"].metadata
    assert A["report"]["state_validity"]["status"] == "valid"
    assert B["report"]["state_validity"]["status"] == "valid"
    assert C["report"]["state_validity"]["status"] == "valid"
    assert "W_final" not in C["state_evidence"]
    assert B["report"]["state_validity"]["finite"]["H_trace"] is True
    assert B["report"]["state_validity"]["finite"]["w_trace"] is True
    assert B["report"]["state_validity"]["bounds"] == {
        "H": [0.1, 10.0],
        "W": [-10.0, 10.0],
    }
    assert B["report"]["total_score"] < A["report"]["total_score"]
    assert abs(B["report"]["rate"] - B["report"]["target_rate"]) < abs(
        A["report"]["rate"] - A["report"]["target_rate"]
    )
    assert result.best_parameters == repeat.best_parameters
    assert result.best_score == repeat.best_score
    assert set(result.summary["theta_0"]) == set(result.best_parameters)
    assert result.summary["theta_best"] == result.best_parameters
    assert result.summary["initial_weight_evidence"]["finite"] is True
    assert result.summary["best_evaluation"]["state_evidence"]["W_final"]["finite"] is True
    assert (
        result.summary["best_evaluation"]["state_evidence"]["W0"]["sha256"]
        != result.summary["best_evaluation"]["state_evidence"]["W_final"]["sha256"]
    )
    assert result.summary["candidate_rejection_count"] == 0
    assert all(
        item["score_status"] == "finite"
        for item in result.summary["candidate_evaluations"]
    )


def test_mcc3_manifest_is_json_safe_and_retains_conditions(mcc3_bundle) -> None:
    manifest = json_safe(mcc3_bundle["manifest"])
    json.dumps(manifest, allow_nan=False)
    conditions = manifest["tuning"]["mcc3_conditions"]
    assert set(conditions) == {"A", "B", "C"}
    assert conditions["A"]["rate"] is not None
    assert conditions["B"]["rate_loss"] is not None
    assert {
        "rate",
        "target_rate",
        "rate_loss",
        "weight_regularizer",
        "H_regularizer",
        "invalid_status",
        "total_score",
    }.issubset(conditions["B"])
    assert conditions["C"]["invalid_status"] is None
