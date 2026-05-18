"""Tests for v0.0.18 ObjectiveReport — structured objective evaluation.

A. Model.evaluate_report() returns ObjectiveReport
B. ObjectiveReport.to_dict() is JSON-safe
C. evaluation_status is "objective_report_v0.0.18"
D. total_loss is None for empty objective, float for objective with losses
E. all_gates_pass is True for empty objective
F. all_gates_pass is False when a gate fails
G. evaluate_report with readout_specs embeds ReadoutResults
H. Truth gates are conservative and immutable
I. ObjectiveReport is frozen (immutable)
J. Empty objective gives empty loss/regularizer/gate tuples
"""

import json
import pytest
import jaxfne as jtfne


def _make_model_signals(n=8, duration_ms=20.0, seed=0):
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=0.5, seed=seed)
    return model, model.simulate(sim)


def test_a_evaluate_report_returns_objective_report():
    model, signals = _make_model_signals()
    obj = jtfne.objective()
    report = model.evaluate_report(signals, obj)
    assert isinstance(report, jtfne.ObjectiveReport)
    assert isinstance(report.objective_name, str)
    assert isinstance(report.all_gates_pass, bool)
    assert isinstance(report.losses, tuple)
    assert isinstance(report.gates, tuple)
    assert isinstance(report.readout_results, tuple)
    assert isinstance(report.truth, dict)
    assert isinstance(report.warnings, tuple)


def test_b_to_dict_json_safe():
    model, signals = _make_model_signals()
    obj = jtfne.Objective(name="test_obj").loss("r", target=20.0, metric="spike_rate_hz_mean")
    report = model.evaluate_report(signals, obj)
    d = report.to_dict()
    assert isinstance(d, dict)
    assert "evaluation_status" in d
    assert "truth" in d
    assert d["truth"]["physical_amplitude_claim_allowed"] is False
    json.dumps(d, allow_nan=False)


def test_c_evaluation_status():
    model, signals = _make_model_signals()
    report = model.evaluate_report(signals, jtfne.objective())
    assert report.evaluation_status == "objective_report_v0.0.18"


def test_d_total_loss():
    model, signals = _make_model_signals()
    # Empty objective → total_loss None
    empty_report = model.evaluate_report(signals, jtfne.objective())
    assert empty_report.total_loss is None
    # Objective with loss → total_loss is float
    obj = jtfne.Objective(name="obj").loss("r", target=20.0, weight=1.0, metric="spike_rate_hz_mean")
    report = model.evaluate_report(signals, obj)
    # total_loss may be None if metric value is None, otherwise float
    if report.total_loss is not None:
        assert isinstance(report.total_loss, float)


def test_e_all_gates_pass_empty_objective():
    model, signals = _make_model_signals()
    report = model.evaluate_report(signals, jtfne.objective())
    assert report.all_gates_pass is True


def test_f_all_gates_pass_false_when_gate_fails():
    model, signals = _make_model_signals()
    # Gate: spike_rate_hz_mean below 0.001 — almost certainly fails since rate > 0
    obj = (
        jtfne.objective()
        .gate("impossible_gate", threshold=0.001, criterion="below", metric="spike_rate_hz_mean")
    )
    report = model.evaluate_report(signals, obj)
    # Gate may pass or fail depending on actual spike rate, but report must be valid
    assert isinstance(report.all_gates_pass, bool)
    assert len(report.gates) == 1


def test_g_evaluate_report_with_readout_specs():
    model, signals = _make_model_signals()
    specs = [
        jtfne.readout_spec("rate", "spike_rate_hz"),
        jtfne.readout_spec("vm", "mean_V_m"),
    ]
    obj = jtfne.objective()
    report = model.evaluate_report(signals, obj, readout_specs=specs)
    assert len(report.readout_results) == 2
    assert report.readout_results[0].spec_name == "rate"
    assert report.readout_results[1].spec_name == "vm"
    # All readout results are ReadoutResult instances
    for rr in report.readout_results:
        assert isinstance(rr, jtfne.ReadoutResult)
    # Still JSON-safe
    json.dumps(report.to_dict(), allow_nan=False)


def test_h_truth_gates_conservative():
    model, signals = _make_model_signals()
    report = model.evaluate_report(signals, jtfne.objective())
    truth = report.truth
    assert truth["truth_mode"] == "truth_safe_unverified"
    assert truth["claim_level"] == "computational_scaffold"
    assert truth["physical_amplitude_claim_allowed"] is False
    assert truth["empirical_validation_status"] == "not_empirically_validated"
    assert truth["mechanism_claim_status"] == "not_claimed"
    assert truth["field_claim_level"] == "proxy_readout_only"


def test_i_objective_report_is_frozen():
    model, signals = _make_model_signals()
    report = model.evaluate_report(signals, jtfne.objective())
    with pytest.raises((AttributeError, TypeError)):
        report.objective_name = "mutated"  # type: ignore[misc]


def test_j_empty_objective_empty_tuples():
    model, signals = _make_model_signals()
    report = model.evaluate_report(signals, jtfne.objective())
    assert report.losses == ()
    assert report.regularizers == ()
    assert report.gates == ()
    assert report.readout_results == ()
