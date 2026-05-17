"""Tests for v0.0.5-P2 Objective/evaluate stack."""

import json

import jaxfne as jtfne


def _model_and_signals(n=12, duration_ms=10.0):
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=0.1, seed=0)
    signals = model.simulate(sim)
    return model, signals


def test_objective_builder_loss_regularizer_gate():
    """Test that Objective builders chain correctly and store manifest-safe dicts."""
    obj = (
        jtfne.objective()
        .loss("spike_rate_loss", target=20.0, weight=1.0, metric="spike_rate_hz_mean")
        .regularizer("vm_reg", target=-65.0, weight=0.1, metric="mean_V_m")
        .gate("rate_gate", threshold=100.0, criterion="below", metric="spike_rate_hz_mean")
    )
    assert len(obj.losses) == 1
    assert len(obj.regularizers) == 1
    assert len(obj.gates) == 1
    assert obj.losses[0]["name"] == "spike_rate_loss"
    assert obj.losses[0]["metric"] == "spike_rate_hz_mean"
    assert obj.losses[0]["target"] == 20.0
    assert obj.regularizers[0]["metric"] == "mean_V_m"
    assert obj.gates[0]["criterion"] == "below"
    assert obj.gates[0]["threshold"] == 100.0


def test_objective_compose():
    """Test that compose() merges two Objective instances without mutation."""
    obj_a = jtfne.objective().loss("loss_a", metric="spike_rate_hz_mean")
    obj_b = jtfne.objective().gate("gate_b", threshold=50.0, criterion="below", metric="spike_rate_hz_mean")
    composed = obj_a.compose(obj_b)
    # Original objects unchanged (frozen dataclass fluent pattern)
    assert len(obj_a.losses) == 1
    assert len(obj_a.gates) == 0
    assert len(obj_b.losses) == 0
    # Composed has both
    assert len(composed.losses) == 1
    assert len(composed.gates) == 1


def test_model_evaluate_empty_objective():
    """Test Model.evaluate() with an empty Objective returns a valid report."""
    model, signals = _model_and_signals()
    obj = jtfne.objective()
    report = model.evaluate(signals, obj)
    assert report["evaluation_status"] == "objective_evaluate_v0.0.5"
    assert report["losses"] == []
    assert report["regularizers"] == []
    assert report["gates"] == []
    assert report["all_gates_pass"] is True
    assert report["acceptance_decision"] == "gates_pass"
    assert report["total_loss"] is None


def test_model_evaluate_known_metrics():
    """Test Model.evaluate() computes known metrics and returns finite values."""
    model, signals = _model_and_signals(n=12, duration_ms=20.0)
    obj = (
        jtfne.objective()
        .loss("rate_loss", target=20.0, weight=1.0, metric="spike_rate_hz_mean")
        .regularizer("vm_reg", target=-65.0, weight=0.1, metric="mean_V_m")
    )
    report = model.evaluate(signals, obj)
    assert report["evaluation_status"] == "objective_evaluate_v0.0.5"
    assert len(report["losses"]) == 1
    assert len(report["regularizers"]) == 1

    loss_r = report["losses"][0]
    assert loss_r["status"] == "ok"
    assert loss_r["value"] is not None
    assert loss_r["weighted_value"] is not None

    reg_r = report["regularizers"][0]
    assert reg_r["status"] == "ok"
    assert reg_r["value"] is not None

    assert report["total_loss"] is not None
    assert isinstance(report["total_loss"], float)


def test_gate_below_above_in_range():
    """Test gate criteria: below, above, in_range."""
    model, signals = _model_and_signals(n=12, duration_ms=20.0)

    # Gate: spike_rate below 10000 should always pass
    obj_below = jtfne.objective().gate("rate_low", threshold=10000.0, criterion="below", metric="spike_rate_hz_mean")
    rep = model.evaluate(signals, obj_below)
    assert rep["gates"][0]["pass"] is True

    # Gate: spike_rate above 0 should always pass (rate >= 0 so above 0.0 can fail if no spikes)
    obj_above = jtfne.objective().gate("rate_above_neg", threshold=-1.0, criterion="above", metric="spike_rate_hz_mean")
    rep = model.evaluate(signals, obj_above)
    assert rep["gates"][0]["pass"] is True

    # Gate: in_range with wide bounds should pass
    obj_range = jtfne.objective().gate("rate_range", threshold=(0.0, 100000.0), criterion="in_range", metric="spike_rate_hz_mean")
    rep = model.evaluate(signals, obj_range)
    assert rep["gates"][0]["pass"] is True

    # Gate: below 0 should always fail (rates are non-negative)
    obj_fail = jtfne.objective().gate("rate_neg", threshold=-1.0, criterion="below", metric="spike_rate_hz_mean")
    rep = model.evaluate(signals, obj_fail)
    assert rep["gates"][0]["pass"] is False
    assert rep["all_gates_pass"] is False
    assert rep["acceptance_decision"] == "gates_fail"


def test_evaluation_report_json_safe():
    """Test that the evaluation report serializes with allow_nan=False."""
    model, signals = _model_and_signals()
    obj = (
        jtfne.objective()
        .loss("rate_loss", target=20.0, weight=1.0, metric="spike_rate_hz_mean")
        .gate("rate_gate", threshold=10000.0, criterion="below", metric="spike_rate_hz_mean")
    )
    report = model.evaluate(signals, obj)
    json_str = json.dumps(report, allow_nan=False)
    assert isinstance(json_str, str)
    loaded = json.loads(json_str)
    assert "evaluation_status" in loaded
    assert "total_loss" in loaded


def test_evaluation_preserves_truth_gates():
    """Test that evaluate() always returns frozen truth gates."""
    model, signals = _model_and_signals()
    report = model.evaluate(signals, jtfne.objective())
    assert report["truth_mode"] == "truth_safe_unverified"
    assert report["claim_level"] == "computational_scaffold"
    assert report["field_claim_level"] == "proxy_readout_only"
    assert report["physical_amplitude_claim_allowed"] is False


def test_unknown_metric_non_strict_warning():
    """Test that unknown metric in non-strict mode adds a warning and sets status."""
    model, signals = _model_and_signals()
    obj = jtfne.objective().gate("bad_gate", threshold=1.0, criterion="below", metric="no_such_metric_xyz")
    report = model.evaluate(signals, obj, strict=False)
    gate_r = report["gates"][0]
    assert gate_r["value"] is None
    assert gate_r["pass"] is False
    assert "unknown_metric" in gate_r["status"]
    assert len(report["warnings"]) > 0
    assert any("no_such_metric_xyz" in w for w in report["warnings"])


def test_unknown_metric_strict_failure_or_status():
    """Test that unknown metric in strict mode still produces a status (not exception)."""
    model, signals = _model_and_signals()
    obj = jtfne.objective().loss("bad_loss", metric="no_such_metric_xyz")
    # strict=True should surface the issue cleanly without raising
    report = model.evaluate(signals, obj, strict=True)
    loss_r = report["losses"][0]
    assert loss_r["value"] is None
    assert "unknown_metric" in loss_r["status"]


def test_no_callable_serialization():
    """Test that Objective specs contain no callable objects."""
    obj = (
        jtfne.objective()
        .loss("rate_loss", target=20.0, weight=1.0, metric="spike_rate_hz_mean")
        .regularizer("vm_reg", target=-65.0, weight=0.1, metric="mean_V_m")
        .gate("rate_gate", threshold=100.0, criterion="below", metric="spike_rate_hz_mean")
    )
    for spec in [*obj.losses, *obj.regularizers, *obj.gates]:
        for v in spec.values():
            assert not callable(v), f"Callable found in spec: {spec}"
    # Objective itself must be JSON-serializable
    obj_dict = {
        "name": obj.name,
        "losses": obj.losses,
        "regularizers": obj.regularizers,
        "gates": obj.gates,
    }
    json_str = json.dumps(obj_dict, allow_nan=False)
    assert isinstance(json_str, str)


def test_objective_can_reference_paradigm_condition_metadata_without_hardcoding_task_claims():
    """Test that paradigm condition metadata can be passed via objective metadata field.

    This verifies that Objective.metadata is forwarded without modification,
    and that no biological/mechanism language is injected automatically.
    """
    paradigm = jtfne.standard_visual_omission()
    aaax = paradigm.condition("AAAX")
    assert aaax is not None

    obj = (
        jtfne.objective()
        .gate(
            "omission_p4_rate_gate",
            threshold=200.0,
            criterion="below",
            metric="spike_rate_hz_mean",
            metadata={"condition": aaax.name, "omission_position": aaax.omission_position},
        )
    )
    gate_spec = obj.gates[0]
    assert gate_spec["metadata"]["condition"] == "AAAX"
    assert gate_spec["metadata"]["omission_position"] == "p4"

    # No mechanism claims in spec
    spec_str = json.dumps(gate_spec, allow_nan=False)
    for bad_term in ["mechanism_proven", "active_inference_validated", "prediction_error_proven"]:
        assert bad_term not in spec_str


def test_objective_name_field():
    """Test that Objective name field is preserved and appears in evaluation report."""
    model, signals = _model_and_signals()
    obj = jtfne.Objective(name="my_experiment_objective")
    report = model.evaluate(signals, obj)
    assert report["objective_name"] == "my_experiment_objective"
