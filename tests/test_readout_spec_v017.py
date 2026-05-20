"""Tests for v0.0.17 ReadoutSpec/ReadoutResult — declarative feature extraction.

A. readout_spec() factory returns ReadoutSpec with correct fields
B. ReadoutSpec.to_dict() is JSON-safe
C. ReadoutResult.to_dict() is JSON-safe, value is float or None
D. Model.compute_readout() returns list of ReadoutResult in spec order
E. spike_rate_hz metric computes non-negative value
F. spike_count and mean_V_m metrics compute correctly
G. csd_abs_mean and lfp_abs_mean metrics compute with field output
H. no_field status when field absent and field metric requested
I. unknown_metric status for unrecognized metric token
J. time_window_ms slice respected — windowed result vs full result may differ
K. n_contacts_slice restricts field depth dimension
L. claim guards: physical_amplitude_claim_allowed False, claim_level scaffold
M. _KNOWN_READOUT_METRICS exported and contains expected entries
"""

import json

import pytest
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne.core import _KNOWN_READOUT_METRICS


def _make_model(n: int = 8):
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    return jtfne.construct(cfg)


def _make_signals(model, duration_ms=20.0, seed=0):
    sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=0.5, seed=seed)
    return model.simulate(sim)


def test_a_readout_spec_factory():
    spec = jtfne.readout_spec("rate", "spike_rate_hz")
    assert isinstance(spec, jtfne.ReadoutSpec)
    assert spec.name == "rate"
    assert spec.metric == "spike_rate_hz"
    assert spec.time_window_ms is None
    assert spec.n_contacts_slice is None
    assert isinstance(spec.metadata, dict)


def test_b_readout_spec_to_dict_json_safe():
    spec = jtfne.readout_spec(
        "windowed_rate", "spike_rate_hz",
        time_window_ms=(5.0, 15.0),
        metadata={"condition": "A"},
    )
    d = spec.to_dict()
    assert d["name"] == "windowed_rate"
    assert d["metric"] == "spike_rate_hz"
    assert d["time_window_ms"] == [5.0, 15.0]
    assert d["metadata"]["condition"] == "A"
    json.dumps(d, allow_nan=False)


def test_c_readout_result_to_dict_json_safe():
    result = jtfne.ReadoutResult(
        spec_name="rate", metric="spike_rate_hz", value=12.5,
    )
    d = result.to_dict()
    assert d["spec_name"] == "rate"
    assert d["value"] == 12.5
    assert d["status"] == "computed"
    assert d["physical_amplitude_claim_allowed"] is False
    json.dumps(d, allow_nan=False)


def test_d_compute_readout_returns_list_in_order():
    model = _make_model()
    signals = _make_signals(model)
    specs = [
        jtfne.readout_spec("r1", "spike_rate_hz"),
        jtfne.readout_spec("r2", "mean_V_m"),
        jtfne.readout_spec("r3", "spike_count"),
    ]
    results = model.compute_readout(signals, specs)
    assert len(results) == 3
    assert results[0].spec_name == "r1"
    assert results[1].spec_name == "r2"
    assert results[2].spec_name == "r3"
    for r in results:
        assert isinstance(r, jtfne.ReadoutResult)
        assert r.status == "computed"


def test_e_spike_rate_hz_nonnegative():
    model = _make_model()
    signals = _make_signals(model, duration_ms=20.0, seed=0)
    specs = [jtfne.readout_spec("rate", "spike_rate_hz")]
    results = model.compute_readout(signals, specs)
    assert results[0].value is not None
    assert results[0].value >= 0.0


def test_f_spike_count_and_mean_vm():
    model = _make_model()
    signals = _make_signals(model)
    specs = [
        jtfne.readout_spec("sc", "spike_count"),
        jtfne.readout_spec("vm", "mean_V_m"),
    ]
    results = model.compute_readout(signals, specs)
    # spike_count must be non-negative
    assert results[0].value >= 0.0
    # mean_V_m should be a finite float in plausible Izhikevich range
    assert results[1].value is not None
    assert -100.0 < results[1].value < 100.0


def test_g_csd_and_lfp_abs_mean_with_field():
    model = _make_model()
    signals = _make_signals(model)
    specs = [
        jtfne.readout_spec("csd", "csd_abs_mean"),
        jtfne.readout_spec("lfp", "lfp_abs_mean"),
    ]
    results = model.compute_readout(signals, specs)
    assert results[0].status == "computed"
    assert results[0].value is not None
    assert results[0].value >= 0.0
    assert results[1].status == "computed"
    assert results[1].value is not None
    assert results[1].value >= 0.0


def test_h_no_field_status_when_field_absent():
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="p", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.5, seed=0, record_fields=False)
    signals = model.simulate(sim)
    specs = [jtfne.readout_spec("csd", "csd_abs_mean")]
    results = model.compute_readout(signals, specs)
    assert results[0].status == "no_field"
    assert results[0].value is None


def test_i_unknown_metric_status():
    model = _make_model()
    signals = _make_signals(model)
    specs = [jtfne.readout_spec("bad", "not_a_real_metric")]
    results = model.compute_readout(signals, specs)
    assert results[0].status == "unknown_metric"
    assert results[0].value is None


def test_j_time_window_slice():
    model = _make_model()
    signals = _make_signals(model, duration_ms=20.0, seed=0)
    full_spec = jtfne.readout_spec("full", "spike_count")
    early_spec = jtfne.readout_spec("early", "spike_count", time_window_ms=(0.0, 5.0))
    results = model.compute_readout(signals, [full_spec, early_spec])
    # Windowed count should be <= full count
    assert results[1].value <= results[0].value + 1e-6  # float tolerance


def test_k_n_contacts_slice():
    model = _make_model()
    signals = _make_signals(model)
    full_spec = jtfne.readout_spec("lfp_full", "lfp_abs_mean")
    slice_spec = jtfne.readout_spec(
        "lfp_top", "lfp_abs_mean", n_contacts_slice=(0, 4)
    )
    results = model.compute_readout(signals, [full_spec, slice_spec])
    assert results[0].status == "computed"
    assert results[1].status == "computed"
    # Both non-negative
    assert results[0].value >= 0.0
    assert results[1].value >= 0.0


def test_l_claim_guards():
    result = jtfne.ReadoutResult(spec_name="r", metric="spike_rate_hz", value=10.0)
    assert result.physical_amplitude_claim_allowed is False
    assert result.claim_level == "computational_scaffold"
    d = result.to_dict()
    assert d["physical_amplitude_claim_allowed"] is False
    assert d["claim_level"] == "computational_scaffold"


def test_m_known_readout_metrics_exported():
    assert isinstance(_KNOWN_READOUT_METRICS, frozenset)
    assert "spike_rate_hz" in _KNOWN_READOUT_METRICS
    assert "spike_count" in _KNOWN_READOUT_METRICS
    assert "mean_V_m" in _KNOWN_READOUT_METRICS
    assert "csd_abs_mean" in _KNOWN_READOUT_METRICS
    assert "lfp_abs_mean" in _KNOWN_READOUT_METRICS
    assert "source_abs_mean" in _KNOWN_READOUT_METRICS


def test_n_readout_result_name_compatibility_alias():
    """Test ReadoutResult.name as compatibility alias for spec_name.

    Allows public README examples to use:
        for result in results:
            print(result.name, result.metric, result.value, result.status)
    """
    result = jtfne.ReadoutResult(
        spec_name="rate", metric="spike_rate_hz", value=10.5, status="computed"
    )
    # Test alias works
    assert result.name == "rate"
    assert result.name == result.spec_name

    # Test in iteration pattern (as used in public examples)
    results = [
        jtfne.ReadoutResult(spec_name="s1", metric="m1", value=1.0),
        jtfne.ReadoutResult(spec_name="s2", metric="m2", value=2.0),
    ]
    for res in results:
        # This should not raise AttributeError
        name = res.name
        metric = res.metric
        value = res.value
        status = res.status
        assert isinstance(name, str)
        assert isinstance(metric, str)
        assert isinstance(value, float)
        assert isinstance(status, str)
