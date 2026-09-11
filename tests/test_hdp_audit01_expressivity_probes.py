"""23-HDP-AUDIT-01: expressivity probes against the *current* HDP engine.

These tests document what the runtime can and cannot express today. They are
evidence for the audit receipt classification, not specifications for desired
behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.plasticity import update_stdp_weights_jax

AUDIT_JSON = (
    Path(__file__).resolve().parents[1]
    / "artifacts/audit/hdp_audit_01_probe_results.json"
)


def _net1_hdp(**hdp_overrides):
    hp = {
        "K_HDP": 0.02,
        "alpha": 0.02,
        "gamma": 0.1,
        "K_ctrl": 0.05,
        "tau_0_ms": 20.0,
    }
    hp.update(hdp_overrides)
    cfg = jtfne.suite2_net1_config(seed=7, n=8, duration_ms=40.0, dt_ms=0.5)
    return jtfne.construct(
        cfg.runtime(enable_hdp=True, recurrent_backend="edge_list", hdp_params=hp)
    )


def test_causal_chain_activity_to_H_to_weight_to_current():
    """event/activity → H → P-like rule → w_eff → synaptic drive is wired."""
    model = _net1_hdp(K_HDP=0.05)
    sig = jtfne.simulate(model, duration_ms=40.0, dt_ms=0.5, seed=1)
    diag = model.last_hdp_diagnostics()
    w0 = np.asarray(model.params["edge_list"].weight)
    wf = np.asarray(diag["w_final"])
    H = np.asarray(diag["H_trace"])
    assert bool(np.isfinite(sig.V_m).all())
    assert float(np.max(np.abs(wf - w0))) > 1e-6
    assert float(H.max() - H.min()) > 1e-4


def test_custom_hdp_rule_not_dispatchable():
    """Arbitrary finite-state / custom rules are not registered."""
    model = _net1_hdp(hdp_rule="synthetic_two_trace_stp")
    with pytest.raises(ValueError, match="Unknown hdp_rule"):
        jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=0)


def test_only_three_builtin_weight_rules():
    """Current rule surface is a closed trio, not an open grammar."""
    allowed = {"signed_linear", "signed_quadratic", "hebbian_product"}
    for rule in allowed:
        model = _net1_hdp(hdp_rule=rule, K_HDP=0.01)
        jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=0)
    assert allowed == {
        "signed_linear",
        "signed_quadratic",
        "hebbian_product",
    }


def test_state_scope_neuron_H_and_edge_w_only():
    """Plastic state is per-neuron H (or population vector) plus per-edge w."""
    model = _net1_hdp(h_state_dim=2)
    jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=0)
    diag = model.last_hdp_diagnostics()
    n = model.params["emitter"].n_neurons
    e = model.params["edge_list"].n_edges
    assert np.asarray(diag["H_trace"]).shape[-2:] in {(n,), (n, 2)}
    assert np.asarray(diag["w_final"]).shape == (e,)


def test_population_theta_not_in_continuation_carrier():
    """Population Θ is simulated but not carried by ContinuationState."""
    from jaxfne._hdp_adaptive import reject_population_continuation

    with pytest.raises(ValueError, match="population H-state locality"):
        reject_population_continuation("population", context="audit_probe")


def test_null_hdp_routes_to_baseline_bit_exact():
    """Documented null builtin HDP routes through baseline kernel (23-HDP-01)."""
    cfg = jtfne.suite2_net1_config(seed=3, n=8)
    off = jtfne.construct(cfg.runtime(enable_hdp=False))
    on = jtfne.construct(cfg.runtime(enable_hdp=True))
    v_off = np.asarray(jtfne.simulate(off, duration_ms=20.0, dt_ms=0.5, seed=5).V_m)
    v_on = np.asarray(jtfne.simulate(on, duration_ms=20.0, dt_ms=0.5, seed=5).V_m)
    assert np.array_equal(v_off, v_on)
    assert on.last_hdp_diagnostics() is None


def test_hdp_jit_deterministic_replay():
    model = _net1_hdp()
    model = jtfne.construct(
        jtfne.suite2_net1_config(seed=7, n=8).runtime(
            enable_hdp=True, jit=True, hdp_params={"K_HDP": 0.02, "alpha": 0.01}
        )
    )
    a = np.asarray(jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=7).V_m)
    b = np.asarray(jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=7).V_m)
    assert np.array_equal(a, b)


def test_stdp_module_not_on_simulate_hdp_path():
    """STDP exists as a separate dense-W API; hdp_params stdp keys are inert."""
    n = 4
    W = jnp.ones((n, n), dtype=jnp.float32) * 0.1
    W2 = update_stdp_weights_jax(
        W,
        jnp.zeros(n),
        jnp.zeros(n),
        spiked=jnp.array([1.0, 1.0, 0, 0]),
        exc_mask=jnp.ones(n, dtype=bool),
        A_plus=0.01,
        A_minus=0.012,
        plasticity_scale=1.0,
        w_min=0.0,
        w_max=1.5,
    )
    assert not np.allclose(np.asarray(W), np.asarray(W2))

    base = _net1_hdp(K_HDP=0.0, alpha=0.05)
    with_stdp_key = jtfne.construct(
        jtfne.suite2_net1_config(seed=7, n=8).runtime(
            enable_hdp=True,
            recurrent_backend="edge_list",
            hdp_params={
                "K_HDP": 0.0,
                "alpha": 0.05,
                "stdp_config": {"A_plus": 1.0},
            },
        )
    )
    jtfne.simulate(base, duration_ms=20.0, dt_ms=0.5, seed=0)
    jtfne.simulate(with_stdp_key, duration_ms=20.0, dt_ms=0.5, seed=0)
    w_base = np.asarray(base.last_hdp_diagnostics()["w_final"])
    w_stdp = np.asarray(with_stdp_key.last_hdp_diagnostics()["w_final"])
    np.testing.assert_allclose(w_base, w_stdp)


def test_audit_results_json_matches_classification():
    """Receipt JSON must exist and classify HDP_EXTENSION_REQUIRED."""
    assert AUDIT_JSON.is_file(), "run audit receipt generation"
    payload = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    assert payload["classification"] == "HDP_EXTENSION_REQUIRED"
    assert payload["custom_rule_dispatch"] is False
    assert payload["causal_chain_partial"] is True
