"""Regression for _simulate_homeostasis_metadata, extracted from Model.simulate()
(jaxfne/core.py) -- the homeostasis metadata sub-block was the one genuinely
branchy, self-contained unit in simulate()'s tail (conditional diag presence,
conditional w_final, three summary-stat computations); the rest of the tail
is flat key-value assembly not worth splitting further.
"""

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.core import _simulate_homeostasis_metadata


def _base_cfg(n=20):
    return (
        jtfne.Configuration()
        .runtime(duration_ms=20.0, dt_ms=0.5, seed=0)
        .column("V1", layers=["L2/3", "L4"], n=n)
        .cell_types({"E": 0.8, "PV": 0.2})
        .connectivity()
        .set_emitter(family="izhikevich")
        .probes(["spikes", "V_m"])
    )


def test_homeostasis_metadata_present_and_correctly_shaped_via_full_simulate():
    cfg = _base_cfg().homeostasis(relative_baseline=2.0)
    model = jtfne.construct(cfg)
    sig = jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=0)

    homeo = sig.metadata["homeostasis"]
    assert homeo["enabled"] is True
    assert homeo["method"] == "minimal_homeostatic_resource_adaptation_controller"
    assert homeo["claim_status"] == "computational_control_proxy_not_biological_mechanism"
    assert homeo["biological_learning_claim"] is False
    assert homeo["mechanism_claim_status"] == "not_claimed"
    assert "g_bias_summary" in homeo
    assert "r_trace_summary" in homeo
    assert set(homeo["g_bias_summary"].keys()) == {"min", "max", "mean", "shape"}


def test_homeostasis_metadata_absent_when_disabled_via_full_simulate():
    cfg = _base_cfg()  # no .homeostasis() call -> enable_homeostasis stays False
    model = jtfne.construct(cfg)
    sig = jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=0)
    assert "homeostasis" not in sig.metadata


def test_simulate_homeostasis_metadata_helper_without_diag():
    from jaxfne.core import RuntimeConfig

    runtime_cfg = RuntimeConfig(homeostasis_params={"k_gain": 0.3, "r_star": 5.0, "eta": 0.0})
    meta = _simulate_homeostasis_metadata(runtime_cfg, diag=None)

    assert meta["enabled"] is True
    assert meta["params"]["k_gain"] == 0.3
    assert meta["synaptic_plasticity_enabled"] is False
    assert "g_bias_summary" not in meta
    assert "w_final_summary" not in meta


def test_simulate_homeostasis_metadata_helper_with_diag_and_plasticity():
    from jaxfne.core import RuntimeConfig
    import jax.numpy as jnp

    runtime_cfg = RuntimeConfig(homeostasis_params={"k_gain": 0.1, "eta": 0.05})
    diag = {
        "g_bias": jnp.array([0.1, 0.2, 0.3]),
        "r_trace": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "w_final": jnp.array([0.5, 0.6]),
    }
    meta = _simulate_homeostasis_metadata(runtime_cfg, diag=diag)

    assert meta["synaptic_plasticity_enabled"] is True
    assert meta["g_bias_summary"]["shape"] == [3]
    assert meta["g_bias_summary"]["min"] == pytest.approx(0.1)
    assert meta["g_bias_summary"]["max"] == pytest.approx(0.3)
    assert meta["r_trace_summary"]["shape"] == [2, 2]
    assert meta["w_final_summary"]["shape"] == [2]
    assert meta["w_final_summary"]["mean"] == pytest.approx(0.55)


def test_simulate_homeostasis_metadata_helper_per_neuron_array_param():
    from jaxfne.core import RuntimeConfig
    import jax.numpy as jnp

    runtime_cfg = RuntimeConfig(homeostasis_params={"k_gain": jnp.array([0.1, 0.2, 0.3])})
    meta = _simulate_homeostasis_metadata(runtime_cfg, diag=None)
    assert meta["params"]["k_gain"] == "per_neuron_array"
