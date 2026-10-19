"""LIF hard refractory window ``t_ref_ms`` (Brunel 2000 ``taurp``).

After a spike at ``t_last``, while ``t_ms - t_last < round(t_ref_ms/dt_ms) *
dt_ms`` the cell's ``V`` is held at ``v_reset_mV`` and it cannot spike;
``t_ref_ms = 0.0`` (default) keeps ``V`` and spikes bit for bit.
"""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitter_registry import check_emitter_params
from jaxfne.emitters_lif import (
    LIF_CELL_TYPE_DEFAULTS,
    _PARAM_FIELDS,
    _check_t_ref_ms,
    lif_step,
    make_lif_params,
)


def _one_cell(t_ref_ms, **fields):
    base = dict(
        tau_m_ms=20.0, v_thresh_mV=20.0, v_reset_mV=10.0, v_min_mV=-1000.0,
        eta_rp_mV=0.0, tau_rp_ms=5.0, tau_sra_ms=100.0, delta_sra_mV=0.0,
        t_ref_ms=t_ref_ms,
    )
    base.update(fields)
    return make_lif_params(["simple"], {"simple": base})


def _run(params, current, dt=0.5, n_steps=6):
    v = jnp.zeros(1)
    a = jnp.zeros(1)
    tl = jnp.full(1, -1.0e30)
    spikes, trace = [], []
    for t in range(n_steps):
        (v, a, tl), spk = lif_step(
            (v, a, tl), jnp.asarray([current]), jnp.asarray([0]),
            float(t * dt), params, dt)
        spikes.append(bool(spk[0]))
        trace.append(float(v[0]))
    return spikes, trace


def test_t_ref_ms_is_a_lif_param_with_zero_default():
    assert "t_ref_ms" in _PARAM_FIELDS, "hard refractory must be a LIF field"
    for ctype, defaults in LIF_CELL_TYPE_DEFAULTS.items():
        assert defaults["t_ref_ms"] == 0.0, \
            f"{ctype} default must keep today's behaviour"


def test_default_column_is_zero():
    params = make_lif_params(["simple"])
    assert params.labels == ("simple",), "fixture must build one LIF column"
    assert np.all(np.asarray(params.t_ref_ms) == 0.0), \
        "undeclared t_ref_ms must read 0.0"


def test_check_accepts_nonnegative_finite():
    assert _check_t_ref_ms(0.0, label="t") == 0.0
    assert _check_t_ref_ms(2, label="t") == 2.0
    assert _check_t_ref_ms(2.0, label="t") == 2.0


@pytest.mark.parametrize("bad", [True, False, -1.0, -0.5, math.inf, float("nan"),
                                  "2.0", None, (2.0,)])
def test_check_refuses_bad_window(bad):
    with pytest.raises(ValueError, match="t_ref_ms"):
        _check_t_ref_ms(bad, label="t")


def test_override_sets_the_column():
    params = make_lif_params(["simple"], {"simple": {"t_ref_ms": 2.0}})
    assert np.all(np.asarray(params.t_ref_ms) == 2.0), \
        "declared window must reach the realized column"


def test_override_refuses_negative():
    with pytest.raises(ValueError, match="t_ref_ms"):
        make_lif_params(["simple"], {"simple": {"t_ref_ms": -2.0}})


def test_non_lif_family_refuses_t_ref_ms():
    for family in ("adex", "izhikevich", "glif"):
        with pytest.raises(ValueError, match="does not consume"):
            check_emitter_params(family, {"t_ref_ms": 2.0})


def test_refractory_suppresses_refire_and_holds_reset():
    spikes, trace = _run(_one_cell(2.0), current=25.0)
    assert spikes[0] is True, "strong drive must spike on the first step"
    assert spikes[1:4] == [False, False, False], \
        "no re-fire within t_ref_ms=2 ms at dt=0.5"
    assert trace[1:4] == pytest.approx([10.0, 10.0, 10.0]), \
        "V must be held at reset while refractory"


def test_refractory_releases_after_the_window():
    spikes, _ = _run(_one_cell(2.0), current=25.0)
    assert spikes[4] is True, "cell must fire again once the window lapses"


def test_zero_window_refires_immediately():
    spikes, _ = _run(_one_cell(0.0), current=25.0)
    assert spikes[0] is True, "fixture must spike on the first step"
    assert spikes[1] is True, "t_ref_ms=0.0 must not suppress any re-fire"


def test_window_rounds_to_whole_steps():
    # round(0.9 / 0.5) = round(1.8) = 2 steps: suppressed at 0.5 ms, free at 1 ms.
    spikes, _ = _run(_one_cell(0.9), current=25.0)
    assert spikes[:3] == [True, False, True], \
        "window must round to whole steps"
    # round(0.6 / 0.5) = round(1.2) = 1 step: no whole step inside the window.
    spikes, _ = _run(_one_cell(0.6), current=25.0)
    assert spikes[:2] == [True, True], \
        "a sub-step window must suppress nothing"


def test_window_holds_at_large_t_in_float32():
    """Float32 rounding of t - t_last late in a run must not move the window edge."""
    params = _one_cell(2.0)
    # dt = 0.3: k*dt is inexact in float32, so t - t_last straddles the edge
    # (dt = 0.1 or 0.5 happen to round exactly and cannot show the defect).
    dt, bad = 0.3, []
    for i0 in range(100_000, 100_300):  # t near 30 s, formed as the scan forms it
        v, a, tl = jnp.zeros(1), jnp.zeros(1), jnp.full(1, -1.0e30, dtype=jnp.float32)
        spikes = []
        for k in range(9):
            t = jnp.asarray(i0 + k, dtype=jnp.int32) * float(dt)
            (v, a, tl), spk = lif_step((v, a, tl), jnp.asarray([25.0]), jnp.asarray([0]),
                                       t, params, dt)
            spikes.append(bool(spk[0]))
        if spikes[:8] != [True] + [False] * 6 + [True]:  # round(2.0 / 0.3) = 7 steps
            bad.append(i0)
    assert not bad, f"window edge moved at {len(bad)} offsets, first {bad[:3]}"


def _tensor_model(emitter_params, drive=25.0):
    from jaxfne import neuronal_tensor as nt

    tensor = nt.NeuronalTensor(
        name="t-ref",
        areas=[nt.Area(name="net", layers=[nt.Layer(
            name="L0", n_neurons=1,
            neuron_types=[nt.NeuronType.make("E", fraction=1.0, emitter="lif",
                                             emitter_params=dict(emitter_params),
                                             drive=drive)])])],
    )
    return jtfne.construct(
        tensor, nt.RuntimeConfiguration(seed=0, duration_ms=20.0, dt_ms=0.5,
                                        emitter="lif"))


def test_constructed_column_matches_declaration():
    model = _tensor_model({"t_ref_ms": 2.0})
    assert np.all(np.asarray(model.params["emitter"].t_ref_ms) == 2.0), \
        "declared window must reach the realized column"


def test_constructed_default_column_is_zero():
    model = _tensor_model({})
    assert np.all(np.asarray(model.params["emitter"].t_ref_ms) == 0.0), \
        "undeclared window must realize 0.0"


def test_constructed_default_spikes_back_to_back():
    sim = jtfne.Simulation(duration_ms=20.0, dt_ms=0.5, seed=0,
                           runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    spikes = np.asarray(_tensor_model({}, drive=25.0).simulate(sim).spikes).reshape(-1)
    idx = np.where(spikes)[0]
    assert len(idx) > 1, "fixture must spike repeatedly"
    assert int(np.diff(idx).min()) == 1, "t_ref_ms=0.0 must allow back-to-back spikes"


def test_constructed_window_separates_spikes():
    sim = jtfne.Simulation(duration_ms=20.0, dt_ms=0.5, seed=0,
                           runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    spikes = np.asarray(
        _tensor_model({"t_ref_ms": 2.0}, drive=25.0).simulate(sim).spikes).reshape(-1)
    idx = np.where(spikes)[0]
    assert len(idx) > 1, "fixture must spike repeatedly"
    assert int(np.diff(idx).min()) >= 4, "re-fire must respect t_ref_ms=2 ms"
