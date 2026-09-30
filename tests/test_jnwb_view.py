"""Signals -> jnwb seam (todo 0d): units, sampling rate, spike times, proxies."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.core import Signals, Simulation, TrialBatch, TrialBatchResult, TrialResult, TrialSpec
from jaxfne.jnwb_view import to_jnwb, to_jnwb_trials


@pytest.fixture(scope="module")
def model():
    return jtfne.construct(jtfne.suite2_four_celltype_config(seed=0))


@pytest.fixture(scope="module")
def sim(model):
    return jtfne.simulate(model, duration_ms=100, dt_ms=0.1, seed=0)


def _signals(n, dt_ms, n_units=2, meta=None, V_m=None, spikes=None, t0_ms=0.0, time_ms=None):
    if time_ms is None:
        time_ms = jnp.float32(t0_ms) + jnp.arange(n, dtype=jnp.float32) * jnp.float32(dt_ms)
    V = jnp.zeros((n, n_units), jnp.float32) if V_m is None else V_m
    S = jnp.zeros((n, n_units), jnp.float32) if spikes is None else jnp.asarray(spikes)
    return Signals(time_ms=time_ms, V_m=V, spikes=S, sources=None, field=None,
                   metadata={"dt_ms": dt_ms} if meta is None else meta)


def _dense(view):
    dense = np.zeros((view.n_steps, len(view.spike_times_s)))
    for u, t in enumerate(view.spike_times_s):
        np.add.at(dense[:, u], np.rint((t - view.t0_s) * view.fs_hz).astype(int), 1)
    return dense


def test_round_trip_on_a_simulation(sim):
    spikes = np.asarray(sim.spikes)
    assert spikes.sum() > 0, "fixture must spike, or the round trip is vacuous"
    view = to_jnwb(sim)
    assert view.fs_hz == pytest.approx(1e4)
    np.testing.assert_array_equal(_dense(view), spikes)
    np.testing.assert_array_equal(view.V_m, np.asarray(sim.V_m))
    np.testing.assert_array_equal(view.lfp_proxy, np.asarray(sim.field.lfp_proxy))
    assert view.field_level == "RELATIVE_PROXY"
    assert view.representation == "relative"


def test_spike_times_carry_t0():
    spikes = np.zeros((20, 1), np.float32)
    spikes[10, 0] = 1
    view = to_jnwb(_signals(20, 0.5, n_units=1, spikes=spikes, t0_ms=500.0))
    assert view.t0_s == 0.5
    assert view.spike_times_s[0][0] == 0.5 + 10 * (0.5 / 1000.0)


def test_spike_times_do_not_inherit_float32_time_rounding():
    n, dt = 200_001, 0.05  # 10 s: a float32 time_ms ulp is ~1e-3 ms here
    t32 = (jnp.arange(n, dtype=jnp.float32) * jnp.float32(dt))
    k = int(np.flatnonzero(np.asarray(t32, np.float64) != np.arange(n) * dt)[-1])
    spikes = np.zeros((n, 2), np.float32)
    spikes[k, 0] = 1
    (t,) = to_jnwb(_signals(n, dt, spikes=spikes)).spike_times_s[0]
    assert t == k * (dt / 1000.0)
    assert float(t32[k]) / 1000.0 != t


def test_a_count_of_two_gives_two_spike_times():
    spikes = np.zeros((10, 2), np.float32)
    spikes[3, 1] = 2
    view = to_jnwb(_signals(10, 0.1, spikes=spikes))
    np.testing.assert_array_equal(view.spike_times_s[1], [3 * (0.1 / 1000.0)] * 2)


def test_bfloat16_simulation_is_accepted_at_its_declared_dt(model):
    sig = jtfne.simulate(model, duration_ms=100, dt_ms=0.1, seed=0, dtype="bfloat16")
    assert np.asarray(sig.time_ms).dtype.itemsize == 2, "fixture must build a bfloat16 axis"
    view = to_jnwb(sig)
    assert view.dt_ms == 0.1
    assert view.V_m.dtype == np.float32
    np.testing.assert_array_equal(_dense(view), np.asarray(sig.spikes, np.float64))


def test_view_cannot_change_the_source():
    meta = {"dt_ms": 0.1, "neuron_metadata": {"cell_type": ["E", "E"]}}
    V = np.zeros((10, 2), np.float32)
    sig = Signals(jnp.arange(10, dtype=jnp.float32) * 0.1, V, np.zeros((10, 2)), None, None, meta)
    view = to_jnwb(sig)
    view.neuron_metadata["cell_type"][0] = "I"
    assert meta["neuron_metadata"]["cell_type"][0] == "E"
    with pytest.raises(ValueError, match="read-only"):
        view.V_m[0, 0] = 99.0


_REFUSALS = {
    "dt_disagrees": (lambda: _signals(100, 0.1, meta={"dt_ms": 0.2}), "not uniform"),
    "non_uniform": (lambda: _signals(4, 0.1, time_ms=jnp.array([0.0, 0.05, 0.2, 0.3])),
                    "not uniform"),
    "decreasing": (lambda: _signals(3, 0.1, time_ms=jnp.array([0.2, 0.1, 0.0])), "increasing"),
    "trial_axis": (lambda: _signals(100, 0.1, V_m=jnp.zeros((3, 100, 2))), "n_steps"),
    "one_sample_no_dt": (lambda: _signals(1, 0.1, meta={}), "unknown"),
    "rate": (lambda: _signals(10, 0.1, spikes=jnp.full((10, 2), 0.25)), "integer counts"),
    "negative": (lambda: _signals(10, 0.1, spikes=-jnp.ones((10, 2))), "integer counts"),
    "nan": (lambda: _signals(10, 0.1, spikes=jnp.full((10, 2), jnp.nan)), "integer counts"),
    "threshold_levels": (lambda: _signals(10, 0.1, meta={"dt_ms": 0.1, "spike_threshold": 0.0}),
                         "threshold"),
    "hh_bridge": (lambda: _signals(10, 0.1, meta={"dt_ms": 0.1,
                                                   "bridge": "jaxley_hh_laminar_field"}),
                  "threshold"),
}


@pytest.mark.parametrize("case", sorted(_REFUSALS))
def test_refusals(case):
    build, match = _REFUSALS[case]
    with pytest.raises(ValueError, match=match):
        to_jnwb(build())


def test_trial_batch_stacks_trials_on_axis_0(model):
    batch = TrialBatch(trials=(TrialSpec("a", seed=0), TrialSpec("b", seed=1)))
    res = model.run_trials(batch, Simulation(duration_ms=20, dt_ms=0.1))
    trials = to_jnwb_trials(res)
    assert trials.trial_ids == ("a", "b")
    V = trials.stack("V_m")
    assert V.shape == (2, 200, np.asarray(res.results[0].signals.V_m).shape[1])
    np.testing.assert_array_equal(V[1], np.asarray(res.results[1].signals.V_m))
    assert not np.array_equal(V[0], V[1]), "seeds must differ, or trial order is untested"
    assert trials.stack("lfp_proxy").shape[:2] == (2, 200)


def test_trial_batch_refusals():
    ok = TrialResult("a", signals=_signals(10, 0.1))
    failed = TrialResult("b", signals=None, success=False, error_message="boom")
    with pytest.raises(ValueError, match="without signals"):
        to_jnwb_trials(TrialBatchResult("x", (ok, failed)))
    other_dt = TrialResult("c", signals=_signals(10, 0.2))
    with pytest.raises(ValueError, match="trial 'c' has dt_ms=0.2"):
        to_jnwb_trials(TrialBatchResult("x", (ok, other_dt)))
    with pytest.raises(ValueError, match="not recorded"):
        to_jnwb_trials(TrialBatchResult("x", (ok,))).stack("lfp_proxy")


def test_jnwb_consumes_the_view(sim):
    jnwb = pytest.importorskip("jnwb")
    view = to_jnwb(sim)
    freqs, psd = jnwb.compute_psd(view.lfp_proxy, view.fs_hz)
    # jnwb segments min(n, fs) samples: 1000 at 10 kHz gives a 10 Hz grid;
    # an fs off by 1000 (kHz for Hz) would give 1 Hz.
    assert freqs[1] - freqs[0] == pytest.approx(10.0)
    assert freqs[-1] == pytest.approx(5000.0)
    assert psd.shape[1] == view.lfp_proxy.shape[1]
    unit = int(np.argmax([t.size for t in view.spike_times_s]))
    centers, rate, _ = jnwb.raster_psth(view.spike_times_s[unit], np.array([view.t0_s]),
                                        (0.0, 100.0), bin_ms=10.0)
    assert np.sum(rate) * 10.0 / 1000.0 == pytest.approx(view.spike_times_s[unit].size)
