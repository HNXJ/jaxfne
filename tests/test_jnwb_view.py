"""Signals -> jnwb seam (todo 0d): units, sampling rate, spike times, proxies."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne import pynwb_compat
from jaxfne.core import RuntimeConfig, Signals, Simulation, TrialBatch, TrialBatchResult, TrialResult, TrialSpec
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
    assert np.asarray(sig.time_ms).dtype.itemsize == 4, "fixture must build a float32 axis (bfloat16 is storage only)"
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
    "non_binary_levels": (lambda: _signals(10, 0.1, spikes=jnp.full((10, 2), 2.0),
                                           meta={"dt_ms": 0.1, "spike_threshold": 0.0}),
                          "0 or 1"),
}


def test_threshold_levels_become_upward_crossings():
    from jaxfne.bridges import jaxley_trace_to_signals

    # Two 1 ms action potentials at dt 0.025 (40 samples above 0 mV each) and a
    # trace that starts above threshold, which has no observed crossing.
    v = np.full((400, 2), -65.0, np.float32)
    v[40:80, 0] = 20.0
    v[200:240, 0] = 20.0
    v[:40, 1] = 20.0
    sig = jaxley_trace_to_signals(v, dt_ms=0.025)
    assert np.asarray(sig.spikes)[:, 0].sum() == 80, "fixture must build threshold levels"
    view = to_jnwb(sig)
    np.testing.assert_array_equal(view.spike_times_s[0], [40 * 0.025e-3, 200 * 0.025e-3])
    assert view.spike_times_s[1].size == 0
    assert view.metadata["jnwb_view_spikes"] == "upward_crossings_of_threshold_levels"


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


def test_welch_psd_hands_float64_to_jnwb_and_keeps_the_rest(monkeypatch):
    from scipy import signal
    from jaxfne.vis import core

    calls = []

    def spy(arr, fs, *, axis, nperseg):
        calls.append((arr.dtype, nperseg))
        return signal.welch(arr, fs=fs, axis=axis, nperseg=nperseg)

    monkeypatch.setattr(core, "_jnwb_compute_psd", lambda: spy)
    x = np.random.default_rng(0).standard_normal((600, 3))
    core.welch_psd(x, 1000.0)
    assert calls == [(np.float64, 256)]
    for kept in (x.astype(np.float32), np.where(np.arange(600)[:, None] == 5, np.nan, x), x[:1]):
        core.welch_psd(kept, 1000.0)
    assert len(calls) == 1, "float32, non-finite and 1-sample input must stay on the local path"
    assert core.welch_psd(x.astype(np.float32), 1000.0)[1].dtype == np.float32


def test_welch_psd_through_jnwb_is_bit_identical():
    from scipy import signal
    from jaxfne.vis import core

    if core._jnwb_compute_psd() is None:
        pytest.skip("installed jnwb has no compute_psd(axis=, nperseg=)")
    rng = np.random.default_rng(1)
    for x in (rng.standard_normal((3000, 4)), rng.standard_normal(3000)):  # plotly passes 1-D
        freqs, pxx = core.welch_psd(x, 2000.0, nperseg=300)
        ref_f, ref_p = signal.welch(x, fs=2000.0, axis=0, nperseg=300)
        np.testing.assert_array_equal(freqs, ref_f)
        np.testing.assert_array_equal(pxx, ref_p)


def test_the_jnwb_probe_falls_back_on_any_failure(monkeypatch):
    import sys
    import types
    from jaxfne.vis import core

    for fake in (types.SimpleNamespace(compute_psd=lambda x, fs, *, nperseg=None: None),  # no axis
                 types.SimpleNamespace()):                                                # no function
        monkeypatch.setitem(sys.modules, "jnwb", fake)
        core._jnwb_compute_psd.cache_clear()
        assert core._jnwb_compute_psd() is None
    core._jnwb_compute_psd.cache_clear()


def _delayed_two_site_rate_model(
    direction: str,
    *,
    duration_ms: float = 1200.0,
    dt_ms: float = 1.0,
    delay_ms: float = 6.0,
    weight: float = 0.6,
):
    rate_params = {
        "tau_ms": 5.0,
        "transfer": "threshold_linear",
        "gain": 1.0,
        "sigma": 0.20,
        "r0": 1.0,
    }
    areas = [
        nt.Area(
            name=name,
            layers=[
                nt.Layer(
                    name="L1",
                    n_neurons=1,
                    neuron_types=[
                        nt.NeuronType.make(
                            "E",
                            fraction=1.0,
                            emitter="rate",
                            emitter_params=rate_params,
                            drive=0.0,
                        )
                    ],
                )
            ],
            inter_connections=(),
        )
        for name in ("SiteA", "SiteB")
    ]
    if direction == "uncoupled":
        conns = ()
    else:
        src, dst = ("SiteA", "SiteB") if direction == "forward" else ("SiteB", "SiteA")
        conns = (
            nt.AreaConnection(
                source_area=src,
                source_layer="L1",
                source_neuron_type="E",
                target_area=dst,
                target_layer="L1",
                target_neuron_type="E",
                mechanism="AMPA",
                static=nt.StaticParams(dT_ms=2.0),
                plastic=nt.PlasticParams(w_mech=weight * math.sqrt(2.0)),
                delay_ms=delay_ms,
            ),
        )
    tensor = nt.NeuronalTensor(
        name=f"jnwb-rate-{direction}",
        areas=areas,
        area_connections=conns,
    )
    runtime = nt.RuntimeConfiguration(
        seed=0,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        dtype="float32",
        emitter="rate",
        device="cpu",
        jit=False,
    )
    return jtfne.construct(tensor, runtime)


def test_delayed_two_site_rate_edge_table_and_trials_seam():
    fwd = _delayed_two_site_rate_model("forward")
    fwd_edges = fwd.edge_table()
    assert len(fwd_edges) == 1
    assert int(fwd_edges[0]["pre"]) == 0 and int(fwd_edges[0]["post"]) == 1
    assert int(fwd_edges[0]["delay_steps"]) == 6
    assert float(fwd_edges[0]["weight"]) == pytest.approx(0.6, rel=1e-6, abs=1e-6)

    rev = _delayed_two_site_rate_model("reverse")
    rev_edges = rev.edge_table()
    assert len(rev_edges) == 1
    assert int(rev_edges[0]["pre"]) == 1 and int(rev_edges[0]["post"]) == 0
    assert int(rev_edges[0]["delay_steps"]) == 6

    unc = _delayed_two_site_rate_model("uncoupled")
    assert len(unc.edge_table()) == 0

    sim = Simulation(
        duration_ms=1200.0,
        dt_ms=1.0,
        seed=0,
        runtime=RuntimeConfig(
            dtype="float32",
            recurrent_backend="edge_list",
            enable_hdp=False,
            jit=False,
        ),
    )
    batch_a = TrialBatch(batch_id="a", trials=(TrialSpec("t0", seed=0), TrialSpec("t1", seed=1)))
    batch_b = TrialBatch(batch_id="b", trials=(TrialSpec("t0", seed=0), TrialSpec("t1", seed=1)))
    trials_a = to_jnwb_trials(fwd.run_trials(batch_a, sim))
    trials_b = to_jnwb_trials(fwd.run_trials(batch_b, sim))

    assert trials_a.fs_hz == pytest.approx(1000.0, abs=1e-12)
    assert trials_a.dt_ms == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(np.diff(trials_a.views[0].time_s), 0.001, rtol=1e-12)
    assert trials_a.views[0].V_m.flags.writeable is False
    assert int(trials_a.views[0].metadata["n_floored_to_one_step"]) == 0
    assert np.array_equal(trials_a.stack("V_m"), trials_b.stack("V_m"))
    assert not np.array_equal(trials_a.views[0].V_m, trials_a.views[1].V_m)
    assert float(np.min(trials_a.stack("V_m"))) > 0.0

    unc_trials = to_jnwb_trials(
        unc.run_trials(
            TrialBatch(
                batch_id="unc",
                trials=tuple(TrialSpec(f"u{i}", seed=200 + i) for i in range(8)),
            ),
            sim,
        )
    )
    unc_rates = np.asarray(unc_trials.stack("V_m")[:, 200:, :], dtype=np.float64)
    assert float(np.min(unc_rates)) > 0.0
    theo_var = (0.20 ** 2) / (2.0 - 1.0 / 5.0)
    assert float(np.var(unc_rates)) == pytest.approx(theo_var, rel=0.25)


def test_delayed_two_site_rate_nwb_roundtrip(tmp_path):
    pynwb = pytest.importorskip("pynwb")
    jnwb = pytest.importorskip("jnwb")

    model = _delayed_two_site_rate_model("forward", duration_ms=200.0)
    sig = model.simulate(
        Simulation(
            duration_ms=200.0,
            dt_ms=1.0,
            seed=0,
            runtime=RuntimeConfig(
                dtype="float32",
                recurrent_backend="edge_list",
                enable_hdp=False,
                jit=False,
            ),
        )
    )
    view = to_jnwb(sig)
    nwb_path = pynwb_compat.write_nwb(
        view,
        tmp_path / "delayed_rate.nwb",
        identifier="jaxfne-delayed-rate-smoke",
    )
    with pynwb.NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        vm_series = nwb.acquisition["V_m"]
        assert not isinstance(vm_series, pynwb.ecephys.ElectricalSeries)
        assert vm_series.unit == (view.representation or "unspecified")
        assert "volt" not in vm_series.unit.lower()
        assert vm_series.rate == pytest.approx(1000.0, abs=1e-12)
        assert vm_series.starting_time == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_array_equal(vm_series.data[:], view.V_m)

    loaded = pynwb_compat.read_nwb(nwb_path)
    assert loaded.identifier == "jaxfne-delayed-rate-smoke"
    assert hasattr(jnwb, "read_nwb")

