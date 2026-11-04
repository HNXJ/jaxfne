"""AdEx route: per-step E->E gain (edge_gain) on masked edges."""

from __future__ import annotations

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._construct_adex import adex_edge_gain_kwargs, adex_external_drive_kwargs, adex_kernel_args
from jaxfne.emitters_adex import simulate_adex, simulate_adex_hdp_registered

N = 60
DT_MS = 0.1
N_STEPS = 2000  # 200 ms
CELL_TYPES = {"E": 0.8, "PV": 0.1, "SOM": 0.1}


def _cfg():
    return (
        jtfne.Configuration()
        .network(name="adex60", n=N, cell_types=CELL_TYPES)
        .emitter(family="adex", adex_decay="exp")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .mechanisms(name="gaba", kind="receptor",
                    params={"tau_ms": 8.0, "reversal_mV": -80.0})
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=0.2, weight=0.5, mechanism="ampa", delay_ms=2.0,
                     sampling="bernoulli")
        .connections(name="e_som", source={"cell_type": "E"}, target={"cell_type": "SOM"},
                     probability=0.2, weight=1.0, mechanism="ampa", delay_ms=1.3,
                     sampling="bernoulli")
        .connections(name="som_e", source={"cell_type": "SOM"}, target={"cell_type": "E"},
                     probability=0.2, weight=1.0, mechanism="gaba", delay_ms=1.5,
                     sampling="bernoulli")
        .connectivity(edge_seed=11)
    )


def _cfg_zero(weight):
    return (
        jtfne.Configuration()
        .network(name="adex-zero01", n=2, cell_types={"E": 0.5, "PV": 0.5})
        .emitter(family="adex", adex_decay="exp")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .connections(name="e_pv0", source={"cell_type": "E"}, target={"cell_type": "PV"},
                      probability=1.0, weight=weight, mechanism="ampa", delay_ms=0.0,
                      intra_step_rounds=4, sampling="bernoulli")
        .connectivity(edge_seed=11)
    )


def _sim(**kwargs):
    return jtfne.Simulation(duration_ms=200.0, dt_ms=DT_MS, **kwargs)


def _drive(targets, seed=7, rate_hz=4.0):
    return {
        "n_sources": 560, "rate_hz": rate_hz, "weight_nS": 2.5, "mechanism": "ampa",
        "target_cell_types": list(targets), "delay_ms": 2.0, "seed": seed,
    }


def _labels(model):
    return np.asarray(model.params["emitter"].labels)


def _kw(model, drive):
    sim = _sim(poisson_drive=drive)
    kw = dict(adex_kernel_args(model, DT_MS, dtype="float32"))
    kw.update(adex_external_drive_kwargs(model, sim))
    kw.update(n_steps=N_STEPS, dt_ms=DT_MS, record_v=True)
    return kw


def _gain(pre="E", post="E", value=1.0, n=N_STEPS):
    return {"pre": pre, "post": post, "schedule": np.full(n, value)}


def test_kernel_gain_absent_args_bit_identical():
    model = jtfne.construct(_cfg())
    drive = _drive(["E", "SOM"])
    kw = _kw(model, drive)
    ref = simulate_adex(**kw)
    got = simulate_adex(**kw, edge_gain_mask=None, edge_gain_schedule=None,
                         edge0_gain_mask=None)
    assert np.array_equal(np.asarray(got["spikes"]), np.asarray(ref["spikes"]))
    assert np.array_equal(np.asarray(got["V_mV"]), np.asarray(ref["V_mV"]))
    assert np.asarray(ref["spikes"]).sum() > 0, "fixture must spike"


def test_kernel_all_ones_gain_bit_identical():
    model = jtfne.construct(_cfg())
    drive = _drive(["E", "SOM"])
    kw = _kw(model, drive)
    ref = simulate_adex(**kw)
    n_edges = int(np.asarray(kw["edge_pre"]).shape[0])
    got = simulate_adex(**kw, edge_gain_mask=np.ones(n_edges, dtype=bool),
                        edge_gain_schedule=np.ones(N_STEPS))
    assert np.array_equal(np.asarray(got["spikes"]), np.asarray(ref["spikes"]))
    assert np.array_equal(np.asarray(got["V_mV"]), np.asarray(ref["V_mV"]))
    nul = simulate_adex(**kw, edge_gain_mask=np.zeros(n_edges, dtype=bool),
                        edge_gain_schedule=np.zeros(N_STEPS))
    assert np.array_equal(np.asarray(nul["spikes"]), np.asarray(ref["spikes"]))
    assert np.asarray(ref["spikes"]).sum() > 0, "fixture must spike"


def test_kernel_zero_gain_on_masked_edges_changes_run():
    model = jtfne.construct(_cfg())
    drive = _drive(["E", "SOM"])
    kw = _kw(model, drive)
    ref = simulate_adex(**kw)
    assert np.asarray(ref["spikes"]).sum() > 0, "fixture must spike"
    n_edges = int(np.asarray(kw["edge_pre"]).shape[0])
    got = simulate_adex(**kw, edge_gain_mask=np.ones(n_edges, dtype=bool),
                        edge_gain_schedule=np.zeros(N_STEPS))
    assert not np.array_equal(np.asarray(got["spikes"]), np.asarray(ref["spikes"]))
    assert not np.array_equal(np.asarray(got["V_mV"]), np.asarray(ref["V_mV"]))


@pytest.mark.parametrize("kwargs,match", [
    ({"edge_gain_mask": np.ones(4, dtype=bool)}, "both given or both None"),
    ({"edge_gain_schedule": np.ones(N_STEPS)}, "both given or both None"),
    ({"edge0_gain_mask": np.ones(1, dtype=bool)}, "both given or both None"),
    ({"edge_gain_mask": np.ones(3, dtype=bool),
      "edge_gain_schedule": np.ones(N_STEPS)}, "shape"),
    ({"edge_gain_mask": np.ones(10, dtype=np.int8),
      "edge_gain_schedule": np.ones(N_STEPS)}, "bool array"),
    ({"edge_gain_mask": np.ones(10, dtype=bool),
      "edge_gain_schedule": np.ones(N_STEPS - 1)}, "shape"),
    ({"edge_gain_mask": np.ones(10, dtype=bool),
      "edge_gain_schedule": np.full(N_STEPS, -1.0)}, "entries >= 0"),
    ({"edge_gain_mask": np.ones(10, dtype=bool),
      "edge_gain_schedule": np.full(N_STEPS, np.nan)}, "finite"),
    ({"edge_gain_mask": np.ones(10, dtype=bool),
      "edge_gain_schedule": np.full(N_STEPS, np.inf)}, "finite"),
    ({"edge_gain_mask": np.ones(10, dtype=bool),
      "edge_gain_schedule": np.ones(N_STEPS),
      "edge0_gain_mask": np.ones(1, dtype=bool)}, "no delay-0 edges"),
])
def test_kernel_gain_refusals(kwargs, match):
    model = jtfne.construct(_cfg())
    drive = _drive(["E", "SOM"])
    kw = _kw(model, drive)
    n_edges = int(np.asarray(kw["edge_pre"]).shape[0])
    call = dict(kwargs)
    if "edge_gain_mask" in call:
        arr = np.asarray(call["edge_gain_mask"])
        if arr.shape in ((4,), (3,)):
            call["edge_gain_mask"] = np.ones(n_edges + 1, dtype=bool)
        elif arr.dtype == np.int8:
            call["edge_gain_mask"] = np.ones(n_edges, dtype=np.int8)
        elif arr.shape == (10,):
            call["edge_gain_mask"] = np.ones(n_edges, dtype=bool)
    with pytest.raises(ValueError, match=match):
        simulate_adex(**kw, **call)


def test_kernel_delay0_gain_ones_bit_identical_zero_matters():
    model = jtfne.construct(_cfg_zero(weight=4.0))
    drive = _drive(["E"], seed=7, rate_hz=8.0)
    sim = _sim(poisson_drive=drive)
    kw = dict(adex_kernel_args(model, DT_MS, dtype="float32"))
    kw.update(adex_external_drive_kwargs(model, sim))
    kw.update(n_steps=N_STEPS, dt_ms=DT_MS, record_v=True)
    ref = simulate_adex(**kw)
    assert np.asarray(ref["spikes"]).sum() > 0, "fixture must spike"
    n_delayed = int(np.asarray(kw["edge_pre"]).shape[0])
    n_zero = int(np.asarray(kw["edge0_pre"]).shape[0])
    assert n_delayed == 0 and n_zero == 1
    ones = simulate_adex(
        **kw, edge_gain_mask=np.zeros(0, dtype=bool),
        edge_gain_schedule=np.ones(N_STEPS),
        edge0_gain_mask=np.ones(n_zero, dtype=bool))
    assert np.array_equal(np.asarray(ones["spikes"]), np.asarray(ref["spikes"]))
    assert np.array_equal(np.asarray(ones["V_mV"]), np.asarray(ref["V_mV"]))
    zeroed = simulate_adex(
        **kw, edge_gain_mask=np.zeros(0, dtype=bool),
        edge_gain_schedule=np.zeros(N_STEPS),
        edge0_gain_mask=np.ones(n_zero, dtype=bool))
    assert not np.array_equal(np.asarray(zeroed["V_mV"]), np.asarray(ref["V_mV"]))


def test_model_gain_zero_schedule_changes_spikes_and_records():
    model = jtfne.construct(_cfg())
    labels = _labels(model)
    drive = _drive(["E", "SOM"])
    ref = model.simulate(_sim(poisson_drive=drive))
    assert np.asarray(ref.spikes)[:, labels == "E"].sum() > 0, "E must spike"
    got = model.simulate(_sim(poisson_drive=drive, edge_gain=_gain(value=0.0)))
    assert not np.array_equal(np.asarray(got.spikes), np.asarray(ref.spikes))
    rec = got.metadata["edge_gain"]
    assert rec["pre"] == "E" and rec["post"] == "E"
    assert rec["n_edges"] > 0
    assert got.metadata["edge_gain"]["schedule"]["n_steps"] == N_STEPS


def test_model_gain_ones_schedule_bit_identical():
    model = jtfne.construct(_cfg())
    drive = _drive(["E", "SOM"])
    ref = model.simulate(_sim(poisson_drive=drive))
    got = model.simulate(_sim(poisson_drive=drive, edge_gain=_gain(value=1.0)))
    assert np.array_equal(np.asarray(got.spikes), np.asarray(ref.spikes))
    assert np.array_equal(np.asarray(got.V_m), np.asarray(ref.V_m))


def test_model_gain_kwargs_split_delay0_mask():
    model = jtfne.construct(_cfg_zero(weight=4.0))
    sim = _sim(edge_gain={"pre": "E", "post": "PV",
                          "schedule": np.full(N_STEPS, 1.0)})
    mask, sched, mask0, rec = adex_edge_gain_kwargs(model, sim)
    assert mask.shape == (0,) and mask0.shape == (1,) and bool(mask0[0])
    assert sched.shape == (N_STEPS,)
    assert rec["n_edges"] == 1 and rec["n_delay0_edges"] == 1


@pytest.mark.parametrize("spec,match", [
    ({"pre": "E", "post": "E"}, "schedule"),
    ({"pre": "E", "schedule": np.full(N_STEPS, 1.0)}, "post"),
    ({"post": "E", "schedule": np.full(N_STEPS, 1.0)}, "pre"),
    ({"pre": "E", "post": "E", "schedule": np.full(N_STEPS, 1.0),
      "bogus": 1}, "unknown key"),
    ({"pre": "EX", "post": "E", "schedule": np.full(N_STEPS, 1.0)}, "unknown cell type"),
    ({"pre": 1, "post": "E", "schedule": np.full(N_STEPS, 1.0)}, "cell-type name"),
    ({"pre": "E", "post": "E", "schedule": np.full(N_STEPS - 1, 1.0)}, "length"),
    ({"pre": "E", "post": "E", "schedule": np.full(N_STEPS, -0.5)}, ">= 0"),
    ({"pre": "E", "post": "E", "schedule": np.full(N_STEPS, np.nan)}, "finite"),
    ({"pre": "E", "post": "E", "schedule": np.full(N_STEPS, np.inf)}, "finite"),
    ({"pre": "E", "post": "E", "schedule": np.ones(N_STEPS, dtype=bool)}, "bool"),
    ({"pre": "E", "post": "E", "schedule": "fast"}, "schedule"),
    ("E->E", "mapping"),
])
def test_bad_edge_gain_is_refused_at_construction_or_simulate(spec, match):
    model = jtfne.construct(_cfg())
    if isinstance(spec, dict) and set(spec) <= {"pre", "post", "schedule"} \
            and spec.get("pre") in ("E", "EX") and spec.get("post") == "E":
        with pytest.raises(ValueError, match=match):
            model.simulate(_sim(edge_gain=spec))
    else:
        with pytest.raises(ValueError, match=match):
            _sim(edge_gain=spec)


def test_gain_refused_off_adex_route():
    izhi = jtfne.construct(
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=12,
                 cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann",
               gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    with pytest.raises(ValueError, match="izhikevich"):
        izhi.simulate(jtfne.Simulation(duration_ms=20.0, dt_ms=DT_MS, seed=0,
                                      edge_gain=_gain(n=200)))
    hh = jtfne.construct(
        jtfne.configuration()
        .network(n=10, cell_types={"E": 0.6, "Ig": 0.2, "Il": 0.2})
        .emitter(family="hh")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0, "tau_r_ms": 0.2,
                            "slope_mV": 5.0, "v_th_mV": -20.0, "s0": 0.1})
        .connections(name="e_all", source={"cell_type": "E"}, target={},
                     probability=None, weight=2.5, mechanism="ampa",
                     sign="excitatory")
        .connectivity(edge_seed=0)
    )
    with pytest.raises(ValueError, match="hh"):
        hh.simulate(jtfne.Simulation(duration_ms=20.0, dt_ms=DT_MS, seed=0,
                                    edge_gain=_gain(n=200)))


def test_hdp_kernel_refuses_gain():
    model = jtfne.construct(_cfg())
    kw = _kw(model, _drive(["E", "SOM"]))
    n = int(np.asarray(model.params["emitter"].C_pF).shape[0])
    base = {k: kw[k] for k in (
        "n_steps", "dt_ms", "decay", "step_order", "record_v",
        "edge_pre", "edge_post", "edge_weight_nS", "edge_receptor",
        "edge_delay_steps", "receptor_reversal_mV", "receptor_tau_ms",
        "ext_targets", "ext_lambda", "ext_weight_nS", "ext_receptor",
        "ext_delay_steps", "key") if k in kw}
    n_edges = int(np.asarray(kw["edge_pre"]).shape[0])
    with pytest.raises(ValueError, match="gain"):
        simulate_adex_hdp_registered(
            model.params["emitter"], **base, hdp_rule="signed_linear",
            edge_gain_mask=np.zeros(n_edges, dtype=bool),
            edge_gain_schedule=np.ones(N_STEPS))
    assert n > 0


def _ee_mask(model, kw):
    labels = _labels(model)
    pre = np.asarray(kw["edge_pre"])
    post = np.asarray(kw["edge_post"])
    return (labels[pre] == "E") & (labels[post] == "E")


def test_kernel_zero_gain_equals_zeroed_weights():
    model = jtfne.construct(_cfg())
    kw = _kw(model, _drive(["E", "SOM"]))
    ref = simulate_adex(**kw)
    mask = _ee_mask(model, kw)
    assert mask.any(), "fixture needs E->E edges"
    pre = np.asarray(kw["edge_pre"])[mask]
    assert np.asarray(ref["spikes"])[:, np.unique(pre)].sum() > 0, \
        "masked E->E edges must carry spikes"
    got = simulate_adex(**kw, edge_gain_mask=mask,
                        edge_gain_schedule=np.zeros(N_STEPS))
    w = np.asarray(kw["edge_weight_nS"])
    w2 = w.copy()
    w2[mask] = 0.0
    exp = simulate_adex(**{**kw, "edge_weight_nS": w2})
    assert np.array_equal(np.asarray(got["spikes"]), np.asarray(exp["spikes"]))
    assert np.array_equal(np.asarray(got["V_mV"]), np.asarray(exp["V_mV"]))


def test_kernel_double_gain_equals_doubled_weights():
    model = jtfne.construct(_cfg())
    kw = _kw(model, _drive(["E", "SOM"]))
    ref = simulate_adex(**kw)
    mask = _ee_mask(model, kw)
    assert mask.any(), "fixture needs E->E edges"
    pre = np.asarray(kw["edge_pre"])[mask]
    assert np.asarray(ref["spikes"])[:, np.unique(pre)].sum() > 0, \
        "masked E->E edges must carry spikes"
    got = simulate_adex(**kw, edge_gain_mask=mask,
                        edge_gain_schedule=np.full(N_STEPS, 2.0))
    w = np.asarray(kw["edge_weight_nS"])
    w2 = w.copy()
    w2[mask] *= 2.0
    exp = simulate_adex(**{**kw, "edge_weight_nS": w2})
    assert np.array_equal(np.asarray(got["spikes"]), np.asarray(exp["spikes"]))
    np.testing.assert_allclose(np.asarray(got["V_mV"]), np.asarray(exp["V_mV"]),
                               rtol=1e-5, atol=1e-6)


def test_kernel_windowed_gain_localizes_change():
    model = jtfne.construct(_cfg())
    kw = _kw(model, _drive(["E", "SOM"], rate_hz=1.5))
    ref = simulate_adex(**kw)
    mask = _ee_mask(model, kw)
    assert mask.any(), "fixture needs E->E edges"
    e_idx = np.nonzero(_labels(model) == "E")[0]
    e_spk = np.asarray(ref["spikes"])[:, e_idx]
    assert e_spk.sum() > 0, "E must spike"
    a, b = 500, 700
    assert e_spk[a:b].sum() >= 10, "E must spike inside the window"
    sched = np.ones(N_STEPS)
    sched[a:b] = 2.0
    got = simulate_adex(**kw, edge_gain_mask=mask, edge_gain_schedule=sched)
    # Gain is indexed by arrival step ("|w| * schedule[t] where it arrives"),
    # so every step before the window is bit-identical to the no-gain run.
    assert np.array_equal(np.asarray(got["V_mV"])[:a],
                          np.asarray(ref["V_mV"])[:a])
    assert np.array_equal(np.asarray(got["spikes"])[:a],
                          np.asarray(ref["spikes"])[:a])
    assert int(np.asarray(got["spikes"])[a:b][:, e_idx].sum()) != \
        int(e_spk[a:b].sum())


def test_kernel_delay0_zero_gain_equals_zeroed_weights():
    model = jtfne.construct(_cfg_zero(weight=4.0))
    drive = _drive(["E"], seed=7, rate_hz=8.0)
    sim = _sim(poisson_drive=drive)
    kw = dict(adex_kernel_args(model, DT_MS, dtype="float32"))
    kw.update(adex_external_drive_kwargs(model, sim))
    kw.update(n_steps=N_STEPS, dt_ms=DT_MS, record_v=True)
    ref = simulate_adex(**kw)
    assert np.asarray(ref["spikes"]).sum() > 0, "fixture must spike"
    n_delayed = int(np.asarray(kw["edge_pre"]).shape[0])
    n_zero = int(np.asarray(kw["edge0_pre"]).shape[0])
    assert n_delayed == 0 and n_zero == 1
    got = simulate_adex(
        **kw, edge_gain_mask=np.zeros(0, dtype=bool),
        edge_gain_schedule=np.zeros(N_STEPS),
        edge0_gain_mask=np.ones(n_zero, dtype=bool))
    w0 = np.asarray(kw["edge0_weight_nS"])
    exp = simulate_adex(**{**kw, "edge0_weight_nS": np.zeros_like(w0)})
    assert np.array_equal(np.asarray(got["spikes"]), np.asarray(exp["spikes"]))
    assert np.array_equal(np.asarray(got["V_mV"]), np.asarray(exp["V_mV"]))


def test_model_gain_pair_without_edges_is_refused():
    model = jtfne.construct(_cfg())
    with pytest.raises(ValueError, match="no edges") as exc:
        model.simulate(_sim(edge_gain={"pre": "PV", "post": "E",
                                       "schedule": np.full(N_STEPS, 1.0)}))
    assert "'PV'" in str(exc.value) and "'E'" in str(exc.value)


def test_model_gain_record_matches_schedule():
    model = jtfne.construct(_cfg())
    sched = np.linspace(0.0, 2.0, N_STEPS)
    got = model.simulate(_sim(poisson_drive=_drive(["E", "SOM"]),
                              edge_gain={"pre": "E", "post": "E",
                                         "schedule": sched}))
    rec = got.metadata["edge_gain"]
    assert rec["pre"] == "E" and rec["post"] == "E"
    assert rec["n_edges"] > 0
    s = rec["schedule"]
    assert s["n_steps"] == N_STEPS
    assert s["min"] == float(sched.min())
    assert s["max"] == float(sched.max())
    assert s["mean"] == float(sched.mean())
