"""AdEx route: Poisson replacement of a cell-type fraction."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._construct_adex import adex_external_drive_kwargs, adex_kernel_args
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


def _cfg_ee(weight):
    return (
        jtfne.Configuration()
        .network(name="adex60", n=N, cell_types=CELL_TYPES)
        .emitter(family="adex", adex_decay="exp")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .mechanisms(name="gaba", kind="receptor",
                    params={"tau_ms": 8.0, "reversal_mV": -80.0})
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=0.2, weight=weight, mechanism="ampa", delay_ms=2.0,
                     sampling="bernoulli")
        .connections(name="e_som", source={"cell_type": "E"}, target={"cell_type": "SOM"},
                     probability=0.2, weight=1.0, mechanism="ampa", delay_ms=1.3,
                     sampling="bernoulli")
        .connections(name="som_e", source={"cell_type": "SOM"}, target={"cell_type": "E"},
                     probability=0.2, weight=1.0, mechanism="gaba", delay_ms=1.5,
                     sampling="bernoulli")
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


def _replace(fractions, rate_hz=None, seed=3):
    rates = rate_hz if rate_hz is not None else {t: 4.0 for t in fractions}
    return {"fractions": dict(fractions), "rate_hz": dict(rates), "seed": seed}


def test_kernel_all_false_replace_mask_equals_no_replace():
    model = jtfne.construct(_cfg())
    drive = _drive(["E", "SOM"])
    kw = _kw(model, drive)
    ref = simulate_adex(**kw)
    n = int(np.asarray(model.params["emitter"].C_pF).shape[0])
    got = simulate_adex(**kw, spike_replace_mask=np.zeros(n, dtype=bool),
                        spike_replace_lam=np.full(n, 0.02),
                        spike_replace_key=jax.random.PRNGKey(3))
    assert np.array_equal(np.asarray(got["spikes"]), np.asarray(ref["spikes"]))
    assert np.array_equal(np.asarray(got["V_mV"]), np.asarray(ref["V_mV"]))
    assert np.asarray(ref["spikes"]).sum() > 0, "fixture must spike"


def test_replaced_cells_emit_no_spike_at_zero_rate():
    model = jtfne.construct(_cfg())
    labels = _labels(model)
    drive = _drive(["E", "SOM"])
    ref = np.asarray(model.simulate(_sim(poisson_drive=drive)).spikes)
    assert ref[:, labels == "SOM"].sum() > 0, "SOM must spike in the unreplaced run"
    got = np.asarray(model.simulate(_sim(
        poisson_drive=drive, replace=_replace({"SOM": 1.0}, {"SOM": 0.0}))).spikes)
    assert got[:, labels == "SOM"].sum() == 0


def test_replace_count_matches_poisson_mean():
    model = jtfne.construct(_cfg())
    labels = _labels(model)
    n_e = int((labels == "E").sum())
    assert n_e == 48, "fixture pins the E count the mean is computed from"
    duration_ms = 500.0
    rate_hz = 20.0
    out = model.simulate(jtfne.Simulation(
        duration_ms=duration_ms, dt_ms=DT_MS, seed=0,
        replace=_replace({"E": 1.0}, {"E": rate_hz})))
    count = int(np.asarray(out.spikes)[:, labels == "E"].sum())
    mean = n_e * (duration_ms / 1000.0) * rate_hz
    assert mean == pytest.approx(480.0)
    assert count == pytest.approx(mean, abs=4.0 * np.sqrt(mean))


def test_replaced_spikes_match_across_weights_while_v_differs():
    drive = _drive(["E", "SOM"])
    outs = []
    for weight in (0.2, 4.0):
        model = jtfne.construct(_cfg_ee(weight))
        labels = _labels(model)
        assert set(labels.tolist()) == {"E", "PV", "SOM"}
        outs.append(model.simulate(_sim(
            poisson_drive=drive, replace=_replace({"E": 1.0}, {"E": 20.0}))))
    e = labels == "E"
    spk0 = np.asarray(outs[0].spikes)[:, e]
    spk1 = np.asarray(outs[1].spikes)[:, e]
    assert spk0.sum() > 0, "replaced E cells must spike under the Poisson draw"
    assert np.array_equal(spk0, spk1)
    assert not np.array_equal(np.asarray(outs[0].V_m)[:, e],
                              np.asarray(outs[1].V_m)[:, e])


def test_n_replaced_per_type_matches_rounding():
    model = jtfne.construct(_cfg())
    drive = _drive(["E", "SOM"])
    for fraction, want in ((0.4, 2), (0.9, 5)):
        out = model.simulate(_sim(poisson_drive=drive,
                                  replace=_replace({"SOM": fraction})))
        assert out.metadata["replace"]["n_replaced_per_type"] == {"SOM": want}


def test_replace_metadata_records_selection():
    model = jtfne.construct(_cfg())
    drive = _drive(["E", "SOM"])
    out = model.simulate(_sim(poisson_drive=drive,
                              replace=_replace({"SOM": 0.5}, {"SOM": 4.0}, seed=3)))
    rec = out.metadata["replace"]
    assert rec["fractions"] == {"SOM": 0.5}
    assert rec["rate_hz"] == {"SOM": 4.0}
    assert rec["seed"] == 3
    assert rec["n_replaced_per_type"] == {"SOM": 3}


@pytest.mark.parametrize("replace,match", [
    ({"fractions": {"XX": 0.5}, "rate_hz": {"XX": 4.0}, "seed": 1}, "unknown cell type"),
    ({"fractions": {"SOM": 1.5}, "rate_hz": {"SOM": 4.0}, "seed": 1}, "in \\[0, 1\\]"),
    ({"fractions": {"SOM": -0.1}, "rate_hz": {"SOM": 4.0}, "seed": 1}, "in \\[0, 1\\]"),
    ({"fractions": {"SOM": 0.5}, "rate_hz": {"SOM": 4.0}}, "seed"),
    ({"fractions": {"SOM": 0.5}, "seed": 1}, "rate_hz"),
    ({"fractions": {"SOM": 0.5}, "rate_hz": {"SOM": -1.0}, "seed": 1}, "rate_hz"),
    ({"fractions": {"SOM": 0.5}, "rate_hz": 4.0, "seed": 1}, "non-empty mapping"),
    ({"fractions": {"SOM": 0.5}, "rate_hz": {}, "seed": 1}, "non-empty mapping"),
    ({"fractions": {"SOM": 0.5, "E": 0.0}, "rate_hz": {"E": 4.0}, "seed": 1},
     "lacks a rate"),
    ({"fractions": {"SOM": 0.5}, "rate_hz": {"SOM": 4.0, "E": 1.0}, "seed": 1},
     "absent from fractions"),
    ({"fractions": {"SOM": 0.0}, "rate_hz": {"SOM": 4.0}, "seed": 1, "bogus": 2},
     "unknown key"),
])
def test_bad_replace_is_refused(replace, match):
    model = jtfne.construct(_cfg())
    with pytest.raises(ValueError, match=match):
        model.simulate(_sim(poisson_drive=_drive(["E", "SOM"]), replace=replace))


def test_replace_per_type_rates_zero_silences_only_that_type():
    model = jtfne.construct(_cfg())
    labels = _labels(model)
    duration_ms = 500.0
    rate_som = 60.0
    out = model.simulate(jtfne.Simulation(
        duration_ms=duration_ms, dt_ms=DT_MS, seed=0,
        replace={"fractions": {"E": 1.0, "SOM": 1.0},
                 "rate_hz": {"E": 0.0, "SOM": rate_som}, "seed": 3}))
    spikes = np.asarray(out.spikes)
    assert spikes[:, labels == "E"].sum() == 0
    n_som = int((labels == "SOM").sum())
    mean = n_som * (duration_ms / 1000.0) * rate_som
    assert mean == pytest.approx(180.0)
    assert spikes[:, labels == "SOM"].sum() == pytest.approx(mean, abs=4.0 * np.sqrt(mean))


def test_replace_draw_key_branch_differs_from_permutation_keys():
    from jaxfne._construct_adex import _REPLACE_TAG, adex_replace_kwargs
    model = jtfne.construct(_cfg())
    sim = _sim(replace={"fractions": {"SOM": 0.5},
                        "rate_hz": {"SOM": 4.0}, "seed": 3})
    _, _, rep_key, _ = adex_replace_kwargs(model, sim)
    key0 = jax.random.fold_in(jax.random.PRNGKey(3), _REPLACE_TAG)
    assert np.array_equal(jax.random.key_data(rep_key),
                          jax.random.key_data(jax.random.fold_in(key0, 0)))
    draw_t1 = jax.random.fold_in(rep_key, 1)
    perm_0 = jax.random.fold_in(key0, 1)
    assert not np.array_equal(jax.random.key_data(draw_t1),
                              jax.random.key_data(perm_0))


def test_replace_refused_off_adex_route():
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
                                       replace=_replace({"E": 0.5})))
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
                                     replace=_replace({"E": 0.5})))
    model = jtfne.construct(_cfg())
    kw = _kw(model, _drive(["E", "SOM"]))
    n = int(np.asarray(model.params["emitter"].C_pF).shape[0])
    base = {k: kw[k] for k in (
        "n_steps", "dt_ms", "decay", "step_order", "record_v",
        "edge_pre", "edge_post", "edge_weight_nS", "edge_receptor",
        "edge_delay_steps", "receptor_reversal_mV", "receptor_tau_ms",
        "ext_targets", "ext_lambda", "ext_weight_nS", "ext_receptor",
        "ext_delay_steps", "key") if k in kw}
    with pytest.raises(ValueError, match="replace"):
        simulate_adex_hdp_registered(
            model.params["emitter"], **base, hdp_rule="signed_linear",
            spike_replace_mask=np.zeros(n, dtype=bool))
