"""AdEx route: output silencing of a cell-type fraction and per-step drive rates."""

from __future__ import annotations

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


def _sim(**kwargs):
    return jtfne.Simulation(duration_ms=200.0, dt_ms=DT_MS, **kwargs)


def _drive(targets, seed=7, rate_hz=4.0):
    return {
        "n_sources": 560, "rate_hz": rate_hz, "weight_nS": 2.5, "mechanism": "ampa",
        "target_cell_types": list(targets), "delay_ms": 2.0, "seed": seed,
    }


def _sched_drive(targets, sched, seed=7):
    base = _drive(targets, seed=seed)
    base = {k: v for k, v in base.items() if k != "rate_hz"}
    base["rate_schedule_hz"] = sched
    return base


def _labels(model):
    return np.asarray(model.params["emitter"].labels)


def _kw(model, drive):
    sim = _sim(poisson_drive=drive)
    kw = dict(adex_kernel_args(model, DT_MS, dtype="float32"))
    kw.update(adex_external_drive_kwargs(model, sim))
    kw.update(n_steps=N_STEPS, dt_ms=DT_MS, record_v=True)
    return kw


def test_kernel_all_false_mask_equals_no_mask():
    model = jtfne.construct(_cfg())
    drive = _drive(["E", "SOM"])
    kw = _kw(model, drive)
    ref = simulate_adex(**kw)
    n = int(np.asarray(model.params["emitter"].C_pF).shape[0])
    got = simulate_adex(**kw, spike_out_mask=np.zeros(n, dtype=bool),
                        spike_out_onset_step=0)
    assert np.array_equal(np.asarray(got["spikes"]), np.asarray(ref["spikes"]))
    assert np.array_equal(np.asarray(got["V_mV"]), np.asarray(ref["V_mV"]))
    assert np.asarray(ref["spikes"]).sum() > 0, "fixture must spike"


def test_silenced_cells_record_no_spike_with_onset_zero():
    model = jtfne.construct(_cfg())
    labels = _labels(model)
    drive = _drive(["E", "SOM"])
    ref = np.asarray(model.simulate(_sim(poisson_drive=drive)).spikes)
    assert ref[:, labels == "SOM"].sum() > 0, "SOM must spike in the unmasked run"
    got = np.asarray(model.simulate(_sim(poisson_drive=drive, silence={
        "fractions": {"SOM": 1.0}, "onset_ms": 0.0, "seed": 3})).spikes)
    assert got[:, labels == "SOM"].sum() == 0


def test_silenced_cells_record_no_spike_after_onset():
    model = jtfne.construct(_cfg())
    labels = _labels(model)
    drive = _drive(["E", "SOM"])
    onset_step = 1000
    got = np.asarray(model.simulate(_sim(poisson_drive=drive, silence={
        "fractions": {"SOM": 1.0}, "onset_ms": onset_step * DT_MS, "seed": 3})).spikes)
    som = got[:, labels == "SOM"]
    assert som[:onset_step].sum() > 0, "SOM must spike before the onset"
    assert som[onset_step:].sum() == 0


def test_silenced_v_matches_until_first_arrival_and_keeps_moving():
    model = jtfne.construct(_cfg())
    labels = _labels(model)
    drive = _drive(["E", "SOM"])
    onset_step = 500
    ref = model.simulate(_sim(poisson_drive=drive))
    got = model.simulate(_sim(poisson_drive=drive, silence={
        "fractions": {"SOM": 1.0}, "onset_ms": onset_step * DT_MS, "seed": 3}))
    ref_v = np.asarray(ref.V_m)
    got_v = np.asarray(got.V_m)
    assert np.asarray(ref.spikes)[:, labels == "SOM"][onset_step:].sum() > 0, \
        "a silenced spike must exist after the onset in the unmasked run"
    assert np.array_equal(got_v[: onset_step + 1], ref_v[: onset_step + 1])
    assert not np.array_equal(got.spikes, ref.spikes)
    first_diff = np.nonzero((got_v != ref_v).any(axis=1))[0]
    assert first_diff.size > 0, "silencing SOM must change the network voltage"
    assert int(first_diff.min()) > onset_step
    after = got_v[onset_step + 1:, labels == "SOM"]
    assert np.std(after, axis=0).max() > 0.0, "silenced V must keep moving, not clamp"


def test_silencing_all_som_changes_e_count_and_seed_matters():
    model = jtfne.construct(_cfg())
    labels = _labels(model)
    drive = _drive(["E", "SOM"])
    ref = np.asarray(model.simulate(_sim(poisson_drive=drive)).spikes)
    assert ref[:, labels == "SOM"].sum() > 0, "SOM must spike in the default run"

    def _e_count(seed):
        out = np.asarray(model.simulate(_sim(poisson_drive=drive, silence={
            "fractions": {"SOM": 1.0}, "onset_ms": 0.0, "seed": seed})).spikes)
        return int(out[:, labels == "E"].sum())

    assert _e_count(3) != int(ref[:, labels == "E"].sum())
    assert _e_count(3) == _e_count(3), "same seed must give the same mask"

    def _half_counts(seed):
        out = np.asarray(model.simulate(_sim(poisson_drive=drive, silence={
            "fractions": {"SOM": 0.5}, "onset_ms": 0.0, "seed": seed})).spikes)
        return out

    runs = [_half_counts(s) for s in range(6)]
    assert any(not np.array_equal(runs[0], r) for r in runs[1:]), \
        "different seeds must give different masks at fraction 0.5"


def test_n_silenced_per_type_matches_rounding():
    model = jtfne.construct(_cfg())
    drive = _drive(["E", "SOM"])
    for fraction, want in ((0.4, 2), (0.9, 5)):
        out = model.simulate(_sim(poisson_drive=drive, silence={
            "fractions": {"SOM": fraction}, "onset_ms": 0.0, "seed": 3}))
        assert out.metadata["silence"]["n_silenced_per_type"] == {"SOM": want}


def test_silence_metadata_records_selection():
    model = jtfne.construct(_cfg())
    drive = _drive(["E", "SOM"])
    out = model.simulate(_sim(poisson_drive=drive, silence={
        "fractions": {"SOM": 0.5}, "onset_ms": 10.0, "seed": 3}))
    rec = out.metadata["silence"]
    assert rec["fractions"] == {"SOM": 0.5}
    assert rec["onset_ms"] == 10.0
    assert rec["onset_step"] == 100
    assert rec["seed"] == 3
    assert rec["n_silenced_per_type"] == {"SOM": 3}


@pytest.mark.parametrize("silence,match", [
    ({"fractions": {"XX": 0.5}, "seed": 1}, "unknown cell type"),
    ({"fractions": {"SOM": 1.5}, "seed": 1}, "in \\[0, 1\\]"),
    ({"fractions": {"SOM": -0.1}, "seed": 1}, "in \\[0, 1\\]"),
    ({"fractions": {"SOM": 0.5}}, "seed"),
    ({"fractions": {"SOM": 0.5}, "seed": 1, "bogus": 2}, "unknown key"),
    ({"fractions": {"SOM": 0.5}, "seed": 1, "onset_ms": -1.0}, "onset_ms"),
])
def test_bad_silence_is_refused(silence, match):
    model = jtfne.construct(_cfg())
    with pytest.raises(ValueError, match=match):
        model.simulate(_sim(poisson_drive=_drive(["E", "SOM"]), silence=silence))


def _izhikevich_cfg():
    return (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=12,
                 cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann",
               gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


def test_silence_refused_on_izhikevich_route():
    model = jtfne.construct(_izhikevich_cfg())
    with pytest.raises(ValueError, match="izhikevich"):
        model.simulate(jtfne.Simulation(duration_ms=20.0, dt_ms=DT_MS, seed=0,
                                        silence={"fractions": {"E": 0.5}, "seed": 1}))


def test_silence_refused_on_hh_route():
    cfg = (
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
    model = jtfne.construct(cfg)
    with pytest.raises(ValueError, match="hh"):
        model.simulate(jtfne.Simulation(duration_ms=20.0, dt_ms=DT_MS, seed=0,
                                        silence={"fractions": {"E": 0.5}, "seed": 1}))


def test_kernel_mask_validation():
    model = jtfne.construct(_cfg())
    kw = _kw(model, _drive(["E", "SOM"]))
    n = int(np.asarray(model.params["emitter"].C_pF).shape[0])
    good = np.zeros(n, dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        simulate_adex(**{**kw, "spike_out_mask": np.zeros(n + 1, dtype=bool),
                         "spike_out_onset_step": 0})
    with pytest.raises(ValueError, match="dtype"):
        simulate_adex(**{**kw, "spike_out_mask": np.zeros(n, dtype=np.int8),
                         "spike_out_onset_step": 0})
    with pytest.raises(ValueError, match="onset"):
        simulate_adex(**{**kw, "spike_out_mask": good,
                         "spike_out_onset_step": N_STEPS})
    with pytest.raises(ValueError, match="onset"):
        simulate_adex(**{**kw, "spike_out_onset_step": 3})


def test_hdp_kernel_refuses_mask():
    model = jtfne.construct(_cfg())
    kw = _kw(model, _drive(["E", "SOM"]))
    n = int(np.asarray(model.params["emitter"].C_pF).shape[0])
    base = {k: kw[k] for k in (
        "n_steps", "dt_ms", "decay", "step_order", "record_v",
        "edge_pre", "edge_post", "edge_weight_nS", "edge_receptor",
        "edge_delay_steps", "receptor_reversal_mV", "receptor_tau_ms",
        "ext_targets", "ext_lambda", "ext_weight_nS", "ext_receptor",
        "ext_delay_steps", "key") if k in kw}
    with pytest.raises(ValueError, match="mask"):
        simulate_adex_hdp_registered(
            model.params["emitter"], **base, hdp_rule="signed_linear",
            spike_out_mask=np.zeros(n, dtype=bool))


def test_constant_schedule_equals_rate_hz_exactly():
    model = jtfne.construct(_cfg())
    drive = _drive(["E", "SOM"])
    ref = np.asarray(model.simulate(_sim(poisson_drive=drive)).spikes)
    assert ref.sum() > 0, "fixture must spike"
    sched = np.full(N_STEPS, drive["rate_hz"])
    got = np.asarray(model.simulate(_sim(
        poisson_drive=_sched_drive(["E", "SOM"], sched))).spikes)
    assert np.array_equal(got, ref)


def test_on_off_schedule_moves_target_count_between_halves():
    model = jtfne.construct(_cfg())
    labels = _labels(model)
    drive = _drive(["E"])
    sched = np.zeros(N_STEPS)
    sched[: N_STEPS // 2] = drive["rate_hz"]
    out = np.asarray(model.simulate(_sim(
        poisson_drive=_sched_drive(["E"], sched))).spikes)
    e = out[:, labels == "E"]
    assert e[: N_STEPS // 2].sum() > 0, "first half must carry driven spikes"
    assert e[N_STEPS // 2:].sum() < e[: N_STEPS // 2].sum()


def test_schedule_metadata_records_summary():
    model = jtfne.construct(_cfg())
    sched = np.linspace(1.0, 8.0, N_STEPS)
    out = model.simulate(_sim(poisson_drive=_sched_drive(["E", "SOM"], sched)))
    rec = out.metadata["poisson_drive"]["rate_schedule_hz"]
    assert rec["n_steps"] == N_STEPS
    assert rec["min"] == pytest.approx(1.0)
    assert rec["max"] == pytest.approx(8.0)
    assert rec["mean"] == pytest.approx(float(sched.mean()))


def test_list_path_schedule_is_realized():
    model = jtfne.construct(_cfg())
    labels = _labels(model)
    sched_e = np.full(N_STEPS, 4.0)
    sched_s = np.zeros(N_STEPS)
    sched_s[: N_STEPS // 2] = 4.0
    entry_e = _sched_drive(["E"], sched_e, seed=7)
    entry_s = _sched_drive(["SOM"], sched_s, seed=8)
    out = model.simulate(_sim(poisson_drive=[entry_e, entry_s]))
    spk = np.asarray(out.spikes)
    assert spk[:, labels == "SOM"][: N_STEPS // 2].sum() > 0, \
        "first half must carry driven SOM spikes"
    recs = out.metadata["poisson_drive"]["entries"]
    assert recs[0]["rate_schedule_hz"]["n_steps"] == N_STEPS
    assert recs[1]["rate_schedule_hz"]["mean"] == pytest.approx(2.0)


def test_grouped_rates_with_schedule_refused():
    model = jtfne.construct(_cfg())
    bad = _drive(["E", "SOM"])
    bad["rate_hz"] = [4.0, 2.0]
    bad["rate_schedule_hz"] = np.full(N_STEPS, 4.0)
    with pytest.raises(ValueError, match="in place of rate_hz"):
        model.simulate(_sim(poisson_drive=bad))


@pytest.mark.parametrize("drive,match", [
    ("both", "in place of rate_hz"),
    ("length", "length"),
    ("negative", "must hold finite values"),
])
def test_bad_schedules_are_refused(drive, match):
    model = jtfne.construct(_cfg())
    base = _drive(["E", "SOM"])
    if drive == "both":
        bad = {**base, "rate_schedule_hz": np.full(N_STEPS, base["rate_hz"])}
    elif drive == "length":
        bad = {k: v for k, v in base.items() if k != "rate_hz"}
        bad["rate_schedule_hz"] = np.full(N_STEPS - 1, 4.0)
    else:
        bad = {k: v for k, v in base.items() if k != "rate_hz"}
        bad["rate_schedule_hz"] = np.full(N_STEPS, -1.0)
    with pytest.raises(ValueError, match=match):
        model.simulate(_sim(poisson_drive=bad))


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


def test_masked_delay0_source_matches_edge_removed():
    model = jtfne.construct(_cfg_zero(weight=4.0))
    ref_model = jtfne.construct(_cfg_zero(weight=0.0))
    labels = _labels(model)
    assert set(labels.tolist()) == {"E", "PV"}
    drive = _drive(["E"], seed=7, rate_hz=8.0)
    sim = _sim(poisson_drive=drive)
    unmasked = model.simulate(sim)
    assert np.asarray(unmasked.spikes)[:, labels == "E"].sum() > 0, \
        "source must spike in the unmasked run"
    masked = model.simulate(_sim(poisson_drive=drive, silence={
        "fractions": {"E": 1.0}, "onset_ms": 0.0, "seed": 3}))
    ref = ref_model.simulate(sim)
    pv = labels == "PV"
    assert np.array_equal(np.asarray(masked.V_m)[:, pv], np.asarray(ref.V_m)[:, pv])
    assert not np.array_equal(np.asarray(unmasked.V_m)[:, pv],
                              np.asarray(ref.V_m)[:, pv])
