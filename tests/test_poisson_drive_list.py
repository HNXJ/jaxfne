"""``Simulation(poisson_drive=[...])``: one current-route drive per population, lif route only.

A list of one mapping is that mapping, bit for bit. A list of entries whose triple targets are
pairwise disjoint realizes the sum of each population's own draw, and every entry is consumed.
The other routes refuse the list by name, and the receipts carry one entry per population.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaxfne as jtfne
import jaxfne._model_simulate as ms
from jaxfne import neuronal_tensor as nt
from jaxfne._signals import (
    _make_poisson_drive,
    poisson_chunk_seed,
    resolve_poisson_drive_target,
)
from jaxfne.inputs import (
    compile_poisson_sources,
    compile_poisson_sources_from_signals_metadata,
)

N = 8
DT = 0.5
DUR = 20.0
N_STEPS = 40
DEFAULT_SEED = 0 + 7919
TRIPLE_E = ("mc", "L23", "E")
TRIPLE_PV = ("mc", "L23", "PV")
# Asymmetric per-population drives: different rates, amplitudes and counts (k_ext).
DRIVE_E = {"rate_hz": 1000.0, "amplitude": 5.0, "counts": 4, "target": TRIPLE_E}
DRIVE_PV = {"rate_hz": 600.0, "amplitude": 2.5, "counts": 2, "target": TRIPLE_PV}
REFUSED = "realized only on the lif route"


def _lif_two_pop_model():
    tensor = nt.NeuronalTensor(
        name="poisson-list",
        areas=[nt.Area(name="mc", layers=[nt.Layer(
            name="L23", n_neurons=N,
            neuron_types=[
                nt.NeuronType.make("E", fraction=0.5, emitter="lif", drive=0.0),
                nt.NeuronType.make("PV", fraction=0.5, emitter="lif", drive=0.0),
            ])])],
    )
    return jtfne.construct(tensor, nt.RuntimeConfiguration(
        seed=0, duration_ms=DUR, dt_ms=DT, emitter="lif"))


def _sim(poisson, seed=0):
    return jtfne.Simulation(
        duration_ms=DUR, dt_ms=DT, seed=seed, poisson_drive=poisson,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))


def _own_draw(model, entry, seed=DEFAULT_SEED):
    """The array ``_make_poisson_drive`` builds for one entry alone: the reference draw."""
    target = entry.get("target", "all")
    idx = resolve_poisson_drive_target(model, target) if isinstance(target, tuple) else None
    return np.asarray(_make_poisson_drive(
        n_steps=N_STEPS, n_neurons=N, rate_hz=float(entry["rate_hz"]),
        amplitude=float(entry["amplitude"]), dt_ms=DT, seed=poisson_chunk_seed(seed, 0),
        target=target, counts=int(entry.get("counts", 1)), target_indices=idx))


def _izhikevich_model():
    return jtfne.construct(jtfne.suite2_net1_config(seed=11, n=N, duration_ms=DUR, dt_ms=DT))


def _izhikevich_sim(poisson):
    return jtfne.Simulation(
        duration_ms=12.0, dt_ms=DT, seed=17, poisson_drive=poisson,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))


def _rate_model():
    types = [nt.NeuronType.make(name, fraction=0.5, emitter="rate") for name in ("E", "I")]
    tensor = nt.NeuronalTensor(
        name="rate-list",
        areas=[nt.Area(name="V1", layers=[nt.Layer(
            name="L4", n_neurons=N, neuron_types=types)])],
    )
    return jtfne.construct(tensor, nt.RuntimeConfiguration(
        seed=0, duration_ms=DUR, dt_ms=DT, emitter="rate"))


# --- realization on the lif route ---


@pytest.mark.parametrize("target", [TRIPLE_E, "all"])
@pytest.mark.parametrize("wrap", [list, tuple])
def test_list_of_one_mapping_equals_the_mapping(target, wrap):
    model = _lif_two_pop_model()
    one = {"rate_hz": 1000.0, "amplitude": 5.0, "counts": 4, "target": target}
    base = model.simulate(_sim(one))
    got = model.simulate(_sim(wrap([one])))
    assert int(np.asarray(base.spikes).sum()) > 0, "fixture must spike"
    assert np.array_equal(np.asarray(got.spikes), np.asarray(base.spikes))
    assert np.array_equal(np.asarray(got.V_m), np.asarray(base.V_m))
    assert got.metadata["poisson_drive"] == [base.metadata["poisson_drive"]]


def test_disjoint_entries_realize_the_sum_of_each_population_draw():
    model = _lif_two_pop_model()
    plan = ms._lif_poisson_plan(model, [DRIVE_E, DRIVE_PV])
    total = np.asarray(ms._lif_poisson_drive(
        plan, n_steps=N_STEPS, n_neurons=N, dt_ms=DT, default_seed=DEFAULT_SEED))
    own_e = _own_draw(model, DRIVE_E)
    own_pv = _own_draw(model, DRIVE_PV)
    assert float(own_e.sum()) > 0 and float(own_pv.sum()) > 0, "fixture must draw events"
    assert np.array_equal(total, own_e + own_pv)
    idx_pv = resolve_poisson_drive_target(model, TRIPLE_PV)
    assert np.array_equal(total[:, idx_pv], own_pv[:, idx_pv])


def test_each_entry_is_consumed_and_the_unperturbed_run_is_bit_identical():
    model = _lif_two_pop_model()
    base = model.simulate(_sim([DRIVE_E, DRIVE_PV]))
    assert int(np.asarray(base.spikes).sum()) > 0, "fixture must spike"
    again = model.simulate(_sim([DRIVE_E, DRIVE_PV]))
    assert np.array_equal(np.asarray(again.V_m), np.asarray(base.V_m))
    silent_b = model.simulate(_sim([DRIVE_E, dict(DRIVE_PV, amplitude=0.0)]))
    assert not np.array_equal(np.asarray(silent_b.V_m), np.asarray(base.V_m)), \
        "entry B amplitude must be consumed"
    fewer_b = model.simulate(_sim([DRIVE_E, dict(DRIVE_PV, counts=1)]))
    assert not np.array_equal(np.asarray(fewer_b.V_m), np.asarray(base.V_m)), \
        "entry B counts must be consumed"


def test_entry_seed_is_its_own_and_the_run_default_otherwise():
    model = _lif_two_pop_model()
    pinned = dict(DRIVE_PV, seed=5)
    out = model.simulate(_sim([DRIVE_E, pinned]))
    assert [r["seed"] for r in out.metadata["poisson_drive"]] == [DEFAULT_SEED, 5]
    plan = ms._lif_poisson_plan(model, [DRIVE_E, pinned])
    total = np.asarray(ms._lif_poisson_drive(
        plan, n_steps=N_STEPS, n_neurons=N, dt_ms=DT, default_seed=DEFAULT_SEED))
    assert np.array_equal(
        total, _own_draw(model, DRIVE_E) + _own_draw(model, pinned, seed=5))


def test_seed_batch_member_matches_the_serial_list_run():
    model = _lif_two_pop_model()
    seeds = [11, 12]
    out = ms.simulate_lif_seed_batch(model, _sim([DRIVE_E, DRIVE_PV]), seeds)
    spk = np.asarray(out["spikes"])
    assert int(spk.sum()) > 0, "fixture must spike"
    for k, s in enumerate(seeds):
        ref = model.simulate(_sim([DRIVE_E, DRIVE_PV], seed=s))
        assert np.array_equal(spk[k], np.asarray(ref.spikes))
        np.testing.assert_allclose(np.asarray(out["V_m"][k]), np.asarray(ref.V_m),
                                   rtol=0.0, atol=1e-4)


@pytest.mark.parametrize("pinned_at", [0, 1])
def test_seed_batch_refuses_a_pinned_seed_on_any_entry(pinned_at):
    model = _lif_two_pop_model()
    entries = [DRIVE_E, DRIVE_PV]
    entries[pinned_at] = dict(entries[pinned_at], seed=5)
    with pytest.raises(ValueError, match="refuses a template pinning poisson_drive"):
        ms.simulate_lif_seed_batch(model, _sim(entries), [11])


# --- refusals ---


@pytest.mark.parametrize("bad", [[], [DRIVE_E, 5.0], "E", 3])
def test_container_shape_is_refused_at_construction(bad):
    with pytest.raises(ValueError, match="must be None, a mapping, or a non-empty list of mappings"):
        _sim(bad)


def test_all_target_is_refused_in_a_multi_entry_list():
    model = _lif_two_pop_model()
    with pytest.raises(ValueError, match="target 'all'"):
        model.simulate(_sim([{"rate_hz": 100.0, "amplitude": 1.0}, DRIVE_PV]))


def test_overlapping_entries_are_refused_naming_both_targets():
    model = _lif_two_pop_model()
    with pytest.raises(ValueError, match="overlapping neurons") as exc:
        model.simulate(_sim([DRIVE_E, dict(DRIVE_E, rate_hz=50.0)]))
    msg = str(exc.value)
    assert "entries 0 and 1" in msg and msg.count(repr(TRIPLE_E)) == 2


def test_entry_with_an_adex_key_is_refused():
    model = _lif_two_pop_model()
    with pytest.raises(ValueError, match="current route does not consume"):
        model.simulate(_sim([DRIVE_E, dict(DRIVE_PV, delay_ms=1.5)]))


def test_list_is_refused_on_izhikevich_simulate():
    with pytest.raises(ValueError, match=REFUSED) as exc:
        _izhikevich_model().simulate(_izhikevich_sim([{"rate_hz": 200.0, "amplitude": 4.0}]))
    assert "simulate (emitter family 'izhikevich')" in str(exc.value)


def test_list_is_refused_on_rate_simulate():
    with pytest.raises(ValueError, match=REFUSED) as exc:
        _rate_model().simulate(_sim([{"rate_hz": 200.0, "amplitude": 4.0}]))
    assert "simulate (emitter family 'rate')" in str(exc.value)


def test_list_is_refused_on_simulate_batch():
    with pytest.raises(ValueError, match=REFUSED) as exc:
        _izhikevich_model().simulate_batch(_izhikevich_sim([{"rate_hz": 200.0, "amplitude": 4.0}]),
                                           n_seeds=2)
    assert "simulate_batch" in str(exc.value)


def test_list_is_refused_on_the_adex_drive_readers():
    from jaxfne._construct_adex import adex_external_drive_kwargs
    from jaxfne.adex_destexhe import subset_drive_kwargs

    sim = jtfne.Simulation(poisson_drive=[DRIVE_E])
    with pytest.raises(ValueError, match=REFUSED) as exc:
        adex_external_drive_kwargs(None, sim)
    assert "the adex route" in str(exc.value)
    with pytest.raises(ValueError, match=REFUSED) as exc:
        subset_drive_kwargs(None, sim, 0.5)
    assert "subset_drive_kwargs" in str(exc.value)


# --- receipts ---


def test_list_receipts_carry_each_entry_with_its_target():
    model = _lif_two_pop_model()
    receipts = model.simulate(_sim([DRIVE_E, DRIVE_PV])).metadata["poisson_drive"]
    assert receipts == [
        {"rate_hz": 1000.0, "amplitude": 5.0, "seed": DEFAULT_SEED,
         "status": "stochastic_drive_applied", "counts": 4, "target": list(TRIPLE_E)},
        {"rate_hz": 600.0, "amplitude": 2.5, "seed": DEFAULT_SEED,
         "status": "stochastic_drive_applied", "counts": 2, "target": list(TRIPLE_PV)},
    ]


def test_mapping_with_a_declared_triple_target_records_it():
    model = _lif_two_pop_model()
    receipt = model.simulate(_sim(DRIVE_E)).metadata["poisson_drive"]
    assert receipt["target"] == list(TRIPLE_E)


def test_mapping_without_a_target_keeps_the_baseline_receipt():
    model = _lif_two_pop_model()
    receipt = model.simulate(_sim({"rate_hz": 200.0, "amplitude": 4.0})).metadata["poisson_drive"]
    assert receipt == {"rate_hz": 200.0, "amplitude": 4.0, "seed": DEFAULT_SEED,
                       "status": "stochastic_drive_applied"}


# --- manifest sources ---


def test_compile_list_yields_one_source_per_population():
    entries = compile_poisson_sources([DRIVE_E, DRIVE_PV])
    assert [e.targets for e in entries] == [TRIPLE_E, TRIPLE_PV]
    assert [e.value for e in entries] == [1000.0, 600.0]
    assert [dict(e.schedule)["counts"] for e in entries] == [4, 2]
    assert [dict(e.schedule)["amplitude"] for e in entries] == [5.0, 2.5]


def test_compile_triple_target_records_its_parts():
    (entry,) = compile_poisson_sources(DRIVE_E)
    assert entry.targets == TRIPLE_E


def test_compile_refuses_an_adex_element_in_a_list():
    adex = {"n_sources": 8, "rate_hz": 50.0, "weight_nS": 0.3, "mechanism": "ampa",
            "target_cell_types": ["E"], "delay_ms": 0.1, "seed": 0}
    with pytest.raises(ValueError, match="current-route only"):
        compile_poisson_sources([DRIVE_E, adex])


def test_compile_recorded_list_round_trips_from_the_receipts():
    model = _lif_two_pop_model()
    metadata = model.simulate(_sim([DRIVE_E, DRIVE_PV])).metadata
    entries = compile_poisson_sources_from_signals_metadata(metadata)
    assert [e.targets for e in entries] == [TRIPLE_E, TRIPLE_PV]
    assert [dict(e.schedule)["counts"] for e in entries] == [4, 2]
