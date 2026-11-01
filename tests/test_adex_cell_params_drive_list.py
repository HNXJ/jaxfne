"""AdEx route additions: per-type ``cell_params`` and ``Simulation(poisson_drive=[...])`` lists."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitters_adex import _PARAM_FIELDS, make_adex_params, simulate_adex

N = 60
DT_MS = 0.1
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
        .connections(name="e_pv", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=0.2, weight=1.0, mechanism="ampa", delay_ms=1.3,
                     sampling="bernoulli")
        .connectivity(edge_seed=11)
    )


def _sim(**kwargs):
    return jtfne.Simulation(duration_ms=50.0, dt_ms=DT_MS, **kwargs)


def _drive(targets, seed=7, rate_hz=4.0):
    return {
        "n_sources": 560, "rate_hz": rate_hz, "weight_nS": 2.5, "mechanism": "ampa",
        "target_cell_types": list(targets), "delay_ms": 2.0, "seed": seed,
    }


def _columns(model):
    return {f: np.asarray(getattr(model.params["emitter"], f)) for f in _PARAM_FIELDS}


def test_a_default_restating_cell_params_is_bit_identical():
    plain = jtfne.construct(_cfg())
    restated = jtfne.construct(_cfg().cell_params({"cell_type": "E"}, {"C_pF": 200.0}))
    for field in _PARAM_FIELDS:
        assert np.array_equal(_columns(plain)[field], _columns(restated)[field],
                              equal_nan=True), field
    ref = np.asarray(plain.simulate(_sim(poisson_drive=_drive(["E", "PV"]))).spikes)
    assert ref.sum() > 0, "fixture must spike"
    got = np.asarray(restated.simulate(_sim(poisson_drive=_drive(["E", "PV"]))).spikes)
    assert np.array_equal(ref, got), "restating a default must not move the spikes"


def test_cell_params_override_touches_only_that_type():
    model = jtfne.construct(_cfg().cell_params({"cell_type": "E"}, {"C_pF": 100.0}))
    cols = _columns(model)
    labels = np.asarray(model.params["emitter"].labels)
    assert np.all(cols["C_pF"][labels == "E"] == 100.0)
    assert np.all(cols["C_pF"][labels != "E"] == 200.0)
    plain_cols = _columns(jtfne.construct(_cfg()))
    for field in _PARAM_FIELDS:
        if field == "C_pF":
            continue
        assert np.array_equal(cols[field][labels != "E"],
                              plain_cols[field][labels != "E"], equal_nan=True), field
    ref = np.asarray(jtfne.construct(_cfg()).simulate(
        _sim(poisson_drive=_drive(["E", "PV"]))).spikes)
    got = np.asarray(model.simulate(_sim(poisson_drive=_drive(["E", "PV"]))).spikes)
    assert not np.array_equal(ref, got), "a live override must move the spikes"


@pytest.mark.parametrize("params, needle", [({"a": 0.1}, "does not realize"),
                                             ({"zzz": 1.0}, "Vpeak_mV")])
def test_cell_params_refuses_izhikevich_only_and_unknown_keys(params, needle):
    with pytest.raises(ValueError, match=needle):
        jtfne.construct(_cfg().cell_params({"cell_type": "E"}, params))


def _drive_kwargs(model, drive):
    from jaxfne._construct_adex import adex_external_drive_kwargs

    return adex_external_drive_kwargs(model, _sim(poisson_drive=drive))


def _assert_kwargs_equal(first, second):
    assert set(first) == set(second)
    for key in first:
        assert np.array_equal(np.asarray(first[key]), np.asarray(second[key])), key


def test_drive_list_of_one_equals_the_mapping_bit_for_bit():
    from jaxfne._construct_adex import adex_kernel_args

    model = jtfne.construct(_cfg())
    entry = _drive(["E", "PV"])
    _assert_kwargs_equal(_drive_kwargs(model, entry), _drive_kwargs(model, [entry]))
    base = adex_kernel_args(model, DT_MS)
    n_steps = int(_sim().n_steps)
    ref = simulate_adex(**base, n_steps=n_steps, dt_ms=DT_MS,
                        **_drive_kwargs(model, entry))["spikes"]
    got = simulate_adex(**base, n_steps=n_steps, dt_ms=DT_MS,
                        **_drive_kwargs(model, [entry]))["spikes"]
    assert int(np.asarray(ref).sum()) > 0, "fixture must spike"
    assert np.array_equal(np.asarray(ref), np.asarray(got))


def test_single_entry_through_extm_path_matches_mapping_bit_for_bit():
    """One entry's ``extm_*`` run equals its ``ext_*`` run: ``bodyM`` pins the delegation."""
    from jaxfne._construct_adex import adex_kernel_args

    model = jtfne.construct(_cfg())
    entry = _drive(["E", "PV"], seed=7)
    map_kw = _drive_kwargs(model, entry)
    n_targets = int(np.asarray(map_kw["ext_targets"]).shape[0])
    assert n_targets > 0
    extm_kw = {
        "extm_targets": map_kw["ext_targets"],
        "extm_lambda": np.full((n_targets,), float(map_kw["ext_lambda"])),
        "extm_weight_nS": np.full((n_targets,), float(map_kw["ext_weight_nS"])),
        "extm_receptor": np.full((n_targets,), int(map_kw["ext_receptor"]), dtype=np.int64),
        "extm_delay_steps": np.full((n_targets,), int(map_kw["ext_delay_steps"]), dtype=np.int64),
        "extm_entry_sizes": np.asarray([n_targets], dtype=np.int64),
        "key": jnp.stack([jax.random.PRNGKey(int(entry["seed"]))]),
    }
    base = adex_kernel_args(model, DT_MS)
    n_steps = int(_sim().n_steps)
    ref = simulate_adex(**base, n_steps=n_steps, dt_ms=DT_MS, record_v=True, **map_kw)
    got = simulate_adex(**base, n_steps=n_steps, dt_ms=DT_MS, record_v=True, **extm_kw)
    assert int(np.asarray(ref["spikes"]).sum()) > 0, "fixture must spike"
    np.testing.assert_array_equal(np.asarray(got["spikes"]), np.asarray(ref["spikes"]))
    np.testing.assert_array_equal(np.asarray(got["V_mV"]), np.asarray(ref["V_mV"]))


def test_singleton_list_records_the_mapping_form():
    """A list of one delegates to the mapping path, so the record matches a bare mapping."""
    model = jtfne.construct(_cfg())
    entry = _drive(["E", "PV"], seed=7)
    solo = model.simulate(_sim(poisson_drive=entry)).metadata["poisson_drive"]
    listed = model.simulate(_sim(poisson_drive=[entry])).metadata["poisson_drive"]
    assert listed == solo, "the delegated record must match what ran"


def test_empty_drive_list_is_refused_for_the_adex_route():
    """An empty list names the adex route's own list contract, never the lif route."""
    import types

    from jaxfne._construct_adex import adex_external_drive_kwargs

    model = jtfne.construct(_cfg())
    # Simulation itself refuses [] generically; the stub reaches the adex route's branch.
    stub = types.SimpleNamespace(poisson_drive=[], dt_ms=DT_MS)
    with pytest.raises(ValueError, match="is empty"):
        adex_external_drive_kwargs(model, stub)


def test_drive_list_with_overlapping_targets_is_refused():
    model = jtfne.construct(_cfg())
    entries = [_drive(["E", "PV"], seed=7), _drive(["E"], seed=8, rate_hz=2.0)]
    with pytest.raises(ValueError, match="disjoint"):
        _drive_kwargs(model, entries)


def test_drive_list_runs_on_adex_and_is_refused_elsewhere():
    from jaxfne._construct_adex import adex_kernel_args

    entries = [_drive(["E"], seed=7), _drive(["PV"], seed=8, rate_hz=2.0)]
    model = jtfne.construct(_cfg())
    kwargs = _drive_kwargs(model, entries)
    base = adex_kernel_args(model, DT_MS)
    out = simulate_adex(**base, n_steps=int(_sim().n_steps), dt_ms=DT_MS, **kwargs)
    assert int(np.asarray(out["spikes"]).sum()) > 0, "the listed drives must spike the fixture"
    izh = jtfne.construct(jtfne.suite2_net1_config(seed=11, n=N, duration_ms=12.0, dt_ms=DT_MS))
    sim = jtfne.Simulation(duration_ms=12.0, dt_ms=DT_MS, seed=17,
                           poisson_drive=[{"rate_hz": 200.0, "amplitude": 4.0}],
                           runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    with pytest.raises(ValueError, match="lif route"):
        izh.simulate(sim)


def test_drive_list_and_cell_params_run_end_to_end_through_simulate():
    cfg = (
        _cfg()
        .cell_params({"cell_type": "E"}, {"C_pF": 100.0})
        .cell_params({"cell_type": "PV"}, {"Delta_mV": 1.0})
    )
    entries = [_drive(["E"], seed=7, rate_hz=4.0), _drive(["PV"], seed=8, rate_hz=2.0)]
    model = jtfne.construct(cfg)
    signals = model.simulate(_sim(poisson_drive=entries))
    spikes = np.asarray(signals.spikes)
    assert spikes.sum() > 0, "fixture must spike"
    labels = np.asarray(model.params["emitter"].labels)
    e_cells = labels == "E"
    assert int(spikes[:, e_cells].sum()) > 0, "the E drive must spike E cells"
    silenced = [_drive(["E"], seed=7, rate_hz=0.0), _drive(["PV"], seed=8, rate_hz=2.0)]
    rerun = np.asarray(model.simulate(_sim(poisson_drive=silenced)).spikes)
    assert int(rerun[:, e_cells].sum()) != int(spikes[:, e_cells].sum()), (
        "silencing the E entry must change the E spike count"
    )
    recorded = signals.metadata["poisson_drive"]
    assert recorded["status"] == "stochastic_drive_applied"
    assert [e["target_cell_types"] for e in recorded["entries"]] == [["E"], ["PV"]]
    assert [e["seed"] for e in recorded["entries"]] == [7, 8]
    assert [e["rate_hz"] for e in recorded["entries"]] == [4.0, 2.0]


def _cfg_feedforward():
    # E -> PV only: nothing feeds back onto E, so E spikes read only the E entry.
    return (
        jtfne.Configuration()
        .network(name="adexep", n=N, cell_types={"E": 0.5, "PV": 0.5})
        .emitter(family="adex", adex_decay="exp")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .connections(name="e_pv", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=0.2, weight=1.0, mechanism="ampa", delay_ms=1.3,
                     sampling="bernoulli")
        .connectivity(edge_seed=11)
    )


def _strong_drive(targets, seed, rate_hz=30.0):
    return {
        "n_sources": 560, "rate_hz": rate_hz, "weight_nS": 2.5, "mechanism": "ampa",
        "target_cell_types": list(targets), "delay_ms": 2.0, "seed": seed,
    }


def test_drive_entries_own_independent_streams():
    """One key per entry: entry 2's seed and entry order move nothing on entry 1."""
    model = jtfne.construct(_cfg_feedforward())
    labels = np.asarray(model.params["emitter"].labels)
    e_cells = labels == "E"
    run = lambda entries: np.asarray(
        model.simulate(_sim(poisson_drive=entries)).spikes)  # noqa: E731
    a = run([_strong_drive(["E"], 7), _strong_drive(["PV"], 8)])
    assert int(a[:, e_cells].sum()) > 0, "fixture must spike E"
    b = run([_strong_drive(["E"], 7), _strong_drive(["PV"], 9)])
    assert np.array_equal(a[:, e_cells], b[:, e_cells]), (
        "entry 2's seed must not move entry 1's cells"
    )
    swapped = run([_strong_drive(["PV"], 8), _strong_drive(["E"], 7)])
    assert np.array_equal(a, swapped), "entry order must not move any spike"
    c = run([_strong_drive(["E"], 10), _strong_drive(["PV"], 8)])
    assert not np.array_equal(a[:, e_cells], c[:, e_cells]), (
        "entry 1's own seed must move its cells"
    )


def test_zero_rate_entry_draws_no_counts_and_leaves_the_other_entry_alone():
    """A zero-rate target is all silent; the live entry matches its solo run."""
    model = jtfne.construct(_cfg_feedforward())
    labels = np.asarray(model.params["emitter"].labels)
    e_cells, pv_cells = labels == "E", labels == "PV"
    run = lambda entries: np.asarray(  # noqa: E731
        model.simulate(_sim(poisson_drive=entries)).spikes)
    two = run([_strong_drive(["E"], 7, rate_hz=0.0), _strong_drive(["PV"], 8)])
    assert int(two[:, e_cells].sum()) == 0, "the zero-rate target must draw no counts"
    one = run([_strong_drive(["PV"], 8)])
    assert np.array_equal(two[:, pv_cells], one[:, pv_cells]), (
        "the live entry must match its solo run bit for bit"
    )


def test_drive_list_window_bounds_the_drive_at_fine_dt():
    """Both entries' counts fall inside the shared window only (dt 0.1 ms)."""
    windowed = dict(_strong_drive(["E"], 7), t_start_ms=10.0, t_stop_ms=20.0)
    windowed_pv = dict(_strong_drive(["PV"], 8), t_start_ms=10.0, t_stop_ms=20.0)
    model = jtfne.construct(_cfg_feedforward())
    spikes = np.asarray(
        model.simulate(_sim(poisson_drive=[windowed, windowed_pv])).spikes)
    t0, t1 = int(10.0 / DT_MS), int(20.0 / DT_MS)
    delay = int(2.0 / DT_MS)
    assert int(spikes[t0 + delay:t1 + delay].sum()) > 0, "fixture must spike in-window"
    assert int(spikes[:t0 + delay].sum()) == 0, "no spikes before the delayed window"
    assert int(spikes[t1 + delay:].sum()) == 0, "no spikes after the delayed window"
    model_plain = jtfne.construct(_cfg())
    off = dict(_drive(["PV"], seed=8), t_start_ms=10.0, t_stop_ms=20.0)
    with pytest.raises(ValueError, match="differ"):
        _drive_kwargs(model_plain, [_drive(["E"], seed=7), off])
    one_bound = dict(_drive(["E"], seed=7), t_start_ms=10.0)
    with pytest.raises(ValueError, match="without the other"):
        _drive_kwargs(model_plain, [one_bound, _drive(["PV"], seed=8)])


def test_drive_key_alone_and_partial_extm_are_refused():
    """A key with no drive group, or with a partial extm group, raises."""
    from jaxfne._construct_adex import adex_kernel_args

    model = jtfne.construct(_cfg())
    base = adex_kernel_args(model, DT_MS)
    n_steps = int(_sim().n_steps)
    with pytest.raises(ValueError, match="all given or all None"):
        simulate_adex(**base, n_steps=n_steps, dt_ms=DT_MS,
                      key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="all given or all None"):
        simulate_adex(**base, n_steps=n_steps, dt_ms=DT_MS,
                      key=jax.random.PRNGKey(0),
                      extm_targets=np.zeros((1,), dtype=np.int64))


@pytest.mark.parametrize("field,value", [
    ("C_pF", -5.0), ("gL_nS", 0.0), ("tauw_ms", 0.0),
    ("Tref_ms", -1.0), ("Delta_mV", -0.5),
])
def test_unphysical_cell_params_are_refused(field, value):
    """Each unphysical override names its field, in the kernel builder and in construct."""
    with pytest.raises(ValueError, match=field):
        make_adex_params(("E",), {"E": {field: value}})
    with pytest.raises(ValueError, match=field):
        jtfne.construct(_cfg().cell_params({"cell_type": "E"}, {field: value}))


def test_el_above_vt_is_a_valid_tonic_regime():
    """EL_mV above VT_mV constructs and runs (tonic firing), in builder and construct."""
    params = make_adex_params(("E",), {"E": {"EL_mV": -40.0}})
    assert float(np.asarray(params.EL_mV)[0]) == -40.0
    model = jtfne.construct(_cfg().cell_params({"cell_type": "E"}, {"EL_mV": -40.0}))
    signals = model.simulate(_sim())
    assert np.asarray(signals.spikes).shape == (int(_sim().n_steps), N)
    assert int(np.asarray(signals.spikes).sum()) > 0, "EL above VT must fire tonically"
