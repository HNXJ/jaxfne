"""T-22 phase 1: declared intra-step fixed point for zero-delay edges (adex route).

A connection declaring ``intra_step_rounds=R`` admits a delay rounding to 0 steps on
that connection only: within one step, round 0 is the ordinary step, then rounds 1..R
fold the newly spiked cells' delay-0 arrivals into the conductance and re-step every
not-yet-spiked cell from the same pre-step ``(V, w, ref)``; spikes are OR-ed, never
undone. Kernel oracles below transcribe that rule with direct ``adex_step`` calls.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._model_manifest import manifest as _run_manifest
from jaxfne.connectivity import compile_connection_rules
from jaxfne.emitters_adex import adex_step, make_adex_params, simulate_adex

DT_MS = 0.1
# Cell 0 is driven hard enough to spike from rest in one step; cells 1.. are quiet
# without synaptic input (observed: I0=60000 spikes cell 0 only; 20000 spikes none).
I_DRIVE_PA = 60000.0
# One delay-0 arrival of this weight spikes a resting E cell (observed).
CHAIN_WEIGHT_NS = 1000.0
RECEPTORS = dict(receptor_reversal_mV=(0.0, -80.0), receptor_tau_ms=(2.0, 8.0))


def _chain(n_cells, weight=CHAIN_WEIGHT_NS):
    """Delay-0 chain 0 -> 1 -> ... -> n-1 on receptor 0, plus cell-0-only drive."""
    I_ext = np.zeros((n_cells,), dtype=np.float64)
    I_ext[0] = I_DRIVE_PA
    edges = dict(
        edge0_pre=np.arange(n_cells - 1, dtype=np.int64),
        edge0_post=np.arange(1, n_cells, dtype=np.int64),
        edge0_weight_nS=np.full(n_cells - 1, weight),
        edge0_receptor=np.zeros(n_cells - 1, dtype=np.int64),
        I_ext_pA=I_ext,
    )
    return make_adex_params(("E",) * n_cells), edges


def _oracle_first_step(params, n_cells, weight, rounds):
    """Independent transcription of the declared rule for step 0: round 0 is the plain
    step from rest; each further round adds the previously-new spikes' arrivals to g
    and re-steps the not-yet-spiked cells from the same pre-step state, OR-ing spikes."""
    n = n_cells
    V0 = jnp.asarray(np.asarray(params.EL_mV))
    w0 = jnp.zeros((n,), jnp.float32)
    ref0 = jnp.zeros((n,), jnp.int32)
    I_ext = jnp.zeros((n,), jnp.float32).at[0].set(I_DRIVE_PA)
    (V1, w1, ref1), spk = adex_step((V0, w0, ref0), I_ext, params, DT_MS)
    g = jnp.zeros((n,), jnp.float32)
    prev_new = spk
    for _ in range(rounds):
        add = jnp.zeros((n,), jnp.float32)
        for pre in range(n_cells - 1):
            add = add.at[pre + 1].add(jnp.where(prev_new[pre], weight, 0.0))
        g = g + add
        I_tot = g * (0.0 - V0) + I_ext
        (Vn, wn, refn), spkn = adex_step((V0, w0, ref0), I_tot, params, DT_MS)
        take = ~spk
        V1 = jnp.where(take, Vn, V1)
        w1 = jnp.where(take, wn, w1)
        ref1 = jnp.where(take, refn, ref1)
        prev_new = spkn & ~spk
        spk = spk | spkn
    return np.asarray(spk), V1, w1, ref1


def _run_chain(n_cells, rounds, **overrides):
    params, edges = _chain(n_cells)
    edges.update(overrides)
    out = simulate_adex(
        params, n_steps=1, dt_ms=DT_MS, intra_step_rounds=rounds,
        **edges, **RECEPTORS, decay="exp",
    )
    return np.asarray(out["spikes"])[0]


def test_chain_round1_reaches_one_hop():
    got = _run_chain(3, 1)
    assert got.sum() > 0, "fixture must spike"
    np.testing.assert_array_equal(got, [True, True, False])
    want, _, _, _ = _oracle_first_step(make_adex_params(("E",) * 3), 3, CHAIN_WEIGHT_NS, 1)
    np.testing.assert_array_equal(got, want)


def test_chain_round2_reaches_two_hops():
    got = _run_chain(3, 2)
    np.testing.assert_array_equal(got, [True, True, True])
    want, _, _, _ = _oracle_first_step(make_adex_params(("E",) * 3), 3, CHAIN_WEIGHT_NS, 2)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("rounds, n_spiked", [(5, 6), (19, 20)])
def test_long_chain_converges_one_hop_per_round(rounds, n_spiked):
    got = _run_chain(20, rounds)
    assert got.sum() > 0, "fixture must spike"
    np.testing.assert_array_equal(got, [(i <= rounds) for i in range(20)])
    assert int(got.sum()) == n_spiked


def test_absent_rounds_runs_the_historical_body_bit_for_bit():
    params = make_adex_params(("E", "E", "E"))
    I_ext = np.zeros((3,), dtype=np.float64)
    I_ext[0] = I_DRIVE_PA
    plain = simulate_adex(
        params, n_steps=3, dt_ms=DT_MS, I_ext_pA=I_ext, record_v=True,
        **RECEPTORS, decay="exp",
    )
    explicit = simulate_adex(
        params, n_steps=3, dt_ms=DT_MS, I_ext_pA=I_ext, record_v=True,
        edge0_pre=None, edge0_post=None, edge0_weight_nS=None, edge0_receptor=None,
        intra_step_rounds=None, **RECEPTORS, decay="exp",
    )
    np.testing.assert_array_equal(np.asarray(explicit["spikes"]), np.asarray(plain["spikes"]))
    np.testing.assert_array_equal(np.asarray(explicit["V_mV"]), np.asarray(plain["V_mV"]))


def test_kernel_rounds_without_edges_refused():
    params, edges = _chain(3)
    edges = {k: v for k, v in edges.items() if not k.startswith("edge0_")}
    with pytest.raises(ValueError, match="needs the delay-0 edge arrays"):
        simulate_adex(params, n_steps=1, dt_ms=DT_MS, intra_step_rounds=2,
                      **edges, **RECEPTORS, decay="exp")


def test_kernel_edges_without_rounds_refused():
    params, edges = _chain(3)
    with pytest.raises(ValueError, match="need intra_step_rounds"):
        simulate_adex(params, n_steps=1, dt_ms=DT_MS, **edges, **RECEPTORS, decay="exp")


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "2"])
def test_kernel_rounds_must_be_a_positive_int(bad):
    params, edges = _chain(3)
    with pytest.raises(ValueError, match="positive integer"):
        simulate_adex(params, n_steps=1, dt_ms=DT_MS, intra_step_rounds=bad,
                      **edges, **RECEPTORS, decay="exp")


def test_float32_fixed_point_matches_float64_spikes():
    params, edges = _chain(3)
    out64 = simulate_adex(params, n_steps=1, dt_ms=DT_MS, intra_step_rounds=2,
                          record_v=True, **edges, **RECEPTORS, decay="exp")
    params32 = params._replace(**{
        f: np.asarray(getattr(params, f), dtype=np.float32)
        for f in params._fields if f != "labels"
    })
    out32 = simulate_adex(params32, n_steps=1, dt_ms=DT_MS, intra_step_rounds=2,
                          record_v=True, **edges, **RECEPTORS, decay="exp")
    np.testing.assert_array_equal(
        np.asarray(out32["spikes"]), np.asarray(out64["spikes"]))
    assert np.asarray(out32["V_mV"]).dtype == np.float32


# Route: Configuration -> construct -> simulate -> manifest.
N = 60
CELL_TYPES = {"E": 0.8, "PV": 0.1, "SOM": 0.1}


def _route_cfg(**over):
    kw = dict(probability=0.2, weight=0.5, delay_ms=0.04, sampling="bernoulli",
              intra_step_rounds=2, edge_seed=11)
    kw.update(over)
    rounds = kw.pop("intra_step_rounds")
    seed = kw.pop("edge_seed")
    conn = dict(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                mechanism="ampa")
    if rounds != "omit":
        conn["intra_step_rounds"] = rounds
    return (
        jtfne.Configuration()
        .network(name="adex60", n=N, cell_types=CELL_TYPES)
        .emitter(family="adex", adex_decay="exp")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .connections(**conn, **kw)
        .connectivity(edge_seed=seed)
    )


def _sim(**kwargs):
    return jtfne.Simulation(duration_ms=10.0, dt_ms=DT_MS, **kwargs)


def test_zero_delay_without_rounds_still_refused():
    model = jtfne.construct(_route_cfg(intra_step_rounds="omit"))
    with pytest.raises(ValueError) as info:
        model.simulate(_sim())
    assert "rounds to 0 steps" in str(info.value)


def test_declared_rounds_admit_zero_delay_end_to_end():
    model = jtfne.construct(_route_cfg())
    signals = model.simulate(_sim())
    st = model.static["adex"]
    n_edges = int(np.asarray(st["pre"]).shape[0])
    assert n_edges > 0, "the fixture must realize delay-0 edges"
    assert signals.metadata["intra_step_fixed_point"] == {
        "rounds": 2, "n_delay0_edges": n_edges,
        "rule": "or-monotone re-step from pre-step state"}


def test_manifest_omits_fixed_point_without_delay0_edges():
    model = jtfne.construct(_route_cfg(delay_ms=2.0, intra_step_rounds="omit"))
    signals = model.simulate(_sim())
    assert "intra_step_fixed_point" not in signals.metadata


def test_rounds_with_nonzero_delay_refused_at_simulate():
    model = jtfne.construct(_route_cfg(delay_ms=2.0))
    with pytest.raises(ValueError) as info:
        model.simulate(_sim())
    msg = str(info.value)
    assert "intra_step_rounds" in msg, msg
    assert "0 steps" in msg, msg


def test_rounds_with_no_realized_edge_refused():
    model = jtfne.construct(_route_cfg(probability=0.0))
    assert int(np.asarray(model.static["adex"]["pre"]).shape[0]) == 0
    with pytest.raises(ValueError) as info:
        model.simulate(_sim())
    assert "no delay-0 edge" in str(info.value)


def test_distinct_rounds_across_rules_refused():
    cfg = (
        jtfne.Configuration()
        .network(name="adex60", n=N, cell_types=CELL_TYPES)
        .emitter(family="adex", adex_decay="exp")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=0.2, weight=0.5, mechanism="ampa", delay_ms=0.04,
                     sampling="bernoulli", intra_step_rounds=1)
        .connections(name="e_pv", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=0.2, weight=0.5, mechanism="ampa", delay_ms=0.04,
                     sampling="bernoulli", intra_step_rounds=2)
        .connectivity(edge_seed=11)
    )
    model = jtfne.construct(cfg)
    with pytest.raises(ValueError) as info:
        model.simulate(_sim())
    assert "distinct" in str(info.value)


def test_rounds_on_a_non_adex_route_refused_at_construct():
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=12,
                 cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes"])
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=0.2, weight=0.5, mechanism="ampa", delay_ms=0.04,
                     sampling="bernoulli", intra_step_rounds=2)
    )
    with pytest.raises(ValueError) as info:
        jtfne.construct(cfg)
    msg = str(info.value)
    assert "intra_step_rounds" in msg, msg
    assert "adex" in msg, msg


def _compiler_rows():
    return [
        {"neuron_id": 0, "area": "V1", "layer": "L4", "cell_type": "E"},
        {"neuron_id": 1, "area": "V1", "layer": "L4", "cell_type": "E"},
    ]


def _compiler_mechs():
    return [{"name": "AMPA", "kind": "AMPA", "params": {"tau_ms": 5.0}}]


def _compiler_rule(**over):
    rule = {"name": "e_e", "source": {"cell_type": "E"}, "target": {"cell_type": "E"},
            "mechanism": "AMPA", "probability": 1.0}
    rule.update(over)
    return rule


def test_compiler_refuses_rounds_off_the_adex_route_by_name():
    with pytest.raises(ValueError) as info:
        compile_connection_rules(
            _compiler_rows(), [_compiler_rule(intra_step_rounds=2)],
            _compiler_mechs(), seed=0)
    msg = str(info.value)
    assert "intra_step_rounds" in msg, msg
    assert "adex" in msg, msg


def test_compiler_carries_rounds_on_the_adex_route():
    res = compile_connection_rules(
        _compiler_rows(), [_compiler_rule(intra_step_rounds=2)],
        _compiler_mechs(), seed=0, allow_intra_step_rounds=True)
    assert res.n_edges > 0


# The run-manifest key is sourced from signals metadata (T-22): present iff the
# run resolved delay-0 edges. Model.manifest() is not realized on the adex route
# (pre-existing refusal), so the impl wiring is pinned on an izhikevich run.
def _izhi_run():
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=12,
                 cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes"])
    )
    model = jtfne.construct(cfg)
    signals = model.simulate(jtfne.simulation(duration_ms=10.0, dt_ms=DT_MS, seed=0))
    return model, signals


def test_manifest_impl_sources_fixed_point_from_signals_metadata():
    import dataclasses

    model, signals = _izhi_run()
    assert "intra_step_fixed_point" not in _run_manifest(model, signals)
    want = {"rounds": 2, "n_delay0_edges": 7,
            "rule": "or-monotone re-step from pre-step state"}
    tagged = dataclasses.replace(
        signals, metadata={**signals.metadata, "intra_step_fixed_point": dict(want)})
    assert _run_manifest(model, tagged)["intra_step_fixed_point"] == want
