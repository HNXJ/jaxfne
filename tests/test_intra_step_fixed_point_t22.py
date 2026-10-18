"""T-22 phase 1: declared intra-step fixed point for zero-delay edges (adex route).

A connection declaring ``intra_step_rounds=R`` admits a delay rounding to 0 steps on
that connection only: within one step, round 0 is the ordinary step, then further
rounds fold the newly spiked cells' delay-0 arrivals into the conductance and re-step
every not-yet-spiked cell from the same pre-step ``(V, w, ref)`` until quiescence
(no newly spiked cell owns a delay-0 out-edge) or R rounds have run; R is a cap,
and a run the cap stops is refused. Spikes are OR-ed, never undone. Kernel oracles
below transcribe that rule with direct ``adex_step`` calls.
"""

from __future__ import annotations

import jax
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


def test_round1_cap_refuses_undelivered_spike():
    # Cap 1 strands cell 1 (its output to cell 2 never delivered): the run is
    # refused instead of returning the partial cascade.
    with pytest.raises(ValueError) as info:
        _run_chain(3, 1)
    msg = str(info.value)
    assert "intra_step_rounds=1" in msg, msg
    assert "step 0" in msg, msg
    assert "raise intra_step_rounds" in msg, msg


def test_chain_round2_reaches_two_hops():
    got = _run_chain(3, 2)
    np.testing.assert_array_equal(got, [True, True, True])
    want, _, _, _ = _oracle_first_step(make_adex_params(("E",) * 3), 3, CHAIN_WEIGHT_NS, 2)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("cap", [19, 30])
def test_long_chain_converges_to_quiescence(cap):
    # A 20-chain needs 19 rounds (the last cell is a sink); a larger cap stops
    # at quiescence with identical outputs.
    out = _chain_out(20, cap)
    got = np.asarray(out["spikes"])[0]
    assert got.sum() > 0, "fixture must spike"
    np.testing.assert_array_equal(got, np.ones(20, bool))
    np.testing.assert_array_equal(
        np.asarray(out["fixed_point_rounds"]), np.array([19], np.int32))
    np.testing.assert_array_equal(
        np.asarray(out["fixed_point_unresolved"]), np.zeros(1, np.int32))


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
    # Cap 64: quiescence on 60 cells needs at most 60 rounds (each round adds
    # at least one newly spiked cell), so the default route is always accepted.
    kw = dict(probability=0.2, weight=0.5, delay_ms=0.04, sampling="bernoulli",
              intra_step_rounds=64, edge_seed=11)
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
    fp = signals.metadata["intra_step_fixed_point"]
    assert fp["rounds"] == 64
    assert fp["n_delay0_edges"] == n_edges
    assert fp["rule"] == (
        "or-monotone re-step from pre-step state until quiescence, rounds capped")
    assert fp["n_unresolved_spikes"] == 0
    assert isinstance(fp["max_rounds_used"], int) and fp["max_rounds_used"] >= 0


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
            "rule": "or-monotone re-step from pre-step state until quiescence, rounds capped",
            "n_unresolved_spikes": 0, "max_rounds_used": 4}
    tagged = dataclasses.replace(
        signals, metadata={**signals.metadata, "intra_step_fixed_point": dict(want)})
    assert _run_manifest(model, tagged)["intra_step_fixed_point"] == want


# T-22 phase 2: zero-delay external Poisson drive ("same_step", adex route).
# A 0-step drive delay is admitted only with ``"same_step": True`` in the drive
# dict: the count drawn for step t acts at step t (external drive has no
# same-step feedback -- its counts never depend on this step's spikes -- so no
# fixed point is needed). Kernel: ``ext_delay_steps=0`` draws first and adds to
# g before the current is computed, under the same one-split-per-step key
# stream, so the drawn counts are the delay>=1 sequence shifted by one step.
SAMEDRIVE = {
    "n_sources": 560, "rate_hz": 4.0, "weight_nS": 2.5, "mechanism": "ampa",
    "target_cell_types": ["E", "PV"], "delay_ms": 0.0, "seed": 7,
}


def _same_sim(**kwargs):
    drive = dict(SAMEDRIVE, same_step=True)
    drive.update(kwargs)
    return _sim(poisson_drive=drive)


def test_zero_delay_drive_needs_same_step():
    model = jtfne.construct(_route_cfg())
    with pytest.raises(ValueError) as info:
        model.simulate(_sim(poisson_drive=dict(SAMEDRIVE)))
    assert "the drive needs >= 1" in str(info.value)
    signals = model.simulate(_same_sim())
    assert int(np.sum(np.asarray(signals.spikes))) > 0, "the same-step drive must spike"


@pytest.mark.parametrize("same_step", [False, 0, 1, "yes"])
def test_zero_delay_drive_same_step_must_be_true(same_step):
    model = jtfne.construct(_route_cfg())
    with pytest.raises(ValueError) as info:
        model.simulate(_sim(poisson_drive=dict(SAMEDRIVE, same_step=same_step)))
    assert "same_step=True" in str(info.value)


def test_same_step_with_nonzero_delay_refused():
    model = jtfne.construct(_route_cfg())
    with pytest.raises(ValueError) as info:
        model.simulate(_sim(poisson_drive=dict(SAMEDRIVE, delay_ms=2.0, same_step=True)))
    msg = str(info.value)
    assert "same_step" in msg, msg
    assert "0 steps" in msg, msg


def test_same_step_with_brian2_refused():
    cfg = (
        jtfne.Configuration()
        .network(name="adex60", n=N, cell_types=CELL_TYPES)
        .emitter(family="adex", adex_decay="exp", adex_step_order="brian2")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     mechanism="ampa", probability=0.2, weight=0.5, delay_ms=0.04,
                     sampling="bernoulli", intra_step_rounds=2)
        .connectivity(edge_seed=11)
    )
    model = jtfne.construct(cfg)
    with pytest.raises(ValueError) as info:
        model.simulate(_sim(poisson_drive=dict(SAMEDRIVE, same_step=True)))
    assert "brian2" in str(info.value)


@pytest.mark.parametrize("variant", ["plain", "window", "edge0"])
def test_same_step_shift_identity(variant):
    """Spikes at delay 0 equal spikes at delay 1 shifted by one step, same key: the
    counts are the same sequence shifted (one count of 30 nS moves V by volts, so
    a count difference cannot hide under the V tolerance). ``window`` gates the
    draw by its step; ``edge0`` runs the intra-step fixed-point branch of both bodies."""
    params = make_adex_params(("E",) * 3)
    n_steps = 60
    sched = np.full(n_steps, 3.0)
    sched[0] = 0.0  # both runs apply no drive at step 0, so trajectories align shifted
    key = jax.random.PRNGKey(11)
    base = dict(
        receptor_reversal_mV=(0.0,), receptor_tau_ms=(2.0,), decay="exp",
        ext_targets=np.arange(3), ext_lambda=sched, ext_weight_nS=30.0,
        ext_receptor=0,
    )
    if variant == "window":
        base.update(ext_t_start_steps=10, ext_t_stop_steps=40)
    elif variant == "edge0":
        base.update(
            ext_targets=np.arange(1), ext_lambda=sched[:, None],
            edge0_pre=np.array([0, 1]), edge0_post=np.array([1, 2]),
            edge0_weight_nS=np.full(2, CHAIN_WEIGHT_NS),
            edge0_receptor=np.zeros(2, dtype=np.int64), intra_step_rounds=2,
        )
    out0 = simulate_adex(
        params, n_steps=n_steps, dt_ms=DT_MS, record_v=True,
        **base, ext_delay_steps=0, key=key,
    )
    out1 = simulate_adex(
        params, n_steps=n_steps, dt_ms=DT_MS, record_v=True,
        **base, ext_delay_steps=1, key=key,
    )
    spk0 = np.asarray(out0["spikes"])
    if variant == "window":
        assert not spk0[:10].any() and spk0[10:40].any(), "window must gate the drive"
    elif variant == "edge0":
        # Only cell 0 is driven; cells 1, 2 spike only through the same-step chain.
        same = spk0[:, 0] & spk0[:, 1] & spk0[:, 2]
        assert same.any(), "the chain must carry a driven spike within its step"
    assert np.asarray(out0["spikes"]).sum() > 0, "fixture must spike"
    np.testing.assert_array_equal(
        np.asarray(out1["spikes"])[1:], np.asarray(out0["spikes"])[:-1])
    # Spikes are bit-exact; V agrees to float32 rounding (the two scan bodies
    # compile separately, so XLA may fuse the same arithmetic differently: a
    # single differing count would move V by volts, not 1e-6).
    np.testing.assert_allclose(
        np.asarray(out1["V_mV"])[1:], np.asarray(out0["V_mV"])[:-1],
        rtol=1e-5, atol=1e-6)


def test_same_step_count_acts_this_step():
    """A 3-cell case: the step-t count changes V at step t, hand-checked."""
    params = make_adex_params(("E", "E", "E"))
    key = jax.random.PRNGKey(4)
    lam, weight = 2.0, 30.0
    out = simulate_adex(
        params, n_steps=1, dt_ms=DT_MS, record_v=True,
        receptor_reversal_mV=(0.0,), receptor_tau_ms=(2.0,), decay="exp",
        ext_targets=np.arange(3), ext_lambda=lam, ext_weight_nS=weight,
        ext_receptor=0, ext_delay_steps=0, key=key,
    )
    _, ks = jax.random.split(key)
    dtype = jnp.asarray(params.C_pF).dtype
    counts = jnp.asarray(np.asarray(jax.random.poisson(ks, lam, shape=(3,))), dtype)
    assert int(jnp.sum(counts)) > 0, "fixture must draw a nonzero count"
    V0 = jnp.asarray(np.asarray(params.EL_mV))
    w0 = jnp.zeros((3,), dtype)
    ref0 = jnp.zeros((3,), jnp.int32)
    g = jnp.asarray(weight, dtype) * counts  # decay of zero g plus the arrival
    (V1, _, _), spk = adex_step((V0, w0, ref0), g * (0.0 - V0), params, DT_MS)
    np.testing.assert_array_equal(np.asarray(out["spikes"])[0], np.asarray(spk))
    np.testing.assert_array_equal(np.asarray(out["V_mV"])[0], np.asarray(V1))


# Byte pin for the unchanged delay>=1 body (Python-level branch): a tiny
# delay-1 drive run, frozen from the implementation (float32, x64 off).
_DELAY1_SPIKES = [
    [False, False, False], [False, False, False], [False, False, False],
    [False, False, False], [False, False, False], [True, False, False],
    [False, False, False], [False, False, False],
]
_DELAY1_V0 = [
    -64.99999237060547, -62.07498550415039, -55.70794677734375,
    -48.63483428955078, -42.05535125732422, -65.0, -65.0, -65.0,
]


def test_delayed_body_pinned_byte_identical():
    params = make_adex_params(("E", "E", "E"))
    out = simulate_adex(
        params, n_steps=8, dt_ms=DT_MS, record_v=True,
        receptor_reversal_mV=(0.0,), receptor_tau_ms=(2.0,), decay="exp",
        ext_targets=np.arange(1), ext_lambda=2.0, ext_weight_nS=30.0,
        ext_receptor=0, ext_delay_steps=1, key=jax.random.PRNGKey(4),
    )
    np.testing.assert_array_equal(np.asarray(out["spikes"]), _DELAY1_SPIKES)
    np.testing.assert_array_equal(
        np.asarray(out["V_mV"])[:, 0],
        np.asarray(_DELAY1_V0, dtype=np.asarray(out["V_mV"]).dtype),
    )
    np.testing.assert_array_equal(
        np.asarray(out["spikes"])[:, 1:], np.zeros((8, 2), bool))


def test_manifest_records_same_step_only_when_declared():
    model = jtfne.construct(_route_cfg())
    tagged = model.simulate(_same_sim())
    assert tagged.metadata["poisson_drive"]["same_step"] is True
    plain = model.simulate(_sim(poisson_drive=dict(SAMEDRIVE, delay_ms=2.0)))
    assert "same_step" not in plain.metadata["poisson_drive"]


def test_kernel_refuses_zero_ext_delay_under_brian2():
    params = make_adex_params(("E", "PV"))
    with pytest.raises(ValueError) as info:
        simulate_adex(
            params, n_steps=10, dt_ms=DT_MS, step_order="brian2",
            receptor_reversal_mV=(0.0,), receptor_tau_ms=(2.0,), decay="exp",
            ext_targets=[0], ext_lambda=1.0, ext_weight_nS=1.0, ext_receptor=0,
            ext_delay_steps=0, key=jax.random.PRNGKey(0),
        )
    assert "brian2" in str(info.value)


def test_hdp_kernel_refuses_zero_ext_delay():
    from jaxfne.emitters_adex import simulate_adex_hdp_registered

    params = make_adex_params(("E", "PV"))
    with pytest.raises(ValueError) as info:
        simulate_adex_hdp_registered(
            params, n_steps=10, dt_ms=DT_MS,
            receptor_reversal_mV=(0.0,), receptor_tau_ms=(2.0,), decay="exp",
            ext_targets=[0], ext_lambda=1.0, ext_weight_nS=1.0, ext_receptor=0,
            ext_delay_steps=0, key=jax.random.PRNGKey(0),
            hdp_rule="synthetic_presyn_gain",
        )
    assert "same-step" in str(info.value)


# T-22 phase 4 (ruling 2026-10-18): the fixed point runs to quiescence with
# intra_step_rounds=R as a cap. A run the cap stops is refused naming the cap,
# the first stranded step and the total; accepted runs report the rounds used.
def _chain_out(n_cells, rounds, n_steps=1, record_v=True, **overrides):
    params, edges = _chain(n_cells)
    edges.update(overrides)
    return simulate_adex(
        params, n_steps=n_steps, dt_ms=DT_MS, intra_step_rounds=rounds,
        record_v=record_v, **edges, **RECEPTORS, decay="exp",
    )


def test_cap1_chain_refused_naming_step_and_cap():
    # Cap 1 strands cell 1 (its output to cell 2 never delivered).
    with pytest.raises(ValueError) as info:
        _chain_out(4, 1)
    msg = str(info.value)
    assert "intra_step_rounds=1" in msg, msg
    assert "step 0" in msg, msg
    assert "1 undelivered" in msg, msg
    assert "raise intra_step_rounds" in msg, msg


def test_resolved_sink_reports_zero():
    out = _chain_out(4, 3)
    np.testing.assert_array_equal(
        np.asarray(out["fixed_point_unresolved"]), np.zeros(1, np.int32))
    np.testing.assert_array_equal(
        np.asarray(out["fixed_point_rounds"]), np.array([3], np.int32))
    assert np.asarray(out["spikes"])[0].tolist() == [True, True, True, True]


def test_longer_than_cap_refused_at_cap_accepted():
    with pytest.raises(ValueError) as info:
        _chain_out(5, 3)
    assert "intra_step_rounds=3" in str(info.value)
    out = _chain_out(5, 4)
    assert np.asarray(out["spikes"])[0].tolist() == [True] * 5
    np.testing.assert_array_equal(
        np.asarray(out["fixed_point_rounds"]), np.array([4], np.int32))
    np.testing.assert_array_equal(
        np.asarray(out["fixed_point_unresolved"]), np.zeros(1, np.int32))


def test_unresolved_absent_without_delay0_edges():
    params = make_adex_params(("E", "E", "E"))
    I_ext = np.zeros((3,), dtype=np.float64)
    I_ext[0] = I_DRIVE_PA
    for record_v in (False, True):
        out = simulate_adex(
            params, n_steps=3, dt_ms=DT_MS, I_ext_pA=I_ext,
            record_v=record_v, **RECEPTORS, decay="exp",
        )
        assert "fixed_point_unresolved" not in out


def test_unresolved_present_without_record_v():
    out = _chain_out(4, 3, record_v=False)
    np.testing.assert_array_equal(
        np.asarray(out["fixed_point_unresolved"]), np.zeros(1, np.int32))
    np.testing.assert_array_equal(
        np.asarray(out["fixed_point_rounds"]), np.array([3], np.int32))
    assert "V_mV" not in out


def test_unresolved_reported_on_the_same_step_drive_body():
    params = make_adex_params(("E",) * 3)
    n_steps = 8
    out = simulate_adex(
        params, n_steps=n_steps, dt_ms=DT_MS, record_v=True,
        receptor_reversal_mV=(0.0,), receptor_tau_ms=(2.0,), decay="exp",
        ext_targets=np.arange(1), ext_lambda=3.0, ext_weight_nS=30.0,
        ext_receptor=0, ext_delay_steps=0, key=jax.random.PRNGKey(11),
        edge0_pre=np.array([0, 1]), edge0_post=np.array([1, 2]),
        edge0_weight_nS=np.full(2, CHAIN_WEIGHT_NS),
        edge0_receptor=np.zeros(2, dtype=np.int64), intra_step_rounds=2,
    )
    unres = np.asarray(out["fixed_point_unresolved"])
    assert unres.dtype == np.int32, unres.dtype
    assert unres.shape == (n_steps,)
    assert bool((unres >= 0).all())
    assert int(unres.sum()) == 0, "a 2-deep chain from cell 0 always resolves"
    rounds = np.asarray(out["fixed_point_rounds"])
    assert rounds.dtype == np.int32, rounds.dtype
    assert bool(((rounds >= 0) & (rounds <= 2)).all())


def test_manifest_carries_unresolved_totals():
    model = jtfne.construct(_route_cfg())
    signals = model.simulate(_sim())
    fp = signals.metadata["intra_step_fixed_point"]
    assert set(fp) == {"rounds", "n_delay0_edges", "rule",
                       "n_unresolved_spikes", "max_rounds_used"}
    assert fp["rounds"] == 64
    assert fp["n_unresolved_spikes"] == 0
    assert isinstance(fp["max_rounds_used"], int) and fp["max_rounds_used"] >= 0


def test_route_refuses_cap_stopped_run():
    # Strong recurrent weight so one delay-0 arrival spikes: cap 1 strands a
    # cascade and the model route refuses (nothing partial is returned).
    model = jtfne.construct(_route_cfg(weight=1000.0, intra_step_rounds=1))
    with pytest.raises(ValueError) as info:
        model.simulate(_same_sim())
    msg = str(info.value)
    assert "intra_step_rounds=1" in msg, msg
    assert "raise intra_step_rounds" in msg, msg


# Byte pin for the quiescence loop: at cap 10 the 4-cell chain reproduces the
# old fixed-R kernel at R = 3 (the cascade depth) bit-for-bit. Spikes and V
# frozen from the pre-loop implementation (float32, x64 off).
_CHAIN_Q_SPIKES = [[True, True, True, True], [False, False, False, False]]
_CHAIN_Q_V = [[-65.0, -65.0, -65.0, -65.0],
              [-65.0, -65.0, -65.0, -65.0]]


def test_quiescence_pinned_byte_identical_to_fixed_r():
    out = _chain_out(4, 10, n_steps=2)
    np.testing.assert_array_equal(np.asarray(out["spikes"]), _CHAIN_Q_SPIKES)
    np.testing.assert_array_equal(
        np.asarray(out["V_mV"]),
        np.asarray(_CHAIN_Q_V, dtype=np.asarray(out["V_mV"]).dtype),
    )
    # The loop stops at quiescence after 3 rounds at step 0; step 1 is quiet.
    np.testing.assert_array_equal(
        np.asarray(out["fixed_point_unresolved"]), np.zeros(2, np.int32))
    np.testing.assert_array_equal(
        np.asarray(out["fixed_point_rounds"]), np.array([3, 0], np.int32))
