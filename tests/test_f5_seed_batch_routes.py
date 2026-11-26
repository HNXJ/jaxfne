"""F5 seed-batch routes (checklist F5, plan row R39): ``simulate_adex_seed_batch`` and
``simulate_lif_seed_batch`` in ``jaxfne/_model_simulate.py``.

Per route: seed parity (row i equals the serial run for seeds[i]), output shapes, gradients,
and refusals (bad seed lists, wrong emitter family, a non-default Simulation seed, and the
plasticity and inhibition refusals the route docstrings and bodies name).

Tolerance: spikes are compared bitwise. V_m is compared with rtol 0 and atol 1e-4 mV, the
bound the seed-batch contract tests in tests/test_adex_route.py and tests/test_lif_tensor_route.py
already use for float32 codegen drift across jax builds. The route docstrings claim bit-identity;
this file asserts only the recorded bound.

Gradients: neither route exposes a differentiable argument. A traced drive amplitude is refused
by the serial and the batch path alike (AdEx: the drive validation names ``weight_nS``; LIF: a
ConcretizationTypeError when ``float()`` meets the traced amplitude). No documented gradient contract exists, so the
gradient tests assert that both paths refuse the same way, and they never invent a gradient.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne._construct_lif import lif_edge_arrays
from jaxfne._model_simulate import simulate_adex_seed_batch, simulate_lif_seed_batch

# AdEx fixture: the plumbing network of tests/test_adex_route.py (60 neurons, 50 ms at 0.1 ms).
ADEX_N = 60
ADEX_DT_MS = 0.1
ADEX_DUR_MS = 50.0
ADEX_DRIVE = {
    "n_sources": 560, "rate_hz": 4.0, "weight_nS": 2.5, "mechanism": "ampa",
    "target_cell_types": ["E", "PV"], "delay_ms": 2.0, "seed": 7,
}

# LIF fixture: one L4 layer of 6 cells (E and PV) with one AMPA edge type, as in
# tests/test_lif_tensor_route.py.
LIF_DT_MS = 0.5
LIF_DUR_MS = 20.0
LIF_N = 6
LIF_DRIVE = {"rate_hz": 50.0, "amplitude": 100.0}

V_RTOL = 0.0
V_ATOL_MV = 1e-4

BAD_SEEDS = [[], 5, [1.5], [True], [[1, 2]], "12", np.array([[1, 2], [3, 4]])]
BAD_SEED_IDS = ["empty", "scalar", "float", "bool", "nested", "string", "array2d"]


def _adex_cfg():
    cell_types = {"E": 0.8, "PV": 0.1, "SOM": 0.1}
    return (
        jtfne.Configuration()
        .network(name="adex60", n=ADEX_N, cell_types=cell_types)
        .emitter(family="adex", adex_decay="exp", adex_step_order="default")
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .mechanisms(name="gaba", kind="receptor", params={"tau_ms": 8.0, "reversal_mV": -80.0})
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=0.2, weight=0.5, mechanism="ampa", delay_ms=2.0,
                     sampling="bernoulli")
        .connections(name="e_pv", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=0.2, weight=1.0, mechanism="ampa", delay_ms=1.3,
                     sampling="bernoulli")
        .connections(name="pv_e", source={"cell_type": "PV"}, target={"cell_type": "E"},
                     probability=0.2, weight=-2.0, mechanism="gaba", delay_ms=0.25,
                     sampling="bernoulli")
        .connections(name="som_e", source={"cell_type": "SOM"}, target={"cell_type": "E"},
                     probability=0.2, weight=1.0, mechanism="gaba", delay_ms=0.8,
                     sampling="bernoulli")
        .connectivity(edge_seed=11)
    )


def _adex_sim(weight_nS=2.5, drive_seed=7, **sim_kwargs):
    """AdEx Simulation with the external drive; ``drive_seed`` is the drive's own seed."""
    drive = dict(ADEX_DRIVE, weight_nS=weight_nS, seed=drive_seed)
    return jtfne.Simulation(duration_ms=ADEX_DUR_MS, dt_ms=ADEX_DT_MS,
                            poisson_drive=drive, **sim_kwargs)


def _nt(*args, **kwargs):
    return nt.NeuronType.make(*args, **kwargs)


def _lif_tensor(local_inhibition=None):
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="E",
        mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0),
        delay_ms=1.0, local_inhibition=local_inhibition,
    )
    geometry = None
    if local_inhibition is not None:
        # One retinotopic point: all cells share one group, so the inhibition cap is live.
        geometry = nt.Geometry3D(x_range=(0.0, 0.0), y_range=(0.0, 0.0))
    layer = nt.Layer(
        name="L4", n_neurons=LIF_N,
        neuron_types=[_nt("E"), _nt("PV")],
        **({"geometry": geometry} if geometry is not None else {}),
    )
    return nt.NeuronalTensor(
        name="lif-f5",
        areas=[nt.Area(name="V1", layers=[layer], inter_connections=[conn])],
    )


def _lif_model(tensor=None):
    runtime = nt.RuntimeConfiguration(
        seed=0, dt_ms=LIF_DT_MS, emitter="lif", duration_ms=LIF_DUR_MS)
    return jtfne.construct(_lif_tensor() if tensor is None else tensor, runtime)


def _lif_sim(seed=None, poisson=None, **kwargs):
    """LIF Simulation with the Poisson drive; ``seed`` is the Simulation seed (serial path)."""
    base = dict(
        duration_ms=LIF_DUR_MS, dt_ms=LIF_DT_MS,
        poisson_drive=dict(LIF_DRIVE if poisson is None else poisson),
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"),
    )
    if seed is not None:
        base["seed"] = seed
    base.update(kwargs)
    return jtfne.Simulation(**base)


@pytest.fixture(scope="module")
def adex_model():
    return jtfne.construct(_adex_cfg())


@pytest.fixture(scope="module")
def lif_model():
    return _lif_model()


# --- AdEx: seed parity ---------------------------------------------------------------------

def test_adex_seed_batch_rows_match_serial_runs(adex_model):
    """Row k equals the serial run with drive seed seeds[k]: spikes bitwise, V_m within the bound.

    The seed order [7, 9, 8] is neither sorted nor reversal-symmetric, so a reversed or
    shifted batch puts the wrong seed in some row and fails."""
    seeds = [7, 9, 8]
    out = simulate_adex_seed_batch(adex_model, _adex_sim(), seeds)
    spk = np.asarray(out["spikes"])
    assert not np.array_equal(spk[0], spk[1]), "seeds 7 and 9 must give different spike trains"
    for k, s in enumerate(seeds):
        ref = adex_model.simulate(_adex_sim(drive_seed=s))
        r_spk = np.asarray(ref.spikes)
        assert int(r_spk.sum()) > 0, f"fixture must spike at seed {s}"
        assert np.array_equal(spk[k], r_spk), f"spike mismatch at row {k} (seed {s})"
        np.testing.assert_allclose(
            np.asarray(out["V_m"][k]), np.asarray(ref.V_m), rtol=V_RTOL, atol=V_ATOL_MV,
            err_msg=f"V_m mismatch at row {k} (seed {s})")


# --- AdEx: shapes --------------------------------------------------------------------------

def test_adex_seed_batch_shapes_lead_with_the_seed_count(adex_model):
    """Leading dimension is len(seeds); trailing dimensions equal the single-run shapes."""
    seeds = [7, 8, 9]
    serial = adex_model.simulate(_adex_sim(drive_seed=7))
    assert int(np.asarray(serial.spikes).sum()) > 0, "fixture must spike"
    single_spk = np.asarray(serial.spikes).shape
    single_v = np.asarray(serial.V_m).shape
    out = simulate_adex_seed_batch(adex_model, _adex_sim(), seeds)
    assert np.asarray(out["spikes"]).shape == (len(seeds),) + single_spk
    assert np.asarray(out["V_m"]).shape == (len(seeds),) + single_v
    # Per-neuron state (V, w, ref) and per-neuron-per-receptor conductance g (ampa, gaba).
    for key in ("V", "w", "ref"):
        assert np.asarray(out["final"][key]).shape == (len(seeds), ADEX_N), key
    assert np.asarray(out["final"]["g"]).shape == (len(seeds), ADEX_N, 2), "g"
    assert [int(s) for s in out["metadata"]["seeds"]] == seeds
    assert int(out["metadata"]["n_steps"]) == single_spk[0]

    one = simulate_adex_seed_batch(adex_model, _adex_sim(), [7])
    assert np.asarray(one["spikes"]).shape == (1,) + single_spk
    assert np.asarray(one["V_m"]).shape == (1,) + single_v


# --- AdEx: gradients -----------------------------------------------------------------------

def test_adex_seed_batch_grad_refused_like_serial(adex_model):
    """jax.grad of a V_m readout w.r.t. the drive weight: the serial and batch paths both refuse.

    The traced weight fails the drive validation (it names poisson_drive and weight_nS). The
    batch must refuse too, never return a gradient the serial path does not give."""
    ref = adex_model.simulate(_adex_sim(drive_seed=7))
    assert int(np.asarray(ref.spikes).sum()) > 0, "fixture must spike"

    def serial_loss(w):
        return jnp.sum(jnp.asarray(adex_model.simulate(_adex_sim(weight_nS=w, drive_seed=7)).V_m))

    def batch_loss(w):
        out = simulate_adex_seed_batch(adex_model, _adex_sim(weight_nS=w), [7, 8])
        return jnp.sum(jnp.asarray(out["V_m"]))

    with pytest.raises(ValueError, match="weight_nS"):
        jax.grad(serial_loss)(2.5)
    with pytest.raises(ValueError, match="weight_nS"):
        jax.grad(batch_loss)(2.5)


# --- AdEx: refusals ------------------------------------------------------------------------

@pytest.mark.parametrize("bad", BAD_SEEDS, ids=BAD_SEED_IDS)
def test_adex_seed_batch_refuses_bad_seed_lists(adex_model, bad):
    with pytest.raises(ValueError, match="non-empty sequence of int seeds"):
        simulate_adex_seed_batch(adex_model, _adex_sim(), bad)


def test_adex_seed_batch_refuses_a_lif_model(lif_model):
    with pytest.raises(ValueError, match="only realized for emitter family 'adex'"):
        simulate_adex_seed_batch(lif_model, _lif_sim(), [7, 8])


def test_adex_seed_batch_refuses_a_nondefault_simulation_seed(adex_model):
    with pytest.raises(ValueError, match="does not consume Simulation.seed"):
        simulate_adex_seed_batch(adex_model, _adex_sim(seed=3), [7, 8])


def test_adex_seed_batch_refuses_plasticity(adex_model):
    sim = _adex_sim(runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True))
    with pytest.raises(ValueError, match="not realized for plasticity on the adex family"):
        simulate_adex_seed_batch(adex_model, sim, [7, 8])


# --- LIF: seed parity ----------------------------------------------------------------------

def test_lif_seed_batch_rows_match_serial_runs(lif_model):
    """Row k equals the serial run with Simulation(seed=seeds[k]): spikes bitwise, V_m within
    the bound. Each serial run must spike, so a silent batch cannot pass."""
    seeds = [11, 13, 12]
    out = simulate_lif_seed_batch(lif_model, _lif_sim(), seeds)
    spk = np.asarray(out["spikes"])
    assert not np.array_equal(spk[0], spk[1]), "seeds 11 and 13 must give different spike trains"
    for k, s in enumerate(seeds):
        ref = lif_model.simulate(_lif_sim(seed=s))
        r_spk = np.asarray(ref.spikes)
        assert int(r_spk.sum()) > 0, f"fixture must spike at seed {s}"
        assert np.array_equal(spk[k], r_spk), f"spike mismatch at row {k} (seed {s})"
        np.testing.assert_allclose(
            np.asarray(out["V_m"][k]), np.asarray(ref.V_m), rtol=V_RTOL, atol=V_ATOL_MV,
            err_msg=f"V_m mismatch at row {k} (seed {s})")


# --- LIF: shapes ---------------------------------------------------------------------------

def test_lif_seed_batch_shapes_lead_with_the_seed_count(lif_model):
    """Leading dimension is len(seeds); spikes and V_m trail the single-run shape; w_final is
    (seeds, edges)."""
    seeds = [11, 12, 13]
    serial = lif_model.simulate(_lif_sim(seed=11))
    assert int(np.asarray(serial.spikes).sum()) > 0, "fixture must spike"
    single_spk = np.asarray(serial.spikes).shape
    assert single_spk[1] == LIF_N
    n_edges = int(np.asarray(lif_edge_arrays(lif_model)["weight"]).shape[0])
    assert n_edges > 0, "fixture must carry edges"
    out = simulate_lif_seed_batch(lif_model, _lif_sim(), seeds)
    assert np.asarray(out["spikes"]).shape == (len(seeds),) + single_spk
    assert np.asarray(out["V_m"]).shape == (len(seeds),) + single_spk
    assert np.asarray(out["w_final"]).shape == (len(seeds), n_edges)
    assert [int(s) for s in out["metadata"]["seeds"]] == seeds

    one = simulate_lif_seed_batch(lif_model, _lif_sim(), [11])
    assert np.asarray(one["spikes"]).shape == (1,) + single_spk


# --- LIF: gradients ------------------------------------------------------------------------

def test_lif_seed_batch_grad_refused_like_serial(lif_model):
    """jax.grad of a V_m readout w.r.t. the Poisson amplitude: the serial and batch paths both
    refuse with a concretization error (a traced amplitude has no concrete value)."""
    ref = lif_model.simulate(_lif_sim(seed=11))
    assert int(np.asarray(ref.spikes).sum()) > 0, "fixture must spike"

    def serial_loss(a):
        sim = _lif_sim(seed=11, poisson={"rate_hz": 50.0, "amplitude": a})
        return jnp.sum(jnp.asarray(lif_model.simulate(sim).V_m))

    def batch_loss(a):
        sim = _lif_sim(poisson={"rate_hz": 50.0, "amplitude": a})
        return jnp.sum(jnp.asarray(simulate_lif_seed_batch(lif_model, sim, [11, 12])["V_m"]))

    with pytest.raises(TypeError, match="concrete value"):
        jax.grad(serial_loss)(100.0)
    with pytest.raises(TypeError, match="concrete value"):
        jax.grad(batch_loss)(100.0)


# --- LIF: refusals -------------------------------------------------------------------------

@pytest.mark.parametrize("bad", BAD_SEEDS, ids=BAD_SEED_IDS)
def test_lif_seed_batch_refuses_bad_seed_lists(lif_model, bad):
    with pytest.raises(ValueError, match="non-empty sequence of int seeds"):
        simulate_lif_seed_batch(lif_model, _lif_sim(), bad)


def test_lif_seed_batch_refuses_an_adex_model(adex_model):
    with pytest.raises(ValueError, match="only realized for emitter family 'lif'"):
        simulate_lif_seed_batch(adex_model, _adex_sim(), [11, 12])


def test_lif_seed_batch_refuses_a_nondefault_simulation_seed(lif_model):
    with pytest.raises(ValueError, match="does not consume Simulation.seed"):
        simulate_lif_seed_batch(lif_model, _lif_sim(seed=3), [11, 12])


def test_lif_seed_batch_refuses_a_pinned_drive_seed(lif_model):
    """A template pinning poisson_drive['seed'] is refused: members ride the seeds argument."""
    sim = _lif_sim(poisson={"rate_hz": 50.0, "amplitude": 100.0, "seed": 5})
    with pytest.raises(ValueError, match="refuses a template pinning poisson_drive"):
        simulate_lif_seed_batch(lif_model, sim, [11, 12])


def test_lif_seed_batch_refuses_hdp_without_a_registered_rule(lif_model):
    sim = _lif_sim(runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list", enable_hdp=True))
    with pytest.raises(ValueError, match="registered hdp_rule"):
        simulate_lif_seed_batch(lif_model, sim, [11, 12])


def test_lif_seed_batch_refuses_non_whole_step_inhibition():
    """Event-order inhibition stays on serial simulate; the batch route names the resolution.

    ``event_scan`` is not used here: its spec carries no ``rounds``, and the tensor build
    then raises KeyError in jaxfne/connectivity.py (see the report), before the route runs."""
    model = _lif_model(_lif_tensor({"rounds": 1, "resolution": "event_order"}))
    with pytest.raises(ValueError, match="not realized for local_inhibition resolution"):
        simulate_lif_seed_batch(model, _lif_sim(), [11, 12])
