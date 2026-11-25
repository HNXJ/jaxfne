"""R46: per-neuron H is inert when HDP is off; it drives state when HDP is on.

- Known-negative: HDP off, H seeded three ways (unseeded, all-ones, non-uniform)
  -> spikes and V_m bitwise identical; sources within rtol 1e-6 (GPU run-to-run,
  see the comment in the test); no HDP diagnostics.
- Known-positive: HDP on (legacy regime), the same two seeds -> outputs and H differ.
- Pure RBD: HDP on with K_HDP = K_w_ctrl = 0 and K_ctrl = 5 -> H evolves, weights fixed.
- Null controller gains: HDP on, K_HDP = K_w_ctrl = K_ctrl = 0 -> H stays at its seed.
- DEFAULT_HDP preset with K_HDP = K_w_ctrl = 0 (K_ctrl = 5) -> H evolves.
"""

import numpy as np

import jaxfne as jtfne

N = 20
LEGACY = {"K_HDP": 0.05, "K_ctrl": 0.3, "K_w_ctrl": 0.002, "alpha": 0.05,
          "gamma": 0.05, "C_spike": 0.02, "noise_scale": 0.0}
PURE_RBD = {"K_HDP": 0.0, "K_w_ctrl": 0.0, "K_ctrl": 5.0, "noise_scale": 0.0}
NULL_GAINS_RBD = {"K_HDP": 0.0, "K_w_ctrl": 0.0, "K_ctrl": 0.0, "noise_scale": 0.0}
DEFAULT_HDP_RBD = {**jtfne.DEFAULT_HDP, "K_HDP": 0.0, "K_w_ctrl": 0.0}


def _model():
    cfg = (
        jtfne.configuration()
        .runtime(seed=0, recurrent_backend="edge_list")
        .network(name="V1", kind="cortical_column", n=N, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _sim(hp=None):
    kw = {} if hp is None else {"enable_hdp": True, "hdp_params": hp}
    return jtfne.simulation(
        duration_ms=60.0, dt_ms=1.0, seed=3, record_sources=True, record_fields=False,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list", jit=False, **kw),
    )


def _seeded_pair(model):
    n = int(model.params["emitter"].n_neurons)
    return (model.with_hdp_initial_state(H0=np.ones(n)),
            model.with_hdp_initial_state(H0=np.linspace(0.2, 4.0, n)))


def test_h_is_inert_when_hdp_is_off():
    model = _model()
    ones, spread = _seeded_pair(model)
    base = jtfne.simulate(model, _sim())
    assert float(np.asarray(base.spikes).sum()) > 0, "fixture must spike"
    for seeded in (ones, spread):
        other = jtfne.simulate(seeded, _sim())
        assert seeded.last_hdp_diagnostics() is None
        assert np.array_equal(np.asarray(base.spikes), np.asarray(other.spikes))
        assert np.array_equal(np.asarray(base.V_m), np.asarray(other.V_m))
        # sources: bitwise identical on CPU (6 repeats x 3 H0, measured). On the default
        # GPU backend (RTX A4000) identical unseeded reruns already differ by up to
        # 4.8e-7 (2 of 1200 entries), so H is not the source of the gap.
        np.testing.assert_allclose(np.asarray(other.sources), np.asarray(base.sources),
                                   rtol=1e-6, atol=0)


def test_h_drives_state_when_hdp_is_on():
    ones, spread = _seeded_pair(_model())
    sig_a = jtfne.simulate(ones, _sim(LEGACY))
    sig_b = jtfne.simulate(spread, _sim(LEGACY))
    # t=0 differs by construction (the seeds); the claim is about the dynamics after it
    assert float(np.abs(np.asarray(sig_a.V_m) - np.asarray(sig_b.V_m)).max()) > 1e-3, (
        "HDP on must make the outputs depend on H")
    da, db = ones.last_hdp_diagnostics(), spread.last_hdp_diagnostics()
    assert float(np.abs(np.asarray(da["H_trace"])[1:] - np.asarray(db["H_trace"])[1:]).max()) > 1e-3
    w = np.asarray(da["w_trace"])
    assert float(np.abs(w - w[:1]).max()) > 0, "legacy HDP must move the weights (live counterpart)"


def test_pure_rbd_moves_h_and_freezes_weights():
    _, spread = _seeded_pair(_model())
    jtfne.simulate(spread, _sim(PURE_RBD))
    diag = spread.last_hdp_diagnostics()
    h = np.asarray(diag["H_trace"])
    w = np.asarray(diag["w_trace"])
    assert w.shape[1] > 0, "fixture must have edges"
    # compare each step with the seed: the seeded spread alone spans 3.8 at t=0
    assert float(np.abs(h - h[:1]).max()) > 1e-3, "H must evolve under RBD"
    assert np.array_equal(w, np.broadcast_to(w[:1], w.shape)), "weights must stay fixed"


def test_null_controller_gains_hold_h_at_its_seed_under_rbd():
    """Observed: with null controller gains (K_ctrl = K_w_ctrl = K_HDP = 0; K_ctrl = 0 is the
    RuntimeConfig default), H has no dynamics even with HDP engaged."""
    _, spread = _seeded_pair(_model())
    jtfne.simulate(spread, _sim(NULL_GAINS_RBD))
    h = np.asarray(spread.last_hdp_diagnostics()["H_trace"])
    assert np.array_equal(h, np.broadcast_to(h[:1], h.shape)), "null gains must freeze H"


def test_default_hdp_preset_moves_h_under_rbd():
    """Observed: the DEFAULT_HDP preset (K_ctrl = 5) moves H with K_HDP = K_w_ctrl = 0."""
    _, spread = _seeded_pair(_model())
    jtfne.simulate(spread, _sim(DEFAULT_HDP_RBD))
    h = np.asarray(spread.last_hdp_diagnostics()["H_trace"])
    assert float(np.abs(h - h[:1]).max()) > 1e-3, "DEFAULT_HDP's K_ctrl must move H"
