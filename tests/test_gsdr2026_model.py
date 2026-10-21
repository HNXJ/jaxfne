"""GSDR lane (Fig. 6 circuit) core-route tests: spec -> construct_hh_model -> kernel.

Reference values: artifacts/etudes/papers/gsdr2026/PORT_SPEC.md sections 1, 4-5
(36 E / 8 Ig / 6 Il; graded AMPA/GABAa; 1800 + 400 + 30 edges; local posts
``choice(PRNGKey(1), arange(50), (5,))``). CPU-only, float32, small grids.
"""

from __future__ import annotations

import numpy as np
import pytest

N_E, N_IG, N_IL, N_CELLS = 36, 8, 6, 50
DT_MS = 0.1


def _local_posts(seed=1):
    import jax
    import jax.numpy as jnp

    return np.asarray(
        jax.random.choice(jax.random.PRNGKey(int(seed)), jnp.arange(N_CELLS),
                          shape=(5,), replace=False)
    )


def _gsdr_spec(local_seed=1, edge_seed=0):
    """The 50-cell Fig. 6 circuit declared through the jaxfne public API only."""
    import jaxfne as jtfne

    posts = _local_posts(local_seed)
    cfg = (
        jtfne.Configuration()
        .network(name="gsdr50", n=N_CELLS,
                 cell_types={"E": N_E / N_CELLS, "Ig": N_IG / N_CELLS,
                             "Il": N_IL / N_CELLS})
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0, "tau_r_ms": 0.2,
                            "slope_mV": 5.0, "v_th_mV": -20.0, "s0": 0.1})
        .mechanisms(name="gaba_global", kind="receptor",
                    params={"tau_ms": 5.0, "reversal_mV": -80.0, "tau_r_ms": 0.5,
                            "slope_mV": 5.0, "v_th_mV": -20.0, "s0": 0.1})
        .mechanisms(name="gaba_local", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": -80.0, "tau_r_ms": 0.5,
                            "slope_mV": 5.0, "v_th_mV": -20.0, "s0": 0.1})
        .connections(name="e_all", source={"cell_type": "E"}, target={},
                     probability=None, weight=2.5, mechanism="ampa",
                     sign="excitatory")
        .connections(name="ig_all", source={"cell_type": "Ig"}, target={},
                     probability=None, weight=5.0, mechanism="gaba_global",
                     sign="inhibitory")
        .connections(name="il_local", source={"cell_type": "Il"},
                     target={"ids": [int(i) for i in posts.tolist()]},
                     probability=None, weight=5.0, mechanism="gaba_local",
                     sign="inhibitory")
    )
    return cfg.connectivity(edge_seed=int(edge_seed)), posts


def test_gsdr_population_counts():
    from jaxfne._construct_hh import construct_hh_model

    model = construct_hh_model(_gsdr_spec()[0])
    labels = model.static["hh"]["labels"]
    assert len(labels) == N_CELLS
    assert labels.count("E") == N_E
    assert labels.count("Ig") == N_IG
    assert labels.count("Il") == N_IL


def test_gsdr_wiring_counts():
    from jaxfne._construct_hh import construct_hh_model

    model = construct_hh_model(_gsdr_spec()[0])
    st = model.static["hh"]
    assert int(st["pre"].shape[0]) == N_E * N_CELLS + N_IG * N_CELLS + N_IL * 5
    rec = np.asarray(st["receptor"])
    assert int((rec == 0).sum()) == N_E * N_CELLS, "AMPA edges, got %r" % int((rec == 0).sum())
    assert int((rec == 1).sum()) == N_IG * N_CELLS, "global GABAa edges"
    assert int((rec == 2).sum()) == N_IL * 5, "local GABAa edges"


def test_gsdr_local_posts_match_reference_mask():
    posts = _local_posts(1)
    assert posts.shape == (5,) and len(set(posts.tolist())) == 5
    assert posts.min() >= 0 and posts.max() < N_CELLS
    from jaxfne._construct_hh import construct_hh_model

    model = construct_hh_model(_gsdr_spec(local_seed=1)[0])
    st = model.static["hh"]
    local_post = set(int(i) for i in np.asarray(st["post"])[np.asarray(st["receptor"]) == 2].tolist())
    assert local_post == set(int(i) for i in posts.tolist())


def test_gsdr_receptor_params():
    from jaxfne._construct_hh import construct_hh_model

    model = construct_hh_model(_gsdr_spec()[0])
    rp = model.static["hh"]["receptor_params"]
    assert rp["reversal_mV"] == (0.0, -80.0, -80.0)
    assert rp["tau_ms"] == (2.0, 5.0, 2.0)
    assert rp["tau_r_ms"] == (0.2, 0.5, 0.5)
    assert rp["slope_mV"] == (5.0, 5.0, 5.0)
    assert rp["v_th_mV"] == (-20.0, -20.0, -20.0)


def test_gsdr_kernel_args_run_a_driven_cell():
    import jax

    from jaxfne._construct_hh import construct_hh_model, hh_kernel_args
    from jaxfne.emitters_hh import simulate_hh

    import jaxfne as jtfne

    cfg = (
        jtfne.Configuration()
        .network(name="mini", n=3, cell_types={"E": 2.0 / 3.0, "Ig": 1.0 / 3.0})
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .connections(name="e_all", source={"cell_type": "E"}, target={},
                     probability=None, weight=4.0, mechanism="ampa")
    ).connectivity(edge_seed=0)
    model = construct_hh_model(cfg)
    kwargs = hh_kernel_args(model, DT_MS)
    assert set(kwargs) == {"params", "edge_pre", "edge_post", "edge_weight",
                           "edge_receptor", "receptor_e_rev", "receptor_tau_r",
                           "receptor_tau_d", "receptor_slope", "receptor_vth",
                           "receptor_s0"}
    n_steps = 3000
    i_ext = np.zeros((n_steps, 3), dtype=np.float32)
    i_ext[:, 0] = 10.0
    out = simulate_hh(**kwargs, n_steps=n_steps, dt_ms=DT_MS, I_ext=i_ext,
                      ou_key=jax.random.PRNGKey(0))
    assert np.asarray(out["spikes"]).shape == (n_steps, 3)
    assert int(np.asarray(out["spikes"])[:, 0].sum()) > 0, \
        "driven cell must spike (fixture must spike)"


def test_gsdr_refuses_delay():
    from jaxfne._construct_hh import construct_hh_model

    cfg, _ = _gsdr_spec()
    cfg.metadata["circuit"]["connections"][0]["delay_ms"] = 1.0
    with pytest.raises(ValueError, match="delay_ms"):
        construct_hh_model(cfg)


def test_gsdr_refuses_negative_weight():
    from jaxfne._construct_hh import construct_hh_model

    cfg, _ = _gsdr_spec()
    cfg.metadata["circuit"]["connections"][0]["weight"] = -1.0
    with pytest.raises(ValueError, match="weight"):
        construct_hh_model(cfg)


def test_gsdr_refuses_unknown_mechanism():
    from jaxfne._construct_hh import construct_hh_model

    cfg, _ = _gsdr_spec()
    cfg.metadata["circuit"]["connections"][0]["mechanism"] = "gabab"
    with pytest.raises(ValueError, match="undeclared mechanism"):
        construct_hh_model(cfg)


def test_gsdr_refuses_emitter_declaration():
    from jaxfne._construct_hh import construct_hh_model

    cfg, _ = _gsdr_spec()
    cfg = cfg.emitter(family="adex", adex_decay="euler")
    with pytest.raises(ValueError, match="no emitter"):
        construct_hh_model(cfg)


def test_gsdr_refuses_non_receptor_mechanism():
    from jaxfne._construct_hh import construct_hh_model

    cfg, _ = _gsdr_spec()
    cfg.metadata["circuit"]["mechanisms"][0]["kind"] = "current"
    with pytest.raises(ValueError, match="kind 'receptor'"):
        construct_hh_model(cfg)


def test_gsdr_kernel_args_refuse_bad_dt():
    from jaxfne._construct_hh import construct_hh_model, hh_kernel_args

    model = construct_hh_model(_gsdr_spec()[0])
    with pytest.raises(ValueError, match="dt_ms"):
        hh_kernel_args(model, 0.0)


def test_gsdr_run_trial_g_override_is_reference_us():
    """``g_override`` passes reference-uS values converted to densities."""
    import importlib.util
    from pathlib import Path

    import math

    path = (Path(__file__).resolve().parents[1] / "artifacts" / "etudes"
            / "papers" / "gsdr2026" / "run" / "run_gsdr.py")
    spec = importlib.util.spec_from_file_location("gsdr2026_run_gsdr_ovr", path)
    rg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rg)

    n_edges = rg.N_AMPA_EDGES + rg.N_GABAA_GLOBAL_EDGES + rg.N_GABAA_LOCAL_EDGES
    assert n_edges == 2230
    override = np.full((n_edges,), 5.0, dtype=np.float32)
    # Metrics need the 100-1400 ms PSD window, so the short trial is 1400 ms.
    result, blobs = rg.run_trial(0.6, g_seed=1, ou_seed=1, t_max_ms=1400.0,
                                 g_override=override)
    want = np.full((n_edges,), 5.0 * (1e5 / (80.0 * math.pi)), dtype=np.float32)
    np.testing.assert_allclose(blobs["g"], want, rtol=1e-5)
    assert np.asarray(blobs["v"]).shape == (14001, rg.N_CELLS)
    assert np.all(np.isfinite(result["rates_hz"]))
    with pytest.raises(ValueError, match="2230"):
        rg.run_trial(0.6, t_max_ms=1400.0,
                     g_override=np.ones((7,), dtype=np.float32))


def test_gsdr_kappa_of_onsets_matches_hand_value():
    """``kappa_of_onsets`` gives the hand-computed -1/3 on the hand-test matrix."""
    import importlib.util
    from pathlib import Path

    path = (Path(__file__).resolve().parents[1] / "artifacts" / "etudes"
            / "papers" / "gsdr2026" / "run" / "run_gsdr.py")
    spec = importlib.util.spec_from_file_location("gsdr2026_run_gsdr_kap", path)
    rg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rg)

    # Same onsets as tests/test_gsdr2026_kappa_hand.py: 10 steps at dt 1 ms,
    # 5 ms bins -> 2 bins with spike counts [2, 1] over 3 units.
    onsets = np.zeros((10, 3), dtype=bool)
    onsets[0, 0] = True
    onsets[5, 0] = True
    onsets[0, 1] = True
    got = rg.kappa_of_onsets(onsets, dt_ms=1.0, bin_ms=5.0)
    assert got == pytest.approx(-1.0 / 3.0, abs=1e-12), "got %r" % got
