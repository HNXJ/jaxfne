"""Endogenous ephaptic field rule ``ephaptic_field`` (jaxfne/hdp_rules_ephaptic.py)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

import jaxfne  # noqa: F401  (registers the rule)
from jaxfne import neuronal_tensor as nt
from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered
from jaxfne._model_simulate import simulate_adex_seed_batch
from jaxfne._runtime_config import RuntimeConfig
from jaxfne.emitters import EdgeList, IzhikevichParams
from jaxfne.emitters_adex import AdExParams, make_adex_params, simulate_adex_hdp_registered
from jaxfne.emitters_hh import HHParams
from jaxfne.emitters_homeostatic_ei import HomeostaticEIParams
from jaxfne.emitters_lif import LIF_CELL_TYPE_DEFAULTS, make_lif_params, simulate_lif_edge
from jaxfne.emitters_rate import RateNetworkParams
from jaxfne.hdp_rule import check_hdp_rule_params
from jaxfne.hdp_rules_ephaptic import (
    EPHAPTIC_RULE_NAME,
    ephaptic_field,
    ephaptic_kernel,
    ephaptic_rule_params,
)
from jaxfne.pcl_topology import (
    EventScanDrive,
    build_s1_scan_tensor,
    event_scan_events,
    event_scan_tables,
    local_inhibition_blocks,
)

N = 40
N_STEPS = 400
DT = 0.1


def _dense(kernel, n):
    M = np.zeros((n, n))
    M[kernel["dst"], kernel["src"]] = kernel["weight"]
    return M


def test_kernel_geometry_and_normalization():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    raw = _dense(ephaptic_kernel(pos, sigma=1.0, cutoff=2.5, normalize="none"), 3)
    expected = np.zeros((3, 3))
    expected[0, 1] = expected[1, 0] = np.exp(-0.5)  # d = 1
    expected[1, 2] = expected[2, 1] = np.exp(-2.0)  # d = 2; d(0,2) = 3 > cutoff
    np.testing.assert_allclose(raw, expected, rtol=1e-12)
    row = _dense(ephaptic_kernel(pos, sigma=1.0, cutoff=2.5), 3)
    np.testing.assert_allclose(row.sum(axis=1), [1.0, 1.0, 1.0], rtol=1e-12)
    np.testing.assert_allclose(row[1], expected[1] / expected[1].sum(), rtol=1e-12)
    assert np.all(np.diag(row) == 0.0)
    lone = _dense(ephaptic_kernel(pos, sigma=1.0, cutoff=0.5), 3)
    assert lone.sum() == 0.0  # no pair within cutoff: rows stay 0, no division by 0


def test_kernel_underflowed_row_stays_zero():
    # d / sigma = 40: the pair is inside the cutoff but exp(-800) underflows to 0 in float64.
    pos = np.array([[0.0, 0.0], [40.0, 0.0]])
    raw = ephaptic_kernel(pos, sigma=1.0, cutoff=50.0, normalize="none")
    assert raw["weight"].size == 2 and np.all(raw["weight"] == 0.0), "fixture must underflow"
    row = ephaptic_kernel(pos, sigma=1.0, cutoff=50.0)
    assert np.all(np.isfinite(row["weight"]))
    assert np.all(row["weight"] == 0.0)


def test_kernel_coincident_positions_give_no_pair():
    # Neurons 0 and 1 share a position (d = 0); both lie at d = 1 from neuron 2.
    pos = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    M = _dense(ephaptic_kernel(pos, sigma=1.0, cutoff=2.0, normalize="none"), 3)
    assert M[0, 1] == 0.0 and M[1, 0] == 0.0
    np.testing.assert_allclose(M[2], [np.exp(-0.5), np.exp(-0.5), 0.0], rtol=1e-12)
    np.testing.assert_allclose(M[:2, 2], [np.exp(-0.5), np.exp(-0.5)], rtol=1e-12)


def test_raw_kernel_is_symmetric():
    rng = np.random.default_rng(2)
    pos = rng.uniform(0.0, 100.0, size=(N, 3))
    M = _dense(ephaptic_kernel(pos, sigma=15.0, cutoff=40.0, normalize="none"), N)
    assert np.count_nonzero(M) > 2 * N, "fixture must have several pairs"
    np.testing.assert_allclose(M, M.T, rtol=1e-12)


def test_kernel_is_permutation_equivariant():
    rng = np.random.default_rng(3)
    pos = rng.uniform(0.0, 100.0, size=(N, 3))
    perm = rng.permutation(N)
    assert not np.array_equal(perm, np.arange(N)), "fixture must permute the positions"
    M = _dense(ephaptic_kernel(pos, sigma=15.0, cutoff=40.0), N)
    assert np.count_nonzero(M) > 2 * N, "fixture must have several pairs"
    Mp = _dense(ephaptic_kernel(pos[perm], sigma=15.0, cutoff=40.0), N)
    P = np.eye(N)[perm]  # row i selects old neuron perm[i]: (P x)_i = x[perm[i]]
    np.testing.assert_allclose(Mp, P @ M @ P.T, rtol=1e-12)


def test_row_normalized_kernel_is_asymmetric():
    # Positions 0, 1, 3 on a line, sigma 1, all pairs in cutoff. Raw row sums are
    # 0.6176 (dst 0) and 0.7418 (dst 1), so row normalization divides the two
    # symmetric pair weights by different sums.
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    raw = _dense(ephaptic_kernel(pos, sigma=1.0, cutoff=5.0, normalize="none"), 3)
    np.testing.assert_allclose(raw, raw.T, rtol=1e-12)
    raw_rows = raw.sum(axis=1)
    assert not np.isclose(raw_rows[0], raw_rows[1]), "fixture rows must be uneven before normalization"
    M = _dense(ephaptic_kernel(pos, sigma=1.0, cutoff=5.0), 3)
    assert abs(M[0, 1] - M[1, 0]) > 0.1, "row normalization must break symmetry (known positive)"


def test_single_neuron_gives_empty_kernel():
    k = ephaptic_kernel(np.zeros((1, 3)), sigma=1.0, cutoff=1.0)
    assert k["src"].size == 0 and k["dst"].size == 0 and k["weight"].size == 0
    assert _dense(k, 1)[0, 0] == 0.0


def test_all_isolated_points_give_empty_kernel():
    # Five points 10 apart on a line; cutoff 2 admits no pair.
    pos = np.column_stack([np.arange(5) * 10.0, np.zeros(5)])
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    assert D[~np.eye(5, dtype=bool)].min() > 2.0, "fixture must be isolated"
    for normalize in ("none", "row"):
        k = ephaptic_kernel(pos, sigma=1.0, cutoff=2.0, normalize=normalize)
        assert k["weight"].size == 0, f"normalize={normalize!r} kept a pair"
        assert np.all(_dense(k, 5) == 0.0)


def test_field_matches_dense_product():
    rng = np.random.default_rng(1)
    pos = rng.uniform(0.0, 100.0, size=(N, 3))
    k = ephaptic_kernel(pos, sigma=15.0, cutoff=40.0)
    assert 0 < len(k["src"]) < N * (N - 1), "fixture must be sparse and non-empty"
    v = rng.normal(-60.0, 5.0, size=N)
    phi = ephaptic_field(jnp.asarray(v), src=k["src"], dst=k["dst"], weight=k["weight"],
                         alpha=0.3, v_rest=-65.0)
    # float32 kernel: the sum over neighbours of (v_j + 65) cancels, so the bound is absolute.
    np.testing.assert_allclose(np.asarray(phi), 0.3 * _dense(k, N) @ (v + 65.0), rtol=1e-5, atol=1e-5)


def test_params_fail_closed():
    with pytest.raises(ValueError, match="unrecognized"):
        check_hdp_rule_params(EPHAPTIC_RULE_NAME, {"alpah": 1.0})
    with pytest.raises(ValueError, match="kernel lacks"):
        ephaptic_rule_params({"src": []}, alpha=1.0, v_rest=0.0)
    with pytest.raises(ValueError, match="v_rest"):
        _lif_run(EPHAPTIC_RULE_NAME, {"alpha": 0.1})


def _network():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 100.0, size=(N, 3))
    pre, post = np.nonzero(rng.random((N, N)) < 0.1)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    weight = np.where(rng.random(pre.size) < 0.8, 2.0, -4.0)
    I_ext = rng.uniform(0.0, 2.0, size=(N_STEPS, N)).astype(np.float64)
    return pos, pre, post, weight, I_ext


def _lif_run(rule, rule_params):
    _, pre, post, weight, I_ext = _network()
    return simulate_lif_edge(
        make_lif_params(("simple",) * N), n_steps=N_STEPS, dt_ms=DT,
        edge_pre=pre, edge_post=post, edge_weight=weight,
        edge_delay_steps=np.ones(pre.size, dtype=int), I_ext_pA=jnp.asarray(I_ext),
        record_v=True, hdp_rule=rule, hdp_rule_params=rule_params,
    )


def _lif_eph(alpha):
    pos = _network()[0]
    k = ephaptic_kernel(pos, sigma=15.0, cutoff=40.0)
    return _lif_run(EPHAPTIC_RULE_NAME, ephaptic_rule_params(k, alpha=alpha, v_rest=0.0)), k


def test_lif_zero_coupling_equals_no_rule():
    base = _lif_run(None, None)
    eph, _ = _lif_eph(0.0)
    assert np.asarray(base["spikes"]).sum() > 0, "fixture must spike"
    np.testing.assert_array_equal(np.asarray(eph["spikes"]), np.asarray(base["spikes"]))
    np.testing.assert_array_equal(np.asarray(eph["V_mV"]), np.asarray(base["V_mV"]))
    assert np.all(np.asarray(eph["b_final"]) == 0.0)
    # Mutant: a nonzero alpha (0.05 moves V_m by about 30 mV in this fixture) must change V_m,
    # or the identity above cannot fail.
    mutant, _ = _lif_eph(0.05)
    assert not np.array_equal(np.asarray(mutant["V_mV"]), np.asarray(base["V_mV"])), (
        "mutant alpha left V_mV unchanged; the V_m identity cannot fail"
    )


def test_lif_field_is_generated_by_activity():
    eph, k = _lif_eph(0.05)
    base = _lif_run(None, None)
    V_last = np.asarray(eph["V_mV"])[-1]
    expected = 0.05 * _dense(k, N) @ V_last
    # The LIF kernel runs in float32.
    np.testing.assert_allclose(np.asarray(eph["aux_final"]), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(eph["b_final"]), np.asarray(eph["aux_final"]))
    assert np.abs(expected).max() > 0, "fixture must produce a nonzero field"
    assert not np.array_equal(np.asarray(eph["spikes"]), np.asarray(base["spikes"])), (
        "a nonzero field must change the spikes in this fixture"
    )


@pytest.mark.parametrize("alpha", [-0.05, 0.02, 0.1])
def test_lif_coupling_sweep_finite_and_reproducible(alpha):
    a, _ = _lif_eph(alpha)
    b, _ = _lif_eph(alpha)
    assert np.all(np.isfinite(np.asarray(a["V_mV"])))
    np.testing.assert_array_equal(np.asarray(a["V_mV"]), np.asarray(b["V_mV"]))


def _izh_params(n):
    return IzhikevichParams(
        a=jnp.full((n,), 0.02), b=jnp.full((n,), 0.2), c=jnp.full((n,), -65.0),
        d=jnp.full((n,), 8.0), drive=jnp.full((n,), 5.0), sign=jnp.ones((n,)),
        W=jnp.zeros((n, n)), v0=jnp.full((n,), -65.0), u0=jnp.full((n,), -13.0),
        source_scale=jnp.ones((n,)), labels=("E",) * n, layer_labels=("L4",) * n,
        source_calibration_status="x",
    )


def _izh_run(alpha):
    pos, pre, post, _, _ = _network()
    edges = EdgeList(
        pre=jnp.asarray(pre, jnp.int32), post=jnp.asarray(post, jnp.int32),
        weight=jnp.full((pre.size,), 0.5), receptor_index=jnp.zeros((pre.size,), jnp.int32),
        tau_ms=jnp.full((pre.size,), 2.0), source_calibration_status="x",
    )
    k = ephaptic_kernel(pos, sigma=15.0, cutoff=40.0)
    return simulate_edge_recurrent_izhikevich_hdp_registered(
        _izh_params(N), edges, N_STEPS, DT, jax.random.PRNGKey(0),
        hdp_rule=EPHAPTIC_RULE_NAME,
        hdp_rule_params=ephaptic_rule_params(k, alpha=alpha, v_rest=-65.0),
    )


def test_izhikevich_field_changes_activity_and_zero_is_inert():
    _, s0, _, d0 = _izh_run(0.0)
    _, s1, _, d1 = _izh_run(0.02)
    assert np.asarray(s0).sum() > 0, "fixture must spike"
    assert np.all(np.asarray(d0["aux_final"]) == 0.0)
    assert np.abs(np.asarray(d1["aux_final"])).max() > 0
    assert not np.array_equal(np.asarray(s0), np.asarray(s1))


# R14 (Solmind plan): what alpha means across dt on the LIF route.
LIF_ALPHA_REF_PER_MS = 0.1  # a: reference coupling rate, 1/ms, fixed across dt
LIF_T_FINAL_MS = 2.0
LIF_V0_MV = np.array([5.0, -3.0])  # v != v_rest (0 mV, the LIF leak target); zero drive
LIF_DT_MS = (0.1, 0.05, 0.025, 0.0125)
LIF_NEG_CONTROL_DT_MS = 0.1  # negative control holds alpha = a * 0.1 ms at every dt
LIF_TAU_MS = LIF_CELL_TYPE_DEFAULTS["simple"]["tau_m_ms"]
EPS32 = float(np.finfo(np.float32).eps)


def _lif_pair_run(dt, alpha, kernel):
    """Voltage trace (n_steps, 2) in mV and spike count of the N=2 LIF pair with the rule."""
    n_steps = int(round(LIF_T_FINAL_MS / dt))
    none = np.zeros(0, dtype=np.int64)
    out = simulate_lif_edge(
        make_lif_params(("simple",) * 2), n_steps=n_steps, dt_ms=dt,
        edge_pre=none, edge_post=none, edge_weight=np.zeros(0), edge_delay_steps=none,
        I_ext_pA=jnp.zeros((n_steps, 2)), V0_mV=jnp.asarray(LIF_V0_MV), record_v=True,
        hdp_rule=EPHAPTIC_RULE_NAME,
        hdp_rule_params=ephaptic_rule_params(kernel, alpha=alpha, v_rest=0.0),
    )
    return np.asarray(out["V_mV"], dtype=np.float64), int(np.asarray(out["spikes"]).sum())


def _lif_lag_map(dt, alpha, M):
    """The kernel's discrete map: exact leak c = exp(-dt/tau), no field at step 1."""
    c = np.exp(-dt / LIF_TAU_MS)
    G = c * np.eye(2) + alpha * M
    x = c * LIF_V0_MV
    for _ in range(int(round(LIF_T_FINAL_MS / dt)) - 1):
        x = G @ x
    return x


def test_lif_alpha_is_per_step_coupling_across_dt():
    """One alpha, one coupling across dt on the LIF route, with positive and negative controls.

    Fixture: N=2 "simple" PCL LIF pair (tau 18 ms, v_thresh 10 mV, v_min -20 mV), v_rest 0 mV
    (the leak target), zero external drive, V(0) = (5, -3) mV, no edges, T = 2 ms. Two points at
    distance 1 with cutoff 2, row-normalized: M = [[0, 1], [1, 0]], M^2 = I.

    Reference: dv/dt = -v/tau + a M v, a = 0.1 /ms fixed. Exact: x(T) = expm(T(-I/tau + aM)) x(0)
    in float64; with M^2 = I this is e^{-T/tau} (cosh(aT) I + sinh(aT) M) x(0).

    Kernel map. lif_step decays V exactly: c = exp(-dt/tau). The field alpha M V is computed from
    the voltages after step k-1 and enters step k (one-step lag). The bias carry starts at zero,
    so step 1 carries no field:
        V(1) = c V(0),    V(k) = (c I + alpha M) V(k-1)  for k >= 2.
    The test asserts that the float32 kernel equals this map, per arm, to 100 eps32 max|V|.

    Expansion. Positive arm, alpha = a dt. For eigenmode m = +-1 of M:
        numeric / exact = 1 + dt [T (m a/tau - a^2/2) - m a] + O(dt^2).
    The -m a dt term is the field lost at step 1. The T(...) term is the one-step lag with
    forward-Euler coupling; the leak is exact. Both are O(dt), so the error halves per halving and
    the ratio tends to 2. The window is applied to the two finest halvings.
    Negative arm, alpha = 0.1 a held: the mode factor is c + 0.1 m a, so the realized rate
    0.1 a / dt doubles per halving and the error against the a solution grows (ratio < 1).

    Guards: a dt < 0.02 and dt/tau < 0.01 at the coarsest dt; no spike in any run; the finest
    positive error exceeds 100 eps32 max|V|.
    """
    assert LIF_ALPHA_REF_PER_MS * max(LIF_DT_MS) < 0.02 and max(LIF_DT_MS) / LIF_TAU_MS < 0.01, (
        "fixture must be in the small-step regime (a dt, dt/tau small)"
    )
    kernel = ephaptic_kernel(np.array([[0.0, 0.0], [1.0, 0.0]]), sigma=1.0, cutoff=2.0)
    M = _dense(kernel, 2)
    assert np.count_nonzero(M) == 2, "fixture must be a coupled pair"
    exact = expm(LIF_T_FINAL_MS * (-np.eye(2) / LIF_TAU_MS + LIF_ALPHA_REF_PER_MS * M)) @ LIF_V0_MV
    runs = {}
    for dt in LIF_DT_MS:
        arms = (("pos", LIF_ALPHA_REF_PER_MS * dt),
                ("neg", LIF_ALPHA_REF_PER_MS * LIF_NEG_CONTROL_DT_MS))
        for arm, alpha in arms:
            V, spikes = _lif_pair_run(dt, alpha, kernel)
            assert spikes == 0, f"fixture must not spike: {arm} arm, dt={dt}, {spikes} spikes"
            runs[arm, dt] = (alpha, V)
    v_scale = max(float(np.max(np.abs(V))) for _, V in runs.values())
    tol = 100 * EPS32 * v_scale  # float32 resolution scale of the kernel's voltages
    for (arm, dt), (alpha, V) in runs.items():
        np.testing.assert_allclose(V[-1], _lif_lag_map(dt, alpha, M), atol=tol,
                                   err_msg=f"{arm} arm, dt={dt}: kernel is not the lag map")
    err = {arm: [float(np.max(np.abs(runs[arm, dt][1][-1] - exact))) for dt in LIF_DT_MS]
           for arm in ("pos", "neg")}
    assert err["pos"][-1] > tol, (
        f"finest positive error {err['pos'][-1]:.3e} is not above float32 resolution {tol:.3e}"
    )
    ratio = {arm: [err[arm][i] / err[arm][i + 1] for i in (1, 2)] for arm in ("pos", "neg")}
    assert all(1.8 <= r <= 2.2 for r in ratio["pos"]), (
        f"positive-control ratios {ratio['pos']} (0.05/0.025, 0.025/0.0125) outside [1.8, 2.2]"
    )
    assert all(r < 1.0 for r in ratio["neg"]), (
        f"negative-control ratios {ratio['neg']} (0.05/0.025, 0.025/0.0125) not below 1"
    )


# R12 refusal half: the ephaptic request on each simulation route that does not realize HDP.
# Refusing routes assert the refusal; routes that drop the request silently are xfail(strict)
# until they refuse (P-069).


def _eph_request(n, alpha=0.02):
    """``hdp_params`` carrying the ephaptic rule for ``n`` neurons (fixed positions)."""
    pos = np.random.default_rng(0).uniform(0.0, 100.0, size=(n, 3))
    k = ephaptic_kernel(pos, sigma=15.0, cutoff=40.0)
    return {"hdp_rule": EPHAPTIC_RULE_NAME,
            "hdp_rule_params": ephaptic_rule_params(k, alpha=alpha, v_rest=-65.0)}


def _adex_model(n=20):
    model = jaxfne.construct(
        jaxfne.Configuration()
        .network(name="adex", n=n, cell_types={"E": 0.8, "PV": 0.2})
        .emitter(family="adex", adex_decay="exp")
        .mechanisms(name="ampa", kind="receptor", params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=0.2, weight=0.5, mechanism="ampa", delay_ms=2.0,
                     sampling="bernoulli")
        .connectivity(edge_seed=11)
    )
    assert isinstance(model.params["emitter"], AdExParams), "fixture must be the adex family"
    return model


def _hh_model():
    model = jaxfne.construct(
        jaxfne.configuration()
        .network(n=10, cell_types={"E": 0.6, "Ig": 0.2, "Il": 0.2})
        .emitter(family="hh")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0, "tau_r_ms": 0.2,
                            "slope_mV": 5.0, "v_th_mV": -20.0, "s0": 0.1})
        .connections(name="e_all", source={"cell_type": "E"}, target={},
                     probability=None, weight=2.5, mechanism="ampa", sign="excitatory")
        .connectivity(edge_seed=0)
    )
    assert isinstance(model.params["emitter"], HHParams), "fixture must be the hh family"
    assert "morphology" not in model.static["hh"], "fixture must be single-compartment"
    return model


def _rate_model(n=4):
    types = [nt.NeuronType.make("E", fraction=0.5, emitter="rate"),
             nt.NeuronType.make("I", fraction=0.5, emitter="rate")]
    conn = nt.InterConnection(
        source_layer="L4", source_neuron_type="E", target_layer="L4", target_neuron_type="I",
        mechanism="AMPA", static=nt.StaticParams(dT_ms=2.0), delay_ms=1.0)
    tensor = nt.NeuronalTensor(
        name="rate-test",
        areas=[nt.Area(name="V1",
                       layers=[nt.Layer(name="L4", n_neurons=n, neuron_types=types)],
                       inter_connections=[conn])])
    model = jaxfne.construct(tensor, nt.RuntimeConfiguration(
        seed=0, duration_ms=100.0, dt_ms=0.5, emitter="rate"))
    assert isinstance(model.params["emitter"], RateNetworkParams), "fixture must be the rate family"
    return model


def _hei_model():
    model = jaxfne.construct(
        jaxfne.configuration().network(n=4).emitter(family="homeostatic_ei").field()
        .probe(name="p"))
    assert isinstance(model.params["emitter"], HomeostaticEIParams), "fixture must be homeostatic_ei"
    return model


def _izh_model():
    model = jaxfne.construct(
        jaxfne.configuration()
        .network(name="V1", kind="cortical_column", n=6, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann",
               gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    assert isinstance(model.params["emitter"], IzhikevichParams), "fixture must be izhikevich"
    return model


def _lif_scan_model():
    """Tensor LIF model with an event-scan local inhibition, and its EventScanDrive."""
    w_s = np.zeros((2, 2, 1, 1), dtype=np.float64)
    w_s[0, 0, 0, 0] = 40.0
    w_s[0, 1, 0, 0] = 4.0
    tables = event_scan_tables(
        nx=1, ny=1, nd=2, cx=1, cy=1, cd=1,
        origins_x=[0], origins_y=[0], size=(1, 1), weights=w_s,
        c_cover={(0, 0): [(0, 0)]}, c_ox=[0], c_oy=[0], c_size=(1, 1),
        c_weights=np.array([[[[40.0, 4.0]]]], dtype=np.float64),
        vw=1, vh=1, inhib_range=(0, 0), lateral_on=False,
    )
    zeros = np.zeros(2, dtype=np.int64)
    events = event_scan_events(
        np.array([100, 200]), zeros, zeros, zeros, dt_ms=0.5, n_steps=1, vw=1, vh=1)
    groups = np.array([0, 0, 1], dtype=np.int64)
    blocks = local_inhibition_blocks(
        groups, li_pre=np.zeros(0, dtype=np.int64), li_post=np.zeros(0, dtype=np.int64),
        li_weight=np.zeros(0, dtype=np.float64))
    drive = EventScanDrive(events, tables, blocks, groups, dt_ms=0.5, n_steps=1)
    model = jaxfne.construct(
        build_s1_scan_tensor(ns=2, nc=1),
        nt.RuntimeConfiguration(seed=0, dt_ms=0.5, duration_ms=0.5, emitter="lif"))
    assert model.static["lif"]["local_inhibition"][0]["resolution"] == "event_scan", (
        "fixture must be the event-scan route")
    return model, drive


def test_lif_event_scan_refuses_ephaptic_request():
    model, drive = _lif_scan_model()
    assert isinstance(drive, EventScanDrive), "fixture must pass an EventScanDrive paradigm"
    sim = jaxfne.Simulation(duration_ms=0.5, dt_ms=0.5, seed=1, runtime=RuntimeConfig(
        recurrent_backend="edge_list", enable_hdp=True, hdp_params=_eph_request(3)))
    with pytest.raises(ValueError, match="not realized with local_inhibition resolution 'event_scan'"):
        model.simulate(sim, paradigm=drive)


def test_adex_model_refuses_ephaptic_request():
    model = _adex_model()
    sim = jaxfne.Simulation(duration_ms=5.0, dt_ms=0.1, seed=0, runtime=RuntimeConfig(
        recurrent_backend="edge_list", enable_hdp=True, hdp_params=_eph_request(20)))
    with pytest.raises(ValueError, match="enable_hdp"):
        model.simulate(sim)


def test_adex_registered_kernel_refuses_drive_bias():
    n = 6
    params = make_adex_params(("E",) * n)
    assert isinstance(params, AdExParams), "fixture must be the adex family"
    request = _eph_request(n)
    with pytest.raises(ValueError, match="needing state the 'adex' family lacks"):
        simulate_adex_hdp_registered(
            params, n_steps=100, dt_ms=0.1,
            edge_pre=np.array([0, 1, 2]), edge_post=np.array([1, 2, 0]),
            edge_weight_nS=np.array([1.0, 1.0, 1.0]), edge_receptor=np.zeros(3, dtype=int),
            edge_delay_steps=np.ones(3, dtype=int),
            receptor_reversal_mV=(0.0,), receptor_tau_ms=(5.0,),
            decay="exp", I_ext_pA=np.full(n, 300.0),
            ext_targets=[0], ext_lambda=1.0, ext_weight_nS=1.0, ext_receptor=0,
            ext_delay_steps=1, key=jax.random.PRNGKey(0),
            hdp_rule=request["hdp_rule"], hdp_rule_params=request["hdp_rule_params"])


def test_hh_model_refuses_ephaptic_request():
    model = _hh_model()
    sim = jaxfne.Simulation(duration_ms=5.0, dt_ms=0.1, seed=0, runtime=RuntimeConfig(
        enable_hdp=True, hdp_params=_eph_request(10)))
    with pytest.raises(ValueError, match="not realized for plasticity on the hh family"):
        model.simulate(sim)


def test_rate_model_refuses_ephaptic_request():
    model = _rate_model(4)
    sim = jaxfne.Simulation(duration_ms=100.0, dt_ms=0.5, seed=0, runtime=RuntimeConfig(
        recurrent_backend="edge_list", enable_hdp=True, hdp_params=_eph_request(4)))
    with pytest.raises(ValueError, match="not realized for emitter family 'rate'"):
        model.simulate(sim)


def test_adex_seed_batch_refuses_ephaptic_request():
    model = _adex_model()
    sim = jaxfne.Simulation(duration_ms=5.0, dt_ms=0.1, seed=0, runtime=RuntimeConfig(
        recurrent_backend="edge_list", enable_hdp=True, hdp_params=_eph_request(20)))
    with pytest.raises(ValueError, match="not realized for plasticity on the adex family"):
        simulate_adex_seed_batch(model, sim, [0, 1])


def test_homeostatic_ei_refuses_ephaptic_request():
    model = _hei_model()
    sim = jaxfne.Simulation(duration_ms=10.0, dt_ms=0.1, seed=0, runtime=RuntimeConfig(
        enable_hdp=True, hdp_params=_eph_request(4)))
    with pytest.raises(ValueError, match="is not realized on the homeostatic_ei route"):
        model.simulate(sim)


def test_izhikevich_refuses_hdp_params_without_enable_hdp():
    model = _izh_model()
    n = int(np.asarray(model.params["emitter"].a).shape[0])
    sim = jaxfne.Simulation(duration_ms=10.0, dt_ms=0.1, seed=0, runtime=RuntimeConfig(
        enable_hdp=False, hdp_params=_eph_request(n)))
    msg = r"\['hdp_rule', 'hdp_rule_params'\] are not realized with enable_hdp=False"
    with pytest.raises(ValueError, match=msg):
        model.simulate(sim)
    with pytest.raises(ValueError, match=msg):
        model.simulate(sim, return_state=True)


def test_izhikevich_keeps_noise_scale_without_enable_hdp():
    model = _izh_model()
    sim = jaxfne.Simulation(duration_ms=10.0, dt_ms=0.1, seed=0, runtime=RuntimeConfig(
        enable_hdp=False, hdp_params={"noise_scale": 0.0}))
    assert model.simulate(sim) is not None
    assert model.simulate(sim, return_state=True) is not None


# R14 (Solmind plan), Izhikevich half: what alpha means across dt on the registered-HDP route.
IZH_ALPHA_REF_PER_MS = 0.1  # a: reference coupling rate, 1/ms, fixed across dt (input enters dv/dt)
IZH_T_FINAL_MS = 2.0
IZH_V0_MV = np.array([-60.0, -70.0])  # v - v_rest = (+5, -5) mV; zero drive
IZH_V_REST_MV = -65.0  # the rule's v_rest; the Izhikevich rest for these parameters is -70 mV
IZH_DT_MS = (0.1, 0.05, 0.025, 0.0125)
IZH_NEG_CONTROL_DT_MS = 0.1  # negative control holds alpha = a * 0.1 ms / dt at every dt


def _izh_pair_params():
    n = 2
    return IzhikevichParams(
        a=jnp.full((n,), 0.02), b=jnp.full((n,), 0.2), c=jnp.full((n,), -65.0),
        d=jnp.full((n,), 8.0), drive=jnp.zeros((n,)), sign=jnp.ones((n,)),
        W=jnp.zeros((n, n)), v0=jnp.asarray(IZH_V0_MV), u0=0.2 * jnp.asarray(IZH_V0_MV),
        source_scale=jnp.ones((n,)), labels=("E",) * n, layer_labels=("L4",) * n,
        source_calibration_status="x",
    )


def _izh_pair_run(dt, alpha, kernel):
    """Voltage trace (n_steps, 2) in mV and spike count of the N=2 Izhikevich pair with the rule."""
    n_steps = int(round(IZH_T_FINAL_MS / dt))
    no_edges = EdgeList(
        pre=jnp.zeros((0,), jnp.int32), post=jnp.zeros((0,), jnp.int32),
        weight=jnp.zeros((0,)), receptor_index=jnp.zeros((0,), jnp.int32),
        tau_ms=jnp.zeros((0,)), source_calibration_status="x",
    )
    V, spikes, _, _ = simulate_edge_recurrent_izhikevich_hdp_registered(
        _izh_pair_params(), no_edges, n_steps, dt, jax.random.PRNGKey(0),
        dtype="float64", noise_scale=0.0,
        hdp_rule=EPHAPTIC_RULE_NAME,
        hdp_rule_params=ephaptic_rule_params(kernel, alpha=alpha, v_rest=IZH_V_REST_MV),
    )
    return np.asarray(V, dtype=np.float64), int(np.asarray(spikes).sum())


@pytest.mark.x64
def test_izhikevich_alpha_is_rate_coupling_across_dt():
    """One alpha, one coupling across dt on the Izhikevich registered-HDP route, with controls.

    Fixture: N=2 Izhikevich pair (a 0.02, b 0.2, c -65, d 8), zero drive, no synaptic edges,
    noise_scale 0 (None draws noise of scale 0.5 at 0.5 ms with shape (n_steps, N), so runs at
    different dt would not share a noise path). V(0) = (-60, -70) mV, u(0) = 0.2 V(0): v - v_rest
    = (+5, -5) mV at v_rest -65 mV. Two points at distance 1, cutoff 2, row-normalized:
    M = [[0, 1], [1, 0]]. T = 2 ms; float64 (x64 marker) keeps float32 rounding out of the ratio.

    Reference: the same kernel with alpha = a = 0.1 /ms at dt_ref = dt_finest / 8 (Richardson).
    The input enters dv/dt, so alpha is a rate: alpha = a at every dt is one coupling.
    Positive arm: alpha = a. Negative arm: alpha = a * 0.1 ms / dt, so the realized rate doubles
    per halving.

    Expansion. The integrator is forward Euler, v <- v + dt dv (_hdp_registrable_kernel.py lines
    389, 428, 567, 608), so it is first order. The field set from the voltages entering step t
    reaches the current of step t + 1 (module docstring, Timing): a fixed number of steps, so the
    lag error is also O(dt). Against the reference, err(dt) = |C (dt - dt_ref)| + O(dt^2), and to
    leading order the ratio of the two finest halvings is (dt - dt_ref) / (dt/2 - dt_ref) = 2.07
    (0.05/0.025) and 2.14 (0.025/0.0125); the window [1.8, 2.2] holds both. Negative arm: the
    realized rate doubles per halving, the error against the a solution grows, and the ratio is
    below 1.

    Guards: a dt < 0.02 at the coarsest dt; no spike in any run; v - v_rest nonzero at t = 0; the
    finest positive error exceeds 100 eps32 max|V|.
    """
    assert IZH_ALPHA_REF_PER_MS * max(IZH_DT_MS) < 0.02, "fixture must be in the small-step regime"
    kernel = ephaptic_kernel(np.array([[0.0, 0.0], [1.0, 0.0]]), sigma=1.0, cutoff=2.0)
    M = _dense(kernel, 2)
    assert np.count_nonzero(M) == 2, "fixture must be a coupled pair"
    assert np.all(IZH_V0_MV != IZH_V_REST_MV), "fixture must put v - v_rest nonzero at t = 0"
    dt_ref = IZH_DT_MS[-1] / 8
    ref_V, ref_spikes = _izh_pair_run(dt_ref, IZH_ALPHA_REF_PER_MS, kernel)
    assert ref_spikes == 0, f"reference must not spike: {ref_spikes} spikes at dt={dt_ref}"
    runs = {}
    for dt in IZH_DT_MS:
        arms = (("pos", IZH_ALPHA_REF_PER_MS),
                ("neg", IZH_ALPHA_REF_PER_MS * IZH_NEG_CONTROL_DT_MS / dt))
        for arm, alpha in arms:
            V, spikes = _izh_pair_run(dt, alpha, kernel)
            assert spikes == 0, f"fixture must not spike: {arm} arm, dt={dt}, {spikes} spikes"
            runs[arm, dt] = (alpha, V)
    v_scale = max([float(np.max(np.abs(ref_V)))]
                  + [float(np.max(np.abs(V))) for _, V in runs.values()])
    tol = 100 * EPS32 * v_scale  # float32 resolution scale of the voltages
    err = {arm: [float(np.max(np.abs(runs[arm, dt][1][-1] - ref_V[-1]))) for dt in IZH_DT_MS]
           for arm in ("pos", "neg")}
    assert err["pos"][-1] > tol, (
        f"finest positive error {err['pos'][-1]:.3e} is not above float32 resolution {tol:.3e}"
    )
    ratio = {arm: [err[arm][i] / err[arm][i + 1] for i in (1, 2)] for arm in ("pos", "neg")}
    assert all(1.8 <= r <= 2.2 for r in ratio["pos"]), (
        f"positive-control ratios {ratio['pos']} (0.05/0.025, 0.025/0.0125) outside [1.8, 2.2]"
    )
    assert all(r < 1.0 for r in ratio["neg"]), (
        f"negative-control ratios {ratio['neg']} (0.05/0.025, 0.025/0.0125) not below 1"
    )
