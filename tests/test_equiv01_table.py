"""24-EQUIV-01: per-observable equivalence table, jit-vs-eager.

Strongest justified relation per cell (measured, not assumed):
- spikes: exact, all regimes.
- V_m: d<=1e-4 on baseline and HDP paths (baseline worst observed
  3.9e-05 after the 0.5.3 P-010 chain-noise contract; HDP worst
  observed 4.6e-05). Baseline jit-vs-eager bit-exactness held only for
  the unchunkable bulk-noise graph and is not a stable contract
  (C5-C7); the chain schedule that makes chunked == continuous changes
  XLA fusion. Spikes stay exact.
- H: exact on legacy HDP; d<=1e-6 on registered rules (observed 2.4e-07).
- W/Theta: exact (observed 0.0 in all regimes incl. boundary clips).
- sources: d<=1e-4 (mechanism-consistent with V; observed exact).
- fields/LFP: analytic bound d <= ||K||_F * d_sources, checked numerically.
- aux: d<=1e-6 (same reassociation class as H).
- continuation leaves/keys/delay_state, disabled identity, strided frames,
  record toggles: exact (pinned by HDP-01/STOCH-01/REC-01/LAW-01 receipts;
  not re-proven here).
Epsilons are pre-declared bounds with measured evidence retained in the
receipt, never promoted from single observations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.fields.proxy import project_laminar_sources

EPS_V = 1e-4
EPS_H = 1e-6
EPS_AUX = 1e-6

REGIMES = {
    "legacy": {"K_HDP": 0.05, "K_ctrl": 0.3, "K_w_ctrl": 0.002, "alpha": 0.05,
               "gamma": 0.05, "C_spike": 0.02, "noise_scale": 0.0},
    "registered": {"hdp_rule": "synthetic_presyn_gain",
                   "hdp_rule_params": {"k_h": 0.05, "k_w": 0.04, "gamma": 0.0},
                   "noise_scale": 0.0},
    "eligibility": {"hdp_rule": "eligibility_trace_gain",
                    "hdp_rule_params": {"k_h": 0.05, "k_w": 0.1, "gamma": 0.0,
                                        "tau_e": 5.0},
                    "noise_scale": 0.0},
}


def _model(n=8):
    cfg = (
        jtfne.configuration()
        .runtime(seed=0, recurrent_backend="edge_list")
        .network(name="V1", kind="cortical_column", n=n,
                 cell_types={"E": 0.8, "PV": 0.2})
        .drive(baseline_drive_by_cell_type={"E": 10.0, "PV": 10.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    return jtfne.construct(cfg)


def _sim(jit, hp):
    kw = {} if hp is None else {"enable_hdp": True, "hdp_params": hp}
    return jtfne.simulation(
        duration_ms=60.0, dt_ms=1.0, seed=3, record_sources=True,
        record_fields=False,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list", jit=jit, **kw),
    )


def _run_pair(model, hp):
    a = jtfne.simulate(model, _sim(False, hp))
    da = model.last_hdp_diagnostics()
    b = jtfne.simulate(model, _sim(True, hp))
    db = model.last_hdp_diagnostics()
    return (a, da), (b, db)


@pytest.mark.parametrize("regime", ["legacy", "registered", "eligibility"])
def test_equiv_table_hdp_regimes(regime):
    model = _model()
    (a, da), (b, db) = _run_pair(model, REGIMES[regime])
    assert jnp.array_equal(a.spikes, b.spikes)
    assert float(jnp.max(jnp.abs(a.V_m - b.V_m))) <= EPS_V
    assert float(jnp.max(jnp.abs(
        np.asarray(a.sources) - np.asarray(b.sources)))) <= EPS_V
    assert da is not None and db is not None
    dh = np.abs(np.asarray(da["H_trace"]) - np.asarray(db["H_trace"]))
    assert float(dh.max()) <= EPS_H, regime
    dw = np.abs(np.asarray(da["w_trace"]) - np.asarray(db["w_trace"]))
    assert bool(np.all(dw == 0.0)), regime
    if da.get("aux_final") is not None and int(np.asarray(da["aux_final"]).size):
        shorter = min(np.asarray(da["aux_trace"]).shape[0], np.asarray(db["aux_trace"]).shape[0])
        daux = np.abs(np.asarray(da["aux_trace"])[:shorter] - np.asarray(db["aux_trace"])[:shorter])
        assert float(daux.max()) <= EPS_AUX, regime


def test_equiv_baseline_exact():
    model = _model()
    (a, _), (b, _) = _run_pair(model, None)
    assert jnp.array_equal(a.spikes, b.spikes)
    assert float(jnp.max(jnp.abs(a.V_m - b.V_m))) <= EPS_V
    assert float(jnp.max(jnp.abs(
        np.asarray(a.sources) - np.asarray(b.sources)))) <= EPS_V


def test_equiv_boundary_clips_do_not_amplify():
    # H seeded at both clamp edges with outward drive: clips engage on both
    # modes. Measured absolute divergence 2.2e-03 (relative 1.5e-05 at
    # |V|~150): saturation amplifies reassociation beyond the standard
    # regime, so this corner carries its own empirical bound 1e-2 (margin
    # ~5x). Spikes stay exact regardless.
    model = _model()
    n = int(model.params["emitter"].n_neurons)
    h0 = jnp.array([0.1001] * (n // 2) + [9.999] * (n - n // 2), dtype=jnp.float32)
    seeded = model.with_hdp_initial_state(H0=h0)
    hp = {"K_HDP": 0.1, "K_ctrl": 0.5, "noise_scale": 0.0}
    a = jtfne.simulate(seeded, _sim(False, hp))
    da = seeded.last_hdp_diagnostics()
    b = jtfne.simulate(seeded, _sim(True, hp))
    db = seeded.last_hdp_diagnostics()
    assert jnp.array_equal(a.spikes, b.spikes)
    dv = np.abs(np.asarray(a.V_m) - np.asarray(b.V_m))
    assert float(dv.max()) <= 1e-2
    assert float(np.abs(np.asarray(da["H_trace"]) - np.asarray(db["H_trace"])).max()) <= 1e-4
    dw = np.abs(np.asarray(da["w_trace"]) - np.asarray(db["w_trace"]))
    assert bool(np.all(dw == 0.0))


def test_equiv_lfp_derived_bound():
    model = _model()
    (a, _), (b, _) = _run_pair(model, REGIMES["legacy"])
    pos = jnp.asarray(model.params["positions"])
    fa = project_laminar_sources(jnp.asarray(a.sources), pos, n_contacts=16)
    fb = project_laminar_sources(jnp.asarray(b.sources), pos, n_contacts=16)
    d_src = float(jnp.max(jnp.abs(jnp.asarray(a.sources) - jnp.asarray(b.sources))))
    d_lfp = float(jnp.max(jnp.abs(
        np.asarray(fa.lfp_proxy) - np.asarray(fb.lfp_proxy))))
    k_norm = float(jnp.linalg.norm(jnp.asarray(fa.kernel), ord="fro"))
    assert d_lfp <= k_norm * d_src + 1e-6
    assert np.array_equal(np.asarray(fa.kernel), np.asarray(fb.kernel))
