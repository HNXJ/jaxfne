"""PCL gate K2: a population-surprise third factor on the HDP kernel (rule ``pcl_stdp_m``).

Fig. 2 network on the kernel (as K0). Phase A trains with predictability 90 / 50 / 10 %
(neurons 1 / 2 / 3); at the switch the predictabilities reverse to 10 / 50 / 90 % and
phase B trains one epoch with the weight and modulator traces recorded. Phase A is shared;
phase B branches into three conditions:
  fixed     g = 0 (plain PCL),
  gated     surprise-gated learning rate (g > 0),
  matched   g = 0 with eta scaled by the gated run's mean multiplier over phase B.
Readout: t_rev, the time after the switch from which the trailing 10 s mean of the weight
onto neuron 3 minus the weight onto neuron 1 stays above 0 for the rest of phase B (K2b;
the first pilot read the raw trace with tau_f = 0.5 s).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcl_fig2 as F  # noqa: E402
import pcl_fig2_hdp as K  # noqa: E402
import pcl_hdp_rule as R  # noqa: E402
from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered as sim  # noqa: E402

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

TAU_F, TAU_S, M_MAX, EPS = 2000.0, 20000.0, 5.0, 1e-4  # ms, ms, -, 1/ms (0.1 Hz); K2b: tau_f 0.5 -> 2 s
STRIDE = 10  # trace every 10 steps (1 ms)
SMOOTH_S = 10.0  # K2b readout: trailing 10 s mean of w3 - w1


def params(s, g, eta_scale=1.0):
    p = R.rule_params(np.full(3, F.TAU_LTP), np.full(3, F.W_MAX * s), np.full(3, F.ETA_LTP * eta_scale),
                      np.full(3, F.ETA_BOUND), np.zeros(3), np.zeros(3), np.arange(3), 3)
    return dict(p, pop=jnp.array([0.0, 1.0, 1.0, 1.0]), tau_f=jnp.float32(TAU_F), tau_s=jnp.float32(TAU_S),
                g=jnp.float32(g), m_max=jnp.float32(M_MAX), eps=jnp.float32(EPS))


def run(w, aux, trains, amp, total_ms, rp, learn, trace=False):
    """One pass; returns |w_final|, aux_final, spike counts, and (if trace) w and aux traces."""
    st = {"v": jnp.full(4, -65.0), "u": jnp.full(4, -13.0), "prev_spikes": jnp.zeros(4),
          "syn_state": jnp.zeros(3), "w_final": -jnp.asarray(w, jnp.float32), "aux_final": jnp.asarray(aux)}
    sched = K.schedule(trains, amp, total_ms)
    _, spikes, _, d = sim(K.izh(4), K.edges(w, 3), int(sched.shape[0]), K.DT, jax.random.PRNGKey(0),
                          drive_schedule=sched, noise_scale=0.0, init_state=st, hdp_rule=R.NAME_M,
                          hdp_rule_params=rp, record_weight_trace=trace, record_stride=STRIDE if trace else 1,
                          plasticity_mask=jnp.full(3, 1.0 if learn else 0.0))
    out = (np.abs(np.asarray(d["w_final"])), np.asarray(d["aux_final"]), np.asarray(spikes).sum(0))
    if not trace:
        return out
    return (*out, np.abs(np.asarray(d["w_trace"])), np.asarray(d["aux_trace"])[::STRIDE, 0, 3:5])


def samples(rng, n, p_replace):
    old = F.P_REPLACE
    F.P_REPLACE = p_replace
    try:
        return [F.make_sample(rng) for _ in range(n)]
    finally:
        F.P_REPLACE = old


def t_rev(w_tr, smooth_s=SMOOTH_S):
    """Seconds after the switch from which the trailing ``smooth_s`` mean of w[2] - w[0] stays > 0.

    ``smooth_s`` = 0 reads the raw trace (the first K2 pilot); inf if never reversed.
    """
    k = max(int(smooth_s * 1000 / (STRIDE * K.DT)), 1)
    diff = np.convolve(w_tr[:, 2] - w_tr[:, 0], np.ones(k) / k, mode="valid")  # diff[i] ends at sample i + k - 1
    bad = np.nonzero(diff <= 0)[0]
    if len(bad) == 0:
        return k * STRIDE * K.DT / 1000.0
    if bad[-1] == len(diff) - 1:
        return float("inf")
    return (bad[-1] + k) * STRIDE * K.DT / 1000.0


def multiplier(m_tr, g):
    m = np.clip((m_tr[:, 0] + EPS) / (m_tr[:, 1] + EPS), 0.0, M_MAX)
    return np.maximum(1.0 + g * (m - 1.0), 0.0), m


def network(seed, g, amp, s, n_train, n_test, epochs_a):
    rng = np.random.default_rng(seed)
    tr_a, t_a = K.concat(samples(rng, n_train, (0.9, 0.5, 0.1)))
    tr_b, t_b = K.concat(samples(rng, n_train, (0.1, 0.5, 0.9)))
    te_b, tt_b = K.concat(samples(rng, n_test, (0.1, 0.5, 0.9)))
    w, aux = np.full(3, s), np.zeros((3, 5), np.float32)
    for _ in range(epochs_a):
        w, aux, _ = run(w, aux, tr_a, amp, t_a, params(s, 0.0), True)
    res = {"w_switch_mV": (w / s).tolist()}
    _, _, _, wg, mg = run(w, aux, tr_b, amp, t_b, params(s, g), True, trace=True)
    mult, m = multiplier(mg, g)
    conds = {"fixed": params(s, 0.0), "gated": params(s, g), "matched": params(s, 0.0, float(mult.mean()))}
    for name, rp in conds.items():
        wf, _, _, w_tr, m_tr = run(w, aux, tr_b, amp, t_b, rp, True, trace=True)
        _, _, on = run(wf, aux, te_b, amp, tt_b, rp, False)
        res[name] = {"t_rev_s": t_rev(w_tr), "w_end_mV": (wf / s).tolist(),
                     "w_avg_mV": (w_tr.mean(0) / s).tolist(), "spikes_test": on.tolist()}
    _, _, off = run(np.zeros(3), aux, te_b, amp, tt_b, params(s, 0.0), False)
    res["spikes_test_no_inh"] = off.tolist()
    res["gated_mult_mean"] = float(mult.mean())
    res["m_first10s_mean"] = float(m[: int(10000 / (STRIDE * K.DT))].mean())
    res["m_last10s_mean"] = float(m[-int(10000 / (STRIDE * K.DT)):].mean())
    return res, wg / s, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--g", type=float, required=True)
    ap.add_argument("--n-train", type=int, default=140)
    ap.add_argument("--n-test", type=int, default=35)
    ap.add_argument("--epochs-a", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    R.register()
    amp, w_cancel = K.calibrate()
    s = w_cancel / 5.3
    runs, traces = [], {}
    for seed in a.seeds:
        res, w_tr, m = network(seed, a.g, amp, s, a.n_train, a.n_test, a.epochs_a)
        runs.append(dict(seed=seed, **res))
        traces[f"w_gated_{seed}"], traces[f"m_gated_{seed}"] = w_tr[::10], m[::10]  # 10 ms resolution
        print(seed, {k: res[k]["t_rev_s"] for k in ("fixed", "gated", "matched")}, "mult", round(res["gated_mult_mean"], 3))
    out = dict(params=dict(vars(a), out=str(a.out), tau_f=TAU_F, tau_s=TAU_S, m_max=M_MAX, eps=EPS,
                           pulse_amp=amp, w_cancel=w_cancel, scale=s), runs=runs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    np.savez_compressed(a.out.with_suffix(".npz"), **traces)


if __name__ == "__main__":
    main()
