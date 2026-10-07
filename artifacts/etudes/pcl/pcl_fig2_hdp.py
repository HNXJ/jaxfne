"""PCL gate K0: Fig. 2 on the jaxfne registrable HDP kernel (Izhikevich neurons, rule ``pcl_stdp``).

Same input protocol as ``pcl_fig2.py``. Each input event is a 1 ms current pulse
(1.5x the smallest pulse that fires a resting neuron, the analogue of the paper's
15 mV input against a 10 mV threshold). Weight units: s = w_cancel / 5.3, where
w_cancel is the smallest 0->k inhibitory weight that blocks a pulse arriving 1 ms
after neuron 0's pulse (the paper's equivalent is (15 - 10) / exp(-1/18) = 5.3 mV).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcl_fig2 as F  # noqa: E402
import pcl_hdp_rule as R  # noqa: E402
from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered as sim  # noqa: E402
from jaxfne.emitters import EdgeList, IzhikevichParams  # noqa: E402

DT = 0.1  # ms
PULSE_MS, GAP_MS, TAU_INH = 1.0, 50.0, 2.0
R.register()


def izh(n):
    f = lambda x: jnp.full((n,), x, jnp.float32)
    return IzhikevichParams(a=f(0.02), b=f(0.2), c=f(-65.0), d=f(8.0), drive=f(0.0), sign=f(1.0),
                            W=jnp.zeros((n, n)), v0=f(-65.0), u0=f(-13.0), source_scale=f(1.0),
                            labels=("E",) * n, layer_labels=None, source_calibration_status="x")


def edges(w, n_post=3):
    return EdgeList(pre=jnp.zeros(n_post, jnp.int32), post=jnp.arange(1, n_post + 1, dtype=jnp.int32),
                    weight=-jnp.asarray(w, jnp.float32), receptor_index=jnp.ones(n_post, jnp.int32),
                    tau_ms=jnp.full(n_post, TAU_INH, jnp.float32), source_calibration_status="x")


def schedule(trains, amp, total_ms):
    n = int(round(total_ms / DT))
    s = np.zeros((n, len(trains)), np.float32)
    w = int(round(PULSE_MS / DT))
    for k, tr in enumerate(trains):
        for t in tr:
            i = int(round(t / DT))
            s[i:i + w, k] = amp
    return jnp.asarray(s)


def run(w, trains, amp, total_ms, rp, learn):
    n_post = len(trains) - 1
    st = {"v": jnp.full(n_post + 1, -65.0), "u": jnp.full(n_post + 1, -13.0), "prev_spikes": jnp.zeros(n_post + 1),
          "syn_state": jnp.zeros(n_post), "w_final": -jnp.asarray(w, jnp.float32),
          "aux_final": jnp.zeros((n_post, 3), jnp.float32)}
    sched = schedule(trains, amp, total_ms)
    _, spikes, _, diag = sim(izh(n_post + 1), edges(w, n_post), int(sched.shape[0]), DT, jax.random.PRNGKey(0),
                             drive_schedule=sched, noise_scale=0.0, init_state=st, hdp_rule=R.NAME,
                             hdp_rule_params=rp, record_weight_trace=False,
                             plasticity_mask=jnp.full(n_post, 1.0 if learn else 0.0))
    return np.abs(np.asarray(diag["w_final"])), np.asarray(spikes).sum(0)


def null_params(n_post):
    z = np.zeros(n_post)
    return R.rule_params(np.full(n_post, F.TAU_LTP), z, z, z, z, z, np.arange(n_post), n_post)


def calibrate():
    """Pulse amplitude (1.5x threshold) and w_cancel by bisection."""
    def fires(amp):
        return run([0.0], [[], [10.0]], amp, 40.0, null_params(1), False)[1][1] > 0
    lo, hi = 0.0, 2000.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if fires(mid) else (mid, hi)
    amp = 1.5 * hi

    def blocked(w):
        return run([w], [[10.0], [10.0 + F.LAG_MS]], amp, 40.0, null_params(1), False)[1][1] == 0
    # Very large weights drive v so negative that the quadratic Izhikevich term
    # fires the neuron anyway; blocking holds from ~30 to >= 1e3 (probe), so cap there.
    lo, hi = 0.0, 1e3
    assert blocked(hi), "no inhibitory weight blocks the pulse"
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if blocked(mid) else (mid, hi)
    return float(amp), float(hi)


def concat(samples):
    """Concatenate samples with gaps; returns trains per neuron and total ms."""
    period = F.T_MS + GAP_MS
    trains = [np.concatenate([s[k] + i * period for i, s in enumerate(samples)]) for k in range(4)]
    return trains, len(samples) * period


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--w0", type=float, default=1.0, help="initial weight in paper mV units")
    ap.add_argument("--n-train", type=int, default=140)
    ap.add_argument("--n-test", type=int, default=35)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    amp, w_cancel = calibrate()
    s = w_cancel / 5.3
    rp = R.rule_params(np.full(3, F.TAU_LTP), np.full(3, F.W_MAX * s), np.full(3, F.ETA_LTP),
                       np.full(3, F.ETA_BOUND), np.zeros(3), np.zeros(3), np.arange(3), 3)
    runs = []
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        train = [F.make_sample(rng) for _ in range(a.n_train)]
        test = [F.make_sample(rng) for _ in range(a.n_test)]
        tr, ttr = concat(train)
        te, tte = concat(test)
        w = np.full(3, a.w0 * s)
        hist = [w / s]
        for _ in range(a.epochs):
            w, _ = run(w, tr, amp, ttr, rp, True)
            hist.append(w / s)
        _, on = run(w, te, amp, tte, rp, False)
        _, off = run(np.zeros(3), te, amp, tte, rp, False)
        runs.append(dict(seed=seed, w_hist_mV=np.array(hist).tolist(), spikes_inhib=on.tolist(),
                         spikes_no_inhib=off.tolist(), suppression=(1 - on[1:] / off[1:]).tolist()))
    sup = np.array([r["suppression"] for r in runs])
    wend = np.array([r["w_hist_mV"][-1] for r in runs])
    res = dict(params=dict(vars(a), out=str(a.out), dt_ms=DT, pulse_amp=amp, w_cancel=w_cancel, scale=s),
               runs=runs, suppression_mean=sup.mean(0).tolist(), w_end_mean_mV=wend.mean(0).tolist())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"pulse amp {amp:.2f}, w_cancel {w_cancel:.3f}, scale {s:.4f} per mV")
    print("neuron (predictability):  1 (90%)  2 (50%)  3 (10%)")
    print("final w (mV-equivalent): ", "  ".join(f"{x:6.2f}" for x in wend.mean(0)))
    print("suppression (mean):      ", "  ".join(f"{x:6.3f}" for x in sup.mean(0)))
    print("suppression (per seed):  ", sup.round(3).tolist())


if __name__ == "__main__":
    main()
