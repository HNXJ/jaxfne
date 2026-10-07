"""PCL gate K0b on the H-state rule: Fig. 2 through ``pcl_stdp_h``.

Same network, calibration, training and test as ``pcl_fig2_hdp.py`` (whose
helpers are reused: neuron/edge/schedule builders, sample concatenation and
the learning-off calibration). The only difference is the rule: per-neuron
traces in H (``pcl_h_rule``) instead of per-edge aux. Weight criterion:
|w| averaged over every step of the last training epoch, ordered 1 > 2 > 3
in every network; suppression criterion unchanged.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.with_name("pcl")))
from jaxfne._hdp_registrable_kernel import simulate_edge_recurrent_izhikevich_hdp_registered as sim  # noqa: E402

import pcl_fig2 as F  # noqa: E402
import pcl_fig2_hdp as K  # noqa: E402
import pcl_hdp_rule as R  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcl_h_rule as RH  # noqa: E402

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

RH.register()


def run(w, trains, amp, total_ms, rp, learn, avg=False):
    """Final |w| and spike counts; with ``avg``, also |w| averaged over every recorded step."""
    n_post = len(trains) - 1
    n = n_post + 1
    st = {"v": jnp.full(n, -65.0), "u": jnp.full(n, -13.0), "prev_spikes": jnp.zeros(n),
          "syn_state": jnp.zeros(n_post), "w_final": -jnp.asarray(w, jnp.float32),
          "H_final": jnp.zeros((n, 2), jnp.float32),
          "aux_final": jnp.zeros((n_post, 2), jnp.float32)}
    sched = K.schedule(trains, amp, total_ms)
    _, spikes, _, diag = sim(K.izh(n), K.edges(w, n_post), int(sched.shape[0]), K.DT, jax.random.PRNGKey(0),
                            drive_schedule=sched, noise_scale=0.0, init_state=st, hdp_rule=RH.NAME,
                            hdp_rule_params=rp, record_weight_trace=avg, record_stride=10 if avg else 1,
                            plasticity_mask=jnp.full(n_post, 1.0 if learn else 0.0))
    out = np.abs(np.asarray(diag["w_final"])), np.asarray(spikes).sum(0)
    return (*out, np.abs(np.asarray(diag["w_trace"])).mean(0)) if avg else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--w0", type=float, default=1.0, help="initial weight in paper mV units")
    ap.add_argument("--n-train", type=int, default=140)
    ap.add_argument("--n-test", type=int, default=35)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    amp, w_cancel = K.calibrate()
    s = w_cancel / 5.3
    rp = R.rule_params(np.full(3, F.TAU_LTP), np.full(3, F.W_MAX * s), np.full(3, F.ETA_LTP),
                       np.full(3, F.ETA_BOUND), np.zeros(3), np.zeros(3), np.arange(3), 3)
    runs = []
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        train = [F.make_sample(rng) for _ in range(a.n_train)]
        test = [F.make_sample(rng) for _ in range(a.n_test)]
        tr, ttr = K.concat(train)
        te, tte = K.concat(test)
        w = np.full(3, a.w0 * s)
        hist = [w / s]
        for _ in range(a.epochs - 1):
            w, _ = run(w, tr, amp, ttr, rp, True)
            hist.append(w / s)
        w, _, w_avg = run(w, tr, amp, ttr, rp, True, avg=True)  # last epoch, time-averaged weight
        hist.append(w / s)
        _, on = run(w, te, amp, tte, rp, False)
        _, off = run(np.zeros(3), te, amp, tte, rp, False)
        runs.append(dict(seed=seed, w_hist_mV=np.array(hist).tolist(), w_avg_last_epoch_mV=(w_avg / s).tolist(),
                         spikes_inhib=on.tolist(), spikes_no_inhib=off.tolist(),
                         suppression=(1 - on[1:] / off[1:]).tolist()))
    sup = np.array([r["suppression"] for r in runs])
    wend = np.array([r["w_hist_mV"][-1] for r in runs])
    wavg = np.array([r["w_avg_last_epoch_mV"] for r in runs])
    res = dict(params=dict(vars(a), out=str(a.out), dt_ms=K.DT, pulse_amp=amp, w_cancel=w_cancel, scale=s,
                           rule=RH.NAME),
               runs=runs, suppression_mean=sup.mean(0).tolist(), w_end_mean_mV=wend.mean(0).tolist(),
               w_avg_mean_mV=wavg.mean(0).tolist())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"pulse amp {amp:.2f}, w_cancel {w_cancel:.3f}, scale {s:.4f} per mV")
    print("neuron (predictability):  1 (90%)  2 (50%)  3 (10%)")
    print("final w (mV-equivalent): ", "  ".join(f"{x:6.2f}" for x in wend.mean(0)))
    print("avg w, last epoch (mV):  ", "  ".join(f"{x:6.2f}" for x in wavg.mean(0)))
    print("avg w ordered per seed:  ", [bool(r[0] > r[1] > r[2]) for r in wavg])
    print("suppression (mean):      ", "  ".join(f"{x:6.3f}" for x in sup.mean(0)))
    print("suppression (per seed):  ", sup.round(3).tolist())


if __name__ == "__main__":
    main()
