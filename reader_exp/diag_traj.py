#!/usr/bin/env python3
"""Per-trial realized vs kernel-predicted dw for the reader HDP leg (O-60).

Rebuilds the training leg exactly as reader_carrier.run_hdp_leg does, but
after each trial records:
  - realized per-class mean dw (w_final diff, pattern vs carrier afferents),
  - kernel-predicted per-class mean dw (all-pairs trace sum on this trial's
    reader and afferent spikes),
  - reader spike count.
Read-only wrt the repo; prints a table.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reader_carrier as rc  # noqa: E402

import jaxfne  # noqa: E402,F401  (needs the clone on PYTHONPATH)

DT_MS = rc.DT_MS
N_A, N_I, K_PAT = rc.N_A, rc.N_I, rc.K_PAT
TAU_P, TAU_M = 17.0, 34.0
A_P, A_M = 0.005, 0.00525


def kernel_balance(a_spk, r_times, t):
    """All-pairs STDP sum per afferent; returns per-class (total, ltp, ltd)."""
    posts = np.concatenate(r_times)
    posts.sort()
    vals = []
    for i in range(a_spk.shape[1]):
        pre = t[np.flatnonzero(a_spk[:, i])]
        dw = ltp = ltd = 0.0
        for p in pre:
            hi = np.searchsorted(posts, p + TAU_P, side="right")
            lo = np.searchsorted(posts, p, side="right")
            if hi > lo:
                ltp += A_P * float(np.exp(-(posts[lo:hi] - p) / TAU_P).sum())
            lo2 = np.searchsorted(posts, p - TAU_M, side="left")
            hi2 = np.searchsorted(posts, p, side="left")
            if hi2 > lo2:
                ltd += A_M * float(np.exp(-(p - posts[lo2:hi2]) / TAU_M).sum())
        vals.append((ltp - ltd, ltp, ltd))
    vals = np.asarray(vals)
    return (float(vals[:K_PAT, 0].mean()), float(vals[:K_PAT, 1].mean()),
            float(vals[:K_PAT, 2].mean()), float(vals[K_PAT:, 0].mean()),
            float(vals[K_PAT:, 1].mean()), float(vals[K_PAT:, 2].mean()))


def main(n_trials=12, seed_h=0):
    chosen = {"w_ar": 0.5, "w_ir": 0.5, "d_R": 3.0}
    w_ai, d_I = 1.0, 2.0
    m, sigma_ms, df_hz, duration_ms = 0.5, 1.0, 0.0, 1000.0
    rule = rc.ensure_reader_rule(2.0)
    model0, _ = rc.build_model(5, True, w_ai, chosen["w_ar"], chosen["w_ir"],
                               d_I, chosen["d_R"], seed_h, duration_ms)
    mask_ar = rc.plasticity_mask_ar(model0)
    n_total = N_A + N_I + 5
    base = np.random.default_rng(seed_h)
    offsets_P = rc.draw_offsets(base)
    rt_plast = rc.hdp_runtime(rule, mask_ar)
    el = model0.params["edge_list"]
    pre = np.asarray(el.pre)
    is_pat = np.isin(pre, np.arange(K_PAT))
    on_ar = (pre < N_A) & (np.asarray(el.post) >= N_A + N_I)
    w_cur = rc.resolved_weights(model0)
    t = np.arange(int(duration_ms / DT_MS)) * DT_MS
    print("trial w_pat w_car | realized dPat dCar | kernel dPat dCar | Rspk")
    for k in range(n_trials):
        trng = np.random.default_rng(rc.trial_seed(seed_h, 500000 + k))
        spike_times, _ = rc.make_trial_input(
            trng, duration_ms, m, df_hz, sigma_ms, offsets_P, "P")
        m_use = model0.with_hdp_initial_state(w0=np.asarray(w_cur))
        sim_cfg = jaxfne.simulation(
            duration_ms=float(duration_ms), dt_ms=float(DT_MS),
            seed=rc.trial_seed(seed_h, 500555 + k), runtime=rt_plast)
        sig = jaxfne.simulate(
            m_use, sim_cfg,
            jaxfne.SpikeTrainInput(spike_times, n_neurons=int(n_total)))
        w_new = np.asarray(m_use.last_hdp_diagnostics()["w_final"], float)
        d = w_new - w_cur
        spk = np.asarray(sig.spikes)
        r_times = [t[np.flatnonzero(spk[:, N_A + N_I + j])]
                   for j in range(5)]
        a_spk = spk[:, :N_A]
        kp, kp_ltp, kp_ltd, kc, kc_ltp, kc_ltd = kernel_balance(a_spk, r_times, t)
        r_spk = int(spk[:, N_A + N_I:].sum())
        print(f"{k:5d} {w_new[on_ar & is_pat].mean():.4f} "
              f"{w_new[on_ar & ~is_pat].mean():.4f} | "
              f"{d[on_ar & is_pat].mean():+.5f} "
              f"{d[on_ar & ~is_pat].mean():+.5f} | "
              f"pat(ltp{kp_ltp:+.5f} ltd{kp_ltd:+.5f} tot{kp:+.5f}) "
              f"car(ltp{kc_ltp:+.5f} ltd{kc_ltd:+.5f} tot{kc:+.5f}) | {r_spk}", flush=True)
        w_cur = w_new


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)