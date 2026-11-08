#!/usr/bin/env python3
"""Reconcile realized dw vs kernel dw using the recorded H trace (O-60).

One training trial; then per A->R edge:
  dw_rec   = sum_b [ A+ * x_rec(b) * S_post(b) - A- * y_rec(b) * S_pre(b) ]
             using the recorded H_trace (the kernel's own view),
  dw_kernel = all-pairs exponential sum from the spike trains alone.
Reports per-class means of dw_rec, dw_kernel, and the LTP/LTD parts of each,
plus the per-bin trace-value comparison at spike bins.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reader_carrier as rc  # noqa: E402

import jaxfne  # noqa: E402

DT_MS = rc.DT_MS
TAU_P, TAU_M = 17.0, 34.0
A_P, A_M = 0.005, 0.00525


def main():
    seed_h = 0
    rule = rc.ensure_reader_rule(2.0)
    model0, _ = rc.build_model(5, True, 1.0, 0.5, 0.5, 2.0, 3.0, seed_h, 1000.0)
    mask_ar = rc.plasticity_mask_ar(model0)
    n_total = rc.N_A + rc.N_I + 5
    base = np.random.default_rng(seed_h)
    offsets_P = rc.draw_offsets(base)
    rt = rc.hdp_runtime(rule, mask_ar)
    el = model0.params["edge_list"]
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    is_pat = np.isin(pre, np.arange(rc.K_PAT))
    on_ar = (pre < rc.N_A) & (post >= rc.N_A + rc.N_I)
    w0 = rc.resolved_weights(model0)

    trng = np.random.default_rng(rc.trial_seed(seed_h, 500000))
    spike_times, _ = rc.make_trial_input(trng, 1000.0, 0.5, 0.0, 1.0,
                                         offsets_P, "P")
    m_use = model0.with_hdp_initial_state(w0=w0)
    sim_cfg = jaxfne.simulation(duration_ms=1000.0, dt_ms=DT_MS,
                                seed=rc.trial_seed(seed_h, 500555), runtime=rt)
    sig = jaxfne.simulate(m_use, sim_cfg,
                          jaxfne.SpikeTrainInput(spike_times, n_neurons=n_total))
    diag = m_use.last_hdp_diagnostics()
    H = np.asarray(diag["H_trace"], float)  # (steps, neurons, 2)
    w_new = np.asarray(diag["w_final"], float)
    spk = np.asarray(sig.spikes)
    n_steps = H.shape[0]
    t = np.arange(n_steps) * DT_MS

    dw_rec = (w_new - w0)[on_ar]
    pre_a, post_a, is_pat_a = pre[on_ar], post[on_ar], is_pat[on_ar]

    # recorded-trace dw per edge (the kernel's own numbers)
    rec_ltp = np.zeros(len(pre_a))
    rec_ltd = np.zeros(len(pre_a))
    for e in range(len(pre_a)):
        x = H[:, pre_a[e], 0]
        y = H[:, post_a[e], 1]
        s_pre = spk[:, pre_a[e]]
        s_post = spk[:, post_a[e]]
        rec_ltp[e] = A_P * float(x[s_post > 0].sum())
        rec_ltd[e] = A_M * float(y[s_pre > 0].sum())

    # spike-pair kernel per edge
    r_spk = spk[:, rc.N_A + rc.N_I:]
    a_spk = spk[:, :rc.N_A]
    ker_ltp = np.zeros(len(pre_a))
    ker_ltd = np.zeros(len(pre_a))
    for e in range(len(pre_a)):
        psp = t[np.flatnonzero(a_spk[:, pre_a[e]])]
        rsp = t[np.flatnonzero(r_spk[:, post_a[e] - rc.N_A - rc.N_I])]
        for p in psp:
            hi = np.searchsorted(rsp, p + TAU_P, side="right")
            lo = np.searchsorted(rsp, p, side="right")
            if hi > lo:
                ker_ltp[e] += A_P * float(np.exp(-(rsp[lo:hi] - p) / TAU_P).sum())
            lo2 = np.searchsorted(rsp, p - TAU_M, side="left")
            hi2 = np.searchsorted(rsp, p, side="left")
            if hi2 > lo2:
                ker_ltd[e] += A_M * float(np.exp(-(p - rsp[lo2:hi2]) / TAU_M).sum())

    for name, msk in (("pattern", is_pat_a), ("carrier", ~is_pat_a)):
        rl, rd, kl, kd = (rec_ltp[msk].mean(), rec_ltd[msk].mean(),
                          ker_ltp[msk].mean(), ker_ltd[msk].mean())
        rr = dw_rec[msk].mean()
        print(f"{name}: realized {rr:+.5f} | rec-trace ltp {rl:+.5f} ltd "
              f"{-rd:+.5f} sum {rl - rd:+.5f} | kernel ltp {kl:+.5f} ltd "
              f"{-kd:+.5f} sum {kl - kd:+.5f}")

    # where do the traces diverge from the pure-spike model? compare at bins
    # just after a pattern spike: x_rec vs exp model for one pattern afferent
    i0 = 0
    x0 = H[:, i0, 0]
    psp0 = t[np.flatnonzero(spk[:, i0])]
    print(f"afferent {i0}: {psp0.size} spikes; x_rec at +1 bin after each of "
          f"the first 6 spikes vs exp model:")
    for p in psp0[:6]:
        b = int(round(p / DT_MS)) + 1
        model = float(np.exp(-(t[b] - p) / TAU_P))
        print(f"  t={p:.2f} x_rec={x0[b]:.4f} model={model:.4f}")


if __name__ == "__main__":
    main()