#!/usr/bin/env python3
"""Per-edge ground truth for one HDP training trial (O-60).

Runs trial 0 of the training leg, then:
  - validates the H layout from H_final (which coord is the pre trace),
  - compares realized per-edge dw (w_final - w0) against the all-pairs kernel
    prediction computed from the same spike trains, per class,
  - reports per-class means and the largest mismatches.
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
    chosen = {"w_ar": 0.5, "w_ir": 0.5, "d_R": 3.0}
    rule = rc.ensure_reader_rule(2.0)
    model0, _ = rc.build_model(5, True, 1.0, chosen["w_ar"], chosen["w_ir"],
                               2.0, chosen["d_R"], seed_h, 1000.0)
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
    print("diag keys:", sorted(diag.keys()))
    for k in ("H_final", "H_trace", "w_final", "w_trace"):
        v = diag.get(k)
        print(f"  {k}: {'None' if v is None else np.asarray(v).shape}")
    H = np.asarray(diag["H_final"], float)
    print("H_final per-neuron coord means: afferents",
          H[:rc.N_A].mean(axis=0).round(4), " I",
          H[rc.N_A:rc.N_A + rc.N_I].mean(axis=0).round(4), " R",
          H[rc.N_A + rc.N_I:].mean(axis=0).round(4))

    w_new = np.asarray(diag["w_final"], float)
    print("w_final shape", w_new.shape, " n_edges(mask)", int(mask_ar.sum()))
    # realized dw on plastic edges
    if w_new.shape[0] == len(pre):
        rw = w_new
    else:
        rw = w_new[:len(pre)]
    dwr = (rw - w0)[on_ar]
    pre_a, post_a, is_pat_a = pre[on_ar], post[on_ar], is_pat[on_ar]

    # spike trains
    spk = np.asarray(sig.spikes)
    t = np.arange(spk.shape[0]) * DT_MS
    a_spk = spk[:, :rc.N_A]
    r_spk = spk[:, rc.N_A + rc.N_I:]

    # per-edge kernel dw (all-pairs, exact exponentials)
    r_times = [t[np.flatnonzero(r_spk[:, j])] for j in range(r_spk.shape[1])]
    posts = np.concatenate(r_times)
    posts.sort()
    dwk = np.zeros(len(pre_a))
    for e in range(len(pre_a)):
        psp = t[np.flatnonzero(a_spk[:, pre_a[e]])]
        rsp = t[np.flatnonzero(r_spk[:, post_a[e] - rc.N_A - rc.N_I])]
        dw = 0.0
        for p in psp:
            hi = np.searchsorted(rsp, p + TAU_P, side="right")
            lo = np.searchsorted(rsp, p, side="right")
            if hi > lo:
                dw += A_P * float(np.exp(-(rsp[lo:hi] - p) / TAU_P).sum())
            lo2 = np.searchsorted(rsp, p - TAU_M, side="left")
            hi2 = np.searchsorted(rsp, p, side="left")
            if hi2 > lo2:
                dw -= A_M * float(np.exp(-(p - rsp[lo2:hi2]) / TAU_M).sum())
        dwk[e] = dw
    for name, msk in (("pattern", is_pat_a), ("carrier", ~is_pat_a)):
        r, k = dwr[msk], dwk[msk]
        print(f"{name}: n={msk.sum()} realized mean {r.mean():+.5f} "
              f"std {r.std():.5f} | kernel mean {k.mean():+.5f} "
              f"std {k.std():.5f} | corr {np.corrcoef(r, k)[0, 1]:+.3f}")
        o = np.argsort(k)[::-1][:3]
        print("  top kernel edges (kernel -> realized):",
              [(round(k[i], 4), round(r[i], 4)) for i in o])
    # realized dw sign structure
    print("realized dw quantiles (pattern):",
          np.quantile(dwr[is_pat_a], [0, .25, .5, .75, 1]).round(5))
    print("realized dw quantiles (carrier):",
          np.quantile(dwr[~is_pat_a], [0, .25, .5, .75, 1]).round(5))
    # unique realized values: is dw discretized/degenerate?
    u = np.unique(dwr.round(8))
    print("unique realized dw values (first 12):", u[:12], " n_unique", u.size)


if __name__ == "__main__":
    main()