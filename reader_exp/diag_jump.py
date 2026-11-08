#!/usr/bin/env python3
"""Measure the effective trace jump and per-bin decay from H_trace (O-60)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reader_carrier as rc  # noqa: E402

import jaxfne  # noqa: E402

DT_MS = rc.DT_MS


def main():
    seed_h = 0
    rule = rc.ensure_reader_rule(2.0)
    model0, _ = rc.build_model(5, True, 1.0, 0.5, 0.5, 2.0, 3.0, seed_h, 1000.0)
    mask_ar = rc.plasticity_mask_ar(model0)
    n_total = rc.N_A + rc.N_I + 5
    base = np.random.default_rng(seed_h)
    offsets_P = rc.draw_offsets(base)
    rt = rc.hdp_runtime(rule, mask_ar)
    trng = np.random.default_rng(rc.trial_seed(seed_h, 500000))
    spike_times, _ = rc.make_trial_input(trng, 1000.0, 0.5, 0.0, 1.0,
                                         offsets_P, "P")
    m_use = model0.with_hdp_initial_state(w0=rc.resolved_weights(model0))
    sim_cfg = jaxfne.simulation(duration_ms=1000.0, dt_ms=DT_MS,
                                seed=rc.trial_seed(seed_h, 500555), runtime=rt)
    sig = jaxfne.simulate(m_use, sim_cfg,
                          jaxfne.SpikeTrainInput(spike_times, n_neurons=n_total))
    diag = m_use.last_hdp_diagnostics()
    H = np.asarray(diag["H_trace"], float)
    spk = np.asarray(sig.spikes)

    # find an afferent spike with no other spike within 40 ms (isolated)
    a0 = spk[:, 0]
    bins = np.flatnonzero(a0)
    isolated = [b for b in bins
                if a0[max(0, b - 800):b + 800].sum() == 1]
    print(f"afferent 0: {bins.size} spikes, {len(isolated)} isolated")
    for b in isolated[:3]:
        seg = H[b - 2:b + 220, 0, 0]  # pre_trace coord
        print(f"  spike bin {b} (t={b * DT_MS:.2f} ms): x before={seg[0]:.4f} "
              f"at-spike={seg[1]:.4f} +1={seg[2]:.4f}")
        # empirical per-bin decay from the tail (bins +20..+200)
        tail = seg[22:202]
        ratio = tail[1:] / tail[:-1]
        print(f"    tail decay/bin: mean {ratio.mean():.6f} "
              f"(exp(-0.05/17)={np.exp(-DT_MS / 17.0):.6f}); "
              f"implied jump = {seg[1] / max(a0[b] * 0 + 1, 1):.4f}")
        ln = np.polyfit(np.arange(20, 200), np.log(tail), 1)
        print(f"    fit decay/bin {np.exp(ln[0]):.6f}, intercept "
              f"{np.exp(ln[0] * 0) * np.exp(ln[1]):.4f} at bin+20")

    # same for a reader's post_trace around its spikes
    r0 = spk[:, rc.N_A + rc.N_I]
    rb = np.flatnonzero(r0)
    riso = [b for b in rb if r0[max(0, b - 1400):b + 1400].sum() == 1]
    print(f"reader 0: {rb.size} spikes, {len(riso)} isolated")
    for b in riso[:3]:
        seg = H[b - 2:b + 1400, rc.N_A + rc.N_I, 1]
        print(f"  spike bin {b}: y before={seg[0]:.4f} at={seg[1]:.4f} "
              f"+1={seg[2]:.4f}")

    # how many bins does an afferent spike register in? (spike width)
    widths = []
    for b in bins[:20]:
        w = int(a0[b:b + 4].sum())
        widths.append(w)
    print("spike bin widths (aff 0, first 20 spikes):", widths)


if __name__ == "__main__":
    main()