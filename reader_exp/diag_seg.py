#!/usr/bin/env python3
"""Dump raw H_trace segments around afferent spikes (O-60)."""
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

    for aff in (0, 25):  # 0 = pattern afferent, 25 = carrier-only
        bins = np.flatnonzero(spk[:, aff])
        print(f"=== afferent {aff}: {bins.size} spikes ===")
        for b in bins[:6]:
            seg = H[b - 3:b + 5, aff, 0]
            prev = spk[:, aff][max(0, b - 400):b].sum()
            print(f"  spike bin {b}: prev spikes in 20 ms: {int(prev)}")
            print("   x[b-3..b+4]:", " ".join(f"{v:.3f}" for v in seg))
    # the very first event of afferent 0 and of a carrier afferent with a
    # long gap after it
    for aff in (0, 25):
        bins = np.flatnonzero(spk[:, aff])
        gaps = np.diff(bins)
        i = int(np.argmax(gaps))
        b = bins[i]
        seg = H[b - 3:b + min(gaps[i], 400) + 3, aff, 0]
        tail = seg[5:]
        if tail.size > 40:
            ln = np.polyfit(np.arange(tail.size), np.log(tail), 1)
            print(f"afferent {aff}: longest gap after bin {b} "
                  f"({gaps[i] * DT_MS:.1f} ms): first x {seg[3]:.4f} "
                  f"{seg[4]:.4f} {seg[5]:.4f}; tail decay/bin "
                  f"{np.exp(ln[0]):.6f} (model {np.exp(-DT_MS / 17.0):.6f})")


if __name__ == "__main__":
    main()