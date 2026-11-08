#!/usr/bin/env python3
"""Deconvolve unrecorded trace events (O-60): find bins where a neuron's
pre_trace jumps without a recorded spike of that neuron, and test what those
bins coincide with (reader spikes, I spikes, other afferents' spikes)."""
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
    decay = np.exp(-DT_MS / 17.0)
    x = H[:, :, 0]  # (steps, neurons) pre_trace

    # jump detector: x[b+1] - decay*x[b] > 0.5  (a jump of ~1.0)
    jump = x[1:] - decay * x[:-1]  # jump occurring in bin b+1
    jbins = jump > 0.5

    for aff in (0, 25):
        own = spk[1:, aff] > 0
        ghost = jbins[:, aff] & ~own
        gb = np.flatnonzero(ghost) + 1
        ob = np.flatnonzero(own) + 1
        print(f"afferent {aff}: recorded jumps {ob.size}, ghost jumps {gb.size}")
        if gb.size:
            print("  first 10 ghost bins (ms):",
                  np.round(gb[:10] * DT_MS, 2).tolist())
            # what else spikes at those bins?
            other_aff = spk[gb, :rc.N_A].sum(axis=1) - spk[gb, aff]
            i_spk = spk[gb, rc.N_A:rc.N_A + rc.N_I].sum(axis=1)
            r_spk = spk[gb, rc.N_A + rc.N_I:].sum(axis=1)
            print("  at ghost bins: other afferents spiking "
                  "(mean/total):", round(float(other_aff.mean()), 2),
                  int(other_aff.sum()), "| I:", round(float(i_spk.mean()), 2),
                  "| R:", round(float(r_spk.mean()), 2), int(r_spk.sum()))

    # do reader spikes appear as jumps in EVERY neuron's pre_trace?
    r_bins = np.flatnonzero(spk[1:, rc.N_A + rc.N_I:].sum(axis=1) > 0) + 1
    if r_bins.size:
        leak = jbins[r_bins - 1, :rc.N_A].mean(axis=0)
        print("reader-spike bins: fraction of reader events that appear as a "
              "pre_trace jump, per afferent (first 12):",
              np.round(leak[:12], 2).tolist(),
              "| mean over afferents:", round(float(leak.mean()), 3))
        i_leak = jbins[r_bins - 1, rc.N_A:rc.N_A + rc.N_I].mean()
        r_leak = jbins[r_bins - 1, rc.N_A + rc.N_I:].mean()
        print("  same for I neurons:", round(float(i_leak), 3),
              "| for readers:", round(float(r_leak), 3))


if __name__ == "__main__":
    main()