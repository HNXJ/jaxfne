#!/usr/bin/env python3
"""O-60 protocol (a) runner: reader STDP retune arms (see O60_protocol_a.md).

Runs one arm on one seed: 60 P training trials with carried weights, then the
frozen before/after P/Q test sets. Rule coefficients are overridden through the
public hdp_params["hdp_rule_params"] transport. Writes protoA_<arm>_seed<s>.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reader_carrier as rc  # noqa: E402

import jaxfne  # noqa: E402

DT_MS = rc.DT_MS
N_A, N_I, K_PAT = rc.N_A, rc.N_I, rc.K_PAT

ARMS = {
    "A1": {"trace_ltp": {"A_plus": 0.0105}},
    "A2": {"trace_ltp": {"A_plus": 0.005},
           "trace_ltd": {"A_minus": 0.005, "tau_minus": 17.0}},
    "A3": {"trace_ltp": {"A_plus": 0.0105},
           "trace_ltd": {"tau_minus": 17.0}},
}


def main(arm, seed, n_train=60):
    t0 = time.perf_counter()
    rule_params = ARMS[arm]
    rule = rc.ensure_reader_rule(2.0)
    model0, _ = rc.build_model(5, True, 1.0, 0.5, 0.5, 2.0, 3.0, seed, 1000.0)
    mask_ar = rc.plasticity_mask_ar(model0)
    n_total = N_A + N_I + 5
    base = np.random.default_rng(seed)
    offsets_P = rc.draw_offsets(base)
    perm = rc.draw_permutation(base)
    offsets_Q = offsets_P[np.asarray(perm)]
    hdp_params = {"hdp_rule": rule, "plasticity_mask": np.asarray(mask_ar),
                  "record_weight_trace": False,
                  "hdp_rule_params": rule_params}
    rt_plast = jaxfne.RuntimeConfig(
        recurrent_backend="edge_list", enable_hdp=True, hdp_params=hdp_params)
    rt_fixed = jaxfne.RuntimeConfig(
        recurrent_backend="edge_list", enable_hdp=True,
        hdp_params={"hdp_rule": rule,
                    "plasticity_mask": np.zeros_like(mask_ar),
                    "record_weight_trace": False,
                    "hdp_rule_params": rule_params})
    el = model0.params["edge_list"]
    pre = np.asarray(el.pre)
    post = np.asarray(el.post)
    is_pat = np.isin(pre, np.arange(K_PAT))
    on_ar = (pre < N_A) & (post >= N_A + N_I)

    def run_set(rng_counters, cls_offsets, n_trials, hdp, w_init):
        w_cur = None if w_init is None else np.asarray(w_init, float).copy()
        spikes, peaks_list, walls = [], [], []
        for t in range(n_trials):
            trng = np.random.default_rng(rc.trial_seed(seed, rng_counters + t))
            st, info = rc.make_trial_input(trng, 1000.0, 0.5, 0.0, 1.0,
                                           cls_offsets, "P")
            m_use = (model0 if w_cur is None
                     else model0.with_hdp_initial_state(w0=w_cur))
            sim_cfg = jaxfne.simulation(
                duration_ms=1000.0, dt_ms=DT_MS,
                seed=rc.trial_seed(seed, rng_counters + 555 + t),
                runtime=rt_plast if hdp else rt_fixed)
            s = time.perf_counter()
            sig = jaxfne.simulate(m_use, sim_cfg,
                                  jaxfne.SpikeTrainInput(st, n_neurons=n_total))
            walls.append(time.perf_counter() - s)
            if hdp:
                w_cur = np.asarray(m_use.last_hdp_diagnostics()["w_final"], float)
            spk = np.asarray(sig.spikes)
            spikes.append(spk)
            peaks_list.append(info["peak_ms"])
        return spikes, peaks_list, walls, w_cur

    # training leg: 60 P trials carrying w
    w_traj_pat, w_traj_car = [], []
    w_cur = rc.resolved_weights(model0)
    w0_pat = float(w_cur[on_ar & is_pat].mean())
    w0_car = float(w_cur[on_ar & ~is_pat].mean())
    for t in range(n_train):
        trng = np.random.default_rng(rc.trial_seed(seed, 500000 + t))
        st, _ = rc.make_trial_input(trng, 1000.0, 0.5, 0.0, 1.0, offsets_P, "P")
        m_use = model0.with_hdp_initial_state(w0=np.asarray(w_cur))
        sim_cfg = jaxfne.simulation(
            duration_ms=1000.0, dt_ms=DT_MS,
            seed=rc.trial_seed(seed, 500555 + t), runtime=rt_plast)
        sig = jaxfne.simulate(m_use, sim_cfg,
                              jaxfne.SpikeTrainInput(st, n_neurons=n_total))
        w_cur = np.asarray(m_use.last_hdp_diagnostics()["w_final"], float)
        w_traj_pat.append(float(w_cur[on_ar & is_pat].mean()))
        w_traj_car.append(float(w_cur[on_ar & ~is_pat].mean()))
        if (t + 1) % 10 == 0:
            print(f"{arm} s{seed} train {t + 1}/{n_train}: "
                  f"w_pat={w_traj_pat[-1]:.4f} w_car={w_traj_car[-1]:.4f}",
                  flush=True)

    # frozen before/after P/Q test sets (6 trials each class)
    def frozen(w_init, tag):
        sp_p, pk_p, _, _ = run_set(700000 + (1000 if tag == "after" else 0),
                                   offsets_P, 6, False, w_init)
        sp_q, pk_q, _, _ = run_set(700000 + (1000 if tag == "after" else 0)
                                   + 500, offsets_Q, 6, False, w_init)
        res = {"P": [{"spikes": s, "peaks": p, "n_cycles": len(p) - 1}
                     for s, p in zip(sp_p, pk_p)],
               "Q": [{"spikes": s, "peaks": p, "n_cycles": len(p) - 1}
                     for s, p in zip(sp_q, pk_q)]}
        return rc.summarize_condition(
            {"P": res["P"], "Q": res["Q"], "idx_R": np.arange(N_A + N_I, n_total),
             "wall": []}, 5, 1000.0)

    acc_before = frozen(rc.resolved_weights(model0), "before")
    acc_after = frozen(w_cur, "after")

    w_end_pat = w_traj_pat[-1]
    w_end_car = w_traj_car[-1]
    passed = (w_end_pat > w_end_car) and (w_end_pat > w0_pat)
    payload = {
        "arm": arm, "seed": seed, "rule_params": rule_params, "n_train": n_train,
        "w0_pat": w0_pat, "w0_car": w0_car,
        "w_end_pat": w_end_pat, "w_end_car": w_end_car,
        "w_traj_pat": w_traj_pat, "w_traj_car": w_traj_car,
        "pass": bool(passed),
        "frozen_before": {"acc": acc_before["lda_accuracy"],
                          "dprime": acc_before["dprime_count"]},
        "frozen_after": {"acc": acc_after["lda_accuracy"],
                         "dprime": acc_after["dprime_count"]},
        "wall_s_total": time.perf_counter() - t0,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"protoA_{arm}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(json.loads(json.dumps(payload, default=float)), f, indent=2)
    print(f"{arm} s{seed}: w_pat {w0_pat:.4f}->{w_end_pat:.4f} "
          f"w_car {w0_car:.4f}->{w_end_car:.4f} pass={passed} "
          f"acc {acc_before['lda_accuracy']}->{acc_after['lda_accuracy']} "
          f"({payload['wall_s_total']:.0f}s) wrote {path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]),
         int(sys.argv[3]) if len(sys.argv) > 3 else 60)