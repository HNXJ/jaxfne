"""PCL Fig. 2 reproduction: causal inhibitory STDP removes the most predictable spikes.

N'dri, Barbier, Teulière, Triesch, "Predictive Coding Light", Nat Commun 16:8880 (2025).
Event-driven, 4 LIF neurons. Neuron 0 inhibits neurons 1-3; neuron k's input
replaces a fraction p_k of its own Poisson spikes with neuron 0's spikes shifted
by 1 ms (p = 0.9, 0.5, 0.1). Parameters: paper Table 2, simple cells, distant
lateral inhibition. Deviations from the paper are listed in README.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Paper Table 2 (simple cells; distant lateral inhibition), times in ms, potentials in mV
V_TH, V_RESET, V_MIN, TAU_M, TAU_RP, ETA_RP = 10.0, -10.0, -20.0, 18.0, 5.0, 10.0
TAU_LTP = TAU_LTD = 7.0
W_MAX, ETA_LTP, ETA_LTD, ETA_BOUND = 50.0, 2.652, -2.652, 0.02
W_EXC = 15.0  # fixed excitatory input weight (paper Fig. 2 legend)
P_REPLACE = (0.9, 0.5, 0.1)
RATE_HZ, T_MS, LAG_MS = 20.0, 1000.0, 1.0


def make_sample(rng):
    """Input spike times (ms) for neurons 0-3, following the paper's generator."""
    def poisson():
        n = rng.poisson(RATE_HZ * T_MS / 1000.0)
        return np.sort(rng.uniform(0.0, T_MS, n))

    t0 = poisson()
    out = [t0]
    for p in P_REPLACE:
        own = poisson()
        n = len(t0)
        n_keep = min(len(own), int(round((1 - p) * n)))
        kept = rng.choice(own, n_keep, replace=False) if n_keep else np.empty(0)
        copied = rng.choice(t0, n - n_keep, replace=False) + LAG_MS
        out.append(np.sort(np.concatenate([kept, copied])))
    return out


class Net:
    def __init__(self, w0: float):
        self.w = np.full(3, w0)  # inhibitory weights 0 -> k (k = 1..3)

    def run(self, inputs, learn: bool, inhib: bool = True):
        """Event-driven pass over one sample; returns output spike counts per neuron."""
        V = np.zeros(4)
        t_last = np.zeros(4)
        t_spk = np.full(4, -np.inf)
        t_spk_prev = np.full(4, -np.inf)
        pending = [[] for _ in range(4)]  # inhibitory event times since the last post spike
        counts = np.zeros(4, int)
        ev = sorted((t, k) for k in range(4) for t in inputs[k])

        def update(k, t, dv):
            V[k] = V[k] * np.exp(-(t - t_last[k]) / TAU_M) + dv - ETA_RP * np.exp(-(t - t_spk[k]) / TAU_RP)
            V[k] = max(V[k], V_MIN)
            t_last[k] = t

        for t, k in ev:
            update(k, t, W_EXC)
            if V[k] < V_TH:
                continue
            V[k] = V_RESET
            t_spk_prev[k], t_spk[k] = t_spk[k], t
            counts[k] += 1
            if k == 0:
                if inhib:
                    for j in (1, 2, 3):
                        update(j, t, -self.w[j - 1])
                        pending[j].append(t)
            elif learn:
                i = k - 1
                for ti in pending[k]:
                    w = self.w[i]
                    w += (W_MAX - w) * ETA_BOUND * ETA_LTP * np.exp(-(t - ti) / TAU_LTP)
                    if np.isfinite(t_spk_prev[k]):
                        w += w * ETA_BOUND * ETA_LTD * np.exp(-(ti - t_spk_prev[k]) / TAU_LTD)
                    self.w[i] = max(w, 0.0)
                pending[k] = []
        return counts


def run_network(seed: int, w0: float, n_train: int, n_test: int, epochs: int):
    rng = np.random.default_rng(seed)
    train = [make_sample(rng) for _ in range(n_train)]
    test = [make_sample(rng) for _ in range(n_test)]
    net = Net(w0)
    w_hist = [net.w.copy()]
    for _ in range(epochs):
        for s in train:
            net.run(s, learn=True)
        w_hist.append(net.w.copy())
    on = sum(net.run(s, learn=False) for s in test)
    off = sum(net.run(s, learn=False, inhib=False) for s in test)
    return dict(seed=seed, w_hist=np.array(w_hist).tolist(), spikes_inhib=on.tolist(),
                spikes_no_inhib=off.tolist(), suppression=(1 - on[1:] / off[1:]).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--w0", type=float, default=1.0)
    ap.add_argument("--n-train", type=int, default=140)
    ap.add_argument("--n-test", type=int, default=35)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs = [run_network(s, a.w0, a.n_train, a.n_test, a.epochs) for s in a.seeds]
    sup = np.array([r["suppression"] for r in runs])
    wend = np.array([r["w_hist"][-1] for r in runs])
    res = dict(params=vars(a) | {"out": str(a.out)}, runs=runs,
               suppression_mean=sup.mean(0).tolist(), w_end_mean=wend.mean(0).tolist())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("neuron (predictability):  1 (90%)  2 (50%)  3 (10%)")
    print("final w (mean):          ", "  ".join(f"{x:6.2f}" for x in wend.mean(0)))
    print("suppression (mean):      ", "  ".join(f"{x:6.3f}" for x in sup.mean(0)))
    print("suppression (per seed):  ", sup.round(3).tolist())


if __name__ == "__main__":
    main()
