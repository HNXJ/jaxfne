"""F1 L5 refutation check: the PCL LTD accumulator factors (README, "L5 is false").

Rule (pcl_h_rule.py): B_e <- B_e + s_i(t) * Y_j(t-1) * a, read at t; then Y_j <- 1 where j spikes,
else Y_j * a; B_e <- 0 where j spikes. Factored: B_e(t) = a^(-tau_j) (z_i(t) - z_i(tau_j)) with
z_i(t) = sum_{t' <= t} s_i(t') a^t' per presynaptic neuron. Exact in rational arithmetic; float64
loses precision as a^(-tau_j) grows.
"""
from fractions import Fraction

import numpy as np


def direct(s, r, a):
    """Accumulator as the rule computes it; the value read at each step, before the reset."""
    Y, B, out = 0 * a, 0 * a, []
    for t in range(len(s)):
        B = B + s[t] * Y * a
        out.append(B)
        Y = 1 + 0 * a if r[t] else Y * a
        if r[t]:
            B = 0 * a
    return out


def factored(s, r, a):
    """Per-neuron z and a per-edge snapshot z(tau) written at post spikes."""
    z, out, tau, p, acc = [], [], None, 1 + 0 * a, 0 * a
    for t in range(len(s)):
        acc = acc + s[t] * p
        z.append(acc)
        out.append(0 * a if tau is None else (z[t] - z[tau]) / a ** tau)
        if r[t]:
            tau = t
        p = p * a
    return out


def trains(seed, T):
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.random(T) < 0.05], [int(v) for v in rng.random(T) < 0.02]


def exact_equal(seed, T=300):
    s, r = trains(seed, T)
    a = Fraction(39, 40)
    d = direct(s, r, a)
    assert max(d) > 0 and sum(r) >= 3, "fixture must reset and accumulate"
    return d == factored(s, r, a)


def float64_rel_err(T, seed=0):
    s, r = trains(seed, T)
    a = float(np.exp(-0.5 / 40))
    d, f = np.array(direct(s, r, a), float), np.array(factored(s, r, a), float)
    return float(np.abs(d - f).max() / np.abs(d).max())


def main():
    for seed in range(3):
        print(f"seed {seed}: exact equality over 300 steps: {exact_equal(seed)}")
    for T in (300, 1000, 3000):
        print(f"float64 T={T}: max relative difference {float64_rel_err(T):.2e}")


if __name__ == "__main__":
    main()
