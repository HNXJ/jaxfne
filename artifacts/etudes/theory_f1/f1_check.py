"""F1 checks: state reduction for HDP traces, as declared in README.md beside this file.

Numpy only. The direct per-edge trace A_e and the factored form (one H trace x per
neuron and class, one snapshot S per edge), with an eager and a lazy snapshot.
Checks 1-4 follow the "Checks" section of README.md.

Run: python artifacts/etudes/theory_f1/f1_check.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE / "f1_check.json"

T = 20000
P_SPIKE = 0.05
P_RESET = 0.02
SEEDS = (0, 1, 2, 3, 4)
TOL_FLOAT64 = 1e-12
NEG_MIN = 1e-3
C_MAX = 10.0
U32 = 2.0 ** -24  # float32 unit roundoff (round to nearest)

A_SLOW = math.exp(-0.5 / 40)
A_FAST = math.exp(-0.5 / 7)
D2_M = np.array([[A_SLOW, 0.1], [0.0, A_FAST]])
D2_B = np.array([1.0, 0.5])


def cases():
    """Edge classes (M, b) for checks 1 and 2."""
    return {
        "scalar_slow": (np.array([[A_SLOW]]), np.array([1.0])),
        "scalar_fast": (np.array([[A_FAST]]), np.array([1.0])),
        "d2": (D2_M, D2_B),
    }


def trains(seed):
    """Presynaptic spikes s (prob P_SPIKE per step) and postsynaptic resets r (prob P_RESET)."""
    rng = np.random.default_rng(seed)
    s = (rng.random(T) < P_SPIKE).astype(np.float64)
    r = (rng.random(T) < P_RESET).astype(np.float64)
    return s, r


def direct(M, b, s, r, dtype=np.float64):
    """Direct trace: Ahat(t) = M A(t-1) + b s(t); A(t) = (1 - r(t)) Ahat(t).

    Returns (Ahat, A), each of shape (T, d).
    """
    M = np.asarray(M, dtype=dtype)
    b = np.asarray(b, dtype=dtype)
    s = np.asarray(s, dtype=dtype)
    r = np.asarray(r, dtype=dtype)
    one = dtype(1)
    n = s.shape[0]
    Ahat = np.zeros((n, b.shape[0]), dtype=dtype)
    A = np.zeros_like(Ahat)
    a_prev = np.zeros(b.shape[0], dtype=dtype)
    for t in range(n):
        ahat = M @ a_prev + b * s[t]
        Ahat[t] = ahat
        a_prev = (one - r[t]) * ahat
        A[t] = a_prev
    return Ahat, A


def factored(M, b, s, r, lazy=False, dtype=np.float64):
    """Factored form: x(t) = M x(t-1) + b s(t), never reset;
    S(t) = (1 - r(t)) M S(t-1) + r(t) x(t); Ahat(t) = x(t) - M S(t-1); A(t) = x(t) - S(t).

    lazy=False updates S every step. lazy=True writes S only at resets of the neuron and
    evaluates S(t) = M^(t - tau) x(tau), tau the last reset at or before t (S = 0 before
    the first reset).

    Returns (Ahat, A), each of shape (T, d).
    """
    M = np.asarray(M, dtype=dtype)
    b = np.asarray(b, dtype=dtype)
    s = np.asarray(s, dtype=dtype)
    r = np.asarray(r, dtype=dtype)
    one = dtype(1)
    n = s.shape[0]
    d = b.shape[0]
    Ahat = np.zeros((n, d), dtype=dtype)
    A = np.zeros_like(Ahat)
    x_prev = np.zeros(d, dtype=dtype)
    zero = np.zeros(d, dtype=dtype)
    if not lazy:
        S_prev = zero
        for t in range(n):
            x = M @ x_prev + b * s[t]
            Ahat[t] = x - M @ S_prev
            S = (one - r[t]) * (M @ S_prev) + r[t] * x
            A[t] = x - S
            x_prev, S_prev = x, S
        return Ahat, A
    tau, snap = None, zero
    for t in range(n):
        x = M @ x_prev + b * s[t]
        S_prev = zero if tau is None else np.linalg.matrix_power(M, t - 1 - tau) @ snap
        Ahat[t] = x - M @ S_prev
        if r[t] != 0:
            tau, snap = t, x
        S = zero if tau is None else np.linalg.matrix_power(M, t - tau) @ snap
        A[t] = x - S
        x_prev = x
    return Ahat, A


def _max_abs(x):
    return float(np.max(np.abs(x)))


def check1_rel(M, b, seed):
    """Check 1: max |direct - factored| over Ahat and A, relative to max |A|."""
    s, r = trains(seed)
    Dh, Da = direct(M, b, s, r)
    Fh, Fa = factored(M, b, s, r)
    diff = max(_max_abs(Dh - Fh), _max_abs(Da - Fa))
    return diff / _max_abs(Da)


def check2_rel(M, b, seed):
    """Check 2: max |eager - lazy| over Ahat and A, relative to max |A| (eager)."""
    s, r = trains(seed)
    Eh, Ea = factored(M, b, s, r, lazy=False)
    Lh, La = factored(M, b, s, r, lazy=True)
    diff = max(_max_abs(Eh - Lh), _max_abs(Ea - La))
    return diff / _max_abs(Ea)


def check3_diff(seed):
    """Check 3, negative control: direct with a' = A_FAST; factored trace built with a = A_SLOW.

    Returns max |A_direct - A_shared| over the run.
    """
    s, r = trains(seed)
    _, A_direct = direct(np.array([[A_FAST]]), np.array([1.0]), s, r)
    _, A_shared = factored(np.array([[A_SLOW]]), np.array([1.0]), s, r)
    return _max_abs(A_direct - A_shared)


def check4_diff(a, seed):
    """Check 4: float32 direct vs float32 factored, scalar decay a. Returns max |A_d - A_f|."""
    s, r = trains(seed)
    M = np.array([[a]])
    b = np.array([1.0])
    _, Ad = direct(M, b, s, r, dtype=np.float32)
    _, Af = factored(M, b, s, r, dtype=np.float32)
    assert Ad.dtype == np.float32 and Af.dtype == np.float32
    return _max_abs(Ad.astype(np.float64) - Af.astype(np.float64))


def main():
    cs = cases()
    c1 = {name: max(check1_rel(M, b, sd) for sd in SEEDS) for name, (M, b) in cs.items()}
    c2 = {name: max(check2_rel(M, b, sd) for sd in SEEDS) for name, (M, b) in cs.items()}
    ok1 = all(v <= TOL_FLOAT64 for v in c1.values())
    ok2 = all(v <= TOL_FLOAT64 for v in c2.values())

    c3_list = [check3_diff(sd) for sd in SEEDS]
    c3_min, c3_max = min(c3_list), max(c3_list)
    ok3 = c3_min > NEG_MIN

    c4 = {}
    for name, a in (("slow", A_SLOW), ("fast", A_FAST)):
        diff = max(check4_diff(a, sd) for sd in SEEDS)
        C = diff / (U32 / (1.0 - a) ** 2)
        c4[name] = {"a": a, "max_abs_diff": diff, "C": C, "pass": C <= C_MAX}
    ok4 = all(v["pass"] for v in c4.values())

    out = {
        "dtype": {"checks_1_3": "float64", "check_4": "float32"},
        "T": T,
        "p_spike": P_SPIKE,
        "p_reset": P_RESET,
        "seeds": list(SEEDS),
        "u_float32": U32,
        "check1": {"max_rel_by_case": c1, "threshold": TOL_FLOAT64, "pass": ok1},
        "check2": {"max_rel_by_case": c2, "threshold": TOL_FLOAT64, "pass": ok2},
        "check3": {"per_seed_max_abs": c3_list, "min_over_seeds": c3_min,
                   "threshold": NEG_MIN, "pass": ok3},
        "check4": {"slow": c4["slow"], "fast": c4["fast"], "threshold_C": C_MAX, "pass": ok4},
    }
    JSON_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    def tag(ok):
        return "PASS" if ok else "FAIL"

    print(f"check 1 L1 factorization float64: max rel diff scalar_slow={c1['scalar_slow']:.3e} "
          f"scalar_fast={c1['scalar_fast']:.3e} d2={c1['d2']:.3e} (threshold {TOL_FLOAT64:.0e}): {tag(ok1)}")
    print(f"check 2 L2 lazy vs eager float64: max rel diff scalar_slow={c2['scalar_slow']:.3e} "
          f"scalar_fast={c2['scalar_fast']:.3e} d2={c2['d2']:.3e} (threshold {TOL_FLOAT64:.0e}): {tag(ok2)}")
    print(f"check 3 L3 negative control float64: min over seeds max abs diff={c3_min:.3e} "
          f"(max {c3_max:.3e}) (threshold >{NEG_MIN:.0e}): {tag(ok3)}")
    print(f"check 4 L4 float32 vs C*u/(1-a)^2: slow a={A_SLOW:.6f} max abs diff={c4['slow']['max_abs_diff']:.3e} "
          f"C={c4['slow']['C']:.3e}; fast a={A_FAST:.6f} max abs diff={c4['fast']['max_abs_diff']:.3e} "
          f"C={c4['fast']['C']:.3e} (threshold C<={C_MAX:g}): {tag(ok4)}")
    return 0 if (ok1 and ok2 and ok3 and ok4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
