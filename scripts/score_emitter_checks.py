#!/usr/bin/env python3
"""Emitter closed-form checks of the jaxfne emitter kernels.

Each check recomputes one agreement pinned by the emitter tests between core
emitter output and an independent reference, with the test's inputs and
tolerances, and records the worst-case absolute deviation beside its tolerance:

- lif_membrane_decay: LIF membrane decay against 5*exp(-t/tau_m)
- lif_refractory_subtraction: per-arrival refractory subtraction against the hand sum
- adex_lif_limit: Delta=0 AdEx step against the hand Euler update in float32 order
- hh_gate_exponential_euler: HH m-gate update against exact exponential Euler at frozen rates
- graded_synapse_steady_state: clamped graded synapse against tauD/(tauD+tauR)
- rate_ou_stationary_variance: rate OU stationary variance against sigma^2/(2-dt/tau) (seed 3)
- rate_transfer_functions: threshold-linear and sigmoid transfer against hand values

Writes artifacts/etudes/core_checks/emitters/score.json (one
{error, tolerance, pass, reference} row per check) and README.md beside it:

    python scripts/score_emitter_checks.py         # write both files
    python scripts/score_emitter_checks.py --check # exit 1 past rtol 1e-9
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from jaxfne.emitters_adex import AdExParams, adex_step
from jaxfne.emitters_hh import (
    graded_syn_step,
    hh_gate_inf,
    hh_step,
    make_hh_params,
)
from jaxfne.emitters_lif import (
    lif_step,
    make_lif_params,
    simulate_lif_edge,
)
from jaxfne.emitters_rate import (
    make_rate_params,
    rate_transfer,
    simulate_rate_edge,
)

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "etudes" / "core_checks" / "emitters"
SCORE_PATH = OUT_DIR / "score.json"
README_PATH = OUT_DIR / "README.md"
RTOL = 1e-9
NOISE_FRACTION = 0.1  # errors below this fraction of the tolerance are compared as rounding residuals

DT_MS = 0.1  # shared step of the AdEx, HH and graded-synapse pins

# One tolerance entry per check: singletons hold the threshold of that row's
# tolerance (absolute, except adex_lif_limit, which holds the factor of
# `factor * |reference|`, and rate_ou_stationary_variance, a relative error bound).
TOLERANCES = {
    "lif_membrane_decay": (1e-5,),
    "lif_refractory_subtraction": (1e-5,),
    "adex_lif_limit": (1e-6,),
    "hh_gate_exponential_euler": (1e-4,),
    "graded_synapse_steady_state": (1e-2,),
    "rate_ou_stationary_variance": (0.05,),
    "rate_transfer_functions": (1e-9,),
}

_LIF_FLOAT_FIELDS = (
    "tau_m_ms", "v_thresh_mV", "v_reset_mV", "v_min_mV", "eta_rp_mV",
    "tau_rp_ms", "tau_sra_ms", "delta_sra_mV",
)


def _simple_f32():
    """One float32 simple LIF cell, as in the pinned test."""
    base = make_lif_params(("simple",))
    return base._replace(
        **{f: jnp.asarray(getattr(base, f), jnp.float32) for f in _LIF_FLOAT_FIELDS}
    )


def _row(error, tolerance, reference, passed):
    return {"error": float(error), "tolerance": float(tolerance),
            "pass": bool(passed), "reference": reference}


def check_lif_membrane_decay():
    """Ten silent steps from 5 mV against 5*exp(-t/tau_m) (atol 1e-5)."""
    p = _simple_f32()
    out = simulate_lif_edge(p, n_steps=10, dt_ms=0.5, V0_mV=jnp.array([5.0], jnp.float32))
    got = float(out["V"][0])
    want = 5.0 * math.exp(-5.0 / 18.0)
    assert not bool(out["spikes"].any()), "fixture must stay silent"
    error = abs(got - want)
    (tolerance,) = TOLERANCES["lif_membrane_decay"]
    return _row(error, tolerance, "closed-form membrane decay 5*exp(-t/tau_m)",
                error <= tolerance)


def check_lif_refractory_subtraction():
    """One arrival on a just-spiked cell against the hand decay-plus-subtraction sum (atol 1e-5)."""
    p = _simple_f32()
    V = jnp.array([-10.0], jnp.float32)  # just spiked and reset
    A = jnp.array([0.0], jnp.float32)
    t_last = jnp.array([0.0], jnp.float32)
    (V1, _, _), spk = lif_step((V, A, t_last), jnp.array([0.0]), jnp.array([1.0]),
                               0.5, p, 0.5)
    want = float(-10.0 * math.exp(-0.5 / 18.0) - 10.0 * math.exp(-0.5 / 5.0))
    assert not bool(spk[0]), "fixture must stay silent"
    error = abs(float(V1[0]) - want)
    (tolerance,) = TOLERANCES["lif_refractory_subtraction"]
    return _row(error, tolerance, "hand decay-plus-refractory sum", error <= tolerance)


def _adex_one_cell(*, delta, v0=-65.0, current=200.0):
    f = np.float32
    params = AdExParams(
        C_pF=jnp.asarray([f(200.0)]),
        gL_nS=jnp.asarray([f(10.0)]),
        EL_mV=jnp.asarray([f(-60.0)]),
        VT_mV=jnp.asarray([f(-50.0)]),
        Delta_mV=jnp.asarray([f(delta)]),
        Vreset_mV=jnp.asarray([f(-60.0)]),
        tauw_ms=jnp.asarray([f(500.0)]),
        a_nS=jnp.asarray([f(0.0)]),
        b_pA=jnp.asarray([f(0.0)]),
        Tref_ms=jnp.asarray([f(5.0)]),
        Vpeak_mV=jnp.asarray([f(float("nan"))]),
        labels=("E",),
    )
    state = (
        jnp.asarray([f(v0)]),
        jnp.asarray([f(0.0)]),
        jnp.asarray([0], dtype=jnp.int32),
    )
    return params, state, jnp.asarray([f(current)])


def _euler_lif(v0, current):
    """Hand Euler step of the LIF limit in float32, same operation order."""
    f = np.float32
    dV = (f(10.0) * (f(-60.0) - f(v0)) + f(current)) / f(200.0) * f(DT_MS)
    return f(v0) + dV


def check_adex_lif_limit():
    """Subthreshold Delta=0 step against the hand float32 Euler update (rtol 1e-6)."""
    params, state, i_ext = _adex_one_cell(delta=0.0, v0=-65.0, current=200.0)
    (V_new, _, _), spk = adex_step(state, i_ext, params, DT_MS)
    assert bool(np.asarray(spk)[0]) is False, "subthreshold cell must not spike"
    want = float(_euler_lif(-65.0, 200.0))
    error = abs(float(np.asarray(V_new)[0]) - want)
    (factor,) = TOLERANCES["adex_lif_limit"]
    tolerance = factor * abs(want)
    return _row(error, tolerance, "hand Euler LIF update in float32 operation order",
                error <= tolerance)


def check_hh_gate_exponential_euler():
    """One m-gate update at frozen rates against exact exponential Euler (atol 1e-4).

    The m gate starts at a hand-set 0.2, away from its steady state: started at m_inf, any
    relaxation rule returns m_inf and the step would go untested. h and n start at the code's
    steady state; the m update does not read them.
    """
    params = make_hh_params(["E"])
    v = jnp.asarray([-70.0], dtype=jnp.float32)
    m0 = (0.2,)
    _, h0, n0 = hh_gate_inf(v)
    state = (v, jnp.asarray([m0[0]], dtype=jnp.float32),
             jnp.asarray([float(h0[0])], dtype=jnp.float32),
             jnp.asarray([float(n0[0])], dtype=jnp.float32))
    inj = jnp.asarray([0.0], dtype=jnp.float32)
    (_, m1, _, _) = hh_step(state, inj, params, DT_MS)
    vv = -70.0 + 65.0
    am = 0.1 * (25.0 - vv) / (np.exp((25.0 - vv) / 10.0) - 1.0)
    bm = 4.0 * np.exp(-vv / 18.0)
    tau = 1.0 / (am + bm)
    expected = float(m0[0]) * np.exp(-DT_MS / tau) + (am * tau) * (1.0 - np.exp(-DT_MS / tau))
    error = abs(float(m1[0]) - expected)
    (tolerance,) = TOLERANCES["hh_gate_exponential_euler"]
    return _row(error, tolerance, "exact exponential-Euler gate step at frozen rates",
                error <= tolerance)


def check_graded_synapse_steady_state():
    """Clamped depolarized synapse against tauD/(tauD+tauR) (atol 0.01)."""
    s = jnp.asarray([0.1], dtype=jnp.float32)
    for _ in range(2000):
        s = graded_syn_step(s, jnp.asarray([20.0]), tau_d=2.0, tau_r=0.2,
                            slope=5.0, v_th=-20.0, dt=DT_MS)
    want = 2.0 / 2.2
    error = abs(float(s[0]) - want)
    (tolerance,) = TOLERANCES["graded_synapse_steady_state"]
    return _row(error, tolerance, "steady state tauD/(tauD+tauR)", error <= tolerance)


def _ou_relative_error(dt_ms):
    """Relative deviation of the stationary variance at one dt (seed 3)."""
    tau, sigma, n = 10.0, 0.2, 64
    base = make_rate_params(["E"] * n)
    params = replace(base, tau_ms=np.full(n, tau), sigma=np.full(n, sigma),
                     gain=np.zeros(n), r0=np.ones(n))  # phi = 1: an OU unit at mu = 1
    n_steps = int(round(20_000.0 / dt_ms))
    empty = np.zeros(0, dtype=np.int64)
    rates = simulate_rate_edge(
        params, n_steps=n_steps, dt_ms=dt_ms, edge_pre=empty, edge_post=empty,
        edge_weight=np.zeros(0), edge_delay_steps=empty, drive=np.zeros(n), seed=3,
    )["rates"]
    tail = rates[n_steps // 10:]
    assert tail.min() > 0.0, "fixture must stay above the rate floor"
    assert abs(float(tail.mean()) - 1.0) < 0.01, "fixture must sit at mu = 1"
    want = sigma ** 2 / (2.0 - dt_ms / tau)
    return abs(float(tail.var()) - want) / want


def check_rate_ou_stationary_variance():
    """Stationary OU variance at dt 0.1 and 1.0 against sigma^2/(2-dt/tau) (rel 0.05, seed 3)."""
    error = max(_ou_relative_error(dt) for dt in (0.1, 1.0))
    (tolerance,) = TOLERANCES["rate_ou_stationary_variance"]
    return _row(error, tolerance,
                "Euler-Maruyama stationary variance sigma^2/(2-dt/tau), seed 3",
                error <= tolerance)


def check_rate_transfer_functions():
    """Threshold-linear floor/slope and sigmoid bounds, midpoint and slope against hand values (atol 1e-9).

    The sigmoid 2/(1+exp(-g x)) at g = 1: 0 and 2 at -100 and 100, 1 at 0, 2/(1+e^-1) at 1.
    """
    tl = rate_transfer(np.array([-2.0, -1.0, 0.0, 1.0]), "threshold_linear", 1.0)
    sig = rate_transfer(np.array([-100.0, 0.0, 1.0, 100.0]), "sigmoid", 1.0)
    sig_want = [0.0, 1.0, 2.0 / (1.0 + math.exp(-1.0)), 2.0]
    error = max(float(np.abs(tl - [0.0, 0.0, 1.0, 2.0]).max()),
                float(np.abs(sig - sig_want).max()))
    (tolerance,) = TOLERANCES["rate_transfer_functions"]
    return _row(error, tolerance,
                "hand transfer values ([0,0,1,2]; sigmoid 0, 1, 2/(1+e^-1), 2)",
                error <= tolerance)


CHECKS = (
    ("lif_membrane_decay", check_lif_membrane_decay),
    ("lif_refractory_subtraction", check_lif_refractory_subtraction),
    ("adex_lif_limit", check_adex_lif_limit),
    ("hh_gate_exponential_euler", check_hh_gate_exponential_euler),
    ("graded_synapse_steady_state", check_graded_synapse_steady_state),
    ("rate_ou_stationary_variance", check_rate_ou_stationary_variance),
    ("rate_transfer_functions", check_rate_transfer_functions),
)


def compute_all():
    """One score row per check, in CHECKS order."""
    checks = {name: fn() for name, fn in CHECKS}
    return {"checks": checks,
            "pass": bool(checks) and all(row["pass"] for row in checks.values()),
            "skipped": []}


def compare(old, new, rtol=RTOL):
    """Mismatch strings between a stored document and a fresh one; [] means agreement."""
    if set(old.get("checks", ())) != set(new.get("checks", ())):
        return [f"checks: {sorted(old.get('checks', ()))} != {sorted(new.get('checks', ()))}"]
    bad = []
    for name in old["checks"]:
        o, n = old["checks"][name], new["checks"][name]
        if o["pass"] != n["pass"]:
            bad.append(f"{name}.pass: {o['pass']!r} != {n['pass']!r}")
        for key in ("error", "tolerance"):
            if math.isclose(o[key], n[key], rel_tol=rtol, abs_tol=1e-15):
                continue
            # A rounding-level residual differs across platforms (BLAS, float32); below a tenth
            # of the tolerance in both copies it cannot change the verdict.
            floor = NOISE_FRACTION * min(o["tolerance"], n["tolerance"])
            if key == "error" and o[key] <= floor and n[key] <= floor:
                continue
            bad.append(f"{name}.{key}: {o[key]!r} != {n[key]!r} past rtol {rtol:g}")
    if old.get("pass") != new.get("pass"):
        bad.append(f"pass: {old.get('pass')!r} != {new.get('pass')!r}")
    return bad


README_TEXT = """# Emitter closed-form checks

Worst-case deviations of the jaxfne emitter kernels (LIF, AdEx, HH, graded
synapse, rate OU, transfer functions) against independent references,
recomputed by `scripts/score_emitter_checks.py` with the inputs and tolerances of
`tests/test_emitters_lif.py`, `tests/test_adex_lif_limit.py`,
`tests/test_emitters_hh.py` and `tests/test_emitters_rate.py`. `score.json` holds one
`{error, tolerance, pass, reference}` row per check; `error` is the worst-case
absolute deviation (`rate_ou_stationary_variance` records the relative deviation and
its relative tolerance) and `tolerance` is the pinned test's tolerance, so a passing
test implies a passing row. These are analytic agreements of a core method, not a
paper replication: there is no paper value beside ours.

| check | reference | tolerance |
|---|---|---|
| lif_membrane_decay | closed-form membrane decay 5*exp(-t/tau_m) | 1e-5 |
| lif_refractory_subtraction | hand decay-plus-refractory sum | 1e-5 |
| adex_lif_limit | hand Euler LIF update in float32 operation order | 1e-6·|want| |
| hh_gate_exponential_euler | exact exponential-Euler gate step at frozen rates | 1e-4 |
| graded_synapse_steady_state | steady state tauD/(tauD+tauR) | 0.01 |
| rate_ou_stationary_variance | Euler-Maruyama stationary variance sigma^2/(2-dt/tau), seed 3 | relative 0.05 |
| rate_transfer_functions | hand transfer values ([0,0,1,2]; sigmoid 0, 1, 2/(1+e^-1), 2) | 1e-9 |

Rerun:

    python scripts/score_emitter_checks.py         # rewrite score.json (this README is left as is)
    python scripts/score_emitter_checks.py --check # recompute and exit 1 past rtol 1e-9

The suite runs under `JAX_PLATFORMS=cpu`.
"""


def write(doc):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORE_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    if not README_PATH.is_file():
        README_PATH.write_text(README_TEXT, encoding="utf-8", newline="\n")


def check(doc):
    try:
        stored = json.loads(SCORE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read {SCORE_PATH}: {exc}", file=sys.stderr)
        return 1
    bad = compare(stored, doc)
    if bad:
        for line in bad:
            print(f"mismatch: {line}", file=sys.stderr)
        return 1
    print(f"fresh: {SCORE_PATH} matches a recomputation within rtol {RTOL:g} "
          f"(errors under {NOISE_FRACTION:g} x tolerance in both copies count as rounding residuals)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score the emitter closed-form checks.")
    parser.add_argument("--check", action="store_true",
                        help="recompute and exit 1 past rtol 1e-9")
    args = parser.parse_args(argv)
    doc = compute_all()
    if args.check:
        return check(doc)
    write(doc)
    print(f"wrote {SCORE_PATH} ({len(doc['checks'])} checks, pass={doc['pass']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
