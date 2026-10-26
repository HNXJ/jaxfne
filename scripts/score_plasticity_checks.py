#!/usr/bin/env python3
"""Plasticity hand-value checks of the jaxfne core.

Each check recomputes one agreement pinned by the plasticity tests between core
output (`jaxfne/`) and an independent reference, with the test's inputs and
tolerances, and records the worst-case deviation beside its tolerance:

- istdp_alpha_paper_value: the derived depression constant against the paper value 0.12
- istdp_pre_depresses: lone-presynaptic-step drives against the hand pairing update
- istdp_post_potentiates: postsynaptic-step drives against the hand pairing update
- hdp_barrier_ratio_100: pure-barrier relaxation against the derived fixed point H*=1
- hdp_barrier_ratio_1: pure-barrier relaxation against the derived fixed point H*=5.05
- rbd_d2a_analytic_bracket: kernel H_K trajectory against the scorer-side closed-form F1 solution

Writes artifacts/etudes/core_checks/plasticity/score.json (one
{error, tolerance, pass, reference} row per check) and README.md beside it:

    python scripts/score_plasticity_checks.py         # write both files
    python scripts/score_plasticity_checks.py --check # exit 1 past rtol 1e-9
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import contextlib
import json
import math

import jax
import jax.numpy as jnp
import numpy as np

from jaxfne.emitters import (
    EdgeList,
    IzhikevichParams,
    simulate_edge_recurrent_izhikevich_dynamic_h_k_recovery as dynamic_h_k_kernel,
    simulate_edge_recurrent_izhikevich_hdp as hdp_kernel,
)
from jaxfne.hdp_rule import HDPRuleContext, get_hdp_rule
from jaxfne.istdp_vogels2011 import (
    ISTDP_RULE_NAME,
    istdp_alpha,
    resolve_istdp_params,
)

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "etudes" / "core_checks" / "plasticity"
SCORE_PATH = OUT_DIR / "score.json"
README_PATH = OUT_DIR / "README.md"
RTOL = 1e-9
NOISE_FRACTION = 0.1  # errors below this fraction of the tolerance are compared as rounding residuals

# Paper and test inputs shared by the iSTDP checks (tests/test_istdp_vogels2011.py).
DT_MS = 0.1
ETA = 1e-2
TAU_MS = 20.0
ALPHA = 2.0 * 3.0 * (TAU_MS / 1000.0)  # 0.12 at the paper values


@contextlib.contextmanager
def _x64():
    """x64 on for one block, restored after (jax.enable_x64 is missing from older jax in the declared range)."""
    orig = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", orig)


# One tolerance entry per check: singletons hold the pinned test's threshold — a
# relative bound for the iSTDP rows (worst per-element relative deviation, zero
# reference entries exact), an absolute bound for the barrier and D2a rows.
TOLERANCES = {
    "istdp_alpha_paper_value": (1e-6,),
    "istdp_pre_depresses": (1e-12,),
    "istdp_post_potentiates": (1e-12,),
    "hdp_barrier_ratio_100": (0.05,),
    "hdp_barrier_ratio_1": (0.1,),
    "rbd_d2a_analytic_bracket": (0.05,),
}


def _row(error, tolerance, reference, passed):
    return {"error": float(error), "tolerance": float(tolerance),
            "pass": bool(passed), "reference": reference}


def _relative_error(got, want):
    """Worst per-element |got - want| / |want|, the pinned tests' own metric.

    The tests compare element-wise at rtol with atol 0, so a reference zero
    admits only an exact zero: any nonzero value there yields inf.
    """
    worst = 0.0
    for g, w in zip(np.asarray(got, dtype=float).flat, np.asarray(want, dtype=float).flat):
        if w == 0.0:
            e = 0.0 if g == 0.0 else float("inf")
        else:
            e = abs(g - w) / abs(w)
        worst = max(worst, e)
    return worst


def check_istdp_alpha_paper_value():
    """Derived alpha against the paper value 0.12 (pytest.approx rel 1e-6)."""
    with _x64():
        got = float(istdp_alpha())
        resolved = float(resolve_istdp_params({})["alpha"])
    paper = 0.12
    error = max(abs(got - paper), abs(resolved - paper))
    (rel,) = TOLERANCES["istdp_alpha_paper_value"]
    tolerance = rel * paper
    return _row(error, tolerance, "paper value alpha = 2*rho0*tau = 0.12 (Vogels et al. 2011)",
                error <= tolerance)


def _istdp_ctx(spikes, pre_sp, aux=None, rule_params=None):
    n = 2
    return HDPRuleContext(
        H=jnp.ones((n,)),
        aux=jnp.zeros((1, 2)) if aux is None else jnp.asarray(aux),  # (A_pre, A_post) of the one edge
        v=jnp.zeros((n,)),
        u=jnp.zeros((n,)),
        spikes=jnp.asarray(spikes, dtype=jnp.float64),
        prev_spikes=jnp.zeros((n,)),
        syn_state=jnp.zeros((1,)),
        w=jnp.asarray([3.0]),
        pre=jnp.asarray([1]),
        post=jnp.asarray([0]),
        exc_mask=jnp.asarray([True]),
        dt=jnp.asarray(DT_MS),
        n_neurons=n,
        rule_params=dict(rule_params or {}),
        key=jax.random.PRNGKey(0),
        pre_sp=None if pre_sp is None else jnp.asarray(pre_sp, dtype=jnp.float64),
    )


def check_istdp_pre_depresses():
    """Lone presynaptic step drives against the hand pairing update (rtol 1e-12)."""
    with _x64():
        _, rule_fn = get_hdp_rule(ISTDP_RULE_NAME)
        upd = rule_fn(_istdp_ctx(spikes=[0.0, 1.0], pre_sp=[1.0]))
        d_aux = np.asarray(upd.d_aux, dtype=float)
        dw = np.asarray(upd.d_theta["edge_weight"], dtype=float)
    want_aux = np.array([[1.0 / DT_MS, 0.0]])
    want_dw = np.array([ETA * (0.0 - ALPHA) / DT_MS])
    error = max(_relative_error(d_aux, want_aux), _relative_error(dw, want_dw))
    (tolerance,) = TOLERANCES["istdp_pre_depresses"]
    return _row(error, tolerance, "hand pairing update w += (0-alpha)*eta, A += 1 (Vogels et al. 2011)",
                error <= tolerance)


def check_istdp_post_potentiates():
    """Postsynaptic step drives against the hand pairing update (rtol 1e-12)."""
    with _x64():
        _, rule_fn = get_hdp_rule(ISTDP_RULE_NAME)
        upd = rule_fn(_istdp_ctx(spikes=[1.0, 0.0], pre_sp=[0.0], aux=[[0.5, 0.0]]))
        d_aux = np.asarray(upd.d_aux, dtype=float)
        dw = np.asarray(upd.d_theta["edge_weight"], dtype=float)
    want_aux = np.array([[-0.5 / TAU_MS, 1.0 / DT_MS]])
    want_dw = np.array([ETA * 0.5 / DT_MS])
    error = max(_relative_error(d_aux, want_aux), _relative_error(dw, want_dw))
    (tolerance,) = TOLERANCES["istdp_post_potentiates"]
    return _row(error, tolerance, "hand pairing update w += A_pre*eta with A_pre = 0.5 (Vogels et al. 2011)",
                error <= tolerance)


def _barrier_relax(barrier_c, barrier_d, H0, n_steps=4000, dt_ms=0.5, tau_0_ms=5.0, N=8, ne=32):
    """Pure-barrier relaxation with the pinned test's inputs (seed 0)."""
    rng = np.random.default_rng(0)
    p = IzhikevichParams(
        a=jnp.full((N,), 0.02), b=jnp.full((N,), 0.2), c=jnp.full((N,), -65.0),
        d=jnp.full((N,), 8.0), drive=jnp.full((N,), 6.0), sign=jnp.ones((N,)),
        W=jnp.zeros((N, N)), v0=jnp.full((N,), -65.0), u0=jnp.full((N,), -13.0),
        source_scale=jnp.ones((N,)), labels=tuple("E" for _ in range(N)),
        layer_labels=tuple("L4" for _ in range(N)), source_calibration_status="x")
    receptor_index = rng.integers(0, 2, ne)
    weight_mag = np.abs(rng.normal(0, 0.3, ne)).astype(np.float32)
    weight = np.where(receptor_index == 0, weight_mag, -weight_mag).astype(np.float32)
    edges = EdgeList(
        pre=jnp.asarray(rng.integers(0, N, ne), jnp.int32),
        post=jnp.asarray(rng.integers(0, N, ne), jnp.int32),
        weight=jnp.asarray(weight),
        receptor_index=jnp.asarray(receptor_index, jnp.int32),
        tau_ms=jnp.full((ne,), 5.0), source_calibration_status="x")
    init_state = {
        "v": p.v0, "u": p.u0,
        "prev_spikes": jnp.zeros((N,), dtype=jnp.float32),
        "syn_state": jnp.zeros((ne,), dtype=jnp.float32),
        "H_final": jnp.full((N,), H0, dtype=jnp.float32),
        "w_final": edges.weight,
    }
    _, _, _, diag = hdp_kernel(
        p, edges, n_steps, dt_ms, jax.random.PRNGKey(0),
        barrier_c=barrier_c, barrier_d=barrier_d, tau_0_ms=tau_0_ms,
        size_scale_override=jnp.ones((N,), dtype=jnp.float32), init_state=init_state,
        alpha=0.0, beta=0.0, gamma=0.0, delta=0.0, C_spike=0.0,
        K_ctrl=0.0, rho_passive=0.0, K_HDP=0.0, noise_scale=0.0,
    )
    H_trace = np.asarray(diag["H_trace"])
    assert bool(np.isfinite(H_trace).all()), "fixture must relax without NaN"
    return float(H_trace[-1].mean())


def check_hdp_barrier_ratio_100():
    """Pure-barrier relaxation against the derived fixed point H* = 1 (atol 0.05)."""
    H_final = _barrier_relax(barrier_c=0.01, barrier_d=1.0, H0=3.0)
    error = abs(H_final - 1.0)
    (threshold,) = TOLERANCES["hdp_barrier_ratio_100"]
    return _row(error, threshold, "derived pure-barrier fixed point H* = 1 at barrier_d/barrier_c = 100 (seed 0)",
                error <= threshold)


def check_hdp_barrier_ratio_1():
    """Pure-barrier relaxation against the derived fixed point H* = 5.05 (atol 0.1)."""
    H_final = _barrier_relax(barrier_c=1.0, barrier_d=1.0, H0=3.0)
    error = abs(H_final - 5.05)
    (threshold,) = TOLERANCES["hdp_barrier_ratio_1"]
    passed = abs(H_final - 1.0) > 1.0 and error <= threshold
    return _row(error, threshold, "derived pure-barrier fixed point H* = 5.05 at barrier_d/barrier_c = 1 (seed 0)",
                passed)


def _f1_closed_form(h0, t_ms, tau_k_ms):
    """Scorer-side closed form H_K(t) = 1 + (H0-1) exp(-t/tau_K), independent of jaxfne."""
    t = np.asarray(t_ms, dtype=np.float64)
    return 1.0 + (float(h0) - 1.0) * np.exp(-t / float(tau_k_ms))


def _isolated_single_cell():
    """One silent neuron with no edges, as in the pinned test: H_K runs autonomously."""
    jdtype = jnp.float32
    params = IzhikevichParams(
        v0=jnp.asarray([-65.0], dtype=jdtype),
        u0=jnp.zeros((1,), dtype=jdtype),
        a=jnp.asarray([0.02], dtype=jdtype),
        b=jnp.asarray([0.2], dtype=jdtype),
        c=jnp.asarray([-65.0], dtype=jdtype),
        d=jnp.asarray([8.0], dtype=jdtype),
        drive=jnp.zeros((1,), dtype=jdtype),
        sign=jnp.ones((1,), dtype=jdtype),
        W=jnp.zeros((1, 1), dtype=jdtype),
        source_scale=jnp.ones((1,), dtype=jdtype),
        labels=("E",),
        layer_labels=("L4",),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    edges = EdgeList(
        pre=jnp.zeros((0,), dtype=jnp.int32),
        post=jnp.zeros((0,), dtype=jnp.int32),
        weight=jnp.zeros((0,), dtype=jdtype),
        receptor_index=jnp.zeros((0,), dtype=jnp.int32),
        tau_ms=jnp.zeros((0,), dtype=jdtype),
    )
    return params, edges


def check_rbd_d2a_analytic_bracket():
    """Kernel H_K trajectory against the scorer-side closed F1 form (atol 0.05)."""
    params, edges = _isolated_single_cell()
    n_steps, dt_ms, tau_k_ms, h0 = 3000, 0.5, 100.0, 0.8
    _, _, _, st = dynamic_h_k_kernel(
        params, edges, n_steps, dt_ms, jax.random.PRNGKey(0),
        h_k0=jnp.asarray([h0], dtype=jnp.float32), tau_k_ms=tau_k_ms, noise_scale=0.0,
    )
    traj = np.asarray(st["H_K_trace"], dtype=float)[:, 0]
    assert traj.shape == (n_steps,), f"kernel must return one H_K per step, got {traj.shape}"
    t = np.array([50.0, 500.0, 1500.0])
    analytic = _f1_closed_form(h0, t, tau_k_ms)
    idx = (t / dt_ms).astype(int) - 1
    error = float(max(abs(traj[ix] - analytic[i]) for i, ix in enumerate(idx)))
    (threshold,) = TOLERANCES["rbd_d2a_analytic_bracket"]
    return _row(error, threshold,
                "scorer-side closed form H_K(t) = 1 + (H0-1) exp(-t/tau_K) (kernel seed 0)",
                error <= threshold)


CHECKS = (
    ("istdp_alpha_paper_value", check_istdp_alpha_paper_value),
    ("istdp_pre_depresses", check_istdp_pre_depresses),
    ("istdp_post_potentiates", check_istdp_post_potentiates),
    ("hdp_barrier_ratio_100", check_hdp_barrier_ratio_100),
    ("hdp_barrier_ratio_1", check_hdp_barrier_ratio_1),
    ("rbd_d2a_analytic_bracket", check_rbd_d2a_analytic_bracket),
)


def compute_all():
    """One score row per check, in CHECKS order."""
    checks = {name: fn() for name, fn in CHECKS}
    live = [row for row in checks.values() if "skipped" not in row]
    return {"checks": checks,
            "pass": bool(live) and all(row["pass"] for row in live),
            "skipped": sorted(name for name, row in checks.items() if "skipped" in row)}


def compare(old, new, rtol=RTOL):
    """Mismatch strings between a stored document and a fresh one; [] means agreement."""
    if set(old.get("checks", ())) != set(new.get("checks", ())):
        return [f"checks: {sorted(old.get('checks', ()))} != {sorted(new.get('checks', ()))}"]
    bad = []
    for name in old["checks"]:
        o, n = old["checks"][name], new["checks"][name]
        if "skipped" in n and "skipped" not in o:
            # This environment lacks the check's optional dependency; the stored row stands.
            continue
        if ("skipped" in o) != ("skipped" in n):
            bad.append(f"{name}: skipped {('skipped' in o)} != {('skipped' in n)}")
            continue
        if "skipped" in o:
            if o["skipped"] != n["skipped"]:
                bad.append(f"{name}.skipped: {o['skipped']!r} != {n['skipped']!r}")
            continue
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


README_TEXT = """# Plasticity hand-value checks

Worst-case deviations of the plasticity core in `jaxfne` (Vogels 2011 inhibitory
STDP rule, barrier-potential H equilibrium, autonomous H_K F1 relaxation) against
independent references, recomputed by `scripts/score_plasticity_checks.py` with the
inputs and tolerances of `tests/test_istdp_vogels2011.py`,
`tests/test_hdp_barrier_equilibrium.py` and
`tests/test_protocol_d_d2a_autonomous_relaxation.py`. `score.json` holds one
`{error, tolerance, pass, reference}` row per check; `error` is the worst-case
deviation in the pinned test's own metric and `tolerance` is the pinned test's
threshold, so a passing test implies a passing row. For the iSTDP step rows the
metric is the worst per-element relative deviation with zero reference entries
exact; elsewhere it is the worst-case absolute deviation. These are hand-value
agreements of a core method, not
a paper replication: the paper value beside ours is the stated constant only.

| check | reference | tolerance |
|---|---|---|
| istdp_alpha_paper_value | paper value alpha = 2*rho0*tau = 0.12 (Vogels et al. 2011) | rel 1e-6 of 0.12 |
| istdp_pre_depresses | hand pairing update w += (0-alpha)*eta, A += 1 (Vogels et al. 2011) | worst per-element relative deviation below 1e-12; zero reference entries exact |
| istdp_post_potentiates | hand pairing update w += A_pre*eta with A_pre = 0.5 (Vogels et al. 2011) | worst per-element relative deviation below 1e-12; zero reference entries exact |
| hdp_barrier_ratio_100 | derived pure-barrier fixed point H* = 1 at barrier_d/barrier_c = 100 (seed 0) | |H - 1| below 0.05 |
| hdp_barrier_ratio_1 | derived pure-barrier fixed point H* = 5.05 at barrier_d/barrier_c = 1 (seed 0) | |H - 5.05| below 0.1, and |H - 1| above 1.0 |
| rbd_d2a_analytic_bracket | scorer-side closed form H_K(t) = 1 + (H0-1) exp(-t/tau_K), against the D2a dynamic-kernel H_K trajectory | max|deviation| below 0.05 |

These iSTDP checks use out-degree 1. The per-synapse trace fix lives in the kernel
LFP lane (todo section 2), not yet on dev; do not add an out-degree check here.

Rerun:

    python scripts/score_plasticity_checks.py         # rewrite score.json (this README is left as is)
    python scripts/score_plasticity_checks.py --check # recompute and exit 1 past rtol 1e-9

The barrier checks run the pure-barrier kernel with seed 0 (4000 steps, dt 0.5 ms).
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
    unchecked = sorted(name for name, row in doc["checks"].items()
                       if "skipped" in row and "skipped" not in stored["checks"][name])
    if unchecked:
        # The stored rows stand; this environment lacks their optional dependency and did not recompute them.
        print(f"not rechecked here: {', '.join(unchecked)}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score the plasticity hand-value checks.")
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
