#!/usr/bin/env python3
"""Core-method numerics checks: integrators and autodiff gradients.

Each check recomputes one agreement pinned by the cited test between core
`jaxfne` output and an independent reference, with that test's inputs, and
records the worst-case absolute deviation beside the test's tolerance:

- euler_scan_exponential_decay: `euler_scan` on dy/dt = -y against the
  closed-form exp(-1) (tests/test_v0331_euler_solver.py)
- euler_solve_ode_decay: `solve_ode` with method="euler" on dy/dt = -y
  against the closed-form discrete iterate y_init * 0.9**20
  (tests/test_solvers.py)
- source_scale_gradient_fd: autodiff d(loss)/d(source_scale) against central
  finite differences (tests/test_gradient_checks_v0417.py)

Every check exercises `jaxfne/` code. Two candidates were dropped:
`test_gradient_checks_v0417.py::test_source_scale_linear_rescale_map_matches_finite_differences`
differentiates a loss defined inside the test (`s * base`), not a jaxfne
function; and `test_gsdr2026_kappa_hand.py` pins `window_kappa` in
`artifacts/etudes/papers/gsdr2026/run/data_labels.py`, lane code, not `jaxfne/`.

Writes artifacts/etudes/core_checks/numerics/score.json (one
{error, tolerance, pass, reference} row per check) and README.md beside it:

    python scripts/score_numerics_checks.py         # write both files
    python scripts/score_numerics_checks.py --check # exit 1 past rtol 1e-9
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne._model import _model_with_scalar_parameter

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "etudes" / "core_checks" / "numerics"
SCORE_PATH = OUT_DIR / "score.json"
README_PATH = OUT_DIR / "README.md"
RTOL = 1e-9
NOISE_FRACTION = 0.1  # errors below this fraction of the tolerance are compared as rounding residuals

# One tolerance entry per check, read by every check_* below: each row holds
# the single absolute threshold of that row's tolerance.
TOLERANCES = {
    "euler_scan_exponential_decay": (2e-3,),
    "euler_solve_ode_decay": (1e-5,),
    "source_scale_gradient_fd": (1e-3,),
}


def _row(error, tolerance, reference, passed, seed=None):
    row = {"error": float(error), "tolerance": float(tolerance),
           "pass": bool(passed), "reference": reference}
    if seed is not None:
        row["seed"] = seed
    return row


def check_euler_scan_exponential_decay():
    """Forward-Euler scan of dy/dt = -y against the closed form exp(-1) (atol 2e-3)."""
    def dydt(y, t):
        return -y

    y_final, trajectory = jtfne.euler_scan(
        jnp.array(1.0), t_start=0.0, dt=0.01, n_steps=100, dydt_fn=dydt
    )
    assert trajectory.shape == (100,), "fixture must return one state per step"
    expected = math.exp(-1.0)  # stdlib closed form, independent of jax
    error = abs(float(y_final) - expected)
    (threshold,) = TOLERANCES["euler_scan_exponential_decay"]
    return _row(error, threshold, "closed-form exponential exp(-1)", error <= threshold)


def check_euler_solve_ode_decay():
    """`solve_ode` with method="euler" on dy/dt = -y against y_init * 0.9**20 (atol 1e-5)."""
    def dydt_fn(y, t):
        return -y

    y_init = np.array([1.0, 2.0])
    config = jtfne.SolverConfig(method="euler", dt=0.1)
    y_final, trajectory = jtfne.solve_ode(config, dydt_fn, jnp.asarray(y_init), 0.0, 2.0)
    assert trajectory.shape == (20, 2), "fixture must return one state per step"
    expected = y_init * (0.9 ** 20)  # closed form of the discrete Euler iterate
    error = float(np.abs(np.asarray(y_final, dtype=float) - expected).max())
    (threshold,) = TOLERANCES["euler_solve_ode_decay"]
    return _row(error, threshold, "closed-form Euler iterate y_init * 0.9**20",
                error <= threshold)


_SIM = None


def _gradient_model():
    cfg = (
        jtfne.configuration()
        .network(n=8, cell_types={"E": 0.75, "PV": 0.25})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="probe", modes=["spikes", "V_m", "source"])
    )
    return jtfne.construct(cfg)


def _gradient_sim():
    global _SIM
    if _SIM is None:
        _SIM = jtfne.simulation(seed=3, duration_ms=10.0, dt_ms=0.1)
    return _SIM


def _raster_at(model, s):
    return np.asarray(
        _model_with_scalar_parameter(model, "source_scale", s).simulate(_gradient_sim()).spikes
    )


def check_source_scale_gradient_fd():
    """Autodiff d(loss)/d(source_scale) against central finite differences (atol 1e-3)."""
    model = _gradient_model()
    s0, h = 1.0, 1e-2

    rasters = [_raster_at(model, s) for s in (s0 - h, s0, s0 + h)]
    assert np.array_equal(rasters[0], rasters[1]), (
        "spike raster changed within FD radius; gradient check domain invalid"
    )
    assert np.array_equal(rasters[1], rasters[2])

    def scalar_loss(s):
        m2 = _model_with_scalar_parameter(model, "source_scale", s)
        return jnp.mean(jnp.abs(jnp.asarray(m2.simulate(_gradient_sim()).field.source_proxy)))

    autodiff = float(jax.grad(scalar_loss)(jnp.asarray(s0)))
    lp = float(scalar_loss(s0 + h))
    lm = float(scalar_loss(s0 - h))
    fd = (lp - lm) / (2.0 * h)
    error = abs(autodiff - fd)
    (threshold,) = TOLERANCES["source_scale_gradient_fd"]
    return _row(error, threshold, "central finite differences", error <= threshold, seed=3)


CHECKS = (
    ("euler_scan_exponential_decay", check_euler_scan_exponential_decay),
    ("euler_solve_ode_decay", check_euler_solve_ode_decay),
    ("source_scale_gradient_fd", check_source_scale_gradient_fd),
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


README_TEXT = """# Core-method numerics checks

Worst-case absolute deviations of core numerics (integrators and the
declared autodiff path) against independent references, recomputed by
`scripts/score_numerics_checks.py` with the inputs and tolerances of the
pinned tests. Every check exercises `jaxfne/` code. `score.json` holds one
`{error, tolerance, pass, reference}` row per check; `error` is the
worst-case absolute deviation, so a passing test implies a passing row.
These are agreements of a core method, not a paper replication: there is no
paper value beside ours.

| check | reference | tolerance |
|---|---|---|
| euler_scan_exponential_decay | closed-form exponential exp(-1) | absolute 2e-3 |
| euler_solve_ode_decay | closed-form Euler iterate y_init * 0.9**20 | absolute 1e-5 |
| source_scale_gradient_fd | central finite differences | absolute 1e-3 |

The gradient row reruns the pinned small-network simulation at seed 3 and
records it in `"seed"`. Two candidates were dropped: the linear-rescale
gradient test differentiates a loss defined inside the test, not a jaxfne
function, and `window_kappa` is lane code, not `jaxfne/`.

Rerun:

    python scripts/score_numerics_checks.py         # rewrite score.json (this README is left as is)
    python scripts/score_numerics_checks.py --check # recompute and exit 1 past rtol 1e-9

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
    parser = argparse.ArgumentParser(description="Score the core-method numerics checks.")
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
