#!/usr/bin/env python3
"""Core-method checks of the field invariants in jaxfne.fields.

Each check recomputes one agreement pinned by a test under tests/, with that test's
inputs, against a reference that does not run the core's code path, and records the
worst-case deviation beside its tolerance:

- operator_superposition: LinearReadout, eeg_proxy_transform and meg_proxy_transform
  superpose and scale (tests/test_global_superposition.py:12, seed 100) against Y = S W^T
- csd_flat_potential_zero: csd_tensor of a constant potential (tests/test_csd_tensor.py:42)
- csd_quadratic_potential: csd_tensor of a quadratic in depth against -phi'' at interior
  contacts (new; the same pinned invariant as tests/test_linear_readouts.py:44 but on the core)
- cable_filter_transfer: cable_filter_sources gain at 5, 20 and 100 Hz against the exact discrete-time
  response of its one-pole cascade (new; tests/test_cable_filter_tensor.py:60 is qualitative only)
- poisson_1d_flux_magnitude: experimental_poisson_1d flux against Q dx (Gauss's law;
  tests/test_experimental_poisson_1d_convergence.py:71)
- poisson_1d_flux_span: flux zero outside the source-sink span and constant inside
  (tests/test_experimental_poisson_1d_convergence.py:56)

Writes artifacts/etudes/core_checks/fields/score.json (one {error, tolerance, pass,
reference} row per check) and README.md beside it:

    python scripts/score_field_invariant_checks.py         # write both files
    python scripts/score_field_invariant_checks.py --check # exit 1 past rtol 1e-9
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

from jaxfne.fields import (
    LinearReadout,
    cable_filter_sources,
    csd_tensor,
    eeg_proxy_transform,
    meg_proxy_transform,
)
from jaxfne.fields.solvers import experimental_poisson_1d

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "etudes" / "core_checks" / "fields"
SCORE_PATH = OUT_DIR / "score.json"
README_PATH = OUT_DIR / "README.md"
RTOL = 1e-9
NOISE_FRACTION = 0.1  # errors below this fraction of the tolerance are compared as rounding residuals

TAU_S = 0.02  # cable time constant of the pinned test (tests/test_cable_filter_tensor.py:75)
DT_MS = 0.5
ORDER = 2


@contextlib.contextmanager
def _x64():
    """x64 on for one block, restored after (jax.enable_x64 is missing from older jax in the declared range)."""
    orig = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", orig)


# One tolerance entry per check. Pairs hold (atol, rtol): tolerance = atol + rtol * scale.
# Singletons hold the single factor: relative to scale, or absolute where the test is absolute.
TOLERANCES = {
    "operator_superposition": (1e-5, 1e-5),  # np.allclose of tests/test_global_superposition.py:29-49
    "csd_flat_potential_zero": (1e-5,),  # absolute, tests/test_csd_tensor.py:46
    "csd_quadratic_potential": (1e-6,),  # relative to max|-phi''|, a new check (float64)
    "cable_filter_transfer": (1e-6,),  # relative to the exact discrete gain: float32 rounding scale
    "poisson_1d_flux_magnitude": (1e-3,),  # relative to Q dx, tests/test_experimental_poisson_1d_convergence.py:82
    "poisson_1d_flux_span": (1e-4,),  # absolute, tests/test_experimental_poisson_1d_convergence.py:67-68
}


def _row(error, tolerance, reference, passed):
    return {"error": float(error), "tolerance": float(tolerance),
            "pass": bool(passed), "reference": reference}


def discrete_cascade_gain(freq_hz, tau_s, dt_ms, order):
    """Exact |H(e^{j 2 pi f dt})| of the realised cascade in cable_filter_sources.

    Each section is y[n] = alpha x[n] + (1 - alpha) y[n-1] with alpha = 1 - exp(-dt/tau), so
    H1(z) = alpha / (1 - (1 - alpha) z^-1) and the order-n cascade is H1**n. On the unit circle
    |H1|**2 = alpha**2 / (1 - 2 (1 - alpha) cos(Omega) + (1 - alpha)**2) with Omega = 2 pi f dt.
    """
    dt_s = dt_ms / 1000.0
    alpha = 1.0 - np.exp(-dt_s / tau_s)
    omega = 2.0 * np.pi * np.asarray(freq_hz, float) * dt_s
    beta = 1.0 - alpha
    mag1 = alpha / np.sqrt(1.0 - 2.0 * beta * np.cos(omega) + beta ** 2)
    return mag1 ** order


def csd_quadratic_reference(a):
    """-phi'' for phi = a z**2 + b z + c: the csd_tensor convention, minus sign and no conductivity."""
    return -2.0 * a


def check_operator_superposition():
    """Superposition and scaling of the three linear readouts (tests/test_global_superposition.py:12, seed 100)."""
    c, n = 4, 10
    rng = np.random.default_rng(100)
    W = rng.standard_normal((c, n)).astype(np.float32)
    s1 = rng.standard_normal((50, n)).astype(np.float32)
    s2 = rng.standard_normal((50, n)).astype(np.float32)
    a = 3.14
    readout = LinearReadout(name="generic_lfp", W=W)
    operators = (
        readout.apply,
        lambda s: eeg_proxy_transform(s, W),
        lambda s: meg_proxy_transform(s, W),
    )
    # Independent reference: the hand formula Y = S W^T in float64 numpy, never the core.
    W64 = W.astype(np.float64)
    ref1 = s1.astype(np.float64) @ W64.T
    ref2 = s2.astype(np.float64) @ W64.T
    scale = float(max(np.abs(ref1).max(), np.abs(ref2).max()))
    assert scale > 0, "fixture must produce a signal"
    atol, rtol = TOLERANCES["operator_superposition"]

    def worst_ratio(got, want):
        # np.allclose element-wise: |got - want| <= atol + rtol |want|; the ratio is <= 1 when it holds.
        return float((np.abs(np.asarray(got, float) - want) / (atol + rtol * np.abs(want))).max())

    error = 0.0
    for op in operators:
        error = max(error, worst_ratio(op(s1 + s2), ref1 + ref2), worst_ratio(op(a * s1), a * ref1))
    tolerance = 1.0
    return _row(error, tolerance,
                "superposition and scaling against Y = S W^T (float64 numpy hand formula; seed 100)",
                error <= tolerance)


def check_csd_flat_potential_zero():
    """A potential constant across contacts has zero second difference (tests/test_csd_tensor.py:42)."""
    phi = jnp.ones((5, 8), dtype=jnp.float32) * 3.7
    assert float(jnp.max(phi) - jnp.min(phi)) == 0.0 and float(phi[0, 0]) != 0.0, "fixture must be a non-zero constant"
    csd = np.asarray(csd_tensor(phi, dz=0.1), float)
    error = float(np.abs(csd).max())
    (atol,) = TOLERANCES["csd_flat_potential_zero"]
    return _row(error, atol, "zero: second difference of a constant (all 8 contacts)", error <= atol)


def check_csd_quadratic_potential():
    """csd_tensor of a quadratic in depth against -phi'' at interior contacts (new check)."""
    dz = 0.1
    z = dz * np.arange(8)
    a = np.array([1.0, -0.5, 2.5])
    b = np.array([0.7, -1.2, 0.0])
    c = np.array([-0.3, 2.0, 0.5])
    phi = a[:, None] * z[None, :] ** 2 + b[:, None] * z[None, :] + c[:, None]
    with _x64():
        csd = np.asarray(csd_tensor(jnp.asarray(phi, dtype=jnp.float64), dz), float)
    ref = np.broadcast_to(csd_quadratic_reference(a)[:, None], phi.shape)
    scale = float(np.abs(ref).max())
    edge_dev = float(np.abs(csd[:, [0, -1]] - ref[:, [0, -1]]).max())
    assert edge_dev > 1e-3 * scale, "fixture must show the edge stencil departs from -phi''"
    error = float(np.abs(csd[:, 1:-1] - ref[:, 1:-1]).max())
    (factor,) = TOLERANCES["csd_quadratic_potential"]
    tolerance = factor * scale
    return _row(error, tolerance,
                "-phi'' = -2a at interior contacts for phi = a z^2 + b z + c (no conductivity, sigma = 1)",
                error <= tolerance)


def check_cable_filter_transfer():
    """cable_filter_sources gain at 5, 20 and 100 Hz against its exact discrete-time response (new check)."""
    T = 4000
    t = np.arange(T) * (DT_MS / 1000.0)
    freqs = np.array([5.0, 20.0, 100.0])
    sources = np.stack([np.sin(2.0 * np.pi * f * t) for f in freqs], axis=1).astype(np.float32)
    tau_s = jnp.full((freqs.size,), TAU_S, dtype=jnp.float32)
    filtered = np.asarray(cable_filter_sources(sources, tau_s, DT_MS, order=ORDER), float)
    # After a 0.4 s transient (20 tau) the window holds whole cycles of each frequency.
    win = slice(800, T)
    gain = np.array([np.std(filtered[win, k]) / np.std(sources[win, k]) for k in range(freqs.size)])
    assert gain[0] > gain[-1], "fixture must show the low-pass ordering"
    ref = discrete_cascade_gain(freqs, TAU_S, DT_MS, ORDER)
    error = float(np.abs(gain / ref - 1.0).max())
    (rtol,) = TOLERANCES["cable_filter_transfer"]
    return _row(error, rtol,
                "exact discrete response of the cascade, |H| = (alpha / |1 - (1-alpha) e^{-j 2 pi f dt}|)^order, "
                "alpha = 1 - exp(-dt/tau), at 5, 20, 100 Hz (tau 20 ms, dt 0.5 ms; relative)",
                error <= rtol)


def _poisson_case(n, q=2.0):
    """The balanced source-sink fixture of tests/test_experimental_poisson_1d_convergence.py:37."""
    dx = 1.0 / (n - 1)
    iface_node = n // 2
    src_node = max(1, n // 6)
    sink_node = min(n - 2, n - n // 6 - 1)
    sigma_face = np.where(np.arange(n - 1) < iface_node, 1.0, 3.0).astype(np.float32)
    sources = np.zeros(n, dtype=np.float32)
    sources[src_node] = q
    sources[sink_node] = -q
    phi, _, manifest = experimental_poisson_1d(jnp.asarray(sources), jnp.asarray(sigma_face), dx)
    assert manifest["convergence_status"] == "converged", f"N={n} must converge"
    phi = np.asarray(phi, dtype=np.float64)
    flux = sigma_face * (phi[1:] - phi[:-1]) / dx
    return dx, src_node, sink_node, flux, float(manifest["residual_norm"])


def check_poisson_1d_flux_magnitude():
    """Interior flux of experimental_poisson_1d equals -Q dx at N = 21, 41, 81 (Gauss's law, signed:
    with the source below the sink the potential falls across the span, so sigma dphi/dx < 0)."""
    q = 2.0
    worst = 0.0
    for n in (21, 41, 81):
        dx, src, sink, flux, residual = _poisson_case(n, q)
        assert residual < 1e-3, f"N={n} residual {residual} past the converged threshold"
        inside_mean = float(np.mean(flux[src:sink]))
        worst = max(worst, abs(inside_mean - (-q * dx)) / (q * dx))
    (rtol,) = TOLERANCES["poisson_1d_flux_magnitude"]
    return _row(worst, rtol,
                "Gauss's law: interior flux = -Q dx at N = 21, 41, 81 (signed, relative)",
                worst <= rtol)


def check_poisson_1d_flux_span():
    """Flux vanishes outside the source-sink span and is constant inside it (N = 81, Gauss's law)."""
    dx, src, sink, flux, residual = _poisson_case(81)
    assert residual < 1e-3, f"N=81 residual {residual} past the converged threshold"
    outside = np.concatenate([flux[:src], flux[sink:]])
    inside = flux[src:sink]
    error = max(float(np.abs(outside).max()), float(np.std(inside)))
    (atol,) = TOLERANCES["poisson_1d_flux_span"]
    return _row(error, atol,
                "Gauss's law: flux zero outside [src, sink) and constant inside (N = 81, absolute)",
                error <= atol)


CHECKS = (
    ("operator_superposition", check_operator_superposition),
    ("csd_flat_potential_zero", check_csd_flat_potential_zero),
    ("csd_quadratic_potential", check_csd_quadratic_potential),
    ("cable_filter_transfer", check_cable_filter_transfer),
    ("poisson_1d_flux_magnitude", check_poisson_1d_flux_magnitude),
    ("poisson_1d_flux_span", check_poisson_1d_flux_span),
)


def compute_all():
    """One score row per check, in CHECKS order."""
    checks = {name: fn() for name, fn in CHECKS}
    return {"checks": checks,
            "pass": bool(checks) and all(row["pass"] for row in checks.values())}


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


README_TEXT = """# Field-invariant core checks

Worst-case deviations of the field and probe operators in `jaxfne.fields` against independent
references, recomputed by `scripts/score_field_invariant_checks.py`. `score.json` holds one
`{error, tolerance, pass, reference}` row per check; `error` is the worst-case deviation and
`tolerance` is the bound named in the table. A passing row means the stated agreement holds at
the stated inputs; these are analytic and hand-formula agreements of a core method, not a
paper replication, and there is no paper value beside ours.

| check | pinned by | reference | tolerance |
|---|---|---|---|
| operator_superposition | tests/test_global_superposition.py:12 | hand formula Y = S W^T in float64 numpy; seed 100; float32 inputs as the test | worst element-wise ratio \\|got - Y\\| / (1e-5 + 1e-5 \\|Y\\|) against 1 (np.allclose as the test) |
| csd_flat_potential_zero | tests/test_csd_tensor.py:42 | zero second difference of a constant (3.7, 5 x 8) | 1e-5 absolute |
| csd_quadratic_potential | new (cf. tests/test_linear_readouts.py:44, which uses a test-local stencil) | -phi'' = -2a at interior contacts for phi = a z^2 + b z + c, float64 | 1e-6 · max\\|-phi''\\| |
| cable_filter_transfer | new (tests/test_cable_filter_tensor.py:60 is qualitative only) | exact discrete response of the cascade at 5, 20, 100 Hz, tau 20 ms, dt 0.5 ms, order 2 | 1e-6 relative |
| poisson_1d_flux_magnitude | tests/test_experimental_poisson_1d_convergence.py:71 | Gauss's law: interior flux = Q dx at N = 21, 41, 81, Q = 2 | 1e-3 relative |
| poisson_1d_flux_span | tests/test_experimental_poisson_1d_convergence.py:56 | Gauss's law: flux zero outside [src, sink), constant inside, N = 81 | 1e-4 absolute |

Conventions and choices, stated once:

- csd_tensor is -(second difference)/dz^2 with edge padding and no conductivity, so the reference is
  -phi'' with sigma = 1 (its docstring: "without the conductivity"). Edge contacts are not scored: the
  edge padding gives a one-sided stencil there, which departs from -phi'' by design (the fixture asserts
  the departure). The reference sign and convention come from the operator's own docstring.
- The cable filter is a fixed linear recursion per section, y[n] = alpha x[n] + (1 - alpha) y[n-1] with
  alpha = 1 - exp(-dt/tau), cascaded `order` times (cable_filter_sources, proxy.py). Its exact response is
  H1(z) = alpha / (1 - (1 - alpha) z^-1), so |H(f)| = (alpha / sqrt(1 - 2 (1 - alpha) cos(Omega) + (1 - alpha)^2))^order
  with Omega = 2 pi f dt. The reference is this closed form, not the continuous RC transfer. The gain is
  measured over 800 to 4000 samples, whole cycles of each frequency, after a 0.4 s transient.
- The tolerances of the new checks (csd_quadratic_potential, cable_filter_transfer) and of
  poisson_1d_flux_magnitude are this scorer's choices, recorded here; the other three are the pinned tests' own.

Rerun:

    python scripts/score_field_invariant_checks.py         # rewrite score.json (this README is left as is)
    python scripts/score_field_invariant_checks.py --check # recompute and exit 1 past rtol 1e-9

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
    parser = argparse.ArgumentParser(description="Score the core field-invariant checks.")
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
