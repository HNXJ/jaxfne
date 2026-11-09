#!/usr/bin/env python3
"""Core-method checks of the spectral Granger causality in jaxfne.analysis.

Each check recomputes one agreement pinned by tests/test_granger_spectral.py, with that test's
inputs, against a closed form that does not run the core's code path, and records the worst case
beside its tolerance:

- spectral_gc_var1: granger_spectral x -> y against the analytic Geweke ratio of a VAR(1)
  (tests/test_granger_spectral.py:43-62); n 200 000, seed 11, coupling 0.6, dt 1 ms, freqs 5, 10,
  40 Hz; error = worst relative error; tolerance 0.05 (rtol, as the test)
- reverse_direction_near_zero: gc_yx of the same fixture (tests/test_granger_spectral.py:63);
  error = max gc_yx; tolerance 0.01 (strict, as the test)

Geweke's time-domain identity is not scored: jaxfne/analysis/granger.py has no time-domain GC.

Writes artifacts/etudes/core_checks/granger/score.json (one {error, tolerance, pass, reference}
row per check) and README.md beside it:

    python scripts/score_granger_checks.py         # write both files
    python scripts/score_granger_checks.py --check # exit 1 past rtol 1e-9
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math

import numpy as np

from jaxfne.analysis import granger_spectral

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "etudes" / "core_checks" / "granger"
SCORE_PATH = OUT_DIR / "score.json"
README_PATH = OUT_DIR / "README.md"
RTOL = 1e-9
NOISE_FRACTION = 0.1  # errors below this fraction of the tolerance are compared as rounding residuals

# The pinned fixture of tests/test_granger_spectral.py:43-63.
N_SAMPLES = 200_000
SEED = 11
COUPLING = 0.6
ORDER = 1
DT_MS = 1.0
FREQS = np.array([5.0, 10.0, 40.0])

TOLERANCES = {
    "spectral_gc_var1": 0.05,  # relative, the rtol of assert_allclose at tests/test_granger_spectral.py:62
    "reverse_direction_near_zero": 0.01,  # absolute, tests/test_granger_spectral.py:63
}


def _row(error, tolerance, reference, passed):
    return {"error": float(error), "tolerance": float(tolerance),
            "pass": bool(passed), "reference": reference}


def drive_x_to_y(n, seed, coupling):
    """The x_to_y fixture of tests/test_granger_spectral.py:15-36: x[t] = 0.5 x[t-1] + e_x[t] and
    y[t] = 0.5 y[t-1] + coupling x[t-1] + e_y[t], with e_x, e_y standard normal from default_rng(seed)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    ex = rng.standard_normal(n)
    ey = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + ex[t]
        y[t] = 0.5 * y[t - 1] + coupling * x[t - 1] + ey[t]
    return x, y


def analytic_gc_xy(freqs_hz, dt_ms, coupling):
    """Geweke ratio ln(S_yy / (S_yy - |H_yx|^2)) of the VAR(1) with A1 = [[0.5, 0], [coupling, 0.5]]
    and unit noise covariance (tests/test_granger_spectral.py:52-59): H = (I - A1 z)^-1, S = H H*."""
    a1 = np.array([[0.5, 0.0], [coupling, 0.5]])
    out = []
    for f in freqs_hz:
        z = np.exp(-2j * np.pi * f * dt_ms / 1000.0)
        h = np.linalg.inv(np.eye(2) - a1 * z)
        s = h @ h.conj().T
        syy = s[1, 1].real
        out.append(np.log(syy / (syy - abs(h[1, 0]) ** 2)))
    return np.asarray(out)


def _pinned_spectra():
    x, y = drive_x_to_y(N_SAMPLES, SEED, COUPLING)
    return granger_spectral(x, y, DT_MS, ORDER, FREQS)


def check_spectral_gc_var1():
    """granger_spectral x -> y against the analytic Geweke ratio at 5, 10, 40 Hz (test line 43)."""
    got = _pinned_spectra()["gc_xy"]
    want = analytic_gc_xy(FREQS, DT_MS, COUPLING)
    assert np.all(want > 0.05), "fixture must carry a clear x -> y spectrum"
    error = float(np.max(np.abs(got - want) / np.abs(want)))
    tolerance = TOLERANCES["spectral_gc_var1"]
    return _row(error, tolerance,
                f"ln(S_yy / (S_yy - |H_yx|^2)) of the VAR(1) with A1 = [[0.5, 0], [{COUPLING}, 0.5]], "
                f"Sigma = I, H = (I - A1 z)^-1, at {', '.join(f'{f:g}' for f in FREQS)} Hz "
                f"(n {N_SAMPLES}, seed {SEED}; relative)",
                error <= tolerance)


def check_reverse_direction_near_zero():
    """gc_yx of the same fixture stays below the test's bound at every probe frequency (test line 63)."""
    got = _pinned_spectra()["gc_yx"]
    error = float(np.max(got))
    tolerance = TOLERANCES["reverse_direction_near_zero"]
    return _row(error, tolerance,
                f"y -> x spectrum of the x -> y fixture reads zero: max gc_yx at "
                f"{', '.join(f'{f:g}' for f in FREQS)} Hz (absolute)",
                error < tolerance)


CHECKS = (
    ("spectral_gc_var1", check_spectral_gc_var1),
    ("reverse_direction_near_zero", check_reverse_direction_near_zero),
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
            # A rounding-level residual differs across platforms (BLAS); below a tenth of the
            # tolerance in both copies it cannot change the verdict.
            floor = NOISE_FRACTION * min(o["tolerance"], n["tolerance"])
            if key == "error" and o[key] <= floor and n[key] <= floor:
                continue
            bad.append(f"{name}.{key}: {o[key]!r} != {n[key]!r} past rtol {rtol:g}")
    if old.get("pass") != new.get("pass"):
        bad.append(f"pass: {old.get('pass')!r} != {new.get('pass')!r}")
    return bad


README_TEXT = """# Spectral Granger causality core checks

Worst-case deviations of `jaxfne.analysis.granger_spectral` against independent references,
recomputed by `scripts/score_granger_checks.py`. `score.json` holds one `{error, tolerance, pass,
reference}` row per check; `error` is the worst-case deviation and `tolerance` is the bound named in
the table. A passing row means the stated agreement holds at the stated inputs; these are analytic
agreements of a core method, not a paper replication, and there is no paper value beside ours.

| check | pinned by | reference | tolerance |
|---|---|---|---|
| spectral_gc_var1 | tests/test_granger_spectral.py:43 | ln(S_yy / (S_yy - \\|H_yx\\|^2)) of the VAR(1) with A1 = [[0.5, 0], [0.6, 0.5]], Sigma = I, H = (I - A1 z)^-1, z = exp(-2 pi i f dt), at 5, 10, 40 Hz; n 200 000, seed 11, dt 1 ms | worst relative error \\|gc - ref\\| / ref <= 0.05 (rtol as the test) |
| reverse_direction_near_zero | tests/test_granger_spectral.py:63 | the y -> x spectrum of the same fixture reads zero | max gc_yx < 0.01 (strict, as the test) |

Conventions and choices, stated once:

- The fixture is the test's x_to_y drive: x[t] = 0.5 x[t-1] + e_x[t], y[t] = 0.5 y[t-1] + 0.6 x[t-1] + e_y[t],
  with e_x, e_y standard normal from numpy default_rng(11). The scorer's copy is pinned to the test's
  own `_drive` by tests/test_score_granger_checks.py.
- The reference uses the true VAR(1) coefficients. The estimate is a least-squares VAR(1) fit to 200 000
  samples, so the 0.05 bound is the test's own statistical allowance for finite-sample error.
- The tolerances are the pinned test's own.
- Check (c), Geweke's time-domain identity (time-domain GC equals the frequency average of spectral GC),
  is absent: `jaxfne/analysis/granger.py` has no time-domain GC function, so there is nothing to score.
  No time-domain GC is added to jaxfne.

Rerun:

    python scripts/score_granger_checks.py         # rewrite score.json (this README is left as is)
    python scripts/score_granger_checks.py --check # recompute and exit 1 past rtol 1e-9

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
    parser = argparse.ArgumentParser(description="Score the spectral Granger core checks.")
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
