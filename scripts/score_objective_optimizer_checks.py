#!/usr/bin/env python3
"""Core-method checks of the objective and optimizer stages (jaxfne.objectives, jaxfne.optim).

Each check recomputes one agreement the core code is meant to satisfy, with small inputs and
fixed seeds, against an independent reference (a closed form, a hand formula, or a NumPy
computation written here). It records the worst-case deviation beside its tolerance and names
its stage ("objective" or "optimizer").

Writes artifacts/etudes/core_checks/objectives/score.json (one {error, tolerance, pass,
reference, stage} row per check, plus the dropped checks with reasons) and README.md beside it:

    python scripts/score_objective_optimizer_checks.py         # write both files
    python scripts/score_objective_optimizer_checks.py --check # exit 1 past rtol 1e-9
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import contextlib
import functools
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxfne import objectives as ob
from jaxfne.optim import core as oc
from jaxfne.optim.bounds import enforce_parameter_bounds
from jaxfne.optim.gsgd import step_gsgd_transform

assert "site-packages" not in sys.modules["jaxfne"].__file__, (
    "P-025: expected the repo jaxfne on sys.path, not site-packages"
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "etudes" / "core_checks" / "objectives"
SCORE_PATH = OUT_DIR / "score.json"
README_PATH = OUT_DIR / "README.md"
RTOL = 1e-9
NOISE_FRACTION = 0.1  # errors below this fraction of the tolerance are compared as rounding residuals

# tolerance = factor * scale, scale being the reference's magnitude (1.0 for ratios and counts).
# sdr, gsdr and gsgd: relative deviation from the closed-form trajectory 0.8^100 (float64 rounding).
# agsdr: element-wise agreement with the independent float64 NumPy recurrence below.
TOLERANCES = {
    "polarity_null_flips_sign": (1e-12,),
    "layer_shuffle_null_permutes": (1e-12,),
    "uniform_gain_null_scales": (1e-12,),
    "synchrony_metric_limits": (1e-12,),
    "null_samples_closed_form": (0.5,),
    "spectrolaminar_score_known_profile": (1e-12,),
    "box_projection_is_clip": (1e-12,),
    "sdr_quadratic_minimum": (1e-6,),
    "gsdr_quadratic_minimum": (1e-6,),
    "agsdr_quadratic_minimum": (1e-12,),
    "gsgd_quadratic_minimum": (1e-6,),
}

SGD_LR = 0.1  # inner optax.sgd learning rate of the sdr, gsdr and agsdr rows
GSGD_ETA = 0.1  # step size of the plain gradient step; contraction per step is 1 - 2 eta = 0.8
XSTAR = np.array([1.0, -2.0, 0.5, 3.0])  # closed-form minimum of f(x) = ||x - x*||^2
X_START = np.zeros(4)
STEPS = 100
KEY_SEED = 0  # PRNG key seed of the stochastic transforms (their noise is switched off)

# (relative standard error, count) with count = ceil(1 + 1 / (2 rse^2)), derived by hand in the README;
# rse 0.4 gives 4.125, so ceil (5), round (4) and floor (4) disagree.
NULL_SAMPLE_CASES = ((0.5, 3), (0.4, 5), (0.25, 9), (0.125, 33))

DROPPED = {
    "phase_null_keeps_power": (
        "phase_randomized_null multiplies each entry by the cosine of an independent uniform angle "
        "(jaxfne/objectives.py). Its inputs are band-power profiles, not time series, so a spectral "
        "invariant does not apply; no check was written for it."
    ),
    "optax_adam_quadratic_minimum": (
        "optax_adam returns an OptimizerSpec declaration (jaxfne/optim/core.py); its update runs only "
        "inside Model.tune(), not on a plain function, so a quadratic run would test optax, not core code."
    ),
}


@contextlib.contextmanager
def _x64():
    """x64 on for one block, restored after (jax.enable_x64 is missing from older jax in the declared range)."""
    orig = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", orig)


def _f64(fn):
    """Run one check under x64 so its float64 fixtures stay float64."""

    @functools.wraps(fn)
    def wrapper():
        with _x64():
            return fn()

    return wrapper


def _row(name, error, scale, reference, stage):
    """One score row; the tolerance is the check's factor times its reference scale."""
    (factor,) = TOLERANCES[name]
    tolerance = factor * scale
    return {"error": float(error), "tolerance": float(tolerance),
            "pass": bool(error <= tolerance), "reference": reference, "stage": stage}


def _readout(seed, n=64):
    """Two random profiles, the shape the null generators take."""
    rng = np.random.default_rng(seed)
    return {"alpha_beta": rng.normal(size=n), "gamma": rng.normal(size=n)}


def _profile_scale(readout):
    return float(max(np.abs(readout["alpha_beta"]).max(), np.abs(readout["gamma"]).max()))


def check_polarity_null_flips_sign():
    """source_polarity_flip_null returns minus each profile (its docstring: invert profiles)."""
    r = _readout(11)
    out = ob.source_polarity_flip_null(r)
    error = max(np.abs(out["alpha_beta_shuffled"] - (-r["alpha_beta"])).max(),
                np.abs(out["gamma_shuffled"] - (-r["gamma"])).max())
    return _row("polarity_null_flips_sign", error, _profile_scale(r),
                "minus its input, exact negation of both profiles", "objective")


def check_layer_shuffle_null_permutes():
    """layer_shuffle_null returns a permutation of each profile's rows (sorted values equal the input's)."""
    r = _readout(12)
    out = ob.layer_shuffle_null(r, rng=np.random.default_rng(13))
    assert not np.array_equal(out["alpha_beta_shuffled"], r["alpha_beta"]), "fixture must move rows"
    error = max(np.abs(np.sort(out["alpha_beta_shuffled"]) - np.sort(r["alpha_beta"])).max(),
                np.abs(np.sort(out["gamma_shuffled"]) - np.sort(r["gamma"])).max())
    return _row("layer_shuffle_null_permutes", error, _profile_scale(r),
                "sorted values of the input, the multiset a permutation keeps", "objective")


def check_uniform_gain_null_scales():
    """uniform_gain_null scales both profiles by one gain, uniform on [gain_min, 1].

    The function returns no gain, so the reference draws it again: the first value of a
    generator with the same seed, which is the value the function draws.
    """
    r = _readout(14)
    seed = 15
    gain = float(np.random.default_rng(seed).uniform(0.1, 1.0))
    assert 0.1 <= gain <= 1.0, "fixture gain must lie in the documented range"
    out = ob.uniform_gain_null(r, gain_min=0.1, rng=np.random.default_rng(seed))
    error = max(np.abs(out["alpha_beta_shuffled"] - gain * r["alpha_beta"]).max(),
                np.abs(out["gamma_shuffled"] - gain * r["gamma"]).max())
    return _row("uniform_gain_null_scales", error, _profile_scale(r),
                "input times the seeded generator's first uniform(0.1, 1) draw", "objective")


def check_synchrony_metric_limits():
    """Identical channels give mean pairwise correlation 1; complementary channels give -1, clipped to 0."""
    col = np.random.default_rng(16).integers(0, 2, size=200).astype(float)
    assert col.min() != col.max(), "fixture must vary in time"
    identical = np.tile(col[:, None], (1, 5))
    complementary = np.column_stack([col, 1.0 - col])
    got_identical = ob.compute_synchrony_metric(identical, bin_ms=1.0, dt_ms=1.0)
    got_complementary = ob.compute_synchrony_metric(complementary, bin_ms=1.0, dt_ms=1.0)
    error = max(abs(got_identical - 1.0), abs(got_complementary - 0.0))
    return _row("synchrony_metric_limits", error, 1.0,
                "closed form: identical channels 1; one complementary pair r = -1, clipped to 0", "objective")


def check_null_samples_closed_form():
    """null_samples_for_sigma_precision gives ceil(1 + 1 / (2 rse^2)) at four precisions."""
    error = max(abs(ob.null_samples_for_sigma_precision(rse) - n) for rse, n in NULL_SAMPLE_CASES)
    return _row("null_samples_closed_form", error, 1.0,
                "n = ceil(1 + 1/(2 rse^2)): 3, 5, 9 and 33 samples at rse 0.5, 0.4, 0.25 and 0.125", "objective")


def check_spectrolaminar_score_known_profile():
    """An ideal readout scores 100; a profile against its layer reversal scores 100 exp(-2.4)."""
    p = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    ideal = ob.spectrolaminar_profile_score({"alpha_beta": p.copy(), "gamma": p.copy()}, p, p)
    reversed_readout = {"alpha_beta": p[::-1].copy(), "gamma": p[::-1].copy()}
    flipped = ob.spectrolaminar_profile_score(reversed_readout, p, p)
    error = max(abs(ideal["profile_score_percent"] - 100.0),
                abs(flipped["profile_score_percent"] - 100.0 * math.exp(-2.4)))
    return _row("spectrolaminar_score_known_profile", error, 100.0,
                "100 for an ideal readout; 100 exp(-3 x 0.8) for the reversed profile (MSE 0.4 per band)",
                "objective")


@_f64
def check_box_projection_is_clip():
    """enforce_parameter_bounds equals np.clip of each proposal row to its bounds."""
    rng = np.random.default_rng(41)
    props = rng.normal(scale=3.0, size=(50, 4))
    lows = np.array([-1.0, -1.0, -1.0, -1.0])
    highs = np.array([1.0, 0.5, 2.0, 1.0])
    assert (props < lows).any() and (props > highs).any(), "fixture must clip both ends"
    want = np.clip(props, lows[None, :], highs[None, :])
    got = np.asarray(enforce_parameter_bounds(jnp.asarray(props), lows, highs))
    return _row("box_projection_is_clip", np.abs(got - want).max(), np.abs(want).max(),
                "np.clip of each row to [lows, highs]", "optimizer")


AGSDR_RECURRENCE = ("independent float64 NumPy recurrence of the transform update "
                    "(U_t = (1 - a_t) D_t, D_t = -0.1 grad_t, a_0 = 0.5, "
                    "v_{t+1} = 0.99 v_t + 0.01 var(D_t), "
                    "a_{t+1} = clip((u + eps) / (u + v_{t+1} + 2 eps)), u = 0, eps = 1e-6; "
                    "loss falls every step so no deselection reset fires); "
                    "error = max|x_core - x_ref| / max|x_0 - x*|")
QUAD_TRAJECTORY = ("closed-form trajectory: ||x_100 - x*|| / ||x_0 - x*|| = 0.8^100 from x_0 = 0, "
                   "x* = (1, -2, 0.5, 3); error = relative deviation from 0.8^100")


def _ratio(x):
    return float(np.linalg.norm(np.asarray(x) - XSTAR) / np.linalg.norm(X_START - XSTAR))


TRAJECTORY = (1.0 - 2.0 * SGD_LR) ** STEPS  # exact contraction 0.8^100 of a gradient step of 0.1


def _trajectory_error(x):
    """Relative deviation of the error ratio from the closed-form 0.8^100: two-sided, so a
    wrong step size fails as surely as a step that does not converge."""
    return abs(_ratio(x) - TRAJECTORY) / TRAJECTORY


def _run_transform(tx):
    """Drive an Optax-style transform on f(x) = ||x - x*||^2; its updates are added to x."""
    key = jax.random.PRNGKey(KEY_SEED)
    params = jnp.asarray(X_START)
    state = tx.init(params)
    for _ in range(STEPS):
        grad = 2.0 * (params - XSTAR)
        loss = float(np.sum((np.asarray(params) - XSTAR) ** 2))
        updates, state = tx.update(grad, state, params, key=key, loss=loss)
        params = params + updates
    return params


def _agsdr_reference_trajectory():
    """Independent float64 NumPy recurrence of the agsdr_transform update (no jaxfne import).

    With stochastic_scale = 0 and inner optax.sgd(0.1), the transform update is
    U_t = (1 - a_t) D_t with D_t = -0.1 grad_t = -0.2 (x_t - x*), from a_0 = 0.5 and
    v_0 = u_0 = 0. After each step v_{t+1} = 0.99 v_t + 0.01 var(D_t), the population
    variance over the flattened update, and a_{t+1} = clip((u + eps) / (u + v + 2 eps))
    with eps = 1e-6; u stays 0 because the stochastic variance is 0. The loss falls every
    step from X_START, which keeps the deselection branch inactive (asserted below).
    """
    x = np.asarray(X_START, dtype=np.float64).copy()
    alpha = 0.5
    var_sup = 0.0
    var_unsup = 0.0
    eps = 1e-6
    decay = 0.99
    prev_loss = float(np.sum((x - XSTAR) ** 2))
    for _ in range(STEPS):
        grad = 2.0 * (x - XSTAR)
        inner = -SGD_LR * grad
        x = x + (1.0 - alpha) * inner
        loss = float(np.sum((x - XSTAR) ** 2))
        assert loss < prev_loss, "fixture must improve every step so no deselection reset fires"
        prev_loss = loss
        var_sup = decay * var_sup + (1.0 - decay) * float(np.var(inner))
        alpha = float(min(1.0, max(0.0, (var_unsup + eps) / (var_unsup + var_sup + 2.0 * eps))))
    return x


def _agsdr_agreement_error(x_core, x_ref):
    """Max element deviation from the recurrence, scaled by the start displacement max|x_0 - x*|.

    Float64 rounding of x near x* is about 1e-16 of this scale; a halved step leaves
    about 3e-5 and a_0 = 0 about 2.5e-11 (README).
    """
    scale = float(np.abs(np.asarray(X_START, dtype=np.float64) - XSTAR).max())
    return float(np.abs(np.asarray(x_core) - x_ref).max() / scale)


@_f64
def check_sdr_quadratic_minimum():
    """sdr_transform reaches the quadratic minimum with its stochastic term off."""
    tx = oc.sdr_transform(inner_optimizer=optax.sgd(SGD_LR), stochastic_scale=0.0)
    return _row("sdr_quadratic_minimum", _trajectory_error(_run_transform(tx)), 1.0,
                QUAD_TRAJECTORY + "; inner optax.sgd(0.1), noise off", "optimizer")


@_f64
def check_gsdr_quadratic_minimum():
    """gsdr_transform reaches the quadratic minimum with its stochastic term off."""
    tx = oc.gsdr_transform(inner_optimizer=optax.sgd(SGD_LR), stochastic_scale=0.0)
    return _row("gsdr_quadratic_minimum", _trajectory_error(_run_transform(tx)), 1.0,
                QUAD_TRAJECTORY + "; inner optax.sgd(0.1), noise off", "optimizer")


@_f64
def check_agsdr_quadratic_minimum():
    """agsdr_transform (EXPERIMENTAL in its docstring) agrees element-wise with the
    independent float64 NumPy recurrence, with noise off."""
    tx = oc.agsdr_transform(inner_optimizer=optax.sgd(SGD_LR), stochastic_scale=0.0)
    x_core = np.asarray(_run_transform(tx), dtype=np.float64)
    return _row("agsdr_quadratic_minimum",
                _agsdr_agreement_error(x_core, _agsdr_reference_trajectory()), 1.0,
                AGSDR_RECURRENCE + "; inner optax.sgd(0.1), noise off; experimental transform",
                "optimizer")


@_f64
def check_gsgd_quadratic_minimum():
    """step_gsgd_transform is plain gradient descent x <- x - eta grad, so the error is (1 - 2 eta)^T."""
    x = jnp.asarray(X_START)
    for _ in range(STEPS):
        x, _ = step_gsgd_transform(x, 2.0 * (x - XSTAR), None, {"eta": GSGD_ETA})
    return _row("gsgd_quadratic_minimum", _trajectory_error(x), 1.0,
                QUAD_TRAJECTORY + "; eta 0.1", "optimizer")


CHECKS = (
    ("polarity_null_flips_sign", check_polarity_null_flips_sign),
    ("layer_shuffle_null_permutes", check_layer_shuffle_null_permutes),
    ("uniform_gain_null_scales", check_uniform_gain_null_scales),
    ("synchrony_metric_limits", check_synchrony_metric_limits),
    ("null_samples_closed_form", check_null_samples_closed_form),
    ("spectrolaminar_score_known_profile", check_spectrolaminar_score_known_profile),
    ("box_projection_is_clip", check_box_projection_is_clip),
    ("sdr_quadratic_minimum", check_sdr_quadratic_minimum),
    ("gsdr_quadratic_minimum", check_gsdr_quadratic_minimum),
    ("agsdr_quadratic_minimum", check_agsdr_quadratic_minimum),
    ("gsgd_quadratic_minimum", check_gsgd_quadratic_minimum),
)


def compute_all():
    """One score row per check, in CHECKS order, plus the dropped checks and their reasons."""
    checks = {name: fn() for name, fn in CHECKS}
    return {"checks": checks,
            "pass": bool(checks) and all(row["pass"] for row in checks.values()),
            "dropped": dict(DROPPED)}


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
            # A rounding-level residual differs across platforms; below a tenth of the tolerance
            # in both copies it cannot change the verdict.
            floor = NOISE_FRACTION * min(o["tolerance"], n["tolerance"])
            if key == "error" and o[key] <= floor and n[key] <= floor:
                continue
            bad.append(f"{name}.{key}: {o[key]!r} != {n[key]!r} past rtol {rtol:g}")
    if old.get("pass") != new.get("pass"):
        bad.append(f"pass: {old.get('pass')!r} != {new.get('pass')!r}")
    return bad


README_TEXT = """# Objective and optimizer core-method checks

Worst-case deviations of the objective stage (`jaxfne/objectives.py`) and the optimizer stage
(`jaxfne/optim/`) against independent references, recomputed by
`scripts/score_objective_optimizer_checks.py` with small inputs and fixed seeds. `score.json`
holds one `{error, tolerance, pass, reference, stage}` row per check; `stage` is `objective` or
`optimizer`. `error` is the worst-case deviation and `tolerance` is the check's factor times its
scale, so a passing test implies a passing row. These are agreements of core methods with
closed forms, not a paper replication: there is no paper value beside ours.

| check | stage | reference | tolerance |
|---|---|---|---|
| polarity_null_flips_sign | objective | minus each input profile | 1e-12 x scale |
| layer_shuffle_null_permutes | objective | sorted values of each input profile | 1e-12 x scale |
| uniform_gain_null_scales | objective | input times the seeded uniform(0.1, 1) gain | 1e-12 x scale |
| synchrony_metric_limits | objective | identical channels 1; complementary channels 0 | 1e-12 |
| null_samples_closed_form | objective | ceil(1 + 1/(2 rse^2)) = 3, 5, 9, 33 | 0.5 (counts) |
| spectrolaminar_score_known_profile | objective | 100; 100 exp(-2.4) | 1e-12 x 100 |
| box_projection_is_clip | optimizer | np.clip to the bounds | 1e-12 x scale |
| sdr_quadratic_minimum | optimizer | closed-form trajectory 0.8^100 | 1e-6 relative |
| gsdr_quadratic_minimum | optimizer | closed-form trajectory 0.8^100 | 1e-6 relative |
| agsdr_quadratic_minimum | optimizer | independent float64 NumPy recurrence of the transform update | 1e-12 x max\\|x_0 - x*\\| |
| gsgd_quadratic_minimum | optimizer | closed-form trajectory 0.8^100 | 1e-6 relative |

## Hand derivations

- **Polarity.** `source_polarity_flip_null` returns `-alpha_beta` and `-gamma`.
- **Layer shuffle.** A permutation keeps the multiset of values, so the sorted output equals the sorted input.
- **Uniform gain.** The function returns no gain. It draws `gain = gen.uniform(gain_min, 1.0)` once from
  the generator it is given. The reference draws the same first value from the same seed, so the check
  tests that the output is that gain times the input.
- **Synchrony.** Identical channels have correlation 1 in every pair of binned counts, so the mean over
  pairs is 1. Two complementary channels `y = 1 - x` with `x` non-constant have `corr(x, y) = -1`; the
  function clips the mean to [0, 1], so the value is 0.
- **Null samples.** For a normal null, `RSE(sigma_hat) ~ 1 / sqrt(2 (n - 1))`, so the smallest n with RSE at
  most r is `1 + 1/(2 r^2)`, rounded up. At r = 0.5, 0.4, 0.25 and 0.125 this gives 1 + 2 = 3,
  ceil(1 + 3.125) = 5, 1 + 8 = 9 and 1 + 32 = 33; the 0.4 case separates rounding up from rounding off. The formula is the large-n approximation stated in the docstring; the check verifies the
  inversion the code implements, not the accuracy of the approximation.
- **Spectrolaminar score.** The default similarity is `100 exp(-3 (MSE_ab + MSE_gamma))`. An ideal readout has
  MSE 0 and scores 100. For the profile p = (1, 0, 0, 0, 0) and its reversal (0, 0, 0, 0, 1) the squared
  differences are (1, 0, 0, 0, 1), so each band has MSE 0.4 and the score is 100 exp(-2.4) = 9.07.
- **Box projection.** `enforce_parameter_bounds` is `jnp.clip` with the bounds broadcast over rows; the
  reference is `np.clip` with the same broadcast.
- **Quadratic minima.** f(x) = ||x - x*||^2 has gradient 2(x - x*) and minimum x* = (1, -2, 0.5, 3). With
  inner `optax.sgd(0.1)` and noise off, each step contracts x - x* by 1 - 2 x 0.1 = 0.8, so the error after
  100 steps is exactly 0.8^100 = 2.04e-10. `step_gsgd_transform` with eta = 0.1 has the same exact
  contraction. The sdr, gsdr and gsgd rows score the relative deviation of the observed ratio from
  0.8^100, so a wrong step size fails as well as a step that does not converge.
- **AGSDR recurrence.** With `stochastic_scale = 0` and inner `optax.sgd(0.1)` the transform update is
  `U_t = (1 - a_t) D_t` with `D_t = -0.2 (x_t - x*)`, from `a_0 = 0.5` and `v_0 = u_0 = 0`. After each
  step `v_{t+1} = 0.99 v_t + 0.01 var(D_t)` (population variance over the flattened update) and
  `a_{t+1} = clip((u + eps) / (u + v_{t+1} + 2 eps))` with `eps = 1e-6`; `u` stays 0 because the
  stochastic variance is 0. The loss falls every step from `x_0 = 0`, which keeps the deselection
  branch inactive (asserted in the reference). The row scores `max|x_core - x_ref| / max|x_0 - x*|`
  against this recurrence: float64 rounding near x* is about 1e-16 of that scale, a halved core
  step leaves about 3e-5 and `a_0 = 0` about 2.5e-11, against the tolerance 1e-12.

## Tolerances and configuration

Each row's tolerance is its factor times its scale (`TOLERANCES` in the scorer). The optimizer rows
use factor 1e-6 at scale 1.0; the agsdr row uses 1e-12 of the start displacement, below the error of a
halved-step or an `a_0 = 0` mutant. Fixed
configuration: x_0 = 0, 100 steps, PRNG key seed 0, inner `optax.sgd(0.1)`.

- `sdr_quadratic_minimum`, `gsdr_quadratic_minimum` and `agsdr_quadratic_minimum` run with
  `stochastic_scale = 0` (the default is 0.1). With the noise off, sdr, gsdr and gsgd reduce to the inner
  gradient step, so these rows test the deterministic update path; the stochastic term and the reset
  branches are not tested here (tests/test_optim_tune.py covers the reset branches). In agsdr the
  stochastic variance is 0 with the noise off, so its unsup EMA stays 0 and the reference exercises
  the mixing-weight recurrence directly.
- `agsdr_transform` is marked EXPERIMENTAL in its docstring (not the canonical AGSDR algorithm, not on the
  `Model.tune()` path). Its row tests that transform only.
- `step_gsgd_transform` is the documented plain step (its docstring: kept for direct use). Its row tests that step.

## Dropped checks

- `phase_null_keeps_power` (objective): `phase_randomized_null` multiplies each entry by the cosine of an
  independent uniform angle. Its inputs are band-power profiles, not time series, so a spectral
  invariant does not apply.
- `optax_adam_quadratic_minimum` (optimizer): `optax_adam` returns an OptimizerSpec declaration; its update
  runs only inside `Model.tune()`, not on a plain function. A quadratic run would test optax, not core code.

## Seeds

Readout profiles: seeds 11, 12 and 14. Layer-shuffle generator: 13. Gain generator: 15. Synchrony column: 16.
Box proposals: 41. PRNG key of the transforms: 0. Requires optax for the sdr, gsdr and agsdr rows.

Rerun:

    python scripts/score_objective_optimizer_checks.py         # rewrite score.json (this README is left as is)
    python scripts/score_objective_optimizer_checks.py --check # recompute and exit 1 past rtol 1e-9
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
    parser = argparse.ArgumentParser(description="Score the objective and optimizer core-method checks.")
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
