# Objective and optimizer core-method checks

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
| agsdr_quadratic_minimum | optimizer | closed-form minimum x* after 100 steps (experimental) | 2 x recorded run |
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

## Empirical tolerances and configuration

The agsdr row uses tolerance = 2 x the error from one run of this scorer (one-sided; its value is in
`score.json`): it guards the transform path against regressions and is not an accuracy claim. Fixed
configuration: x_0 = 0, 100 steps, PRNG key seed 0, inner `optax.sgd(0.1)`.

- `sdr_quadratic_minimum`, `gsdr_quadratic_minimum` and `agsdr_quadratic_minimum` run with
  `stochastic_scale = 0` (the default is 0.1). With the noise off, sdr, gsdr and gsgd reduce to the inner
  gradient step, so these rows test the deterministic update path; the stochastic term and the reset
  branches are not tested here (tests/test_optim_tune.py covers the reset branches). In agsdr the
  reward variance is 0 with the noise off, so its mixing weight collapses after the first step and the
  mixing is not tested either.
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
