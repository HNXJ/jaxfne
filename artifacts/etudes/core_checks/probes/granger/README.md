# Spectral Granger causality core checks

Worst-case deviations of `jaxfne.analysis.granger_spectral` against independent references,
recomputed by `scripts/score_granger_checks.py`. `score.json` holds one `{error, tolerance, pass,
reference}` row per check; `error` is the worst-case deviation and `tolerance` is the bound named in
the table. A passing row means the stated agreement holds at the stated inputs; these are analytic
agreements of a core method, not a paper replication, and there is no paper value beside ours.

| check | pinned by | reference | tolerance |
|---|---|---|---|
| spectral_gc_var1 | tests/test_granger_spectral.py:43 | ln(S_yy / (S_yy - \|H_yx\|^2)) of the VAR(1) with A1 = [[0.5, 0], [0.6, 0.5]], Sigma = I, H = (I - A1 z)^-1, z = exp(-2 pi i f dt), at 5, 10, 40 Hz; n 200 000, seed 11, dt 1 ms | worst relative error \|gc - ref\| / ref <= 0.05 (rtol as the test) |
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
