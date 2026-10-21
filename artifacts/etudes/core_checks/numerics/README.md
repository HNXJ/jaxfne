# Core-method numerics checks

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
