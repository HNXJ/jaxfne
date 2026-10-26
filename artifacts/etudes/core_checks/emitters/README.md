# Emitter closed-form checks

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
