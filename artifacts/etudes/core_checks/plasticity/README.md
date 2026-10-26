# Plasticity hand-value checks

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
