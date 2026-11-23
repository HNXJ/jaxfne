# B1 preregistration, amendment 2 (before any scored run)

Status: ruled by Hamm 2026-11-23 ("Approve all recommended", plan `solmind_0510_plan.md` §10.3 item 14, options
(i) and (iv)). No scored B1 or B2 run exists.

Amends `PREREG_B1_0510.md` (signed 2026-11-06, pinned) together with amendment 1 and its erratum (pinned, left
as written).

| # | Signed text | Amended text | Why |
|---|---|---|---|
| 1 | E3: "neuron-steps/s of `simulate_batch` (vmap over seeds) at batch B" | E3: neuron-steps/s of a sequential per-seed loop of `Model.simulate` at batch B, as `b1/b1_harness.py` runs it | `simulate_batch` refuses the lane Poisson and Gaussian drives (`jaxfne/_model_simulate.py:3495-3503`), so no vmap path runs lane-faithful M1 or M2 |
| 2 | "BLAS threads pinned and recorded" (no count) | `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` = 1 in each cell process, recorded per cell | the signed text names no count; one process per cell |

Baseline tools installed 2026-11-23 (WSL `~/venv_jaxfne`, Python 3.12.3): brian2 2.10.1, NEST 3.10.0 (pip wheels),
jax 0.10.1. The signed text pins no baseline version; each cell records the version it ran.
