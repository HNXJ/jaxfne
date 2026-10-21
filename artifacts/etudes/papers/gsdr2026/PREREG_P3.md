# GSDR lane P3 preregistration (draft): Fig. 6 circuit on the jaxfne core

## Claim

The GSDR paper Fig. 6 circuit (Nejat, Sherfey, Bastos 2026, PLoS One 21(8):
e0354021; code run `misc/main.py`, read as data per `SOURCE_P1.md`) is
realized on jaxfne core modules without changing any existing public
function: `jaxfne/emitters_hh.py` (HH + OU + graded-synapse kernel),
`jaxfne/_construct_hh.py` (`construct_hh_model` + `hh_kernel_args`), one
export line in `jaxfne/__init__.py`. The etude runner is
`artifacts/etudes/gsdr2026/run_gsdr.py`.

## Method (frozen before the baseline run)

- Spec: `run_gsdr.py:build_spec` (PORT_SPEC.md §§1, 4–5, §13 core map);
  local posts `choice(PRNGKey(1), arange(50), (5,))`; `edge_seed=0`.
- Init: gAMPA ~ U(−4,10)→[0,8], gGABAa ~ U(2,10)→[0,10], `g_seed` per trial
  (A3: base 0). OU tau ~ U(10,50), sigma = clip(U(−0.1,0.1),0,0.05) (D5 run
  values); `ou_seed` per trial (A2: base 0).
- Grid: dt 0.1 ms, t_max 1500 ms (15001 steps), float32 on CUDA (D7).
- Stimulus: gated 120 Hz sine, amps (0.0, 0.6, 0.6), every-other-E (D1–D3,
  D8–D9 as declared).
- Trials: 3 (one per batch amp). Outputs: `scratch/gsdr2026/baseline/`
  (V traces, conductances, `manifest.json`); one code SHA per directory.

## Acceptance (baseline, unscored)

1. Wiring asserts pass in `run_trial`: 1800 / 400 / 30 edges per receptor.
2. All three trials complete; `manifest.json` records SHA, JAX version,
   devices, seeds, wall time.
3. Report: per-trial mean rate, rate penalty (5–50 Hz), 15–60 Hz model PSD,
   kappa. No loss is scored: the Fig. 6 loss needs data label spectra, and
   there are no MT/MST data in the lane (data-label PSD loss blocked).

## Fidelity check (RESIDUAL_PROTOCOL.md)

Baseline rates/PSDs are compared against the reference run's qualitative
targets (stim-driven E activity, I stabilization). A miss triggers the
declared fidelity suspects in order: D9 (stimulus sign), A1/D10 (HH/C_m
assumed values); a paper/code disagreement on a claim-changing value stops
the lane.

## Stop conditions

Paper/code disagree on a claim-changing value; a core addition needs to
change an existing public function; any run exceeds 2 h GPU.

## Not in P3

GSDR training loop (P4+; `fig6_loss`/`mcdp_factors` in `run_gsdr.py` are the
untrained skeleton), calibration claims (`CALIBRATION.md` records the
baseline only), data-label loss.
