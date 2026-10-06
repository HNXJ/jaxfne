# AT-10-R5 protocol proposal (for human review; nothing run)

Status: choices approved 2026-10-06 (see Choices). Row: `AT-10-R5` in `atlas_coverage.json`
("bounded != returning != homeostatically stabilized"). Freeze this file
before any result, then run; do not tune after seeing the outcome (new
protocol if changed).

## Tools already present

- `boundedness_report` and `stability_report` (twin trajectories differing
  only in initial H; tiers RETURNING < 0.5, PERSISTING <= 2, DIVERGING > 2;
  zero early distance is refused). Currently test-local in
  `tests/test_long_diagnostics_053.py`; promote to a shared module (the
  0.5.5 measurement-vector item) so the Atlas run can import them.
- `artifacts/atlas/at10_n20_055.py::run_phases` (20-area run, per-area PSD,
  boundedness, spread).
- Intervention grammar (`artifacts/programme/intervention_053.md`).

## Reuse finding (2026-10-06, before any run)

`at10_n20_055.py::run_assay` already runs, on the one realized N20 network
and per noise seed: reference (engaged) vs H0-perturbed and W-kicked arms,
each with HDP engaged and HDP disabled, scored on window firing rates
(`late_dev_hz`, verdicts STABILIZED / NOT_STABILIZED / NO_LASTING_EFFECT),
and `run_phases` already reports boundedness (R3). What it lacks for R5 is
the separation in H space: `stability_report` on H twins (RETURNING /
PERSISTING / DIVERGING) beside `boundedness_report`, per arm. The runner
should therefore extend `run_assay` (keep `H_trace` per arm, apply the two
reports) before adding a new set of arms; the three-arm design below is the
fallback if the existing arms cannot give the A/B/C contrast. Decision for
the reviewer: extend `run_assay` (cheaper, same seeds) or run the new arms.

## Mismatch to resolve before any run (found 2026-10-06)

The approved choices below (K_ctrl = 5, tau_0 = 200 ms, 1 s horizon,
0.05 offset) come from the 0.5.3 fixture, not from `AT-10-N20`. The N20
parameters in `at10_n20_055.py` are `HP_HEBB` = {K_HDP 0.005, K_ctrl 0.15,
K_w_ctrl 0.05, alpha 0.05, tau_0_ms 5.0}, phases of 10 s, an existing H0
perturbation to 0.0 (not a 0.05 offset), and W kicks of 0.8x / 1.3x. With
tau_0 = 5 ms, a 1 s horizon is 200 tau_0, and K_ctrl = 5 would be a change
to the validated model, not a reuse. Options: (a) keep N20's own HP_HEBB,
the existing H0 = 0.0 and kicks, horizon = the 1 s windows `run_assay`
already scores; (b) impose the 0.5.3 values as a new, separately named arm.
Needs a human choice; (a) keeps R3's validated model unchanged.

## Design

1. System: `AT-10-N20`, same manifest and seed as the validated R3 run.
2. Three arms, each run as a twin pair (base vs perturbed):
   - A fixed W, no restoring (K_ctrl = 0): expected bounded, not returning.
   - B fixed W with restoring (K_ctrl > 0): expected bounded and returning.
   - C plastic W (HDP on): reported as observed; no expected tier.
3. Perturbation: one pre-declared H offset (fraction of H range) applied to
   a pre-declared area subset at t0; the twin differs in nothing else.
4. Readouts, kept separate: `boundedness_report` (H, W inside declared
   bounds), `stability_report` ratio and tier per arm, and per-area firing
   spread. A tier is never inferred from boundedness or the reverse.
5. Acceptance: arms A and B land in different tiers with the same boundedness
   verdict (the non-inference proof at Atlas scale); C is reported either
   way. A zero early distance voids that arm (refused, not scored).
6. Not claimed: "homeostatically stabilized" is stated only for arm B and
   only as RETURNING under the declared perturbation, never as a general
   property.

## Choices (approved by Hamm 2026-10-06)

| Choice | Value |
|---|---|
| Offset size / area subset | 0.05 of H range (as in 0.5.3), 2 of 20 areas |
| Horizon | 5 restoring time constants (tau_0 = 200 ms) = 1 s after t0 |
| K_ctrl for arm B | the 0.5.3 value (5) |
| Seeds | R3 seed plus 2 more for an ordering check |

Status: APPROVED, not run. Next: freeze this file in a commit, then
implement the runner (imports `jaxfne._long_diagnostics`).
