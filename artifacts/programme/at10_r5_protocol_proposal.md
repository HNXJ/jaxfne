# AT-10-R5 protocol proposal (for human review; nothing run)

Status: PROPOSED 2026-10-06. Row: `AT-10-R5` in `atlas_coverage.json`
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

## Open choices for the human

| Choice | Proposed default |
|---|---|
| Offset size / area subset | 0.05 of H range (as in 0.5.3), 2 of 20 areas |
| Horizon | long enough for 5 restoring time constants (tau_0 = 200 ms) after t0 |
| K_ctrl for arm B | the 0.5.3 value (5) |
| Seeds | 1 (R3 seed) plus 2 more for an ordering check |
