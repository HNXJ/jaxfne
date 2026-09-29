# Literature reproduction études: plan

Status: plan only. No paper is selected and no citation is recorded here; every
target enters through P1 below. Timing: after the 0.5.5 seal, own todo stack.

## Aim

Reproduce recent results on neuronal biophysics and cortical microcircuits, as
close as possible to field and neural models, with jaxfne, each in two arms:
**HDP off** (fixed `W`, `H` inert: the paper's own model) and **HDP on**.
Report what each arm reproduces, where HDP changes the outcome, and where it
changes nothing.

## Decisions (human, 2026-09-29)

- Window 2022–2026; about 12 candidates screened to 4–6 targets.
- "Reproduce" = predeclared qualitative observables with tolerances. No
  re-tuning after the outcome is seen; a miss is reported, not repaired.
- Paired arms per target: same seeds, configuration and drives; both
  preregistered; both reported.
- Starts after 0.5.5; own plan and stack.

## Pipeline (one target moves through six stages)

| Stage | Output | Gate |
|---|---|---|
| P0 Screen | candidate table: paper, model class, field readout, what jaxfne lacks | none; nothing here is cited yet |
| P1 Verify | primary-source read; observables and parameters quoted from the paper itself | citation primary-verified (fact); claim from memory is not cited |
| P2 Spec | jaxfne configuration from the paper's model, mapping table paper → jaxfne, gaps listed | every declared field consumed or refused (fact); gaps stay gaps |
| P3 Preregister | observables, tolerances, seeds, arms, analysis, frozen before any run | human sign-off on the preregistration |
| P4 Run | both arms, all seeds, manifests | seed changes output; HDP-off arm shown inert (below) |
| P5 Report | étude bundle `artifacts/etudes/<name>/`, docs page, notebook, ledger rows | claim wording no stronger than evidence; negatives reported |

## Screening criteria (P0)

Keep a paper when at least three hold:
1. Cortical microcircuit or single-neuron biophysics with an explicit model.
2. A field readout (LFP, CSD, spectrolaminar power, EEG-like) or a stated
   prediction for one.
3. Observables that a proxy readout can express (laminar profile, band power,
   rate, PSD shape, sink/source pattern, response to omission or oddball).
4. Parameters given in the paper or its supplement, not only "tuned".
5. E/PV/SST/VIP structure, laminar layers, or feedforward/feedback bands.
6. A place where slow adaptation or homeostasis could matter, so the HDP arm
   is a real question and not decoration.

Reject: no explicit model; parameters unavailable; needs a solved field (the
shipped path is proxy); needs structural plasticity (outside the HDP grammar).

## HDP off / on

- **Off arm.** The paper's model as declared: fixed `W`, `H` carries no
  dynamics. Test before use: off-arm output equals a run with no HDP declared,
  bit for bit (`Ẇ = 0` is valid, fact).
- **On arm.** Same configuration plus one declared HDP rule (state, law,
  target set, gain), chosen at P3 from the paper's own adaptation or
  homeostasis mechanism, not tuned to the outcome.
- Same seeds, drives and duration in both arms; the arms differ in one
  declaration.
- Comparison: each predeclared observable in each arm against its tolerance,
  then arm against arm. Report four cells per observable: reproduced in off /
  on, and whether the arms differ.

## Per-target template (P2/P3 document)

| Field | Content |
|---|---|
| Paper | authors, year, venue, DOI, primary-verified date |
| Model class | emitter family, populations, connectivity, drives |
| Observables | name, definition, paper value, tolerance, source locator |
| Mapping | paper parameter → jaxfne field, or GAP with reason |
| Field readout | proxy kernel and what it does not claim |
| HDP rule | state, law, targets, gain, and why this rule |
| Seeds and duration | declared, dt in ms |
| Controls | no-stimulus null, control-window, selection status |
| Outcome | per arm, per observable: pass / miss / not applicable |

## Deliverables and order

1. This plan, then the P0 candidate table (about 12 rows).
2. P1 verification of the shortlist; the human confirms 4–6 targets.
3. One pilot target end to end (P2–P5) to fix the template and the off-arm
   inertness test.
4. Remaining targets in parallel worktrees, one writer each.
5. An études index page and a cross-target table: which observables HDP
   changes, by how much, in which targets.

## Risks

| Risk | Handling |
|---|---|
| Paper model needs features jaxfne lacks | GAP row; target dropped or reduced, stated |
| Proxy readout cannot express the observable | observable marked not applicable, not approximated |
| HDP rule picked after seeing outcomes | rule fixed at P3; changing it needs a new preregistration |
| Only agreements reported | four-cell report per observable; misses stay in the table |
| Citation drift or secondhand claims | P1 gate: primary source only |
