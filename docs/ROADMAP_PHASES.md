# jaxfne 20-Phase Master Roadmap

> **Audience:** opencode worker agent + maintainer (Perplexity/Claude alignment sessions).
> **Rule:** Do not copy internal-roadmap language into `docs/` public files.
> This file lives under `docs/` but is agent-facing; it does **not** appear in the
> mkdocs nav and must never be linked from public-facing pages.
>
> **Before every phase:** read README, AGENTS.md, the relevant skill, search existing
> tests, search existing APIs. **Search first. Reuse second. Create last.**
>
> **After every phase:** run the validation block from AGENTS.md, confirm exit
> criteria, update `.lab/` graph nodes, append a one-line note to
> `artifacts/developer/AGENT_CHANNEL.md` (local-only, gitignored), then stop
> and align with Perplexity before starting the next phase.

---

## Coordinate system

| Symbol | Meaning |
|--------|---------|
| `[DONE]` | Completed and merged to `dev` as of 2026-08-04 |
| `[ACTIVE]` | In progress on `dev` |
| `[NEXT]` | Next queued item — start here after alignment |
| `[BLOCKED]` | Cannot start until dependency phase exits |
| `[SKIP-IF]` | Conditional — skip if guard check passes |
| `[GATE]` | Hard acceptance test — phase does not exit until this passes |
| `[NULL-CTRL]` | Null-control variant required alongside positive test |
| `[DOCONLY]` | Documentation or comment change, no logic change |
| `[EVIDENCE]` | Requires a real command + output receipt, not prose |

---

## Phase A — Repository Hygiene & Alignment

**Goal:** `main` and `dev` are aligned; CI is green; no stale open PRs blocking forward work.
**Status: `[DONE]`** (PR #76 merged 2026-08-04, commit `aa3a1a1`).

### Tasks

- [DONE] A-01: Confirm `main` and `dev` SHA match via `git rev-parse`.
- [DONE] A-02: Merge any open hygiene PRs (PR #76 was the outstanding one).
- [DONE] A-03: Run validation block; confirm zero failures on smoke suite
  (`test_api_smoke.py`, `test_root_import_lightweight.py`, `test_signals_get_v0329.py`).
- [DONE] A-04: Confirm `.lab/` graph nodes for emitter modules are not stale
  (status `confirmed`, not `unconfirmed`). If stale, update before proceeding.

### Exit criteria

- `main` SHA == `dev` SHA.
- CI smoke suite: 0 failures.
- No open PRs with merge conflicts.

### Alignment checkpoint A

Report `git rev-parse HEAD` on both branches and CI status before moving to Phase B.

---

## Phase B — H(t) Convergence Gate (emitters_homeostatic_ei)

**Goal:** Prove that `simulate_homeostatic_ei` with `freeze_H=False` and
`homeostasis_rule="cubic_penalty"` reaches a genuine interior equilibrium.
All downstream G-plasticity and Izhikevich-layer weight updates are blocked
until this gate is green.
**Status: `[ACTIVE]`** (Stage 0 test written 2026-08-04, commit `5c4b200`).

### Stage 0 — Minimal convergence test `[DONE]`

- [DONE] B-01: Read `emitters_homeostatic_ei.py` in full. Map every `homeostasis_rule`
  and its documented convergence behaviour.
- [DONE] B-02: Read every existing test whose name contains `homeostatic_ei`.
  Confirm no existing test already gates `freeze_H=False` + interior equilibrium.
- [DONE] B-03: Write `tests/test_phaseB_stage0_H_convergence.py`:
  - `test_H_converges_cubic_penalty_freeze_G`: G frozen, H live.
    Assertions: finite, inside bounds, late-window |ΔH| < 0.05, not collapsed to H_min.
  - `test_H_converges_cubic_penalty_full_dynamics`: G and H both live.
    Same four assertions. Different seed (42 vs 0).

### Stage 1 — Rule survey `[NEXT]`

- [ ] B-04: `[GATE]` Run `pytest tests/test_phaseB_stage0_H_convergence.py -v`.
  Both tests must pass. Record stdout as `[EVIDENCE]`. If either fails:
  - Diagnose first (don't patch blindly).
  - If `freeze_G` test fails: the H-ODE itself diverges → fix `cubic_penalty`
    rule in `emitters_homeostatic_ei.py` and add regression note in docstring.
  - If `full_dynamics` test fails while `freeze_G` passes: G-feedback is the
    source → add a `K_ctrl`-style damping term to the conductance update and
    document the finding.
  - Re-run until both pass, then record receipt.
- [ ] B-05: Parametric sweep over `homeostasis_rule` in `{"linear", "logistic",
  "cubic_penalty", "cubic_penalty_coupled"}` with `freeze_G=True`.
  For each rule: record whether H converges to interior, collapses to H_min,
  or saturates at H_max. Write findings as a table in a docstring in
  `emitters_homeostatic_ei.py` (not in tests; docstring is the right home).
- [ ] B-06: `[NULL-CTRL]` Confirm that `homeostasis_rule="linear"` does NOT
  converge to an interior point (it should collapse to H_min). If it does
  converge, update the documented rationale — the finding would change
  which rules are safe to use downstream.
- [ ] B-07: Write `tests/test_phaseB_stage1_rule_survey.py` with one test per
  rule documenting its confirmed behaviour (pass = expected behaviour, not
  necessarily convergence; the linear/logistic collapse is the expected result).

### Stage 2 — G-adaptation stability `[BLOCKED on Stage 1]`

- [ ] B-08: With `freeze_G=False, freeze_H=False, homeostasis_rule="cubic_penalty"`,
  sweep `conductance_rule` over `{"hebbian", "bcm", "linear", "hebbian_pairwise"}`.
  For each: record whether the G trajectory stays inside `[G_min, G_max]` and
  whether the full 3-timescale system converges.
- [ ] B-09: Write `tests/test_phaseB_stage2_G_stability.py`.
  Four tests, one per conductance rule. Each test: G finite, G bounded, late-window
  |ΔG| check, no assertion that bcm converges (documented: bcm saturates at G_max).
- [ ] B-10: `[DOCONLY]` Update `simulate_homeostatic_ei` docstring to include a
  2-by-4 compatibility table (homeostasis_rule × conductance_rule) with the
  confirmed convergence outcomes.

### Stage 3 — N-scaling `[BLOCKED on Stage 2]`

- [ ] B-11: Use `make_minimal_ei_params(n=8)` and `make_minimal_ei_params(n=16)`.
  Confirm `bound_mode="stable"` prevents the documented N=16 divergence.
  `[EVIDENCE]` required (pytest receipt).
- [ ] B-12: Write `tests/test_phaseB_stage3_N_scaling.py`.
  Tests: N=4, N=8, N=16, `bound_mode="stable"`, `homeostasis_rule="cubic_penalty"`,
  full dynamics. All four assertions (finite, bounded, settled, not collapsed).

### Exit criteria — Phase B

- `[GATE]` All 8+ Phase B tests pass on `dev`.
- `[EVIDENCE]` Full pytest receipt stored in `AGENT_CHANNEL.md` (local-only).
- Docstring compatibility table written.
- No regressions in pre-existing `test_homeostatic_ei_*` tests.

### Alignment checkpoint B

Report: which rules converge, which collapse, N-scaling verdict, pytest receipt.

---

## Phase C — Izhikevich-Layer H(t) Carry

**Goal:** The Izhikevich/edge-list HDP kernel (`simulate_edge_recurrent_izhikevich_hdp`
in `emitters.py`) must cleanly carry H(t) across multi-chunk simulations using
`DynamicState` so that pause/resume produces output byte-identical to a single
continuous run.
**Entry gate:** Phase B exit criteria met.

### Tasks

- [ ] C-01: Read `simulate_edge_recurrent_izhikevich_hdp` in `emitters.py` in full.
  Map every field in `final_state` / `init_state`. Confirm whether `H` is in both.
- [ ] C-02: Read `jaxfne._pipeline` (`compile_step_fn`, `scan_network`, `DynamicState`).
  Confirm `DynamicState` has all six fields: `v, u, prev_spikes, syn_state, H, w`.
  If any field is missing: add it, add a test, document in `FRICTIONS_STACK.md`.
- [ ] C-03: `[SKIP-IF]` Read `tests/test_sanity_delta_backup_resume.py` and
  `tests/test_sanity_delta_resume_equivalence_full.py`. If they already gate
  byte-identical H carry: record that, mark C-03 done, skip C-04.
- [ ] C-04: Write `tests/test_phaseC_H_carry_resume.py`:
  - Run 2000-step simulation in one chunk, record `H_trace`.
  - Run same simulation as two 1000-step chunks using `init_state` from chunk 1.
  - Assert `jnp.allclose(one_chunk_H[-1000:], two_chunk_H, atol=1e-5)`.
  - `[NULL-CTRL]` Run with `k_gain=0` (homeostasis disabled) and assert H stays
    at its initial value throughout both chunks.
- [ ] C-05: `[GATE]` Both tests in C-04 pass. `[EVIDENCE]` required.
- [ ] C-06: `[DOCONLY]` Update AGENTS.md "Three build paths" table to note that
  `DynamicState` is the canonical carry for multi-chunk HDP runs.

### Exit criteria — Phase C

- Pause/resume byte-identical (atol 1e-5) for H field.
- No regressions in `test_sanity_delta_*`.
- `DynamicState` documented with all six fields.

### Alignment checkpoint C

Report: DynamicState field list, resume test result, any field gap found and fixed.

---

## Phase D — Source Schema Stability

**Goal:** The Source object produced by every emitter path has a stable,
documented schema. Downstream Field and Probe operators must not infer schema
from shape alone.
**Entry gate:** Phase C exit criteria met.

### Tasks

- [ ] D-01: Read `jaxfne/sources.py` (or wherever Source/SourceTensor is defined).
  List every field, its dtype, shape convention, and `source_calibration_status` string.
- [ ] D-02: Read `_construct_population.py`. Confirm that every emitter path
  (`simulate_eig_izhikevich`, `simulate_edge_recurrent_izhikevich`,
  `simulate_edge_recurrent_izhikevich_homeostatic`,
  `simulate_edge_recurrent_izhikevich_hdp`,
  `simulate_homeostatic_ei`) produces a Source whose
  `source_calibration_status` string is propagated correctly.
- [ ] D-03: Search all tests for `source_calibration_status`. List which paths
  are already covered and which are not.
- [ ] D-04: Write `tests/test_phaseD_source_schema.py`.
  One test per emitter path: assert `source.source_calibration_status` is not
  `None`, is a non-empty string, and matches the documented value for that path.
  `[NULL-CTRL]` Assert that `physical_amplitude_calibrated=False` for all paths
  (no path has been calibrated).
- [ ] D-05: `[GATE]` All source schema tests pass.
- [ ] D-06: `[DOCONLY]` Write `docs/api/source_schema.md` (add to mkdocs nav).
  Table: emitter path | `source_calibration_status` value | `physical_amplitude_calibrated`.
  Language: use "proxy"/"scaffold" — never "physical"/"calibrated".

### Exit criteria — Phase D

- Every emitter path has a confirmed, tested `source_calibration_status`.
- `docs/api/source_schema.md` exists and builds clean under `mkdocs build --strict`.
- No regressions.

### Alignment checkpoint D

Report: schema table populated, any uncovered paths found, mkdocs build status.

---

## Phase E — Field Schema Stability

**Goal:** The Field object produced by every solver path has a stable, documented
schema. Probe operators must be able to select field components by name, not by
axis index.
**Entry gate:** Phase D exit criteria met.

### Tasks

- [ ] E-01: Read `jaxfne/fields/` directory listing. Map every solver class and
  its output schema.
- [ ] E-02: Read `test_field_admissibility_v020.py` and
  `test_field_proxy_admissibility_v024.py`. Confirm what is already gated.
- [ ] E-03: For each field solver: assert the output has a `field_claim_level`
  metadata key. Confirm it is `"proxy_readout"` (current default). If any solver
  sets it differently, document the discrepancy.
- [ ] E-04: Write `tests/test_phaseE_field_schema.py`.
  One test per solver: assert output has `field_claim_level`, `solver_status`,
  `physical_amplitude_calibrated=False`.
- [ ] E-05: `[GATE]` All field schema tests pass.
- [ ] E-06: `[DOCONLY]` Write `docs/api/field_schema.md` (add to mkdocs nav).
  Table: solver | `field_claim_level` | `solver_status` | notes.

### Exit criteria — Phase E

- Every solver path has a confirmed, tested field schema.
- `docs/api/field_schema.md` builds clean.
- No regressions.

### Alignment checkpoint E

Report: field schema table, any solver with unexpected `field_claim_level`.

---

## Phase F — Solver Acceptance Criteria

**Goal:** Define and implement acceptance criteria for the linear field solver
before any higher-fidelity solver is added. "Acceptance" means a set of tests
that any solver must pass to be admitted to the public API.
**Entry gate:** Phase E exit criteria met.
**Note:** Per AGENTS.md known fragility #3 and the "Do not build major solver
systems before Source schema stable + Field schema stable + Acceptance criteria
exist" rule — Phases D, E, F are the prerequisite chain.

### Tasks

- [ ] F-01: Read `jaxfne/fields/` in full. Confirm the current linear solver's
  implementation: what equations, what assumptions, what inputs/outputs.
- [ ] F-02: Write a solver acceptance checklist as a comment block at the top of
  `jaxfne/fields/__init__.py`:
  - Input: Source tensor (n_neurons × T) + positions (n_neurons × 3) + probe positions.
  - Output: Field tensor at probe positions (n_probes × T).
  - Must: produce finite output for any finite source input.
  - Must: scale linearly with source amplitude (superposition).
  - Must: be `jax.jit`-compatible.
  - Must: carry `field_claim_level` metadata.
  - Must not: claim physical amplitude without calibration.
- [ ] F-03: Write `tests/test_phaseF_solver_acceptance.py`.
  Tests implementing each checklist item above for the current linear solver.
  One test per criterion. `[NULL-CTRL]` Zero-source input → zero-field output.
- [ ] F-04: `[GATE]` All solver acceptance tests pass.
- [ ] F-05: `[DOCONLY]` Write `docs/guides/solver_acceptance.md`.
  Reproduces the checklist in human-readable form. Add to mkdocs nav.

### Exit criteria — Phase F

- Acceptance checklist exists in code and in docs.
- All acceptance tests pass.
- The path to adding a second solver (e.g., FEM, BEM) is documented.

### Alignment checkpoint F

Report: checklist written, tests passing, path to next solver documented.

---

## Phase G — Discoverability: Public API Catalog

**Goal:** Every public function and class in `jaxfne` is findable from
`skills/catalog-glossary-jaxfne/SKILL.md` with its correct signature,
a one-line description, and the correct status tag
(`[STABLE]`, `[EXPERIMENTAL]`, `[STUB]`).
**Entry gate:** Phase F exit criteria met.
**Rule:** Do not add new public API during this phase — catalog what exists,
correct what is wrong, mark what is stub.

### Tasks

- [ ] G-01: Run `python3 -c "import jaxfne; print(dir(jaxfne))"`.
  Capture full public namespace. Diff against `SKILL.md`. Record gaps.
- [ ] G-02: For each gap (present in namespace, missing from SKILL.md):
  read its implementation, confirm its status, add a SKILL.md entry.
- [ ] G-03: For each entry in SKILL.md: confirm the signature matches the
  current implementation. Correct any mismatch. Add a `[CHANGED]` note if
  the signature changed since the last documented version.
- [ ] G-04: Tag all known stubs (`GLIFEmitter`, `LIFEmitter`, `write_nwb`,
  `read_nwb`) as `[STUB]` in SKILL.md with a one-line note on what they
  would need to be promoted.
- [ ] G-05: `[GATE]` Run `tests/test_agent_api_catalog.py`. Zero failures.
- [ ] G-06: `[GATE]` Run `tests/test_public_api_snapshot_v034.py`. Zero failures.
  If either fails due to a real API gap (not a test artifact), fix the API
  first, then re-run.
- [ ] G-07: `[DOCONLY]` Update `docs/api/` index to surface the SKILL.md
  catalog as a human-readable reference page.

### Exit criteria — Phase G

- Every public name has a SKILL.md entry with status tag.
- `test_agent_api_catalog.py` and `test_public_api_snapshot_v034.py` pass.
- No invented signatures in SKILL.md.

### Alignment checkpoint G

Report: gap count, corrected entries, stub count, both catalog tests passing.

---

## Phase H — Reusable Tutorial Utilities

**Goal:** Promote any workflow that appears ≥2 times across notebooks/scripts
into a public API function in `jaxfne.tutorial_utils` (or the correct
submodule). Remove the duplicate inline implementations from the notebooks.
**Entry gate:** Phase G exit criteria met.
**Rule:** Notebooks are for demonstration. Package APIs are for reusable logic.
Do not put simulators, optimizers, field solvers, or scientific engines in notebooks.

### Tasks

- [ ] H-01: List every function defined inline in `examples/` and
  `tutorials/` notebooks. Use `grep -r "^def " examples/ tutorials/`.
  Record the function name and the file it appears in.
- [ ] H-02: For each inline function that appears in ≥2 files: confirm
  whether an equivalent already exists in the public API (search SKILL.md
  first). If yes: replace the inline definition with a call to the public
  function. If no: promote it.
- [ ] H-03: For each promoted function:
  - Add to `jaxfne/tutorial_utils.py` (or correct submodule).
  - Add a unit test in `tests/`.
  - Add a SKILL.md entry.
  - Remove the inline definition from all notebooks.
- [ ] H-04: `[GATE]` Run `tests/test_tutorial_utils.py` (create if not exists).
  One test per promoted function.
- [ ] H-05: Confirm no notebook defines a function that duplicates a public API.
  Use `grep -r "^def " examples/ tutorials/` again. Any remaining inline
  function must be either a one-liner adapter (≤3 lines, no logic) or
  documented as notebook-local in a comment.

### Exit criteria — Phase H

- Zero duplicated inline functions across notebooks.
- Every promoted function has a unit test.
- SKILL.md updated.

### Alignment checkpoint H

Report: functions promoted, notebooks cleaned, test count added.

---

## Phase I — Documentation Consolidation

**Goal:** Every page in `docs/` is accurate, buildable under
`mkdocs build --strict`, and cross-linked. Remove stale/duplicate pages.
**Entry gate:** Phase H exit criteria met.

### Tasks

- [ ] I-01: Run `python3 -m mkdocs build --strict`. Record every warning and
  error. Triage: fix broken links first, then stale content, then missing pages.
- [ ] I-02: Run `tests/test_docs_links_v0330.py`. Fix every broken link.
- [ ] I-03: Run `python3 scripts/audit_public_docs_language.py --check`.
  Fix every violation. Language rule: never "validated"/"physical"/"mechanism"
  without a receipt; always "proxy"/"scaffold"/"relative".
- [ ] I-04: Read every `.md` file under `docs/api/`. For each: confirm the
  documented function signature matches the current implementation.
  Correct mismatches. Add a `[UPDATED yyyy-mm-dd]` tag in the file's
  YAML front-matter.
- [ ] I-05: Remove or merge any duplicate doc pages (same content, different
  filenames). Use git blame to find the authoritative version.
- [ ] I-06: `[GATE]` `mkdocs build --strict` with zero errors and zero warnings.
- [ ] I-07: `[GATE]` `tests/test_docs_links_v0330.py` passes.
- [ ] I-08: `[GATE]` `scripts/audit_public_docs_language.py --check` passes.

### Exit criteria — Phase I

- `mkdocs build --strict` clean.
- Zero broken links.
- Zero language violations.
- Every API doc page has a confirmed-current signature.

### Alignment checkpoint I

Report: warning count before/after, broken link count, language violation count.

---

## Phase J — Spectrolaminar API Consolidation

**Goal:** The two spectrolaminar metric functions (`spectrolaminar_motif_score`
and `spectrolaminar_similarity_kernel_jax`) have clean, non-overlapping
documented use cases. No third variant is introduced before the two existing
ones are fully documented and tested.
**Entry gate:** Phase I exit criteria met.

### Tasks

- [ ] J-01: Read `spectrolaminar_motif_score` and
  `spectrolaminar_similarity_kernel_jax` implementations in full.
  Write a side-by-side comparison in their respective docstrings:
  - Input shape
  - Output shape and meaning
  - When to use each
  - What the other one does that this one does not
- [ ] J-02: Read every test whose name contains `spectrolaminar`.
  Confirm which metric each test exercises. Record any gap.
- [ ] J-03: Write `tests/test_phaseJ_spectrolaminar_api.py`.
  - One test per metric confirming input/output shape contract.
  - One test confirming the two metrics are not accidentally identical
    (distinct outputs on the same synthetic input).
  - `[NULL-CTRL]` Zero-amplitude input → metric is finite and defined.
- [ ] J-04: `[GATE]` All new spectrolaminar tests pass, plus all pre-existing
  `test_spectrolaminar_*` tests pass without modification.
- [ ] J-05: `[DOCONLY]` Write `docs/guides/spectrolaminar_guide.md`:
  when to use motif score, when to use kernel, worked example for each.
  Add to mkdocs nav.

### Exit criteria — Phase J

- Both spectrolaminar functions have complete, non-overlapping docstrings.
- Shape contracts are tested.
- Guide page exists and builds clean.

### Alignment checkpoint J

Report: docstring diff summary, new test count, guide page built.

---

## Phase K — HDP Weight Drift Guard (AGENTS.md fragility #3)

**Goal:** Fix the documented `K_w_ctrl=0.0` permits unbounded weight drift
fragility (AGENTS.md known fragility #3). Introduce a safe default or a
clear documented guard that prevents runaway weight growth on long HDP runs.
**Entry gate:** Phase J exit criteria met.

### Tasks

- [ ] K-01: Read `simulate_edge_recurrent_izhikevich_hdp` in full.
  Trace the weight-update path when `K_w_ctrl=0.0`. Confirm the runaway
  scenario: under what drive/duration does weight magnitude diverge?
  `[EVIDENCE]` required: run a 10 000-step simulation and record
  `max(|w|)` at t=0 and t=final.
- [ ] K-02: Determine the correct fix:
  - Option A: Change the default to `K_w_ctrl=1e-3` (gentle weight decay).
  - Option B: Add a `w_ceiling` hard clip that is active by default.
  - Option C: Document the constraint and add a runtime warning if
    `K_w_ctrl=0.0` and simulation exceeds 5000 steps.
  Discuss with maintainer during alignment — do not decide unilaterally.
- [ ] K-03: Implement the agreed fix. Update `DEFAULT_HDP` preset if it uses
  `K_w_ctrl`.
- [ ] K-04: Write `tests/test_phaseK_weight_drift_guard.py`.
  - Positive test: with guard active, `max(|w|)` stays below `w_ceiling`
    after 10 000 steps.
  - `[NULL-CTRL]` With `K_HDP=0` (HDP disabled), weights do not change at all.
  - Regression: pre-existing short-run HDP tests still pass (guard does not
    change behaviour for runs ≤ 5000 steps).
- [ ] K-05: `[GATE]` All K tests pass.
- [ ] K-06: `[DOCONLY]` Update AGENTS.md known fragility #3 to read `[FIXED]`
  with a one-line description of the fix and the test that guards it.

### Exit criteria — Phase K

- Weight drift guard implemented and tested.
- AGENTS.md fragility #3 marked `[FIXED]`.
- No regressions in pre-existing HDP tests.

### Alignment checkpoint K

Report: chosen option (A/B/C), `max(|w|)` before/after receipt, test result.

---

## Phase L — Multi-Area Source Projector Stability

**Goal:** The multi-area source projector produces stable, finite output for
any valid multi-area configuration. The inter-area coupling path through
`simulate` is fully tested end-to-end.
**Entry gate:** Phase K exit criteria met.

### Tasks

- [ ] L-01: Read `tests/test_multi_area_source_projector.py` and
  `tests/test_multi_area_emitter_runtime.py` in full.
  Confirm what is already gated.
- [ ] L-02: Run those tests. If any fail: fix before adding new tests.
  `[EVIDENCE]` required (pytest receipt).
- [ ] L-03: Write `tests/test_phaseL_multi_area_e2e.py`.
  - Two-area network: Area A (E/PV) → Area B (E/PV), with inter-area
    excitatory projection.
  - Run 1000 steps. Assert: output finite, both areas spike, inter-area
    current reaches Area B neurons (mean |drive| > 0 on at least one neuron).
  - `[NULL-CTRL]` Zero inter-area weight → Area B output is identical
    to a single-area simulation.
- [ ] L-04: `[GATE]` All L tests pass, including pre-existing multi-area tests.

### Exit criteria — Phase L

- Multi-area E2E test with null control passes.
- No regressions in pre-existing multi-area tests.

### Alignment checkpoint L

Report: two-area test result, null control result.

---

## Phase M — Config Path Completeness

**Goal:** Every public `Configuration` method (fluent builder) that is
documented produces a valid `Model` via `construct()`. No undocumented
path reaches `construct()` without error handling.
**Entry gate:** Phase L exit criteria met.

### Tasks

- [ ] M-01: Read `jaxfne/_config.py` in full. List every public method
  of `Configuration`.
- [ ] M-02: For each method: confirm there is at least one test that
  calls it and asserts the resulting `Model` is valid.
  List uncovered methods.
- [ ] M-03: Write `tests/test_phaseM_config_completeness.py`.
  One test per uncovered method. Each test: build a config, call `construct()`,
  assert `model is not None` and output of `model.simulate(n_steps=10)` is finite.
- [ ] M-04: `[GATE]` All M tests pass.
- [ ] M-05: `[DOCONLY]` Update `docs/guides/configuration_grammar.md`.
  Add a method reference table: method | purpose | example call | status.

### Exit criteria — Phase M

- Every `Configuration` method has at least one test.
- Configuration grammar doc has method reference table.

### Alignment checkpoint M

Report: uncovered method count, tests added, doc table written.

---

## Phase N — NeuronalTensor Path Completeness

**Goal:** Every `NeuronalTensor` construct path (multi-area, JSON roundtrip,
HDP via `RuntimeConfig`) is tested end-to-end. The `HDPColumnConfig` narrow
tier is fully documented.
**Entry gate:** Phase M exit criteria met.

### Tasks

- [ ] N-01: Read `skills/jaxfne-neural-tensor/SKILL.md` in full.
  List every canonical construct path described.
- [ ] N-02: For each path: confirm a test exists that exercises it. List gaps.
- [ ] N-03: Confirm `RuntimeConfig(enable_hdp=True, hdp_params={...})` path
  works end-to-end via a test (`test_sanity_delta_plasticity.py` or equivalent).
  Run it. `[EVIDENCE]` required.
- [ ] N-04: Confirm `HDPColumnConfig` produces a valid `Model` via `construct()`.
  If no test exists for this exact path: write one.
- [ ] N-05: `[GATE]` All N tests pass.
- [ ] N-06: `[DOCONLY]` Update `docs/api/neuronal_tensor.md`.
  Add a "HDP via RuntimeConfig" section with a minimal code example.

### Exit criteria — Phase N

- Every NeuronalTensor canonical path has a test.
- `HDPColumnConfig` + `construct()` tested.
- `docs/api/neuronal_tensor.md` has HDP section.

### Alignment checkpoint N

Report: gap count, tests added, HDP section written.

---

## Phase O — Optimizer & Manifest Stability

**Goal:** The optimizer path (`Objective → Optimizer → Manifest`) produces
stable, reproducible output. Manifests are JSON-roundtrippable. The optimizer
does not silently produce NaN gradients.
**Entry gate:** Phase N exit criteria met.

### Tasks

- [ ] O-01: Read `tests/test_optim_manifests.py`, `tests/test_manifest_v005.py`,
  `tests/test_optim_tune.py` in full. Confirm what is already gated.
- [ ] O-02: Run `tests/test_differentiable_scalar_soft_rate_tune.py` and
  `tests/test_differentiable_source_scale_tune.py`. Record results.
  `[EVIDENCE]` required.
- [ ] O-03: `[SKIP-IF]` If both O-02 tests pass: skip O-03 and note it.
  If either fails: diagnose gradient path. Fix before proceeding.
- [ ] O-04: Write `tests/test_phaseO_optimizer_stability.py`.
  - One test: 50-step gradient descent on `source_scale`, assert loss decreases
    monotonically over the first 20 steps.
  - `[NULL-CTRL]` Zero learning rate → loss does not change.
  - One test: manifest JSON roundtrip → re-loaded manifest produces identical
    loss on the same input.
- [ ] O-05: `[GATE]` All O tests pass.
- [ ] O-06: `[DOCONLY]` Write `docs/guides/optimizer_guide.md`:
  canonical optimizer loop, manifest save/load example, gradient safety notes.

### Exit criteria — Phase O

- Optimizer convergence tested (loss decreasing).
- Manifest JSON roundtrip tested.
- Optimizer guide written.

### Alignment checkpoint O

Report: gradient test result, roundtrip result, guide written.

---

## Phase P — Probe Operator Coverage

**Goal:** Every probe operator (LFP, CSD, EEG, MEG, EMM) has a tested
shape contract and a `[NULL-CTRL]` (zero source → zero probe output).
**Entry gate:** Phase O exit criteria met.

### Tasks

- [ ] P-01: Read `tests/test_probe_operators_v021.py` and
  `tests/test_probe_operators_eeg_meg_emm.py`. List which operators
  are covered and which are not.
- [ ] P-02: Run all pre-existing probe tests. `[EVIDENCE]` required.
- [ ] P-03: For each uncovered operator: write a test in
  `tests/test_phaseP_probe_coverage.py`.
  Shape contract: `output.shape == (n_probes, T)`. Finite. `[NULL-CTRL]`.
- [ ] P-04: `[GATE]` All probe tests pass (new + pre-existing).
- [ ] P-05: `[DOCONLY]` Update `docs/api/probes.md`:
  operator table with input/output shape, status, and a one-line use case.

### Exit criteria — Phase P

- Every probe operator has a shape-contract test and a null-control test.
- `docs/api/probes.md` operator table complete.

### Alignment checkpoint P

Report: operators covered, operators added, null-control results.

---

## Phase Q — Performance Baseline

**Goal:** Establish a reproducible performance baseline for the three core
compute-critical paths: (1) Izhikevich dense scan, (2) Izhikevich sparse
edge scan, (3) `simulate_homeostatic_ei`. Record wall-clock and compile time.
This is a measurement phase, not an optimization phase.
**Entry gate:** Phase P exit criteria met.
**Rule:** Do not optimize before profiling. Do not claim speedup without a
before/after measurement.

### Tasks

- [ ] Q-01: Read `tests/test_scaling_benchmark.py`. Confirm what is already
  measured. Record current numbers.
- [ ] Q-02: Write `scripts/benchmark_core_paths.py`.
  Three functions, one per path. Each: warm up JIT (3 runs), then time
  10 runs, report median ± std. Parameters:
  - Dense: N=128, T=1000, dt=0.5ms.
  - Sparse: N=128, ~5% connectivity, T=1000, dt=0.5ms.
  - HomeostaticEI: N=8, T=5000, dt=0.5ms.
  Output: a JSON receipt `artifacts/benchmarks/baseline_<date>.json`
  (local-only, gitignored).
- [ ] Q-03: Run `python3 scripts/benchmark_core_paths.py`. Record receipt.
  `[EVIDENCE]` required.
- [ ] Q-04: Write `tests/test_phaseQ_performance_regression.py`.
  For each path: assert median time < 2× the baseline recorded in Q-03.
  (A 2× regression is a canary, not a hard cap.)
- [ ] Q-05: `[GATE]` All Q tests pass.
- [ ] Q-06: `[DOCONLY]` Write `docs/guides/performance.md`.
  Baseline table, profiling methodology, how to re-run the benchmark.

### Exit criteria — Phase Q

- Baseline JSON exists and is cited in `performance.md`.
- Regression tests pass.
- No optimization attempted yet.

### Alignment checkpoint Q

Report: baseline numbers for all three paths, regression test result.

---

## Phase R — vmap / lax.scan Hardening

**Goal:** The two most performance-critical paths (dense scan and sparse edge
scan) are confirmed `vmap`-compatible and `lax.scan`-optimal. Any remaining
Python-loop hotspot inside a scan body is removed.
**Entry gate:** Phase Q exit criteria met.
**Rule:** Prefer `O(N log N)`, prefer `vmap`, prefer `lax.scan`, prefer `jit`,
use explicit PRNG keys. Do not optimize before profiling (Phase Q is the profile).

### Tasks

- [ ] R-01: Read `simulate_eig_izhikevich` and
  `simulate_edge_recurrent_izhikevich` scan bodies.
  List every operation that is not a pure JAX primitive (any Python
  control flow, any `np.*` call inside the scan body).
- [ ] R-02: For each non-primitive operation: either move it outside the scan
  (pre-compute) or replace it with a JAX equivalent. Document the change.
- [ ] R-03: Write `tests/test_phaseR_vmap_compat.py`.
  - `jax.vmap` over a batch of 4 PRNG keys on both scan paths.
  - Assert outputs are batch-correct (each element matches its
    single-key equivalent).
  - `[NULL-CTRL]` Batch of 1 matches single run exactly.
- [ ] R-04: Re-run `scripts/benchmark_core_paths.py` after hardening.
  Record new receipt. `[EVIDENCE]` required. Report speedup (if any).
- [ ] R-05: `[GATE]` `tests/test_phaseR_vmap_compat.py` passes.
  `tests/test_jit_equivalence_v036.py` passes (pre-existing, must not regress).
- [ ] R-06: `[DOCONLY]` Update `docs/guides/performance.md` with post-hardening
  numbers and a note on vmap usage pattern.

### Exit criteria — Phase R

- Zero Python control flow inside scan bodies.
- vmap-over-batch test passes.
- Post-hardening benchmark receipt recorded.

### Alignment checkpoint R

Report: hotspots removed, vmap test result, speedup (if any).

---

## Phase S — Release Preparation (0.5.0)

**Goal:** Cut a clean `0.5.0` release on `main`. All Phases A–R must be
complete. `CHANGELOG.md` updated. Version bumped. No open issues tagged
`blocking-release`.
**Entry gate:** Phases A–R all exit criteria met.

### Tasks

- [ ] S-01: Run full test suite: `pytest tests/ -q --tb=short`.
  Zero failures, zero errors. `[EVIDENCE]` required (full receipt).
- [ ] S-02: Run `mkdocs build --strict`. Zero warnings, zero errors.
- [ ] S-03: Run `python3 scripts/audit_public_docs_language.py --check`. Clean.
- [ ] S-04: Bump version in `jaxfne/__init__.py` and `pyproject.toml` to `0.5.0`.
- [ ] S-05: Write `CHANGELOG.md` entry for `0.5.0`:
  - Phases A–R summary in plain language.
  - Breaking changes (if any).
  - Language: proxy/scaffold — not calibrated/physical.
- [ ] S-06: Open PR `dev → main` titled "Release 0.5.0".
  PR body: link to this ROADMAP_PHASES.md, pytest receipt from S-01.
- [ ] S-07: `[GATE]` PR passes CI. Maintainer approves. Merge.
- [ ] S-08: Tag `v0.5.0` on `main`. Push tag.

### Exit criteria — Phase S

- `v0.5.0` tag exists on `main`.
- Full test suite green.
- `mkdocs build --strict` clean.
- `CHANGELOG.md` updated.

### Alignment checkpoint S

Report: test count, failure count, tag SHA, CI status.

---

## Phase T — Forward Plan (0.6.x)

**Goal:** Define the next major capability increment after 0.5.0 is stable.
This phase is a planning phase only — no code is written.
**Entry gate:** Phase S exit criteria met.

### Candidate directions (to be prioritised at alignment checkpoint T)

These are options, not commitments. Ranked by current evidence:

1. **Physical-amplitude calibration bridge** — add an optional
   `CalibrationBridge` object that maps native Izhikevich current units to
   physical µA/mm². Required before any claim beyond `proxy_readout`.
   Gate: a real electrode recording or a validated compartmental model
   provides the calibration constant.

2. **Second field solver** — add a BEM or FEM solver as a second option
   alongside the linear solver. Gate: Phase F acceptance criteria must
   pass for the new solver before it is added to the public API.

3. **Jaxley full-neuron interoperability** — complete the `jaxley` bridge
   beyond the current stub level. Gate: `test_jaxley_emitter_bridge_e2e.py`
   and `test_jaxley_trace_bridge.py` must pass with a real Jaxley model,
   not a placeholder.

4. **Streaming / chunked simulation API** — productize `compile_step_fn` /
   `scan_network` as a first-class public API. Gate: pause/resume
   byte-identical (Phase C) must be confirmed for all six `DynamicState`
   fields.

5. **PySNF export** — `write_nwb` promoted from stub to working implementation.
   Gate: round-trip test (write → read → compare).

### Tasks

- [ ] T-01: At alignment checkpoint T: rank the five candidates by
  scientific priority and implementation readiness.
- [ ] T-02: Write the 0.6.x roadmap as a new `docs/ROADMAP_06x.md`.
  Same format as this file. Phases numbered U onward.
- [ ] T-03: Update `.lab/` graph nodes for any 0.5.0-era nodes that
  should transition to `status: historical`.

### Exit criteria — Phase T

- 0.6.x roadmap document written.
- Candidate ranking agreed at alignment checkpoint.

### Alignment checkpoint T (FINAL)

Report: candidate ranking, 0.6.x roadmap path, any open risks from 0.5.0.

---

## Appendix: Standing Rules (worker agent must follow)

### Before every task

```
1. Read the relevant skill (skills/catalog-glossary-jaxfne/SKILL.md or the task-specific skill).
2. Search existing tests for the function/behaviour you are about to test.
3. Search existing APIs for the function you are about to write.
4. If you find it already exists: reuse it, document it, do not duplicate it.
```

### Code quality

- Implementation → Tests → Documentation. In that order, always.
- Never put a simulator, optimizer, or field solver in a notebook.
- Prefer `vmap` over Python loops. Prefer `lax.scan` over Python `for` loops.
  Prefer `jit`. Use explicit PRNG keys.
- Every new public function gets a one-line docstring minimum and a SKILL.md entry.
- `[GATE]` items are hard stops — do not proceed past them on a failure.

### Evidence standard

- `[EVIDENCE]` = a real command + its stdout/stderr, not prose.
- `status=done` requires `achieved_score >= target_score` AND a real receipt.
- JSON edits ≠ work done.

### Language

- Public docs: `proxy` / `scaffold` / `relative` — never `validated` /
  `physical` / `mechanism` / `calibrated` without a receipt.
- Agent-facing: be precise about what is confirmed vs. inferred.
  Treat every node in `.lab/` as a claim, not a fact, until verified.

### Truth gates (never escalate)

```
claim_level: computational_scaffold
field_claim_level: proxy_readout
physical_amplitude_calibrated: false
```

These never change without an explicit maintainer decision AND a real
calibration receipt. Do not escalate them in code, docs, or comments.

### Do not add

- Duplicate APIs
- Duplicate skills
- Duplicate tutorials
- Duplicate implementations
- New top-level folders (repo root is frozen)
- Any internal roadmap language in public docs

### Alignment rule

At the end of each phase, **stop**. Report exit criteria status. Wait for
Perplexity to confirm before starting the next phase. A phase is not done
until both the worker agent and Perplexity agree on the exit criteria.
