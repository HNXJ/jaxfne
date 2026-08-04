# jaxfne — Full Canonical Roadmap

> **Canonical file:** `docs/fullroadmap.md`
> **Agent-facing only.** Not in MkDocs nav, not linked from README or public API docs.
> Last edited: 2026-08-04

---

## Purpose & Scope

This file is the single ordered roadmap for the jaxfne project, covering:

- Framework engineering (core, emitters, fields, vis, API)
- Simulation stages for the H(t) homeostasis / omission-response study
- Publication pipeline (figures → manuscript → submission)
- Documentation and agent infrastructure

All prior roadmap sources (`artifacts/developer/ROADMAP.md` — now deleted) are merged here.
No roadmap language lives in README, API docs, source docstrings, or public tutorial pages.

---

## Operating Rules

1. **One canonical file.** This is it. Any agent or human editing the roadmap edits only this file.
2. **One action = one file.** Every action targets exactly one file. Composite changes become multiple consecutive actions on the same file.
3. **Consecutive sibling rule.** If two or more actions affect the same file, they are placed consecutively in the master action list.
4. **Issue insertion rule.** If an issue is discovered during execution: either fix it immediately as the current action, or insert a new numbered action at the correct dependency position. No floating TODOs outside the ordered list.
5. **No roadmap leakage.** Nothing from this file gets pasted into `docs/` public pages, `README.md`, `AGENTS.md` prose, or source docstrings — the public surface gets only the *result* ("Relative", "proxy_readout", etc.), never the planning language.
6. **Done rule.** `[DONE]` requires a real command/output receipt as `Evidence`. Prose claims with no command shown are not evidence.
7. **Phase end requirement.** Every phase closes with an exit-criteria block, evidence required, and alignment checkpoint.
8. **Global numbering.** Actions are numbered globally (0001, 0002 …) across all phases, never reset per phase.
9. **Completed work preserved.** Completed actions stay in the list as `[DONE]` — history is not deleted.
10. **Not in MkDocs nav.** This file must never appear under `nav:` in `mkdocs.yml`.

---

## Status Legend

| Tag | Meaning |
|-----|---------|
| `[DONE]` | Completed; evidence on file |
| `[ACTIVE]` | In progress this session |
| `[NEXT]` | Next to execute; all dependencies met |
| `[QUEUED]` | Queued; dependencies not yet met |
| `[BLOCKED]` | Blocked on external dependency |
| `[HOLD]` | Deferred by explicit decision |
| `[DOCONLY]` | Documentation-only change, no code change |
| `[TEST]` | Test-only action |
| `[BUG]` | Inserted bug-fix action |
| `[REFACTOR]` | Code-quality refactor, no behavior change |
| `[EVIDENCE]` | Evidence-collection action (run + record) |
| `[ALIGN]` | Alignment checkpoint action |

---

## File-Target Convention

Every action specifies `Target file:` as a repo-root-relative path.
Examples: `jaxfne/emitters.py`, `tests/test_phaseB_stage0_H_convergence.py`, `docs/fullroadmap.md`.
Actions that delete a file set `Target file: <path> (DELETE)`.

---

## Insertion Convention for Newly Discovered Issues

When an issue is found during execution:

1. Assign the next available global action number.
2. Insert the new action immediately after the action that uncovered it and before any action that depends on the fix.
3. Tag it `[BUG]` or `[REFACTOR]` as appropriate.
4. Add a backlink in the **Known Issues Register** below (format: `Issue-NNN → Action XXXX`).
5. Append a one-line entry to the **Change Log** section.

---

## Global Dependency Rules

- An action may not begin until all actions in its `Depends on:` list are `[DONE]`.
- `Depends on: none` is valid only for the first action of an independent track.
- Phase B cannot begin until Phase A exit criteria are met.
- Stage N+1 of the simulation cannot begin until Stage N produces passing tests and recorded evidence.
- Figures cannot be frozen until all simulation stages feeding that figure are `[DONE]`.
- Manuscript cannot be drafted until figures are frozen.
- Any `[BUG]` action inserted mid-phase becomes a dependency of all subsequent actions in that phase.

---

## Phase Overview Table

| Phase | Letter | Title | Status | Gate |
|-------|--------|-------|--------|------|
| 1 | A | Theory & Setup | [DONE] | H(t) formulation locked, references anchored, title decided |
| 2 | B | H(t) Emitter Implementation | [ACTIVE] | Stage 0 + Stage 1 passing |
| 3 | C | All Simulation Stages | [QUEUED] | Stage 8 (omission) passing — PRIMARY RESULT gate |
| 4 | D | Figure Freeze | [QUEUED] | ≥5 seeds per condition, all panels rendered |
| 5 | E | Manuscript Draft | [QUEUED] | Full draft written |
| 6 | F | Internal Review | [QUEUED] | Reviewer comments resolved |
| 7 | G | Submission | [QUEUED] | Submitted to PLOS Comp Biol (primary) / eLife (alt) |
| 8 | H | Framework: Core Defragmentation | [DONE] | _model.py/_construct.py split complete (v0.4.8–0.4.48) |
| 9 | I | Framework: Emitter Hardening | [ACTIVE] | F-028 closed; Metal backend fix merged |
| 10 | J | Framework: Fields & Vis | [QUEUED] | All field operators backend-portable |
| 11 | K | Framework: API Stabilization | [QUEUED] | Public API audit passes |
| 12 | L | Framework: HDP/Homeostasis Track | [ACTIVE] | H(t) emitter wired; K_w_ctrl default fixed |
| 13 | M | Agent Infrastructure | [ACTIVE] | Labyrinth graph tracked; PRP local-only |
| 14 | N | CI / Lint / Docs Build | [ACTIVE] | ruff pinned; mkdocs --strict passes |
| 15 | O | Documentation: Public Surface | [QUEUED] | All public docs pass audit_public_docs_language.py |
| 16 | P | Documentation: Agent Surface | [ACTIVE] | AGENTS.md + for_ai_agents.md coherent; roadmap canonical |
| 17 | Q | Test Coverage | [QUEUED] | 19 pre-existing failures resolved; coverage gate set |
| 18 | R | Ablation Suite | [QUEUED] | All 6 ablation conditions have ≥5 seed evidence |
| 19 | S | Cross-Model Generalization | [QUEUED] | Stage 15 (LIF/HH-Jaxley) passing |
| 20 | T | Long-Term & Multi-Area | [QUEUED] | Stage 10 (Colab GPU) + Stage 11 (multi-area) passing |

---

## Ordered Master Action List

---

### CONSOLIDATION BLOCK (Roadmap Infrastructure)

---

## Action 0001 — Create docs/fullroadmap.md
- Status: [DONE]
- Phase: P
- Target file: docs/fullroadmap.md
- Depends on: none
- Why now: No canonical single-file roadmap existed; two partial roadmap files needed merging into one agent-facing document per operating rules.
- Change: Create this file with all required top-level sections, status legend, phase table, and ordered action list. Merge content from `artifacts/developer/ROADMAP.md`.
- Validation: `curl -s https://api.github.com/repos/HNXJ/jaxfne/contents/docs/fullroadmap.md | python3 -c "import sys,json; d=json.load(sys.stdin); print('exists:', d['name'])"` or verify SHA in GitHub API.
- Evidence: Commit `docs(roadmap): create canonical docs/fullroadmap.md` on branch main, 2026-08-04.
- If blocked: N/A — this is the creation action itself.

---

## Action 0002 — Merge artifacts/developer/ROADMAP.md into docs/fullroadmap.md
- Status: [DONE]
- Phase: P
- Target file: docs/fullroadmap.md
- Depends on: 0001
- Why now: `artifacts/developer/ROADMAP.md` contained the canonical H(t) simulation-stage sequence (Stages 0–15), phase map (A–G), figure list, ablation set, and reference table. All content must live here.
- Change: All `artifacts/developer/ROADMAP.md` sections (Central Question, H(t) formulation, Simulation Stages, Ablation Set, Figures, Phases A–G, Claim Level, References) absorbed into this file's action list and phase table.
- Validation: `grep -c 'Stage 8' docs/fullroadmap.md` → ≥2 matches.
- Evidence: Same commit as Action 0001.
- If blocked: N/A.

---

## Action 0003 — Add operating rules block to docs/fullroadmap.md
- Status: [DONE]
- Phase: P
- Target file: docs/fullroadmap.md
- Depends on: 0001
- Why now: "One action = one file" and insertion rules must be codified in the canonical file itself so any agent reading it knows the constraints without external context.
- Change: Operating Rules section (10 rules) present in this file.
- Validation: `grep -c 'One action = one file' docs/fullroadmap.md` → ≥1.
- Evidence: Same commit as Action 0001.
- If blocked: N/A.

---

## Action 0004 — Add insertion rule for newly discovered bugs/issues
- Status: [DONE]
- Phase: P
- Target file: docs/fullroadmap.md
- Depends on: 0003
- Why now: Without an explicit insertion protocol, agents insert ad-hoc TODOs or append at end rather than at correct dependency position.
- Change: "Insertion Convention for Newly Discovered Issues" section present.
- Validation: `grep -c 'Insertion Convention' docs/fullroadmap.md` → ≥1.
- Evidence: Same commit as Action 0001.
- If blocked: N/A.

---

## Action 0005 — Delete artifacts/developer/ROADMAP.md
- Status: [NEXT]
- Phase: P
- Target file: artifacts/developer/ROADMAP.md (DELETE)
- Depends on: 0001, 0002
- Why now: After full content merger into docs/fullroadmap.md, the source file becomes a duplicate. Keeping it risks agents consulting a stale roadmap.
- Change: File deleted from repo. `artifacts/developer/` directory may become empty (acceptable).
- Validation: `curl -s -o /dev/null -w "%{http_code}" https://api.github.com/repos/HNXJ/jaxfne/contents/artifacts/developer/ROADMAP.md` → 404.
- Evidence: GitHub API returns 404 for path after commit; deletion commit SHA recorded here: _pending_.
- If blocked: Do not delete until 0001 and 0002 are [DONE].

---

## Action 0006 — Update docs/for_ai_agents.md — remove roadmap ambiguity
- Status: [NEXT]
- Phase: P
- Target file: docs/for_ai_agents.md
- Depends on: 0001, 0005
- Why now: `docs/for_ai_agents.md` currently lists `artifacts/developer/` as a live location ("PRP backlog (`plans.json`, `progress.json`, `review.json`) and handoff notes") but makes no mention of `docs/fullroadmap.md` as the canonical roadmap. After deletion of `artifacts/developer/ROADMAP.md`, agents need a pointer to this file.
- Change: Add one-line entry to the "Start here (agents)" section: `**Roadmap:** docs/fullroadmap.md — canonical ordered action list (agent-facing, not in MkDocs nav).` Narrow edit only; no other changes.
- Validation: `grep -c 'fullroadmap' docs/for_ai_agents.md` → ≥1.
- Evidence: Commit SHA recorded here: _pending_.
- If blocked: Can proceed independently of 0005 once 0001 is done.

---

## Action 0007 — Verify AGENTS.md — no roadmap ambiguity
- Status: [NEXT]
- Phase: P
- Target file: AGENTS.md
- Depends on: 0001
- Why now: `AGENTS.md` references `artifacts/developer/` PRP backlog but does not reference any roadmap path. Since it correctly says "depth lives in skills/ and docs/", and now docs/fullroadmap.md is the canonical roadmap location, verify AGENTS.md does not contain conflicting roadmap pointers.
- Change: If `AGENTS.md` contains any text pointing to `artifacts/developer/ROADMAP.md` specifically, add a narrow one-line pointer to `docs/fullroadmap.md` in the module-map or PRP-backlog section. If no conflict: no change needed — mark [DONE] with evidence `grep -r 'ROADMAP' AGENTS.md → 0 matches`.
- Validation: `grep 'ROADMAP.md' AGENTS.md | wc -l` → 0 (no stale pointer).
- Evidence: Verification output or commit SHA if edit was needed: _pending_.
- If blocked: N/A — read-only verification is always possible.

---

### PHASE A — Theory & Setup

---

## Action 0008 — Lock H(t) formulation
- Status: [DONE]
- Phase: A
- Target file: artifacts/developer/ROADMAP.md (now merged → docs/fullroadmap.md)
- Depends on: none
- Why now: Foundation for all simulation work. Central question, H(t) ODE, free parameters, and timescale must be fixed before any emitter code is written.
- Change: H(t) formulation locked: `τ_H · dH/dt = r_target - r(t)`. Primary mode: additive offset on Izhikevich threshold `b`. Ablation mode: multiplicative gain on input current.
- Validation: Formulation present in simulation code and matched to references (Cannon & Miller 2016, τ_θ ~30 s).
- Evidence: `artifacts/developer/ROADMAP.md` committed 2026-08-04 (SHA: abdfa59d). Now mirrored in this file.
- If blocked: N/A — theory work.

---

## Action 0009 — Anchor literature references
- Status: [DONE]
- Phase: A
- Target file: artifacts/developer/ROADMAP.md (now merged → docs/fullroadmap.md)
- Depends on: 0008
- Why now: Claims about novelty require explicit foil references before writing simulation code (to avoid post-hoc justification).
- Change: Four anchoring references locked: Turrigiano & Nelson 2004 (motivation); Cannon & Miller 2016 (formal match); Abbott & Varela 1997 (fast-mechanism foil); Yaron et al. 2025 (empirical omission target).
- Validation: References appear in `artifacts/developer/ROADMAP.md` reference table.
- Evidence: Same commit as Action 0008.
- If blocked: N/A.

---

## Action 0010 — Lock paper title and 5-sentence abstract
- Status: [DONE]
- Phase: A
- Target file: (docs/publication/* — now removed from repo per chore commits 2026-08-04)
- Depends on: 0008, 0009
- Why now: Title and abstract decisions must be frozen before Stage 8 figure generation to ensure narrative alignment.
- Change: Title locked: "Omission Responses in a Multi-Area Laminar Cortical Hierarchy Emerge from Simple Firing-Rate Homeostasis". 5-sentence summary, 3 novelties documented in session 2026-08-04.
- Validation: Title matches `artifacts/developer/ROADMAP.md` → "## Title" section.
- Evidence: Commit `docs(publication): add full publication plan + title/H(t) decisions (2026-08-04)` (SHA: 3b8cd946). Note: docs/publication/ subsequently removed from repo (chore commits 2026-08-04) since internal content is not for docs/. Title record preserved here.
- If blocked: N/A.

### Phase A — Exit Criteria

- [x] H(t) ODE and parameter set locked.
- [x] Primary and foil references anchored.
- [x] Paper title and 5-sentence abstract frozen.
- [x] Claim levels set: `computational_scaffold` / `proxy_readout` / `physical_amplitude_calibrated=False`.

**Evidence required:** Actions 0008–0010 all [DONE] with commit SHAs.

**Alignment checkpoint:** → See Action 0034 (Phase A [ALIGN]).

---

### PHASE B — H(t) Emitter Implementation

---

## Action 0011 — Implement H(t) emitter (Izhikevich, threshold-offset mode)
- Status: [ACTIVE]
- Phase: B
- Target file: jaxfne/emitters.py
- Depends on: 0008
- Why now: Phase B gate is Stage 0 + Stage 1 passing. Emitter must exist before any test can run.
- Change: `simulate_edge_recurrent_izhikevich_hdp` (or equivalent) extended / confirmed to carry `H(t)` as the additive offset on Izhikevich threshold `b`. `DynamicState` carries all 6 fields: `v, u, prev_spikes, syn_state, H, w`.
- Validation: `python3 -c "import jaxfne as jtfne; print(hasattr(jtfne._pipeline, 'scan_network'))"` → True.
- Evidence: _pending_.
- If blocked: Check `skills/jaxfne-neural-tensor/SKILL.md` and `skills/FRICTIONS_STACK.md` F-031.

---

## Action 0012 — Write test: Stage 0 — single-neuron H(t) convergence
- Status: [NEXT]
- Phase: B
- Target file: tests/test_phaseB_stage0_H_convergence.py
- Depends on: 0011
- Why now: Stage 0 is the minimum proof that H(t) reaches steady state for a single neuron. Must pass before Stage 1.
- Change: Test file created or verified present. Asserts: H(t) converges to a value within tolerance of `(r_target - r_init) * τ_H + H_init` in long-run limit; no NaN/Inf in `v`; mean rate within [8, 25] Hz plausible range.
- Validation: `pytest tests/test_phaseB_stage0_H_convergence.py -v` → all tests pass.
- Evidence: Pytest output showing PASSED for all tests, no failures. _pending_.
- If blocked: H(t) emitter (Action 0011) must be [DONE] first.

---

## Action 0013 — Run Stage 0 and record evidence
- Status: [NEXT]
- Phase: B
- Target file: docs/evidence_artifacts/ (new evidence file)
- Depends on: 0012
- Why now: Stage 0 evidence must be recorded before Stage 1 begins.
- Change: `pytest tests/test_phaseB_stage0_H_convergence.py -v` output saved to `docs/evidence_artifacts/phaseB_stage0_evidence.txt` (or equivalent).
- Validation: File exists and contains PASSED lines.
- Evidence: _pending_.
- If blocked: Action 0012 must be [DONE].

---

## Action 0014 — Write test: Stage 1 — repeated-pulse adaptation
- Status: [QUEUED]
- Phase: B
- Target file: tests/test_phaseB_stage1_pulse_adaptation.py
- Depends on: 0013
- Why now: Stage 1 validates firing-rate decay and recovery time under repeated pulses — direct test of H(t) dynamic behavior.
- Change: Test asserts: firing rate drops after initial pulse burst; H(t) rises monotonically during sustained drive; recovery time is within expected range after stimulus offset.
- Validation: `pytest tests/test_phaseB_stage1_pulse_adaptation.py -v` → all pass.
- Evidence: _pending_.
- If blocked: Stage 0 must be [DONE].

---

## Action 0015 — Run Stage 1 and record evidence
- Status: [QUEUED]
- Phase: B
- Target file: docs/evidence_artifacts/ (new evidence file)
- Depends on: 0014
- Why now: Phase B gate requires both Stage 0 AND Stage 1 passing.
- Change: `pytest tests/test_phaseB_stage1_pulse_adaptation.py -v` output saved to evidence artifacts.
- Validation: File exists, all PASSED.
- Evidence: _pending_.
- If blocked: Action 0014 must be [DONE].

### Phase B — Exit Criteria

- [ ] `pytest tests/test_phaseB_stage0_H_convergence.py -v` → all pass.
- [ ] `pytest tests/test_phaseB_stage1_pulse_adaptation.py -v` → all pass.
- [ ] Evidence files present in `docs/evidence_artifacts/`.
- [ ] H(t) emitter confirmed for both threshold-offset and multiplicative-gain modes (ablation readiness).

**Evidence required:** Actions 0013 and 0015 [DONE] with pytest output receipts.

**Alignment checkpoint:** → See Action 0035 (Phase B [ALIGN]).

---

### PHASE C — All Simulation Stages (Stage 2 → Stage 8 gate)

---

## Action 0016 — Stage 2: Population adaptation
- Status: [QUEUED]
- Phase: C
- Target file: tests/test_phaseC_stage2_population_adaptation.py
- Depends on: 0015
- Why now: Population-level H distribution and synchrony are prerequisites for the frequency-sweep stages.
- Change: Test asserts synchrony index and H(t) distribution across population; ≥5 seeds.
- Validation: `pytest tests/test_phaseC_stage2_population_adaptation.py -v` → pass.
- Evidence: _pending_.
- If blocked: Phase B must be complete.

---

## Action 0017 — Stage 3: Frequency sweep (τ_H calibration, 1–40 Hz)
- Status: [QUEUED]
- Phase: C
- Target file: tests/test_phaseC_stage3_frequency_sweep.py
- Depends on: 0016
- Why now: τ_H calibration against stimulus frequency is required before amplitude/duration sweeps.
- Change: Test sweeps input frequency 1–40 Hz; records H(t) at steady state per condition; validates τ_H range.
- Validation: `pytest tests/test_phaseC_stage3_frequency_sweep.py -v` → pass.
- Evidence: _pending_.
- If blocked: Stage 2 must be [DONE].

---

## Action 0018 — Stage 4: Amplitude sweep
- Status: [QUEUED]
- Phase: C
- Target file: tests/test_phaseC_stage4_amplitude_sweep.py
- Depends on: 0017
- Why now: Amplitude sweep fills parameter space needed for robustness figure (Figure 12).
- Change: Sweep input amplitude; record adaptation index per amplitude level.
- Validation: `pytest tests/test_phaseC_stage4_amplitude_sweep.py -v` → pass.
- Evidence: _pending_.
- If blocked: Stage 3 must be [DONE].

---

## Action 0019 — Stage 5: Duration sweep
- Status: [QUEUED]
- Phase: C
- Target file: tests/test_phaseC_stage5_duration_sweep.py
- Depends on: 0018
- Why now: Duration sweep closes the parameter sweep trio.
- Change: Sweep stimulus duration; record H(t) plateau and recovery.
- Validation: `pytest tests/test_phaseC_stage5_duration_sweep.py -v` → pass.
- Evidence: _pending_.
- If blocked: Stage 4 must be [DONE].

---

## Action 0020 — Stage 6: Random stimulus trains
- Status: [QUEUED]
- Phase: C
- Target file: tests/test_phaseC_stage6_random_trains.py
- Depends on: 0019
- Why now: Random trains test H(t) under naturalistic drive before oddball paradigm.
- Change: Generate Poisson-distributed stimulus trains; verify H(t) stability; ≥5 seeds.
- Validation: `pytest tests/test_phaseC_stage6_random_trains.py -v` → pass.
- Evidence: _pending_.
- If blocked: Stage 5 must be [DONE].

---

## Action 0021 — Stage 7: Classical oddball (SSA index vs. Abbott/Varela null)
- Status: [QUEUED]
- Phase: C
- Target file: tests/test_phaseC_stage7_classical_oddball.py
- Depends on: 0020
- Why now: SSA index validation against Abbott/Varela foil is the control condition before the primary omission paradigm.
- Change: Oddball paradigm (standard/deviant sequences); compute SSA index; compare to synaptic-depression null model.
- Validation: `pytest tests/test_phaseC_stage7_classical_oddball.py -v` → pass.
- Evidence: _pending_.
- If blocked: Stage 6 must be [DONE].

---

## Action 0022 — Stage 8: Omission paradigm ← PRIMARY RESULT GATE
- Status: [QUEUED]
- Phase: C
- Target file: tests/test_phaseC_stage8_omission_paradigm.py
- Depends on: 0021
- Why now: This is the PRIMARY RESULT of the study. All prior stages exist to support this. Phase C cannot exit without Stage 8 passing.
- Change: Implement omission paradigm matched to Yaron et al. 2025 in-vivo protocol. Simulate LFP proxy; record H(t) trace at omission onset; laminar profile; ablation null control; spectrolaminar motif. ≥5 seeds per condition.
- Validation: `pytest tests/test_phaseC_stage8_omission_paradigm.py -v` → pass; ablation null shows no omission response.
- Evidence: _pending_.
- If blocked: All Stages 0–7 must be [DONE].

---

## Action 0023 — Stage 9: Global-local oddball
- Status: [QUEUED]
- Phase: C
- Target file: tests/test_phaseC_stage9_global_local_oddball.py
- Depends on: 0022
- Why now: Global-local extends the omission result to a more complex hierarchical context.
- Change: Implement global-local paradigm; compare H(t) trace to Stage 8 baseline.
- Validation: `pytest tests/test_phaseC_stage9_global_local_oddball.py -v` → pass.
- Evidence: _pending_.
- If blocked: Stage 8 must be [DONE].

### Phase C — Exit Criteria

- [ ] Stages 2–9 all have passing pytest suites.
- [ ] Stage 8 ablation null confirmed (no omission response when H(t) is removed).
- [ ] ≥5 seeds per condition for Stages 7, 8, 9.
- [ ] Evidence artifacts present for each stage.

**Evidence required:** Actions 0016–0023 all [DONE].

**Alignment checkpoint:** → See Action 0036 (Phase C [ALIGN]).

---

### PHASE D — Figure Freeze

---

## Action 0024 — Render Figure 8 panels (PRIMARY FIGURE — omission paradigm)
- Status: [QUEUED]
- Phase: D
- Target file: docs/evidence_artifacts/figure8_omission_paradigm.py (or scripts/figures/fig8_omission.py)
- Depends on: 0022
- Why now: Figure 8 is the primary result figure. Freeze it before any other figure work.
- Change: Generate all 6 panels: A (in-vivo LFP reference), B (simulated LFP proxy), C (laminar profile), D (H(t) trace at omission onset), E (ablation null), F (spectrolaminar motif comparison). Export as PNG at ≥300 DPI.
- Validation: All 6 panel files exist in output directory; no NaN in data arrays.
- Evidence: _pending_.
- If blocked: Stage 8 (Action 0022) must be [DONE].

---

## Action 0025 — Render Figures 1–7 and 9–15
- Status: [QUEUED]
- Phase: D
- Target file: scripts/figures/ (multiple figure scripts)
- Depends on: 0016, 0017, 0018, 0019, 0020, 0021, 0023
- Why now: All supporting figures must be frozen before manuscript drafting.
- Change: Render all 15 figures. See figure list in SIMULATION STAGES & FIGURES section below.
- Validation: All 15 figure output files exist; no rendering errors.
- Evidence: _pending_.
- If blocked: Respective stage tests must be [DONE].

### Phase D — Exit Criteria

- [ ] All 15 figures rendered at ≥300 DPI.
- [ ] ≥5 seeds per condition for all stochastic panels.
- [ ] Figure 8 peer-reviewed by maintainer.
- [ ] No NaN/Inf in any exported data.

**Evidence required:** Actions 0024–0025 [DONE]; figure file inventory present.

**Alignment checkpoint:** → See Action 0037 (Phase D [ALIGN]).

---

### PHASE E — Manuscript Draft

---

## Action 0026 — Draft manuscript
- Status: [QUEUED]
- Phase: E
- Target file: (external — manuscript file, not tracked in repo)
- Depends on: 0024, 0025
- Why now: All figures must be frozen before writing so prose reflects actual results.
- Change: Full draft covering: Abstract, Introduction, Methods (H(t) formulation, jaxfne pipeline, paradigm details), Results (Stages 0–9), Discussion (foil comparison, limitations), References.
- Validation: Draft exists externally; all figures cited.
- Evidence: _pending_.
- If blocked: Phase D must be complete.

### Phase E — Exit Criteria

- [ ] Full draft written with all 15 figures cited.
- [ ] All amplitude claims stated as Relative (no uncalibrated absolute values).
- [ ] claim_level, field_claim_level, physical_amplitude_calibrated stated correctly in Methods.

**Evidence required:** Action 0026 [DONE].

**Alignment checkpoint:** → See Action 0038 (Phase E [ALIGN]).

---

### PHASE F — Internal Review

---

## Action 0027 — Internal review and revision
- Status: [QUEUED]
- Phase: F
- Target file: (external — manuscript file)
- Depends on: 0026
- Why now: Independent review catches narrative drift and validates ablation logic.
- Change: Reviewer comments addressed; ablation interpretations verified against null results.
- Validation: Review comments resolved; no open major issues.
- Evidence: _pending_.
- If blocked: Phase E must be complete.

### Phase F — Exit Criteria

- [ ] All reviewer comments addressed.
- [ ] Ablation logic correct and clearly stated.
- [ ] Final figures re-checked against revised text.

**Evidence required:** Action 0027 [DONE].

**Alignment checkpoint:** → See Action 0039 (Phase F [ALIGN]).

---

### PHASE G — Submission

---

## Action 0028 — Submit to PLOS Computational Biology (primary)
- Status: [QUEUED]
- Phase: G
- Target file: (external — journal submission)
- Depends on: 0027
- Why now: Primary venue. eLife is the alternative.
- Change: Manuscript, figures, and supplementary materials submitted.
- Validation: Submission confirmation received.
- Evidence: _pending_.
- If blocked: Phase F must be complete.

### Phase G — Exit Criteria

- [ ] Submission confirmation from PLOS Comp Biol or eLife.
- [ ] All code and data necessary for reproducibility archived (Zenodo or equivalent).

**Evidence required:** Action 0028 [DONE] with submission ID.

**Alignment checkpoint:** → See Action 0040 (Phase G [ALIGN]).

---

### PHASE H — Framework: Core Defragmentation

---

## Action 0029 — Defragment _model.py and _construct.py (v0.4.8–0.4.48)
- Status: [DONE]
- Phase: H
- Target file: jaxfne/_model.py, jaxfne/_construct.py (multiple — split pattern)
- Depends on: none (independent framework track)
- Why now: Phase 2 defragmentation completed; both files are now thin re-export aggregators.
- Change: `_model.py` re-exports from `_model_simulate.py`, `_model_readout.py`, `_model_evaluate.py`, `_model_tune.py`, `_model_manifest.py`. `_construct.py` re-exports from `_construct_population.py`, `_construct_connectivity.py`, `_construct_presets.py`, `_construct_core.py`, `_construct_extras.py`.
- Validation: `python3 -c "from jaxfne.core import Model, construct; print('ok')"` → ok.
- Evidence: AGENTS.md "Module map" section documents this split as of 2026-07-20.
- If blocked: N/A — already done.

### Phase H — Exit Criteria

- [x] `_model.py` and `_construct.py` are thin aggregators; direct internal imports blocked by convention.
- [x] `from jaxfne.core import Model, construct` works.

**Evidence required:** Action 0029 [DONE].

**Alignment checkpoint:** → See Action 0041 (Phase H [ALIGN]).

---

### PHASE I — Framework: Emitter Hardening

---

## Action 0030 — Extract _izhikevich_dv_du helper (close F-028)
- Status: [DONE]
- Phase: I
- Target file: jaxfne/emitters.py
- Depends on: none
- Why now: F-028 flagged duplication of Izhikevich derivatives across 11 of 12 closures as highest-value refactor in the file. Session 2026-07-30 completed this.
- Change: `_izhikevich_dv_du` pure function extracted. 12th site (dataclass-carry closure) deliberately left untouched. 26/26 tests in `test_v033_two_neuron_ei.py` pass unchanged.
- Validation: `pytest tests/test_v033_two_neuron_ei.py -v` → 26 passed.
- Evidence: Commit `refactor(emitters): extract _izhikevich_dv_du, closing F-028's duplication` (SHA: b165072). `docs/frictions` updated (SHA: f37f9893).
- If blocked: N/A — already done.

---

## Action 0031 — Fix cable_filter_sources for Metal (Apple Silicon) backend
- Status: [DONE]
- Phase: I
- Target file: jaxfne/fields/ (cable_filter_sources)
- Depends on: none
- Why now: JAX Metal backend (Apple Silicon) cannot legalize FFT for complex f32. Fix was urgent for local development.
- Change: FFT-based filter replaced with causal time-domain cascaded one-pole IIR (matched-pole, lax.scan, real arithmetic). Band-power ratio ~1.00–1.02 ≤150 Hz; deep:superficial gamma ratio 0.0996 → 0.0993. Validated band claims preserved. ruff pinned to 0.15.17 in pyproject [dev] to fix CI Lint-gate failure.
- Validation: `python3 -c "import jaxfne; print('ok')"` on Metal backend → ok.
- Evidence: Commit `fix(fields): port cable_filter_sources off FFT for Metal support` (SHA: 9a0d034). Lab graph commit (SHA: 0bce7613).
- If blocked: N/A — already done.

### Phase I — Exit Criteria

- [x] F-028 closed (duplication removed from emitters.py).
- [x] Metal backend: cable_filter_sources passes without FFT.
- [x] 26/26 emitter tests pass.
- [ ] Remaining 19 pre-existing test failures resolved (→ Phase Q).

**Evidence required:** Actions 0030–0031 [DONE].

**Alignment checkpoint:** → See Action 0042 (Phase I [ALIGN]).

---

### PHASE J — Framework: Fields & Vis

---

## Action 0032 — Audit all field operators for backend portability
- Status: [QUEUED]
- Phase: J
- Target file: jaxfne/fields/ (all modules)
- Depends on: 0031
- Why now: Metal fix revealed that FFT-based operations are a class risk. Full audit needed.
- Change: Grep all `jnp.fft` / `jax.numpy.fft` uses in `jaxfne/fields/`; verify each is either Metal-safe or replaced with IIR equivalent.
- Validation: `grep -r 'jnp.fft\|jax.numpy.fft' jaxfne/fields/` → 0 hits or all hits documented as Metal-safe.
- Evidence: _pending_.
- If blocked: N/A.

### Phase J — Exit Criteria

- [ ] All field operators backend-portable (no unguarded FFT on complex types).
- [ ] `jaxfne.vis.spectrolaminar_suite` runs on Metal without error.

**Evidence required:** Action 0032 [DONE].

**Alignment checkpoint:** → See Action 0043 (Phase J [ALIGN]).

---

### PHASE K — Framework: API Stabilization

---

## Action 0033 — Run public API audit
- Status: [QUEUED]
- Phase: K
- Target file: scripts/audit_public_docs_language.py (run only)
- Depends on: 0032
- Why now: API stabilization requires clean audit before any version bump.
- Change: `python3 scripts/audit_public_docs_language.py --check` passes with 0 violations. Known stubs (GLIFEmitter, LIFEmitter, write_nwb, read_nwb) documented as NotImplementedError per commit 9a0d034.
- Validation: `python3 scripts/audit_public_docs_language.py --check` → exit 0.
- Evidence: _pending_.
- If blocked: N/A.

### Phase K — Exit Criteria

- [ ] `audit_public_docs_language.py --check` → exit 0.
- [ ] No internal roadmap/competitive/agent-to-agent language in any `docs/` or `README.md` file.

**Evidence required:** Action 0033 [DONE].

**Alignment checkpoint:** → See Action 0044 (Phase K [ALIGN]).

---

### PHASE L — Framework: HDP / Homeostasis Track

---

## Action 0034a — Verify K_w_ctrl default does not cause weight runaway
- Status: [QUEUED]
- Phase: L
- Target file: jaxfne/_runtime_config.py (or equivalent DEFAULT_HDP definition)
- Depends on: 0011
- Why now: AGENTS.md "Known fragilities" item 3: `DEFAULT_HDP`'s `K_w_ctrl=0.0` permits unbounded weight drift on long/custom HDP runs outside specific verified presets.
- Change: Either raise the default, add a guard/warning, or document the safe usage boundary explicitly in the DEFAULT_HDP docstring.
- Validation: `grep -n 'K_w_ctrl' jaxfne/_runtime_config.py` shows either a non-zero default or an explicit safety annotation.
- Evidence: _pending_.
- If blocked: Action 0011 must be [DONE].

### Phase L — Exit Criteria

- [ ] H(t) emitter wired and tested (Actions 0011–0015).
- [ ] K_w_ctrl default risk resolved (Action 0034a).
- [ ] Multi-turn state continuity verified via `DynamicState` (all 6 fields).

**Evidence required:** Actions 0011, 0034a [DONE].

**Alignment checkpoint:** → See Action 0045 (Phase L [ALIGN]).

---

### PHASE M — Agent Infrastructure

---

## Action 0034b — Track .lab/ structural graph across machines
- Status: [DONE]
- Phase: M
- Target file: artifacts/.lab/ (104 tracked nodes)
- Depends on: none
- Why now: Graph was fully gitignored, losing hand-authored notes/status on second machines.
- Change: 104 structural nodes tracked. plan-*.json, Archive/, .formation_backups/, suggestions.json, metrics.json excluded. Verified: no secrets, no roadmap/competitive content in tracked nodes.
- Validation: `git ls-files artifacts/.lab/ | wc -l` → ≥104.
- Evidence: Commit `feat(lab): track the .lab structural graph` (SHA: 8f0e5d9a). Audit confirmed 0 excluded paths staged.
- If blocked: N/A — done.

---

## Action 0034c — Make Labyrinth use mandatory per prompt
- Status: [DONE]
- Phase: M
- Target file: AGENTS.md
- Depends on: 0034b
- Why now: Agents were treating .lab/ graph as optional bookkeeping, discarding compounding value.
- Change: "Labyrinth is not optional" section added to AGENTS.md. "Measure, don't vibe" instruction added (cite `repo_mapper.py --target .` numbers, never impressions).
- Validation: `grep -c 'Labyrinth is not optional' AGENTS.md` → ≥1.
- Evidence: Commits SHA: 64eeef46, d71aa8e2.
- If blocked: N/A — done.

### Phase M — Exit Criteria

- [x] .lab/ graph tracked (104 nodes).
- [x] Labyrinth use mandatory per AGENTS.md.
- [x] PRP backlog (plans.json, progress.json, etc.) gitignored and local-only.
- [ ] AGENT_CHANNEL.md handoff protocol clarified (no fixed schema yet — deferred).

**Evidence required:** Actions 0034b–0034c [DONE].

**Alignment checkpoint:** → See Action 0046 (Phase M [ALIGN]).

---

### PHASE N — CI / Lint / Docs Build

---

## Action 0034d — Pin ruff and fix CI Lint gate
- Status: [DONE]
- Phase: N
- Target file: pyproject.toml
- Depends on: none
- Why now: Unpinned ruff caused CI I001 failure (local 0.15.17 passes; CI's newer ruff failed).
- Change: `ruff==0.15.17` pinned in `[dev]` extras of pyproject.toml.
- Validation: CI Lint job passes; `ruff check . --select I001` → 0 violations.
- Evidence: Commit SHA: 9a0d034 (same commit as Metal fix).
- If blocked: N/A — done.

### Phase N — Exit Criteria

- [x] ruff pinned; CI Lint gate passes.
- [ ] `python3 -m mkdocs build --strict` → 0 errors.
- [ ] `python3 -m compileall -q jaxfne tests scripts` → 0 errors.

**Evidence required:** Action 0034d [DONE] + mkdocs build receipt.

**Alignment checkpoint:** → See Action 0047 (Phase N [ALIGN]).

---

### PHASE O — Documentation: Public Surface

---

## Action 0034e — Normalize stale version labels to 0.4.8
- Status: [DONE]
- Phase: O
- Target file: docs/ (multiple files — bulk)
- Depends on: none
- Why now: Pre-v0.4.7 version labels scattered across docs created reader confusion.
- Change: All stale `v0.3.*` / `v0.4.[0-7]` labels normalized to `0.4.8`.
- Validation: `grep -r 'v0.3\.' docs/ | grep -v changelog` → 0 hits (changelog exempted).
- Evidence: Commit `docs(cleanup): normalize stale pre-v0.4.7 version labels to 0.4.8` (SHA: 6dfe83df).
- If blocked: N/A — done.

---

## Action 0034f — Remove docs/publication/* (internal content, not for docs/)
- Status: [DONE]
- Phase: O
- Target file: docs/publication/ (DELETE — multiple files)
- Depends on: none
- Why now: docs/publication/ contained internal planning content (title decisions, competition analysis, publication timeline) that must not appear in the public docs surface per AGENTS.md policy.
- Change: All files under docs/publication/ deleted in a series of commits 2026-08-04.
- Validation: `ls docs/publication/ 2>&1` → "No such file or directory".
- Evidence: Chore commits removing docs/publication (SHAs: ecfa1ef, 6102f4e1, d6f13e81 and prior).
- If blocked: N/A — done.

### Phase O — Exit Criteria

- [x] No internal planning content in docs/.
- [x] Version labels normalized.
- [ ] `python3 scripts/audit_public_docs_language.py --check` → exit 0 (→ Phase K gate).

**Evidence required:** Actions 0034e–0034f [DONE].

**Alignment checkpoint:** → See Action 0048 (Phase O [ALIGN]).

---

### PHASE P — Documentation: Agent Surface

---

(Actions 0001–0007 are the Phase P work — see CONSOLIDATION BLOCK above.)

### Phase P — Exit Criteria

- [x] docs/fullroadmap.md exists (this file).
- [ ] artifacts/developer/ROADMAP.md deleted (→ Action 0005).
- [ ] docs/for_ai_agents.md updated with pointer to docs/fullroadmap.md (→ Action 0006).
- [ ] AGENTS.md verified — no stale roadmap pointer (→ Action 0007).

**Evidence required:** Actions 0005–0007 [DONE].

**Alignment checkpoint:** → See Action 0049 (Phase P [ALIGN]).

---

### PHASE Q — Test Coverage

---

## Action 0034g — Resolve 19 pre-existing test failures
- Status: [QUEUED]
- Phase: Q
- Target file: tests/ (multiple)
- Depends on: 0030, 0031
- Why now: 19 pre-existing failures present as of 2026-07-30 (docs-hygiene text rules, release-script tests, ModuleNotFoundError in scaling benchmarks). Must be resolved before coverage gate can be set.
- Change: Triage each failure; fix or document as intentional skip. Failure categories: docs-hygiene text rules, release-script tests, scaling benchmark import error.
- Validation: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest tests/ -q --tb=short` → 0 unexpected failures.
- Evidence: _pending_.
- If blocked: Emitter hardening (Phase I) and Metal fix must be [DONE] first.

### Phase Q — Exit Criteria

- [ ] 0 unexpected test failures.
- [ ] Coverage gate defined and enforced in CI.

**Evidence required:** Action 0034g [DONE] with pytest output.

**Alignment checkpoint:** → See Action 0050 (Phase Q [ALIGN]).

---

### PHASE R — Ablation Suite

---

## Action 0034h — Implement full ablation set (6 conditions)
- Status: [QUEUED]
- Phase: R
- Target file: tests/test_phaseR_ablations.py
- Depends on: 0022
- Why now: Ablation null controls are required for Stage 8 evidence standard and primary figure panel E.
- Change: Implement all 6 ablation conditions: (1) H(t) fully removed (α=0, τ_H→∞); (2) intrinsic coupling only; (3) synaptic gain only; (4) fast τ_H=100 ms (Abbott/Varela regime); (5) slow τ_H sweep (1, 5, 10, 30 s); (6) H(t) present, zero drive. ≥5 seeds per condition, mean ± CI, null comparison.
- Validation: `pytest tests/test_phaseR_ablations.py -v` → all pass; ablation null shows no omission response.
- Evidence: _pending_.
- If blocked: Stage 8 (Action 0022) must be [DONE].

### Phase R — Exit Criteria

- [ ] All 6 ablation conditions produce ≥5 seed results.
- [ ] Null comparison confirms H(t) removal eliminates omission response.

**Evidence required:** Action 0034h [DONE].

**Alignment checkpoint:** → See Action 0051 (Phase R [ALIGN]).

---

### PHASE S — Cross-Model Generalization

---

## Action 0034i — Stage 12: Jaxley/HH emitter generalization
- Status: [QUEUED]
- Phase: S
- Target file: tests/test_phaseS_stage12_jaxley_hh.py
- Depends on: 0022
- Why now: Generalization to Jaxley/HH demonstrates H(t) result is not model-specific (Figure 13).
- Change: Run omission paradigm with JaxleyBridge emitter; compare omission response to Izhikevich result.
- Validation: `pytest tests/test_phaseS_stage12_jaxley_hh.py -v` → pass.
- Evidence: _pending_.
- If blocked: Stage 8 must be [DONE]; Jaxley stubs must be unblocked.

---

## Action 0034j — Stage 15: Cross-model comparison (Izhikevich / LIF / HH-Jaxley)
- Status: [QUEUED]
- Phase: S
- Target file: tests/test_phaseS_stage15_cross_model.py
- Depends on: 0034i
- Why now: Cross-model comparison closes the generalization track.
- Change: Compare omission responses across all three emitter types; record mean ± CI.
- Validation: `pytest tests/test_phaseS_stage15_cross_model.py -v` → pass.
- Evidence: _pending_.
- If blocked: Stage 12 must be [DONE]; LIFEmitter must be unblocked from stub status.

### Phase S — Exit Criteria

- [ ] Stages 12 and 15 passing.
- [ ] H(t) result replicates across ≥2 emitter types.

**Evidence required:** Actions 0034i–0034j [DONE].

**Alignment checkpoint:** → See Action 0052 (Phase S [ALIGN]).

---

### PHASE T — Long-Term & Multi-Area

---

## Action 0034k — Stage 10: Long-term adaptation (Colab GPU)
- Status: [QUEUED]
- Phase: T
- Target file: tests/test_phaseT_stage10_longterm.py (or Colab notebook)
- Depends on: 0022
- Why now: Long-term adaptation requires GPU resources beyond local Metal; Colab is the designated platform.
- Change: Run 10+ minute simulation with H(t) active; record drift in H(t), rate, and weight norm. Verify K_w_ctrl guard (Action 0034a) prevents runaway.
- Validation: Simulation completes without NaN/Inf; H(t) remains bounded.
- Evidence: _pending_.
- If blocked: Action 0034a (K_w_ctrl fix) must be [DONE]; Colab access required.

---

## Action 0034l — Stage 11: Multi-area propagation
- Status: [QUEUED]
- Phase: T
- Target file: tests/test_phaseT_stage11_multiarea.py
- Depends on: 0034k
- Why now: Multi-area propagation tests laminar hierarchy interactions required for the spectrolaminar motif claim.
- Change: Two-area simulation (e.g. V1 → PFC); verify H(t) in each area reaches correct steady state; LFP proxy shows spectrolaminar motif.
- Validation: `pytest tests/test_phaseT_stage11_multiarea.py -v` → pass.
- Evidence: _pending_.
- If blocked: Stage 10 must be [DONE].

---

## Action 0034m — Stage 13: Optimization with AGSDR
- Status: [QUEUED]
- Phase: T
- Target file: tests/test_phaseT_stage13_agsdr.py
- Depends on: 0034l
- Why now: AGSDR optimization is the learning track; validates that H(t) can be trained toward a target.
- Change: Run AGSDR optimizer on H(t) parameters; record convergence.
- Validation: `pytest tests/test_phaseT_stage13_agsdr.py -v` → pass.
- Evidence: _pending_.
- If blocked: Stage 11 must be [DONE].

---

## Action 0034n — Stage 14: Parameter recovery (τ_H, r_target identifiability)
- Status: [QUEUED]
- Phase: T
- Target file: tests/test_phaseT_stage14_param_recovery.py
- Depends on: 0034m
- Why now: Identifiability analysis is required for the Methods section claim about parameter interpretability.
- Change: Generate synthetic data with known τ_H and r_target; recover via optimization; report identifiability intervals.
- Validation: `pytest tests/test_phaseT_stage14_param_recovery.py -v` → pass.
- Evidence: _pending_.
- If blocked: Stage 13 must be [DONE].

### Phase T — Exit Criteria

- [ ] Stages 10–11 passing on Colab GPU.
- [ ] Stages 13–14 passing locally.
- [ ] H(t) bounded in all long-run conditions.

**Evidence required:** Actions 0034k–0034n [DONE].

**Alignment checkpoint:** → See Action 0053 (Phase T [ALIGN]).

---

## Per-Phase Alignment Checkpoints

## Action 0034 — [ALIGN] Phase A alignment checkpoint
- Status: [DONE]
- Phase: A
- Target file: docs/fullroadmap.md
- Depends on: 0008, 0009, 0010
- Why now: Verify theory is complete and locked before emitter work begins.
- Change: Record: H(t) ODE locked; references anchored; title frozen; claim levels set.
- Validation: All Phase A exit criteria checked above.
- Evidence: Phase A exit block completed 2026-08-04.
- If blocked: N/A.

## Action 0035 — [ALIGN] Phase B alignment checkpoint
- Status: [QUEUED]
- Phase: B
- Target file: docs/fullroadmap.md
- Depends on: 0013, 0015
- Why now: Verify H(t) emitter is functional and Stages 0+1 both pass before moving to Stage 2.
- Change: Update Action 0035 Evidence field with pytest output SHAs.
- Validation: Phase B exit criteria all checked.
- Evidence: _pending_.
- If blocked: Actions 0013 and 0015 must be [DONE].

## Action 0036 — [ALIGN] Phase C alignment checkpoint
- Status: [QUEUED]
- Phase: C
- Target file: docs/fullroadmap.md
- Depends on: 0016, 0017, 0018, 0019, 0020, 0021, 0022, 0023
- Why now: Stage 8 (PRIMARY RESULT) must be fully verified before figure freeze.
- Change: Update Action 0036 Evidence with all Stage pytest receipts.
- Validation: Phase C exit criteria all checked.
- Evidence: _pending_.
- If blocked: All Stage actions must be [DONE].

## Action 0037 — [ALIGN] Phase D alignment checkpoint
- Status: [QUEUED]
- Phase: D
- Target file: docs/fullroadmap.md
- Depends on: 0024, 0025
- Change: Confirm all 15 figures frozen; record file inventory.
- Validation: Phase D exit criteria checked.
- Evidence: _pending_.
- If blocked: Actions 0024–0025 must be [DONE].

## Action 0038 — [ALIGN] Phase E alignment checkpoint
- Status: [QUEUED]
- Phase: E
- Target file: docs/fullroadmap.md
- Depends on: 0026
- Change: Confirm draft complete; all amplitude claims Relative.
- Evidence: _pending_.

## Action 0039 — [ALIGN] Phase F alignment checkpoint
- Status: [QUEUED]
- Phase: F
- Target file: docs/fullroadmap.md
- Depends on: 0027
- Change: Confirm all reviewer comments resolved.
- Evidence: _pending_.

## Action 0040 — [ALIGN] Phase G alignment checkpoint
- Status: [QUEUED]
- Phase: G
- Target file: docs/fullroadmap.md
- Depends on: 0028
- Change: Record submission ID.
- Evidence: _pending_.

## Action 0041 — [ALIGN] Phase H alignment checkpoint
- Status: [DONE]
- Phase: H
- Target file: docs/fullroadmap.md
- Depends on: 0029
- Change: Confirm split complete per AGENTS.md module map.
- Evidence: AGENTS.md "Module map" section documents split as of 2026-07-20.

## Action 0042 — [ALIGN] Phase I alignment checkpoint
- Status: [ACTIVE]
- Phase: I
- Target file: docs/fullroadmap.md
- Depends on: 0030, 0031
- Change: Confirm F-028 closed and Metal fix merged.
- Evidence: Commits b165072 and 9a0d034 on main.

## Action 0043 — [ALIGN] Phase J alignment checkpoint
- Status: [QUEUED]
- Phase: J
- Target file: docs/fullroadmap.md
- Depends on: 0032
- Change: Confirm field audit complete.
- Evidence: _pending_.

## Action 0044 — [ALIGN] Phase K alignment checkpoint
- Status: [QUEUED]
- Phase: K
- Target file: docs/fullroadmap.md
- Depends on: 0033
- Change: Confirm API audit passes.
- Evidence: _pending_.

## Action 0045 — [ALIGN] Phase L alignment checkpoint
- Status: [QUEUED]
- Phase: L
- Target file: docs/fullroadmap.md
- Depends on: 0011, 0034a
- Change: Confirm H(t) fully wired and K_w_ctrl risk resolved.
- Evidence: _pending_.

## Action 0046 — [ALIGN] Phase M alignment checkpoint
- Status: [DONE]
- Phase: M
- Target file: docs/fullroadmap.md
- Depends on: 0034b, 0034c
- Change: Confirm .lab/ graph tracked and Labyrinth use mandatory.
- Evidence: Commits 8f0e5d9a, 64eeef46, d71aa8e2.

## Action 0047 — [ALIGN] Phase N alignment checkpoint
- Status: [ACTIVE]
- Phase: N
- Target file: docs/fullroadmap.md
- Depends on: 0034d
- Change: Confirm ruff pinned and CI lint passes. mkdocs build still pending.
- Evidence: Commit 9a0d034.

## Action 0048 — [ALIGN] Phase O alignment checkpoint
- Status: [DONE]
- Phase: O
- Target file: docs/fullroadmap.md
- Depends on: 0034e, 0034f
- Change: Confirm docs/publication removed and version labels normalized.
- Evidence: Commits 6dfe83df, ecfa1ef, 6102f4e1, d6f13e81.

## Action 0049 — [ALIGN] Phase P alignment checkpoint
- Status: [ACTIVE]
- Phase: P
- Target file: docs/fullroadmap.md
- Depends on: 0001, 0005, 0006, 0007
- Change: Confirm canonical roadmap live; duplicate deleted; agent docs updated.
- Evidence: This commit + Actions 0005–0007 pending.

## Action 0050 — [ALIGN] Phase Q alignment checkpoint
- Status: [QUEUED]
- Phase: Q
- Target file: docs/fullroadmap.md
- Depends on: 0034g
- Change: Confirm 0 unexpected test failures.
- Evidence: _pending_.

## Action 0051 — [ALIGN] Phase R alignment checkpoint
- Status: [QUEUED]
- Phase: R
- Target file: docs/fullroadmap.md
- Depends on: 0034h
- Change: Confirm 6-condition ablation suite passing.
- Evidence: _pending_.

## Action 0052 — [ALIGN] Phase S alignment checkpoint
- Status: [QUEUED]
- Phase: S
- Target file: docs/fullroadmap.md
- Depends on: 0034i, 0034j
- Change: Confirm cross-model generalization stages passing.
- Evidence: _pending_.

## Action 0053 — [ALIGN] Phase T alignment checkpoint
- Status: [QUEUED]
- Phase: T
- Target file: docs/fullroadmap.md
- Depends on: 0034k, 0034l, 0034m, 0034n
- Change: Confirm long-term and multi-area stages passing.
- Evidence: _pending_.

---

## Simulation Stages & Figure Reference

### Simulation Stages

| Stage | Description | Phase | Primary Figure |
|-------|-------------|-------|----------------|
| 0 | Single neuron equilibrium — H(t) convergence to steady-state | B | Fig 4 |
| 1 | Repeated pulse adaptation — firing-rate decay, recovery time | B | Fig 4 |
| 2 | Population adaptation — synchrony, H distribution | C | Fig 5 |
| 3 | Frequency sweep — τ_H calibration (1–40 Hz) | C | Fig 6 |
| 4 | Amplitude sweep | C | Fig 12 |
| 5 | Duration sweep | C | Fig 12 |
| 6 | Random stimulus trains | C | — |
| 7 | Classical oddball — SSA index vs. Abbott/Varela null | C | Fig 7 |
| 8 | **Omission paradigm ← PRIMARY RESULT** | C | **Fig 8** |
| 9 | Global-local oddball | C | Fig 9 |
| 10 | Long-term adaptation (Colab GPU) | T | Fig 10 |
| 11 | Multi-area propagation | T | — |
| 12 | Jaxley/HH emitter generalization | S | Fig 13 |
| 13 | Optimization with AGSDR | T | — |
| 14 | Parameter recovery (τ_H, r_target identifiability) | T | Fig 14 |
| 15 | Cross-model comparison (Izhikevich / LIF / HH-Jaxley) | S | Fig 13 |

### Figure List

| Fig | Description |
|-----|-------------|
| 1 | Conceptual diagram — H(t) in multi-area laminar hierarchy |
| 2 | H(t) equation + TFNE-Izhikevich coupling |
| 3 | jaxfne operator pipeline |
| 4 | Single-neuron H(t) dynamics (Stages 0–1) |
| 5 | Population adaptation (Stage 2) |
| 6 | Frequency dependence (Stage 3) |
| 7 | Classical oddball / SSA (Stage 7) |
| **8** | **Omission paradigm (Stage 8) ← PRIMARY FIGURE** — Panels: A in-vivo LFP ref, B simulated LFP proxy, C laminar profile, D H(t) trace at omission onset, E ablation null, F spectrolaminar motif comparison |
| 9 | Global-local oddball (Stage 9) |
| 10 | Long-term adaptation (Stage 10) |
| 11 | Ablation summary |
| 12 | Robustness sweep |
| 13 | Cross-model generalization (Stage 15) |
| 14 | Parameter recovery |
| 15 | Summary schematic |

---

## H(t) Formulation Reference

```
τ_H · dH/dt = r_target - r(t)

r(t)      — instantaneous population firing rate
r_target  — homeostatic set-point (free parameter, matched to in-vivo baseline)
τ_H       — slow timescale (seconds to tens of seconds)
H(t)      — additive offset on Izhikevich threshold b  [primary mode]
            OR multiplicative gain on input current     [ablation mode]
```

Plausible Izhikevich sanity: rest ≈ −66 mV; spike peak ≈ +30 mV; mean rate ≈ 8–25 Hz.
`|Vm| > 150` or NaN/Inf = blowup.

Claim level: `computational_scaffold` · `field_claim_level: proxy_readout` · `physical_amplitude_calibrated: false`

---

## Ablation Set Reference

| Condition | Description |
|-----------|-------------|
| 1 | H(t) fully removed (α=0, τ_H→∞) |
| 2 | Intrinsic coupling only (threshold offset, no synaptic gain) |
| 3 | Synaptic gain only (no threshold offset) |
| 4 | Fast timescale (τ_H=100 ms — Abbott/Varela regime) |
| 5 | Slow timescale sweep (τ_H = 1, 5, 10, 30 s) |
| 6 | H(t) present, zero drive (r_target = r(t=0)) |

Evidence standard: ≥5 seeds per condition, mean ± CI, null comparison included.

---

## Literature References

| Reference | Role |
|-----------|------|
| Turrigiano & Nelson, Nat Rev Neurosci 2004 | Foundational homeostasis motivation |
| Cannon & Miller, PLOS Comp Biol 2016 | Closest formal match to H(t) (τ_θ ~30 s) |
| Abbott & Varela, Science 1997 | Fast-mechanism foil (synaptic depression) |
| Yaron et al., Neuron 2025 | Empirical omission target |

---

## Known Issues Register

| Issue | Backlink | Status |
|-------|----------|--------|
| K_w_ctrl=0.0 default allows unbounded weight drift on long/custom HDP runs | → Action 0034a | [QUEUED] |
| 19 pre-existing test failures (docs-hygiene, release-script, scaling benchmark import) | → Action 0034g | [QUEUED] |
| 9 unvalidated depends_on edges in .lab/ graph (hand-added, no grammar check) | → Action 0034b (resolved) / AGENTS.md note | [ACTIVE] |
| ruff unpinned (CI I001 failure) | → Action 0034d | [DONE] |
| cable_filter_sources FFT incompatibility on Metal backend | → Action 0031 | [DONE] |
| F-028: Izhikevich dv/du duplicated across 11/12 emitter closures | → Action 0030 | [DONE] |

---

## Change Log

| Date | Editor | Change |
|------|--------|---------|
| 2026-08-04 | HNXJ + agent | Initial creation: merged artifacts/developer/ROADMAP.md; established all 20 phases; 53 actions numbered globally; deleted docs/publication/ internal content; roadmap now canonical at this path. |
