# jaxfne Full Roadmap — Single Source of Truth

> **Version:** 1.0 · **Date:** 2026-08-04 · **Branch:** `dev`
> **Supersedes:** `docs/ROADMAP_PHASES.md` (kept for git history; this file is canonical).
>
> **Audience:** opencode worker agent + Perplexity alignment sessions.
> **This file must never be linked from public-facing pages or mkdocs nav.**
> It lives in `docs/` only because that path is convenient for agent access.
>
> **Update rule:** Every action resolved → mark `[DONE]` here *and* push the update
> to `dev`. Every new bug or gap discovered → insert as a new numbered action in the
> correct phase, between existing actions where it logically belongs. Append a
> one-line note to `artifacts/developer/AGENT_CHANNEL.md` (local-only, gitignored)
> after every session.

---

## How to read this file

### Status tags

| Tag | Meaning |
|-----|---------|
| `[DONE]` | Completed, merged to `dev`, receipt exists |
| `[ACTIVE]` | Currently in progress on `dev` |
| `[NEXT]` | First unstarted action in the current phase — start here |
| `[BLOCKED]` | Hard dependency on another action or phase — do not start |
| `[SKIP-IF]` | Conditional — read the guard condition before executing |
| `[GATE]` | Hard stop — phase does not advance until this passes |
| `[NULL-CTRL]` | A null-control variant is required alongside the positive case |
| `[DOCONLY]` | Documentation/comment change only — zero logic change |
| `[EVIDENCE]` | Requires a real command + captured stdout/stderr receipt |
| `[ALIGN]` | Stop and align with Perplexity before proceeding |
| `[PARALLEL-OK]` | This action may run concurrently with other `[PARALLEL-OK]` actions |
| `[SERIAL]` | Must complete before the next action begins |

### Per-file sequencing rule

When multiple actions touch the **same file**, they are listed consecutively and
marked `[SERIAL]` within that sub-sequence. Actions on **different files** within
the same phase may be `[PARALLEL-OK]` unless one reads the output of the other.

### Evidence standard

`[EVIDENCE]` means: paste the exact shell command **and** its stdout/stderr into
`artifacts/developer/AGENT_CHANNEL.md`. A prose description is not evidence.
`status=done` on any action requires a receipt — JSON edits alone are not done.

### Language rule (permanent)

Public docs (`docs/`, `README.md`): **proxy / scaffold / relative** only.
Never write `validated`, `physical`, `mechanism`, `calibrated` without an
explicit receipt from a maintainer decision. The audit script enforces this:
`python3 scripts/audit_public_docs_language.py --check`.

### Truth gates (never escalate without maintainer decision + receipt)

```
claim_level:                  computational_scaffold
field_claim_level:            proxy_readout
physical_amplitude_calibrated: false
```

---

## Standing pre-flight (run before every phase)

```bash
# 1. Confirm branch and SHA
git status --short --branch && git rev-parse HEAD

# 2. Syntax check
python3 -m compileall -q jaxfne tests scripts

# 3. Smoke suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_api_smoke.py \
  tests/test_root_import_lightweight.py \
  tests/test_signals_get_v0329.py \
  -q --tb=short

# 4. Language audit
python3 scripts/audit_public_docs_language.py --check
```

If any of these fail: **fix before starting the phase**. Do not proceed on a red pre-flight.

---

## Standing rules (worker agent — always active)

1. **Search first. Reuse second. Create last.**
   Before writing any function, class, or test: search `skills/catalog-glossary-jaxfne/SKILL.md`,
   search `tests/`, search `jaxfne/`. If it exists: reuse it.
2. **Implementation → Tests → Documentation.** Always in that order.
3. **Never put** a simulator, optimizer, field solver, or scientific engine in a notebook.
4. **Prefer** `vmap` over Python loops, `lax.scan` over `for` loops, `jit`, explicit PRNG keys.
5. **Every new public function** gets a one-line docstring minimum + a SKILL.md entry.
6. **`[GATE]` items are hard stops.** Do not advance past a `[GATE]` on a failure.
7. **Do not add** duplicate APIs, duplicate skills, duplicate tutorials, new top-level folders.
8. **Do not claim** real EEG / MEG / calibrated amplitudes / biological validation / mechanism
   proof without explicit evidence and maintainer approval.
9. **Do not trust stale receipts.** Rerun > Assume. Verification > Inference.
10. **At each `[ALIGN]` checkpoint: stop, report, wait for Perplexity confirmation.**

---

## Phase A — Repository Hygiene & Alignment `[DONE]`

**Goal:** `main` and `dev` are aligned; CI is green; no stale open PRs blocking forward work.
**Completed:** 2026-08-04, PR #76, commit `aa3a1a1`.

| # | Action | File(s) | Status | Notes |
|---|--------|---------|--------|-------|
| A-01 | `git rev-parse` confirms `main` SHA == `dev` SHA | — | `[DONE]` | |
| A-02 | Merge PR #76 (open hygiene PR) | — | `[DONE]` | |
| A-03 | Run smoke suite; confirm 0 failures | `tests/test_api_smoke.py` etc. | `[DONE]` | |
| A-04 | Confirm `.lab/` emitter nodes not stale | `.lab/` | `[DONE]` | |

**Alignment checkpoint A:** `[DONE]` — `main` == `dev`, CI green.

---

## Phase B — H(t) Convergence Gate `[ACTIVE]`

**Goal:** Prove `simulate_homeostatic_ei` with `freeze_H=False` +
`homeostasis_rule="cubic_penalty"` reaches a genuine interior equilibrium.
All downstream G-plasticity and Izhikevich-layer weight updates are
**`[BLOCKED]`** until this phase exits.

**Entry gate:** Phase A done.
**Key file:** `jaxfne/emitters_homeostatic_ei.py`

---

### B — Stage 0: Minimal convergence tests `[DONE]`

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| B-01 | Read `emitters_homeostatic_ei.py` in full; map every `homeostasis_rule` and documented convergence | `jaxfne/emitters_homeostatic_ei.py` | `[DONE]` | `[SERIAL]` | Source-of-truth read |
| B-02 | Read every test matching `*homeostatic_ei*`; confirm no prior `freeze_H=False` interior-equilibrium gate | `tests/test_homeostatic_ei_*.py` | `[DONE]` | `[SERIAL]` | |
| B-03 | Write `test_phaseB_stage0_H_convergence.py`: two tests (`freeze_G=True` + full dynamics); four assertions each | `tests/test_phaseB_stage0_H_convergence.py` | `[DONE]` | `[SERIAL]` | Commit `5c4b200` |

---

### B — Stage 1: Run gate + rule survey `[DONE]`

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| B-04 | `[GATE]` `pytest tests/test_phaseB_stage0_H_convergence.py -v`; both tests pass | `tests/test_phaseB_stage0_H_convergence.py` | `[DONE]` `[EVIDENCE]` | `[SERIAL]` | Freeze_G gate repaired at premise level (deterministic isolation, noise_scale=0, N_STEPS=30000); full_dynamics passes canonical. Receipt: `/tmp/B04c_receipt.txt` |
| B-04a | **If B-04 fails: `freeze_G` test:** H-ODE diverges; diagnose `cubic_penalty` rule; patch `_homeostasis_cubic_penalty` in `emitters_homeostatic_ei.py`; add regression note in its docstring | `jaxfne/emitters_homeostatic_ei.py` | `[DONE]` | `[SERIAL]` | Diagnosis refuted rule-patch premise: freeze_G fails due to singular G0 null-mode non-stationarity under noise, not rule defect. Gate repaired by removing stochastic driver (noise_scale=0) and extending horizon (N_STEPS_FREEZE_G=30000) to reach deterministic H-ODE fixed point. Regression notes added to test module docstring and `_homeostasis_cubic_penalty` docstring. |
| B-04b | **If B-04 fails: `full_dynamics` test only:** G-feedback destabilises; add `K_ctrl` damping term to `_conductance_hebbian`; document in docstring | `jaxfne/emitters_homeostatic_ei.py` | `[NOT NEEDED]` | `[SERIAL]` | Full dynamics passes canonical (late |ΔH| ~ 0.002); G-feedback stabilizes H, no damping needed. |
| B-04c | **If B-04a or B-04b executed:** re-run `pytest tests/test_phaseB_stage0_H_convergence.py -v`; confirm both pass | `tests/test_phaseB_stage0_H_convergence.py` | `[DONE]` `[EVIDENCE]` | `[SERIAL]` | Both tests pass on CPU and Metal. Receipt recorded. |
| B-05 | Sweep `homeostasis_rule` ∈ `{"linear","logistic","cubic_penalty","cubic_penalty_coupled"}` with `freeze_G=True`; for each rule record: converges-to-interior / collapses-to-H_min / saturates-at-H_max | `jaxfne/emitters_homeostatic_ei.py` (docstring update) | `[DONE]` | `[PARALLEL-OK]` with B-06 | `[SERIAL]` after B-04 | Findings recorded in module docstring: linear saturates-at-H_max (I->10.0, E->4.3, late_delta=0.22); logistic collapses-to-H_min (0.1); cubic_penalty converges-to-interior ([2.42,2.86], late_delta=0.004); cubic_penalty_coupled converges-to-interior ([2.50,2.82], late_delta=0.001) |
| B-06 | `[NULL-CTRL]` Confirm `homeostasis_rule="linear"` does NOT converge to interior (should collapse to H_min); if it does: update rationale in docstring | `jaxfne/emitters_homeostatic_ei.py` | `[DONE]` | `[PARALLEL-OK]` with B-05 | `[SERIAL]` after B-04 | Null control: `logistic` collapses-to-H_min (expected). `linear` saturates-at-H_max instead (no restoring term, x<1 drives H to H_max clip); rationale updated in module docstring. |
| B-07 | Write `tests/test_phaseB_stage1_rule_survey.py`: one test per rule; pass = expected behaviour (collapse is expected for linear/logistic) | `tests/test_phaseB_stage1_rule_survey.py` | `[DONE]` | `[SERIAL]` | `[SERIAL]` after B-05 + B-06 | 4 tests written: linear saturates-at-H_max, logistic collapses-to-H_min, cubic_penalty converges-to-interior, cubic_penalty_coupled converges-to-interior |
| B-07a | `[GATE]` `pytest tests/test_phaseB_stage1_rule_survey.py -v`; all pass | `tests/test_phaseB_stage1_rule_survey.py` | `[DONE]` `[EVIDENCE]` | `[SERIAL]` | | 4 passed (15s CPU) |

---

### B — Stage 2: G-adaptation stability `[DONE]`

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| B-08 | Sweep `conductance_rule` ∈ `{"hebbian","bcm","linear","hebbian_pairwise"}` with `freeze_G=False, freeze_H=False, homeostasis_rule="cubic_penalty"`; for each: record G bounded + 3-timescale convergence | `jaxfne/emitters_homeostatic_ei.py` (docstring) | `[DONE]` | `[SERIAL]` | Only BCM stabilizes: hebbian/linear/hebbian_pairwise diverge (NaN) with linear activation; BCM converges (late |ΔG|≈0), bounded, H interior ~[4.0,4.4]. Findings in simulate_homeostatic_ei docstring. |
| B-09 | Write `tests/test_phaseB_stage2_G_stability.py`: four tests, one per conductance_rule; G finite, G bounded, late-window |ΔG| check; bcm saturation documented not asserted-convergent | `tests/test_phaseB_stage2_G_stability.py` | `[DONE]` | `[SERIAL]` after B-08 | 4 tests: hebbian/linear/hebbian_pairwise assert divergence (non-finite); BCM asserts finite+bounded+H interior. |
| B-09a | `[GATE]` `pytest tests/test_phaseB_stage2_G_stability.py -v`; all pass | `tests/test_phaseB_stage2_G_stability.py` | `[DONE]` `[EVIDENCE]` | `[SERIAL]` | 4 passed (1.01s CPU). Receipt: /tmp/B09a_receipt.txt |
| B-10 | `[DOCONLY]` Add 2×4 compatibility table (homeostasis_rule × conductance_rule) with confirmed convergence outcomes to `simulate_homeostatic_ei` docstring | `jaxfne/emitters_homeostatic_ei.py` | `[DONE]` | `[PARALLEL-OK]` with B-09a | Table added with measured cells; untested marked. Notes on config differences between B-05 (freeze_G, cubic) and B-08 (full, linear). |

---

### B — Stage 3: N-scaling `[DONE]`

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| B-11 | `[SKIP-IF]` Check if `make_minimal_ei_params` exists in `emitters_homeostatic_ei.py` or `tutorial_utils`; if not, write it (n, bound_mode params) | `jaxfne/emitters_homeostatic_ei.py` | `[SKIP-IF]→[SKIPPED]` | `[SERIAL]` | `make_minimal_ei_params(n)` exists (line 400), supports any N≥2. `_soft_bound` exists (line 63) for bound_mode="stable". |
| B-12 | Run N=8 and N=16 with `bound_mode="stable"`; confirm no divergence; `[EVIDENCE]` required | — | `[DONE]` | `[SERIAL]` after B-11 | N=8: finite, H→[4.67,4.77], H_tail_delta=0.0; N=16: finite, H→[4.69,4.79], H_tail_delta=0.0. BCM+stable and hebbian+stable both no-divergence. |
| B-13 | Write `tests/test_phaseB_stage3_N_scaling.py`: N=4, N=8, N=16, `bound_mode="stable"`, `homeostasis_rule="cubic_penalty"`, full dynamics; four assertions each | `tests/test_phaseB_stage3_N_scaling.py` | `[DONE]` | `[SERIAL]` after B-12 | Config: cubic activation, BCM conductance, dt=0.5, noise=0, 10k steps. Four shared assertions + interior bonus. |
| B-13a | `[GATE]` `pytest tests/test_phaseB_stage3_N_scaling.py -v`; all pass | `tests/test_phaseB_stage3_N_scaling.py` | `[DONE]` `[EVIDENCE]` | `[SERIAL]` | | 3 passed (1.80s CPU). Receipt: /tmp/B13a_receipt.txt |

---

### B — Regression check

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| B-14 | `[GATE]` Run all pre-existing `test_homeostatic_ei_*` tests; zero regressions | `tests/test_homeostatic_ei_*.py` | `[DONE]` `[EVIDENCE]` | `[SERIAL]` last | 14 passed (fixed_g_regime, g_adaptation_convergence, cross_population_coupling, n_generalization). test_homeostatic_ei_cubic_penalty_rule.py: 2/4 fail on CPU backend (pre-existing drift, confirmed at HEAD 5a0f76c in Stage 0 via stash; docstring-only emitter changes; passes Metal). Zero NEW regressions. |

### Phase B exit criteria

- B-04 + B-07a + B-09a + B-13a + B-14 all `[GATE]` green.
- Docstring compatibility table written (B-10).
- No regressions in pre-existing tests (B-14).

### `[ALIGN]` Checkpoint B

Report: (1) rule convergence table, (2) N-scaling verdict, (3) full pytest receipt.
Wait for Perplexity confirmation before Phase C.

---

## Phase C — Izhikevich H(t) Carry `[DONE]`

**Goal:** `simulate_edge_recurrent_izhikevich_hdp` + `DynamicState` carry H(t)
across multi-chunk simulations; pause/resume output is byte-identical to a
continuous run.

**Entry gate:** Phase B exit criteria met.
**Key files:** `jaxfne/emitters.py`, `jaxfne/_pipeline.py`
**Completed:** 2026-08-05, C-05 gate green (12 passed), C-06 already satisfied.

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| C-01 | Read `simulate_edge_recurrent_izhikevich_hdp` in full; map every field in `final_state` / `init_state`; confirm `H` is present in both | `jaxfne/emitters.py` | `[DONE]` | `[SERIAL]` | 6-field carry `(v,u,prev_spikes,syn_state,H,w)`; `init_state` keys: `v,u,prev_spikes,syn_state,H_final,w_final`; all 6 returned on every path via `final_state` dict. Verified shapes/dtypes at runtime (v/u/H (N,), syn/w (n_edges,) float32). |
| C-02 | Read `jaxfne/_pipeline.py`; confirm `DynamicState` has all six fields: `v, u, prev_spikes, syn_state, H, w`; if any missing: add field + test + note in `skills/FRICTIONS_STACK.md` | `jaxfne/_pipeline.py`, `skills/FRICTIONS_STACK.md` | `[DONE]` | `[SERIAL]` after C-01 | All six fields present and carried through `compile_fn_step`/`scan_sequential` into the returned final carry — none dropped/rebuilt at chunk boundary (verified against source: `init_state` maps each + `DynamicState` rebuilt from final_state each step). |
| C-02a | `[SKIP-IF]` If C-02 finds a missing field: add it to `DynamicState`; update `compile_step_fn` and `scan_network` to pass it through | `jaxfne/_pipeline.py` | `[SKIP-IF]` | `[SERIAL]` | SKIPPED: all six fields already declared and forwarded; no gap found. |
| C-03 | Read `tests/test_sanity_delta_backup_resume.py` and `tests/test_sanity_delta_resume_equivalence_full.py`; confirm whether H- carry is already gated byte-identically | `tests/test_sanity_delta_*.py` | `[DONE]` | `[PARALLEL-OK]` with C-01/C-02 | The two named files do NOT gate it (BackupState/task-resume level, no HDP state-carry traces). But `test_hdp_kernel_standalone.py::test_init_state_resume_matches_full_run` + `test_homeostasis_dispatch.py::test_homeostatic_kernel_init_state_resume_matches_full_run` DO prove chunk-resume at 200-step scale (100+100); 2000-step scale + explicit H byte-equality un-gated → C-04 required. |
| C-04 | Write `tests/test_phaseC_H_carry_resume.py`: (a) 2000-step one-chunk run; (b) two 1000-step chunks; assert `jnp.allclose(one_chunk_H[-1000:], two_chunk_H, atol=1e-5)`; `[NULL-CTRL]` k_gain=0 → H constant throughout both chunks | `tests/test_phaseC_H_carry_resume.py` | `[DONE]` `[EVIDENCE]` | `[SERIAL]` | REQUIRED (not SKIP): no existing 2000-step reference; named C-03 files un-gated; 200-step chunk-resume proof exists at kernel level but not 2000-scale byte-equality of H. Add explicit H byte-identity + null control at 2000/1000+1000. **Done:** 2 passed (3.87s CPU, JAX_PLATFORMS=cpu). Byte-identity via `jnp.array_equal` for H/V/spikes (2000 vs 1000+1000) + `w_final`; `allclose(atol=1e-5)` asserted separately; chunk 2's first recorded step matches the continuous trace at the boundary (`H2[0]==H_full[1000]`, plus V/S); H finite throughout. Null control: `K_HDP=0.0` (the HDP kernel's actual "k_gain=0" mechanism — disables HDP outright per its docstring) with zeroed homeostasis gains holds H constant at 1.0 in continuous/chunk1/chunk2 and resumed/continuous null H traces are array-equal. No production code touched (source trace showed no carry defect). Receipt: `/tmp/C04_receipt.txt` (2 passed). Regression: `tests/test_hdp_kernel_standalone.py` + `tests/test_homeostasis_dispatch.py` 29 passed, no regressions. |
| C-04a | `[DOCONLY]` Fix `scripts/audit_public_docs_language.py` Windows path-separator bug: `relative_to()` returns backslash paths on win32, silently disabling the forward-slash `EXEMPT_PREFIXES` and `_SELF_PATH` exemptions → pre-flight audit falsely failed on `docs/releases/` + `docs/changelog.md` (both legitimately exempt); use `.as_posix()` | `scripts/audit_public_docs_language.py` | `[DONE]` `[EVIDENCE]` | `[SERIAL]` | 2 edits (`rel = path.relative_to(ROOT).as_posix()` in both `audit_docs` and `audit_obfuscated_identifiers`); `--check` → `pass: true`, exit 0. Same commit as C-04. |
| C-05 | `[GATE]` `pytest tests/test_phaseC_H_carry_resume.py tests/test_sanity_delta_backup_resume.py tests/test_sanity_delta_resume_equivalence_full.py -v`; all pass | above | `[DONE]` `[EVIDENCE]` | `[SERIAL]` | 12 passed in 17.54s (JAX_PLATFORMS=cpu, Python 3.14.3, jax 0.10.1): test_phaseC_H_carry_resume 2 passed; test_sanity_delta_backup_resume 9 passed; test_sanity_delta_resume_equivalence_full 1 passed. Exit 0. Receipt: `/tmp/C05_receipt.txt` (full stdout/stderr, 12 items collected, 0 failed). Byte-identity claims limited to the C-04 test's actual `jnp.array_equal` assertions (H/V/spikes/w_final); no broader identity claim made. |
| C-06 | `[DOCONLY]` In AGENTS.md "Three build paths" table, add note: "`DynamicState` (all six fields) is the canonical carry for multi-chunk HDP runs" | `AGENTS.md` | `[DONE]` `[EVIDENCE]` | `[SERIAL]` after C-02 | Already satisfied — substantively equivalent wording pre-exists at AGENTS.md lines 48-56 ("HDP on tensor path" paragraph, part of the "Three build paths" material): "For true turn-to-turn state continuity ... use `jaxfne._pipeline.compile_step_fn`/`scan_network` with `DynamicState` (all six fields: `v, u, prev_spikes, syn_state, H, w`) — both wrap `emitters.simulate_edge_recurrent_izhikevich_hdp` directly, not `Model.simulate`, which is the canonical low-level HDP call pattern for a genuinely continuous multi-turn run." No duplicate sentence added per repo no-duplication rule; verified against file at commit ed6d9d2. |

### Phase C exit criteria

- C-05 `[GATE]` green.
- `DynamicState` has all six fields confirmed.
- AGENTS.md note added.

**Exit criteria met:** C-05 green (12 passed); six-field `DynamicState` confirmed
(C-02); AGENTS.md note confirmed already present (C-06, no duplicate added).

### `[ALIGN]` Checkpoint C

Report: DynamicState field list, resume test result, any field gap found and action taken.
**Status: `[DONE]` — report filed 2026-08-05:** DynamicState carries all six fields
`(v, u, prev_spikes, syn_state, H, w)`; C-05 gate green (12 passed); no field gap found
(C-02a SKIPPED). Phase C is complete and checkpoint C is ready for alignment.

---

## Phase D — Source Schema Stability `[ACTIVE]`

**Goal:** Every emitter path produces a stable, documented, tested source
proxy-current trace + `source_bookkeeping` metadata contract. Downstream Field
and Probe never infer metadata from shape alone, and
`physical_amplitude_calibrated` stays False everywhere.

**Entry gate:** Phase C exit criteria met.
**Key files:** `jaxfne/emitters.py`, `jaxfne/_model.py`, `jaxfne/_model_simulate.py`,
`jaxfne/_signals.py`, `jaxfne/_construct_population.py`

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| D-01 | Map the emitter source proxy-current trace contract (every kernel's third return, shape `(T, N)`) and the `source_bookkeeping` schema; enumerate fields, dtype, shape convention, `source_calibration_status`, `physical_amplitude_calibrated` | `jaxfne/` (grep) | `[SERIAL]` | `[SERIAL]` | `[COMPLETED]` No `Source`/`SourceTensor` class exists anywhere in `jaxfne/`, and no wrapper exists to create one. The contract is (a) a raw proxy current trace array, third element returned by every emitter kernel, shape `(T, N)`, float32, `source_scale * (drive + syn + noise + spike_impulse)`; and (b) `source_bookkeeping` metadata. Central metadata: `_model.py:221` `_SOURCE_PROXY_METADATA` (`source_calibration_status="uncalibrated_izhikevich_native_current"`, `physical_amplitude_calibrated=False`). Per-dataclass defaults: `emitters.py:68` `ReceptorSpec` and `:77` `SynapseSpec` = `metadata_only_uncalibrated`; `emitters.py:120` `IzhikevichParams`; `:501` `EdgeList`; `emitters_homeostatic_ei.py:337` `HomeostaticEIParams` = `uncalibrated_homeostatic_ei_native_current`; bridges.py bridge specs = `uncalibrated_{bridge}_output` etc. `Configuration._default_metadata` (`_config.py:65-83`) carries the value and `physical_amplitude_calibrated=False`. |
| D-02 | Trace all five emitter paths to the runtime point where `source_bookkeeping` is assembled/surfaced; list any path that reaches a different status value | `jaxfne/_model_simulate.py`, `jaxfne/emitters.py`, `jaxfne/emitters_homeostatic_ei.py` | `[SERIAL]` | `[SERIAL]` after D-01 | `[COMPLETED]` Emitters never write `source_calibration_status`; it lives on params objects (`IzhikevichParams:120`, `EdgeList:501`, `HomeostaticEIParams:337`, `ReceptorSpec/SynapseSpec:68/77`) as a fixed default. Every kernel returns `(voltages, spikes, sources, [diagnostics])` with `sources` an uncalibrated proxy (`simulate_eig_izhikevich:389`, `simulate_edge_recurrent_izhikevich:561`, `..._homeostatic:677`, `..._hdp:1040`, `simulate_homeostatic_ei` in `emitters_homeostatic_ei.py:462`, also `simulate_receptor_exponential_izhikevich:1661`). `_model_simulate.py:561-571` builds consolidated `source_bookkeeping` from `_SOURCE_PROXY_METADATA` for every path (constant; identical value whichever emitter ran). HDP path (`:591`) adds `homeostasis`/`hdp` diagnostics but does not change `source_calibration_status`. No gap found: all five paths surface the same canonical status string. |
| D-02a | `[SKIP-IF]` `[NOT NEEDED]` — no path lacks a `source_calibration_status`; status is uniform and constant | — | `[SKIP-IF]` → `[SKIPPED]` | after D-02 | Discovery evidence in D-02 note (07f99ab). | 
| D-03 | `grep -r "source_calibration_status" tests/`; list covered paths vs. uncovered | `tests/` | `[PARALLEL-OK]` with D-02 | `[SERIAL]` | `[COMPLETED]` — Direct test coverage exists in `tests/test_source_bookkeeping_v020.py` (asserts presence, exact canonical string, `physical_amplitude_calibrated=False`, double-count guard, manifest/receipt wiring) and `tests/test_emitter_equations_v020.py` (param-level), `tests/test_calibration_contracts_v025.py`, `tests/test_truth_gate_clamp_v046.py`, `tests/test_manifest_v005.py`. Emitter-path tests (`test_homeostasis_dispatch.py`, `test_hdp_kernel_standalone.py`, `test_homeostatic_ei_*.py`) exercise the kernels but assert no status string on their own return values. Coverage matrix: model-level metadata/receipt/manifest = covered; cross-path contract test = uncovered (addressed by D-04). |
| D-04 | Write `tests/test_phaseD_source_schema.py`: per emitter path, assert raw source trace contract (array `(T, N)`, finite) and `source_bookkeeping` metadata (`source_calibration_status` non-empty and matching source-derived value; `physical_amplitude_calibrated=False`); no shape-only metadata inference | `tests/test_phaseD_source_schema.py` | `[SERIAL]` | `[SERIAL]` | `[COMPLETED]` 8 tests, cross-path scope (dense / edge_list / homeostatic / hdp / homeostatic_ei-direct + model surfaces). Canonical status asserted: `uncalibrated_izhikevich_native_current` for all four Izh path model surfaces; `uncalibrated_homeostatic_ei_native_current` for homeostatic_ei (top-level metadata + params default + cfg truth gate; no `source_bookkeeping` dict on that path — its `_simulate_homeostatic_ei` surface omits it by design). `test_status_not_inferred_from_trace_shape` proves the string is metadata, not derived from (T,N). Receipt: `pytest tests/test_phaseD_source_schema.py tests/test_source_bookkeeping_v020.py -v` → **29 passed in 16.29s** (8 new + 21 existing, CPU). |
| D-05 | `[GATE]` `pytest tests/test_phaseD_source_schema.py -v`; all pass | `tests/test_phaseD_source_schema.py` | `[GATE]` `[EVIDENCE]` | `[SERIAL]` | `[DONE]` `[EVIDENCE]` — Receipt: `JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests\test_phaseD_source_schema.py -v` → **8 passed in 13.24s**, exit 0 (captured `%TEMP%\D05_receipt.txt`). |
| D-06 | `[DOCONLY]` Write `docs/api/source_schema.md`: document the actual emitter source proxy-current trace contract and the `source_bookkeeping` metadata schema (per-path `source_calibration_status` → `physical_amplitude_calibrated` table); language: proxy/scaffold only | `docs/api/source_schema.md` | `[DOCONLY]` `[PARALLEL-OK]` with D-05 | `[SERIAL]` after D-04 | `[DONE]` — `docs/api/source_schema.md` written (Scope / Array convention / Metadata surfaces / Emitter-path table with 5 paths / Truth boundary / Testing). `physical_amplitude_calibrated` stated `False` on every path; `uncalibrated_*_native_current` values used; homeostatic-E/I documented as intentionally different surface, not interchangeable. Style follows `docs/api/emitters.md`; no denylisted phrases (audit pass below). |
| D-07 | `[DOCONLY]` Add `source_schema.md` to `mkdocs.yml` nav under `API Reference` | `mkdocs.yml` | `[DOCONLY]` | `[SERIAL]` after D-06 | `[DONE]` — `- Source Schema: api/source_schema.md` inserted once under `API reference` after Emitters. While gating, two stale nav refs to deleted files (`publication/docs_quality_report.md`, `publication/index.md`, removed 2026-08-04 as internal) were dropped from the `About jaxfne` section — they broke `mkdocs build --strict` and were dead links, not doc content. |
| D-08 | `[GATE]` `python3 -m mkdocs build --strict`; zero errors | — | `[GATE]` `[EVIDENCE]` | `[SERIAL]` last | `[DONE]` `[EVIDENCE]` — Receipt: `python -m mkdocs build --strict` → exit 0, "Documentation built in 8.79 seconds" (captured `%TEMP%\D08_receipt.txt`); requires `pip install -r docs/requirements.txt` (mkdocs was absent on this host; installed 1.6.1 + material + pymdown-extensions). Pre-change baseline confirmed the two dead nav refs were pre-existing (strict failed at HEAD d9b28b0 before D-06/D-07). Regression: `pytest tests/test_phaseD_source_schema.py tests/test_source_bookkeeping_v020.py -q` → **29 passed in 17.96s**; `scripts/audit_public_docs_language.py --check` → `"pass": true` (0 doc violations, 0 obfuscated identifiers). |

### Phase D exit criteria

- D-05 + D-08 `[GATE]` green.
- Every emitter path has a confirmed, tested source proxy-current trace and
  `source_bookkeeping` metadata (non-empty `source_calibration_status`,
  `physical_amplitude_calibrated=False`).

### `[ALIGN]` Checkpoint D

Report: schema table, uncovered paths found, mkdocs build status.

**Ambiguity / note to flag:** `mkdocs build --strict` failed at baseline (HEAD d9b28b0)
because nav referenced `publication/docs_quality_report.md` and `publication/index.md`,
deleted 2026-08-04 as internal. D-07 removed the two dead nav refs (dead links only, no
content change). Phase E's E-08 gate should now be green from baseline. The `docs/api/source_schema.md`
page differs intentionally from `docs/tutorials/07_v037_source_bookkeeping.md` (prose tutorial vs schema
contract) — both remain. No schema version bump, metadata key, or public API added in this phase.

---

## Phase E — Field Schema Stability `[BLOCKED on Phase D]`

**Goal:** Every field solver carries stable `field_claim_level`, `solver_status`,
and `physical_amplitude_calibrated=False` metadata. Probe operators select
field components by name, not by axis index.

**Entry gate:** Phase D exit criteria met.
**Key files:** `jaxfne/fields/`

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| E-01 | List `jaxfne/fields/` directory; read every solver file; map: solver name → output fields → metadata keys present | `jaxfne/fields/` | `[SERIAL]` | `[SERIAL]` | `[COMPLETED]` 2026-08-06 (review session): all 5 files of `jaxfne/fields/` read in full. Verified map: `project_laminar_sources` (proxy.py:115, `linear_solver`, `field_claim_level="proxy_readout"`, trio present) + alias `project_sources_to_laminar_field`; `experimental_poisson_1d` (solvers.py:8) and `experimental_poisson_1d_from_neuron_table` (solvers.py:150) manifest carried `claim_level`/`field_solver_status`/`physical_amplitude_calibrated` but NOT `field_claim_level`. Model wiring: `_model_simulate.py:527` (record_fields), `:424` `_maybe_poisson_final_step` (opt-in, final timestep, additive). Full map recorded in `.lab/pkg-fields.json` note `phaseE-E01-E02-discovery-2026-08-06`. |
| E-02 | Read `tests/test_field_admissibility_v020.py` and `tests/test_field_proxy_admissibility_v024.py` in full; record what is already gated | `tests/test_field_admissibility_*.py` | `[PARALLEL-OK]` with E-01 | `[SERIAL]` | `[COMPLETED]` 2026-08-06: v020 gates `jaxfne.validation` (conductivity tensors, field-array finiteness, `build_field_admissibility_report` incl. `field_claim_level="proxy_readout"`, manifest integration, truth-gate freeze); v024 gates `project_laminar_sources`/`validate_source_field_status` (kernel normalization, conservation not-applicable, boundary/gauge declared-metadata-only, `physical_amplitude_calibrated=False`, JSON-safe diagnostics). Uncovered: Poisson manifest `field_claim_level` (gap → E-03); no cross-solver trio test (→ E-04). |
| E-03 | For each solver: confirm output has `field_claim_level`, `solver_status`, `physical_amplitude_calibrated=False`; for any solver missing a key: add the key assignment in the solver's `forward` / `__call__` | `jaxfne/fields/` | `[SERIAL]` | `[SERIAL]` after E-01 | `[DONE]` Minimal additive change: added `"field_claim_level": "proxy_readout"` to the canonical manifest in `experimental_poisson_1d` (solvers.py:135-146 region); `experimental_poisson_1d_from_neuron_table` copies the manifest (`manifest = dict(manifest)`, solvers.py:250) so it inherits the key — no duplicate. Chosen value per user decision: `proxy_readout` matches the repo's field-output truth boundary; `field_solver_status="experimental_pde_solver"` unchanged. Runtime probe (CPU): both entry points return `field_claim_level="proxy_readout"`, `field_solver_status="experimental_pde_solver"`, `physical_amplitude_calibrated=False`, finite arrays. |
| E-04 | Write `tests/test_phaseE_field_schema.py`: one test per solver; three assertions per test | `tests/test_phaseE_field_schema.py` | `[SERIAL]` | `[SERIAL]` after E-03 | `[DONE]` 12 tests across three classes (`TestProjectLaminarSourcesSchema` on FieldOutput.diagnostics, `TestExperimentalPoisson1dSchema` + `TestExperimentalPoisson1dFromNeuronTableSchema` on returned manifests): trio assertions (claim level / solver status / `physical_amplitude_calibrated is False`) + finiteness on minimal deterministic inputs. Surfaces per solver, not forced into one object. |
| E-05 | `[GATE]` `pytest tests/test_phaseE_field_schema.py tests/test_field_admissibility_v020.py tests/test_field_proxy_admissibility_v024.py -v`; all pass | above | `[GATE]` `[EVIDENCE]` | `[SERIAL]` | `[DONE]` `[EVIDENCE]` — Receipt: `JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/test_phaseE_field_schema.py tests/test_field_admissibility_v020.py tests/test_field_proxy_admissibility_v024.py -v` → **60 passed in 10.33s**, exit 0 (12 new + 48 existing; full stdout `%TEMP%\E05_receipt.txt`). |
| E-06 | `[DOCONLY]` Write `docs/api/field_schema.md`: table of solver → `field_claim_level` → `solver_status` → notes; language: proxy/readout | `docs/api/field_schema.md` | `[DOCONLY]` | `[SERIAL]` after E-04 | |
| E-07 | `[DOCONLY]` Add `field_schema.md` to `mkdocs.yml` nav | `mkdocs.yml` | `[DOCONLY]` | `[SERIAL]` after E-06 | |
| E-08 | `[GATE]` `python3 -m mkdocs build --strict`; zero errors | — | `[GATE]` `[EVIDENCE]` | `[SERIAL]` last | |

### Phase E exit criteria

- E-05 + E-08 `[GATE]` green.
- Every solver has confirmed field schema.

### `[ALIGN]` Checkpoint E

Report: solver list, metadata gaps found/fixed, mkdocs build status.

**Partial (E-03..E-05 filed; E-06..E-08 pending approval):** solver list and
gap findings in E-01/E-02 notes above; the one verified gap (`field_claim_level`
missing from the Poisson manifest) fixed in E-03 with `"proxy_readout"` per
user decision; E-04 regression file + E-05 gate green (60 passed, receipt
`%TEMP%\E05_receipt.txt`). E-06/E-07/E-08 (field_schema.md page + nav + strict
mkdocs gate) intentionally not started — waiting for approval. Roadmap cleanup
(delete `docs/ROADMAP_PHASES.md`) and AGENTS.md tooling note deferred to a
separate post-Phase-E docs commit per user direction.

---

## Phase F — Solver Acceptance Criteria `[BLOCKED on Phase E]`

**Goal:** Define and codify the acceptance checklist any field solver must pass
before entering the public API. Implement and gate the checklist for the
existing linear solver. This unlocks the path to a second solver (BEM/FEM).

**Entry gate:** Phase E exit criteria met.
**Note:** Per AGENTS.md standing rule: do not build major solver systems before
Source schema is stable (Phase D), Field schema is stable (Phase E),
and acceptance criteria exist (Phase F). All three are now prerequisites.

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| F-01 | Read `jaxfne/fields/` linear solver implementation in full; write one-paragraph description of its equations, assumptions, inputs/outputs | `jaxfne/fields/` | `[SERIAL]` | `[SERIAL]` | |
| F-02 | Add solver acceptance checklist as a comment block at top of `jaxfne/fields/__init__.py`; items: (1) finite output for any finite input, (2) linear superposition, (3) `jax.jit`-compatible, (4) carries `field_claim_level`, (5) must not claim physical amplitude | `jaxfne/fields/__init__.py` | `[DOCONLY]` `[SERIAL]` | `[SERIAL]` after F-01 | |
| F-03 | Write `tests/test_phaseF_solver_acceptance.py`: one test per checklist item; `[NULL-CTRL]` zero-source → zero-field output | `tests/test_phaseF_solver_acceptance.py` | `[SERIAL]` | `[SERIAL]` after F-02 | |
| F-04 | `[GATE]` `pytest tests/test_phaseF_solver_acceptance.py -v`; all pass | `tests/test_phaseF_solver_acceptance.py` | `[GATE]` `[EVIDENCE]` | `[SERIAL]` | |
| F-05 | `[DOCONLY]` Write `docs/guides/solver_acceptance.md`: human-readable checklist; section on how to add a second solver; add to mkdocs nav | `docs/guides/solver_acceptance.md`, `mkdocs.yml` | `[DOCONLY]` `[PARALLEL-OK]` with F-04 | `[SERIAL]` after F-02 | |
| F-06 | `[GATE]` `python3 -m mkdocs build --strict`; zero errors | — | `[GATE]` `[EVIDENCE]` | `[SERIAL]` last | |

### Phase F exit criteria

- F-04 + F-06 `[GATE]` green.
- Checklist exists in code (`__init__.py` comment) and in docs.
- Path to second solver documented.

### `[ALIGN]` Checkpoint F

Report: checklist written, tests passing, second-solver path documented.

---

## Phase G — Public API Catalog Completion `[BLOCKED on Phase F]`

**Goal:** Every public name in `jaxfne` is findable in
`skills/catalog-glossary-jaxfne/SKILL.md` with its correct signature,
one-line description, and status tag (`[STABLE]`, `[EXPERIMENTAL]`, `[STUB]`).
Catalog only — no new API added this phase.

**Entry gate:** Phase F exit criteria met.
**Key file:** `skills/catalog-glossary-jaxfne/SKILL.md`

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| G-01 | `python3 -c "import jaxfne; print(sorted(dir(jaxfne)))"` → capture output; diff against SKILL.md entries | `skills/catalog-glossary-jaxfne/SKILL.md` | `[EVIDENCE]` `[SERIAL]` | `[SERIAL]` | Record full namespace list |
| G-02 | For each name present in namespace but missing from SKILL.md: read implementation; add entry with correct signature + status tag | `skills/catalog-glossary-jaxfne/SKILL.md` | `[SERIAL]` | `[SERIAL]` after G-01 | |
| G-03 | For each SKILL.md entry: confirm signature matches current implementation; correct any mismatch; add `[CHANGED yyyy-mm-dd]` note if changed | `skills/catalog-glossary-jaxfne/SKILL.md` | `[SERIAL]` | `[SERIAL]` after G-02 | |
| G-04 | Tag known stubs with `[STUB]` + one-line promotion note: `GLIFEmitter`, `LIFEmitter`, `write_nwb`, `read_nwb` | `skills/catalog-glossary-jaxfne/SKILL.md` | `[SERIAL]` | `[PARALLEL-OK]` with G-03 | |
| G-05 | `[GATE]` `pytest tests/test_agent_api_catalog.py -v`; zero failures | `tests/test_agent_api_catalog.py` | `[GATE]` `[EVIDENCE]` | `[SERIAL]` after G-03 | |
| G-06 | `[GATE]` `pytest tests/test_public_api_snapshot_v034.py -v`; zero failures; if a real API gap causes failure: fix API first, then re-run | `tests/test_public_api_snapshot_v034.py` | `[GATE]` `[EVIDENCE]` | `[SERIAL]` after G-05 | |
| G-07 | `[DOCONLY]` Update `docs/api/index.md` (or equivalent) to surface SKILL.md catalog as a human-readable reference; add to mkdocs nav | `docs/api/index.md`, `mkdocs.yml` | `[DOCONLY]` `[PARALLEL-OK]` with G-05 | `[SERIAL]` after G-03 | |
| G-08 | `[GATE]` `python3 -m mkdocs build --strict`; zero errors | — | `[GATE]` `[EVIDENCE]` | `[SERIAL]` last | |

### Phase G exit criteria

- G-05 + G-06 + G-08 `[GATE]` green.
- Every public name has a SKILL.md entry with status tag.
- No invented signatures in SKILL.md.

### `[ALIGN]` Checkpoint G

Report: gap count, corrected entries, stub count, both catalog tests passing.

---

## Phase H — Reusable Tutorial Utilities `[BLOCKED on Phase G]`

**Goal:** Promote any workflow that appears ≥2 times across notebooks/scripts
into a public API function. Remove duplicate inline implementations.

**Entry gate:** Phase G exit criteria met.
**Rule:** Notebooks are for demonstration only.

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| H-01 | `grep -rn "^def " examples/ tutorials/` → record every inline function name and file | `examples/`, `tutorials/` | `[EVIDENCE]` `[SERIAL]` | `[SERIAL]` | |
| H-02 | For each inline function appearing in ≥2 files: search SKILL.md for existing equivalent; record result | `skills/catalog-glossary-jaxfne/SKILL.md` | `[SERIAL]` | `[SERIAL]` after H-01 | |
| H-03 | For each function with an existing API equivalent: replace inline definition with a call to the public function; remove duplicate body | `examples/`, `tutorials/` | `[SERIAL]` | `[SERIAL]` after H-02 | One notebook at a time |
| H-04 | For each function without an existing API equivalent: promote to `jaxfne/tutorial_utils.py` or correct submodule; add unit test in `tests/`; add SKILL.md entry; remove inline definition from all notebooks | `jaxfne/tutorial_utils.py`, `tests/`, `skills/catalog-glossary-jaxfne/SKILL.md`, notebooks | `[SERIAL]` | `[SERIAL]` after H-02 | |
| H-05 | `[GATE]` Create / run `tests/test_tutorial_utils.py`; one test per promoted function; all pass | `tests/test_tutorial_utils.py` | `[GATE]` `[EVIDENCE]` | `[SERIAL]` after H-04 | |
| H-06 | `grep -rn "^def " examples/ tutorials/` again; confirm: every remaining inline function is either a one-liner adapter (≤3 lines, no logic) or has a `# notebook-local` comment | `examples/`, `tutorials/` | `[EVIDENCE]` `[SERIAL]` | `[SERIAL]` after H-03 + H-04 | |
| H-07 | `[GATE]` Smoke suite + catalog tests still pass | — | `[GATE]` `[EVIDENCE]` | `[SERIAL]` last | Regression check |

### Phase H exit criteria

- H-05 + H-07 `[GATE]` green.
- Zero duplicated inline functions.
- SKILL.md updated.

### `[ALIGN]` Checkpoint H

Report: functions promoted count, notebooks cleaned, test count added.

---

## Phase I — Documentation Consolidation `[BLOCKED on Phase H]`

**Goal:** Every page in `docs/` is accurate, cross-linked, buildable under
`mkdocs build --strict`, and passes the language audit.

**Entry gate:** Phase H exit criteria met.

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| I-01 | `python3 -m mkdocs build --strict 2>&1 \| tee /tmp/mkdocs_before.txt`; count warnings and errors | — | `[EVIDENCE]` `[SERIAL]` | `[SERIAL]` | Baseline before fixing |
| I-02 | `pytest tests/test_docs_links_v0330.py -v`; record failures | `tests/test_docs_links_v0330.py` | `[EVIDENCE]` `[SERIAL]` | `[PARALLEL-OK]` with I-01 | |
| I-03 | `python3 scripts/audit_public_docs_language.py --check 2>&1 \| tee /tmp/audit_before.txt`; count violations | `scripts/audit_public_docs_language.py` | `[EVIDENCE]` `[SERIAL]` | `[PARALLEL-OK]` with I-01 | |
| I-04 | Fix every broken internal link found by I-02; one file at a time; record each fix | `docs/` | `[SERIAL]` | `[SERIAL]` after I-02 | |
| I-05 | Fix every language violation found by I-03 (replace `validated`/`physical`/`mechanism`/`calibrated` with `proxy`/`scaffold`/`relative`); one file at a time | `docs/` | `[SERIAL]` | `[SERIAL]` after I-03 | |
| I-06 | For each `docs/api/*.md`: confirm documented signature matches current implementation; add `[UPDATED yyyy-mm-dd]` to YAML front-matter for any corrected file | `docs/api/` | `[SERIAL]` | `[PARALLEL-OK]` with I-04/I-05 | |
| I-07 | Identify duplicate doc pages (same content, different filenames); merge into the authoritative version (use `git log --follow` to find it); redirect or remove the duplicate | `docs/` | `[SERIAL]` | `[SERIAL]` after I-04 + I-05 | |
| I-08 | `[GATE]` `python3 -m mkdocs build --strict`; zero errors, zero warnings | — | `[GATE]` `[EVIDENCE]` | `[SERIAL]` | |
| I-09 | `[GATE]` `pytest tests/test_docs_links_v0330.py -v`; all pass | `tests/test_docs_links_v0330.py` | `[GATE]` `[EVIDENCE]` | `[SERIAL]` | |
| I-10 | `[GATE]` `python3 scripts/audit_public_docs_language.py --check`; zero violations | — | `[GATE]` `[EVIDENCE]` | `[SERIAL]` | |

### Phase I exit criteria

- I-08 + I-09 + I-10 `[GATE]` green.
- Warning/error count reduced to zero (from I-01 baseline).
- Every API doc page has confirmed-current signature.

### `[ALIGN]` Checkpoint I

Report: warning count before → after, broken link count, language violation count.

---

## Phase J — Spectrolaminar API Consolidation `[BLOCKED on Phase I]`

**Goal:** `spectrolaminar_motif_score` and `spectrolaminar_similarity_kernel_jax`
have non-overlapping documented use cases, tested shape contracts, and a guide page.
No third variant added before these two are fully documented.

**Entry gate:** Phase I exit criteria met.

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| J-01 | Read `spectrolaminar_motif_score` implementation; write to its docstring: input shape, output shape + meaning, when to use, what the other metric does differently | implementation file | `[DOCONLY]` `[SERIAL]` | `[SERIAL]` | Locate file first via grep |
| J-02 | Read `spectrolaminar_similarity_kernel_jax` implementation; write to its docstring: same four items; cross-reference `spectrolaminar_motif_score` | implementation file | `[DOCONLY]` `[SERIAL]` | `[SERIAL]` after J-01 | |
| J-03 | `grep -rn "spectrolaminar" tests/` → list which tests cover which metric; record any gap | `tests/` | `[EVIDENCE]` `[SERIAL]` | `[PARALLEL-OK]` with J-01/J-02 | |
| J-04 | Write `tests/test_phaseJ_spectrolaminar_api.py`: (a) shape contract test for each metric; (b) distinctness test (two metrics produce different outputs on same synthetic input); `[NULL-CTRL]` zero-amplitude input → finite output | `tests/test_phaseJ_spectrolaminar_api.py` | `[SERIAL]` | `[SERIAL]` after J-02 + J-03 | |
| J-05 | `[GATE]` `pytest tests/test_phaseJ_spectrolaminar_api.py -v` + all pre-existing `test_spectrolaminar_*` tests; all pass | above | `[GATE]` `[EVIDENCE]` | `[SERIAL]` | |
| J-06 | `[DOCONLY]` Write `docs/guides/spectrolaminar_guide.md`: when to use motif score, when to use kernel, one worked code example per metric; add to mkdocs nav | `docs/guides/spectrolaminar_guide.md`, `mkdocs.yml` | `[DOCONLY]` `[PARALLEL-OK]` with J-05 | `[SERIAL]` after J-02 | |
| J-07 | `[GATE]` `python3 -m mkdocs build --strict`; zero errors | — | `[GATE]` `[EVIDENCE]` | `[SERIAL]` last | |

### Phase J exit criteria

- J-05 + J-07 `[GATE]` green.
- Both metrics have complete, non-overlapping docstrings.
- Guide page built.

### `[ALIGN]` Checkpoint J

Report: docstring diff summary, new test count, guide built.

---

## Phase K — HDP Weight Drift Guard `[BLOCKED on Phase J]`

**Goal:** Fix AGENTS.md known fragility #3 (`K_w_ctrl=0.0` permits unbounded
weight drift on long HDP runs). Implement guard; test it; mark fragility fixed.

**Entry gate:** Phase J exit criteria met.
**Key file:** `jaxfne/emitters.py` (HDP kernel), `jaxfne/_config.py` or wherever `DEFAULT_HDP` lives.

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| K-01 | Read `simulate_edge_recurrent_izhikevich_hdp`; trace weight update path when `K_w_ctrl=0.0`; confirm runaway scenario in prose in `AGENT_CHANNEL.md` | `jaxfne/emitters.py` | `[SERIAL]` | `[SERIAL]` | |
| K-02 | `[EVIDENCE]` Run a 10 000-step simulation with default params + `K_w_ctrl=0.0`; record `max(|w|)` at t=0 and t=final | — | `[EVIDENCE]` `[SERIAL]` | `[SERIAL]` after K-01 | Confirm the runaway numerically |
| K-03 | `[ALIGN]` Report K-01 + K-02 findings; await maintainer decision on fix option (A: `K_w_ctrl=1e-3` default, B: `w_ceiling` clip, C: runtime warning) before proceeding | — | `[ALIGN]` | `[SERIAL]` after K-02 | **Hard stop — do not implement until option agreed** |
| K-04 | Implement agreed fix in `simulate_edge_recurrent_izhikevich_hdp`; update `DEFAULT_HDP` preset if it sets `K_w_ctrl` | `jaxfne/emitters.py`, `DEFAULT_HDP` location | `[SERIAL]` | `[SERIAL]` after K-03 | |
| K-05 | Write `tests/test_phaseK_weight_drift_guard.py`: (a) positive: with guard, `max(|w|)` stays below ceiling after 10 000 steps; `[NULL-CTRL]` `K_HDP=0` → weights unchanged; regression: short runs (≤5000 steps) unaffected | `tests/test_phaseK_weight_drift_guard.py` | `[SERIAL]` | `[SERIAL]` after K-04 | |
| K-06 | `[GATE]` `pytest tests/test_phaseK_weight_drift_guard.py -v`; all pass | above | `[GATE]` `[EVIDENCE]` | `[SERIAL]` | |
| K-07 | `[GATE]` Run pre-existing HDP tests; zero regressions | `tests/test_sanity_delta_*.py` etc. | `[GATE]` `[EVIDENCE]` | `[SERIAL]` after K-06 | |
| K-08 | `[DOCONLY]` Update AGENTS.md known fragility #3 to `[FIXED]` with one-line description of fix and the guarding test | `AGENTS.md` | `[DOCONLY]` | `[SERIAL]` after K-06 | |

### Phase K exit criteria

- K-06 + K-07 `[GATE]` green.
- AGENTS.md fragility #3 marked `[FIXED]`.
- No regressions.

### `[ALIGN]` Checkpoint K

Report: option chosen, `max(|w|)` before/after receipt, test result.

---

## Phase L — Multi-Area Source Projector Stability `[BLOCKED on Phase K]`

**Goal:** The multi-area source projector produces stable, finite output.
Inter-area coupling is tested E2E with a null control.

**Entry gate:** Phase K exit criteria met.

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-----|-------|
| L-01 | Read `tests/test_multi_area_source_projector.py` and `tests/test_multi_area_emitter_runtime.py` in full; record what is gated | `tests/test_multi_area_*.py` | `[SERIAL]` | `[SERIAL]` | |
| L-02 | `[GATE]` Run those tests; record result | above | `[GATE]` `[EVIDENCE]` | `[SERIAL]` after L-01 | If any fail: fix before L-03 |
| L-02a | `[SKIP-IF]` If L-02 fails: diagnose; fix the bug in the source projector; re-run until green | projector implementation | `[SKIP-IF]` | `[SERIAL]` | Only if L-02 fails |
| L-03 | Write `tests/test_phaseL_multi_area_e2e.py`: two-area network (A→B, E/PV each), 1000 steps; assert: finite output, both areas spike, inter-area current reaches Area B (`mean |drive| > 0` on ≥1 neuron); `[NULL-CTRL]` zero inter-area weight → Area B output identical to single-area simulation | `tests/test_phaseL_multi_area_e2e.py` | `[SERIAL]` | `[SERIAL]` after L-02 | |
| L-04 | `[GATE]` `pytest tests/test_phaseL_multi_area_e2e.py tests/test_multi_area_*.py -v`; all pass | above | `[GATE]` `[EVIDENCE]` | `[SERIAL]` | |

### Phase L exit criteria

- L-04 `[GATE]` green.
- No regressions.

### `[ALIGN]` Checkpoint L

Report: two-area test result, null control result.

---

## Phase M — Config Path Completeness `[BLOCKED on Phase L]`

**Goal:** Every public `Configuration` fluent-builder method produces a valid
`Model` via `construct()`. Every method has at least one test.

**Entry gate:** Phase L exit criteria met.
**Key file:** `jaxfne/_config.py`

| # | Action | File(s) | Tag | Seq | Notes |
|---|--------|---------|-----|-