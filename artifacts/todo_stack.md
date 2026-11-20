# todo stack

### T-AI-DOC — Explain AI-assisted and autonomous data-inspired modeling direction

[A then H] Proposed documentation alignment, not a new scientific feature or release criterion. Writes: `README.md`, existing `docs/reference/project/scope_and_status.md` and canonical modeling/quickstart documentation only. Blocked by: owner ruling if product scope, scientific claims or public acceptance would change.

JaxFNE is the biophysical neural simulation/modeling tool for researcher-directed and AI-assisted/autonomous model construction, simulation, falsifiable comparison with neural data, and iterative discovery. Keep the tested chain explicit: model assumptions and units → simulation → observable spikes/population/fields → data comparison, uncertainty and residual structure → constrained mechanistic revision. State that JaxFNE does not independently validate an empirical hypothesis, that OCMA governs multi-agent research strategy, and that Jomission is a downstream scientific project rather than a built-in package workflow. Do not claim a working autonomous research agent unless evidenced. Check: every public capability maps to current API/docs/tests, no duplicate definition or unsupported promise, original source-field identity and JDNA reduction semantics preserved, native docs/link gates and independent reviewer PASS.


Ordered smallest to largest by effort, end to end. `→` is the next step, `⇒` a branch on the outcome.
`[A]` agent, `[H]` Hamm. Each item names the command that closes it. A `∥` item runs beside the path in its own
lane (at most 3 lanes, one writer per file set, clones at `E:/repos/_wt/jaxfne/<lane>`).
Done items are deleted; history is in git.

Problems: `open_stack.md`; issues: `assurance/issue_log/ISSUE_LOG.md`; rules B1-B8:
`control/harness/evidence_and_workflow_rules.md`; paper goal G-0: `programme/design/plans/arxiv_paper_plan.md`.

Audit 2026-11-12: 0.5.10 checklist holds 40 of 100 (packages 40 of 70, gates 0 of 30).
- 10E PCL Fig 3 is fully scored: E1 R5-approximate scored FAIL under signed R4b
  (`pcl/prereg/P3_fig3_r5_approx_result.md`, `scores/fig3_r5_approx.json`); oracle arm compared (1246 equal,
  1922 differing; `scores/fig3_r5_approx_armcheck.json`); E2 P-055 attributed in `ISSUE_LOG.md`. 10E 10/10.
- PCL attribution closed 2026-11-12 (`pcl/prereg/P3_fig3_r5_attribution_addendum.md`): the complex-cell orientation
  collapse is the `no-local` inhibition setting, not tree drift — a v4-era-tree `no-local` run reproduces the R5
  oracle arm array-for-array over all 3168 files. T-85 closed.
- 10B B4 `build_config` repair (2 pts) done (`scripts/cortical_column_localized_workflow.py`, `tests/test_build_config_declared_layers.py`).
- 10D D4 S16 Destexhe 2009 panel (2 pts) done (`scripts/publication_figures/supplement.py::write_s16`, `tests/test_supplement_generators.py`).
- 10F F4 GSDR log-scale parameters (1 pt) done (`artifacts/etudes/papers/gsdr2026/PORT_SPEC.md` §10.1).
- 10G G3 Reproducibility audit §7 (1 pt) done (`artifacts/programme/paper/reproducibility_audit.md` §7).
- Dashboard R5 row linked under `pcl-fig3` (T-73); `replications.json` and figures fresh.
- B1 drive unification (T-79) and baseline parameter match (T-80) done (`b1_harness.py`, `b1_brian2.py`, `b1_nest.py`, `b1_jaxley.py`; `--smoke` verified).
- `artifacts/state.md` corrected (`artifacts_check.py` ACCEPT).

## Tier 1: blocked on a ruling

- T-FIG05 [H then A] re-freeze equivalence-gate fig05 and its C3 receipt under change control, or render panel C from
  frozen V_m; print stdout in `_run_gate` failures. Closes when: `tests/test_equivalence_gate_v20260815.py` passes on
  darwin (O-FIG05, `assurance/mac_gate_attribution.md`).
- T-NUM [H then A] rule the remedy for the bit-exact XLA assertions and the izh2003 known-answer receipts (record
  platform and jaxlib). Closes when: the five composition and Gaussian-drive tests and `izh2003_core_path` pass on
  the Mac and CI with the ruled assertions (O-NUM).
- T-MATCH [A] run `python scripts/run_test_gate.py broad` unsandboxed on a matching host (Linux, jax 0.10.2) at the
  integrating SHA. Closes when: it reads 0 failed and CI is green on that SHA.
## Tier 2: 1 to 2 hours each

2. T-D3 [A then H] Mejias 2016 G6 — re-tiered 2026-11-12 from "1 to 2 hours" to needs-a-signed-prereg. G6 is not a
   dashboard metric: `SOURCE_P1.md:70` defines it as the paper's full numerics unknown (time constants, weights,
   noise, delays, dt, seeds), and `PREREG_P3.md:1` is still "draft, UNSIGNED" with `(fill)` estimands at `:10` and
   `:26`. Missing: digitized figure targets, a verdict scorer, a rate-route driver, a P4-passing manifest, receipts.
   Closes when: a filled P3 is signed and G6 is scored under it; the inventory is in the lane's records.
5. TASK-UNITS-CSD (T-102) [A then H] CSD definition corrected and the unit map written (plan, Stage 2 gaps,
   2026-11-19). Left: P-066 (z stays a depth fraction while x, y are mm; distance rules mix them), which needs a
   ruling. Closes when: P-066 is repaired with a before/after table of affected rules, reviewed.

## Tier 3: half a day

- TASK-EPHAPTIC (T-98) [A] Ephaptic operator: rename it a distance-weighted lateral voltage coupling, take it out of HDP
  semantics, fix alpha's units and lag per route. Closes when: a test pins the same alpha meaning on LIF and
  Izhikevich.
- TASK-SOURCE-SCOPE (T-99) [A] Source scope: write S's domain as (X, input) and the observation-claim scope of LFP, CSD, EEG/MEG
  from Q; name kernel LFP a separate factorization. Closes when: the scope text is reviewed.
- TASK-HINV (T-100) [A] H invariance: separate continuous-time invariance (barrier, `methods.md` Eq. 7) from discrete
  clamping and from stability. Outside the core claim. Closes when: the text states each with its evidence.
- TASK-DMAP (T-103) [A] Discrete-map semantics: the discrete map defines the system; each claimed rule gets a stated
  consistency order and a dt-refinement test. Closes when: the tests pass for the core rules.
- TASK-N2-PLANT (T-104) [A] N2 extension table: each extension with its same-backend identity test and a planted break that
  fails it. Closes when: every planted break turns its test red.
- TASK-REPL-TFNE (T-105) [A] Route at least two replication lanes through `jaxfne.tfne` and the provenance map, or present the
  record as simulator validation only. Closes when: the plan states which, with the lanes' receipts.
- TASK-RECEIPTS (T-106) [A] Receipt durability: copy or hash-anchor each scored lane's receipts in a tracked location (Brunel's
  are on C:). Closes when: each scored G3 record names a tracked receipt path or sha256 that resolves.

14. T-F5 [A] T-32 candidate batching for AdEx, LIF and `Model.tune` (`_model_tune.py:451`) (checklist F5,
    1.5 pts). Closes when: the batching tests pass across seeds.
17. T-90 [A] P-056 kernel-LFP deep-layer power. Closes when: a settled column shows deep alpha-beta power or the
    cause is named. 2026-11-15 step 1 done: the pilot's estimator, band, layer map, rates and N are not retained
    (only the diff at tag `archive/kernel-lfp`), so the deficit is unreproduced as measured; the chain
    kernel_lfp -> Welch -> 8-25 Hz power is validated on a known source (`tests/test_kernel_lfp_band_power.py`: tone
    power = analytic, x4 for x2 drive, shuffle 0.7%, zero 1e-14; a wrong-frequency expectation would miss by 14-18%).
    Spec draft: `programme/design/capabilities/p056_diagnostic_spec_v1.md` (estimand, controls, sizing; unsigned). Resource smokes done 2026-11-15 (N 1000 reviewed ACCEPT; N 2000 within limits; receipts in spec 5b, 5d): say nothing about P-056. iSTDP-off control (spec 5e, 5g): `hdp_rule_params={"eta": 0.0}` keeps I-to-E weights at 0 and removes the E-rate change under PV or SOM drive that default iSTDP produces (E ratio 0.876 and 0.969, 5 seeds, far outside spread); resource-smoke network only. Independent review of 5d to 5g: ACCEPT 2026-11-15, six minor corrections applied; open: a fixed-weight control at the grown mean separates plasticity from nonzero weights (unrun). Blocked on owner inputs: population map, per-class N, weights, drive, deep-power threshold; no P-056 intervention until set.
    Next: rebuild the column pilot from the AdEx route with its metric fixed first (bands, L2/3 normalization,
    layer map), record per-layer rates and N, then discriminate candidates (rates, deep kernel amplitude,
    coherence radius) by single-variable interventions; keep the observed deficit as the baseline row.
18. T-95 [A] P-058 `_code_sha` worktree portability (`run_r2_reference.py`): resolve HEAD through git. Closes when:
    tests for a regular clone, a worktree and a loose ref pass, and the PCL lane passes in a worktree.

## Tier 4: a day or more

- TASK-MIXFAM [A] Expansion after the core (Hamm 2026-11-19, "start narrowed core, we will expand as we go"):
  mixed-family circuits through one factorization; two families in one tensor are refused as of 2026-11-19
  (`jaxfne/neuronal_tensor.py:2953-2959`). Starts after the core's stage 2 gaps close. Closes when: a two-family
  circuit runs through `field()`/`probe()` with a test, and the plan's headline is re-reviewed.

18. T-87 [A] F4 scored benchmark across N on CPU and the A4000 with recording controls (checklist B1-B3, 10 pts),
    after T-79 and T-80. Closes when: the receipt records all 6 metrics and is evaluated against the purchase gate.
19. T-B5 [A] LIF edge kernel on GPU (checklist B5, 2 pts). Closes when: the GPU receipt equals CPU within the
    declared tolerance.
20. T-C1 [A] Augmentation ensemble study with provenance and the S7 panel (checklist C1, 4 pts). Closes when: S7
    renders from pinned receipts.
21. T-92 [A] 0.5.10 exit gates 1-5 (30 pts): broad gate, CI, replications audit and dashboard fresh on one SHA;
    version and changelog; second review receipt; then Hamm's tag and publish. Closes when:
    `python scripts/run_test_gate.py broad` and `python scripts/check_replications.py --check` exit 0 on one SHA
    and the review receipt is committed.

## 0.5.10 rows still open, carried with their checks

22. T-26 [A] Seed and candidate batching for AdEx, LIF and `Model.tune`; Mejias G6; the LIF edge kernel on GPU;
    F2, F4, S16 per the release plan. Closes when: each lands with its test or receipt.
23. T-R28 [A] Hamm's 2026-10-28 rulings left: dashboard P1 then P2 (ruling text not in the repo,
    `execution_plans_0.5.9-0.5.10.md:72`); the jitter reader on larger sets; C4's arbitrary-angle S3 panel; C6
    `geometry="laminar"` with Brian2 and Jaxley baselines. Closes when: each has a commit with its test or receipt.
24. T-31 [A] AGSDR log-scale bounds (landed 733b3960) used in the GSDR reproduction; a new fit pilot needs a new
    prereg. Closes when: the GSDR lane names its log-scale parameters.
25. T-32 [A] 0.5.11 lanes picked 2026-11-08 (O-87; `etudes/papers/{lee2025,mazzoni2015}/NOTES.md`): Lee 2025 Fig 6
    and Mazzoni 2015 Fig 5 RWS proxy. Closes when: each has a signed P3 prereg.
26. T-54 [A then H] awesome-jax PR after 0.5.10 and 7 replicated papers. Closes when: the PR is open with Hamm's go.
27. T-11 CAMINOS NWB excerpt for jnwb `tests/`, closes when `git -C D:/repos/jnwb grep -n caminos -- tests/`
    returns its test; T-RG komposer `run_guard.py stop` cwd fallback, closes when that test passes from a foreign
    cwd.
27. T-94 [A] Ephaptic coupling (Hamm 2026-11-17 spec; placement ruled: membrane via `drive_bias`): rule
    `ephaptic_field` and `ephaptic_kernel` (`jaxfne/hdp_rules_ephaptic.py`, `tests/test_hdp_rules_ephaptic.py`),
    outside review ACCEPT 2026-11-17, on dev since 32a81d86 (broad gate 0 failed in a regular clone). Left: the
    exogenous-field contrast (imposed field through the drive schedule vs the rule, same kernel); the poster's
    hypotheses (shared variability, decoding) as a signed prereg. P-057 closed 2026-11-18 (backend mismatch; P-062).
    Closes when: the contrast test passes.

## Owner queue (asked one at a time, in this order)

28. [H] OCMA round 4: re-rule the rater with no reachable second model family on this host. Cause observed
    2026-11-12: Group Policy sets `ExecutionPolicy=AllSigned` so `cursor-agent.ps1` cannot run; the direct
    `node.exe index.js` launch gets past that but Cursor refuses `--sandbox` on Windows; agy produced no output.
    Options: a macOS/Linux host with Cursor; opencode Muse as the second family; defer the round.
29. [H] T-72 NeuroAI 2027: the anonymous code mirror, then your read and submit. Deadline 2026-11-20 23:59 AoE.
30. [A] T-25 retired names in dated records: history allowlist (asked in jchat, jynx keeper).

## Beside the path (∥, at most 3 lanes)

31. ∥ [A] T-8 standing: each new score updates `dashboard/sources.json`, regenerates, pushes dev.
32. ∥ [A] T-10b jnwb proxy-claim wording gate or a dated drop; T-42 CI plan steps at the first rc; T-49 closed lane
    folders by kind.
33. ∥ [A] T-O68 remaining dirty worktrees (`base053`, `bisw1`, `frozen-etude`, `w3`, `jaxfne-wt-p022`,
    `jaxfne-wt-suites`).

## Crazy Sailor M2 (merged to dev 2026-11-20; contract §12)

Scope and decisions: `programme/design/plans/crazy_sailor/m0_contract.md` §12.9. Ordered by effort.

- CS-1 [A] Test hardening: a rollout test with nonzero y-channel counts (the y decode path is covered only by the unit test). Closes when: it passes on HEAD and fails under a y-sign mutant.
- CS-2 [A] Test hardening: the chunked-vs-continuous test asserts that spikes occur in its fixture. Closes when: the assertion holds on HEAD and the test still fails under the dropped-continuation mutant.
- CS-3 [A] Generic `controller.init/step` adapter (§12.3.1). Closes when: a test drives the M2 controller through the generic interface and reproduces the rollout trajectory on the fixture.
- CS-4 [A] 60 s closed-loop episode (3000 control intervals) with spikes, motor commands, positions and distance recorded. Closes when: one reproducible run records SHA, config and seed, with 3000 intervals.
- CS-5 [A] Dashboard page "The Crazy Sailor" from the CS-4 run. Closes when: it renders from that run's data with provenance. Publishing needs Hamm's authorization.
- CS-6 [A] GIFs of the CS-4 trajectory. Starts after CS-4 and its data pipeline pass. Closes when: the GIFs render from that run.
- CS-7 [A] Spike-level feedback validation. Measured by the exploration lanes (EX-1, EX-2; contract §13.1 L2–L3). Closes when: a declared regime shows a reproducible nonzero change in motor counts with distance, or the null is recorded.
- CS-8 [A] Goal-directed navigation. Measured by EX-4 and EX-5 (§13.1 L3). Closes when: the decoded force points toward `p*` over the declared episode set, against the F0 and F5 controls.
- CS-9 [H] Empirical calibration of `b_i`, `g_i`, `d0`, the E/I weights, and fixed vs learned weights (§12.6). Hamm supplies the values with sources; the agent does not choose them. Closes when: the values are recorded in the contract and the M2 tests are re-frozen.
- CS-10 [A] Repository broad gate: matching-host broad gate (`T-MATCH`) on dev without weakening the gate. Closes when: `python scripts/run_test_gate.py broad` exits 0 on one SHA.

## Crazy Sailor exploration (contract §13; exploratory, not confirmatory)

Goal and checklist: `programme/design/plans/crazy_sailor/m0_contract.md` §13.1, §13.5. Lanes run in parallel, at most three.

Done (results in contract §13.8 registry): Lane A (F2, F2b, F4 vs F1, not F0), Lane B (F3, F5), Lane C (F0 zero-encoder, F1, F6); each re-derived from its JSON.

- EX-15 [A] Seeds are not replicates: the M2 fixture has no stochastic input, so every seed gives one trajectory. Add seeded start jitter (§13.7). Closes when: EX-12 passes and the held-out set is generated. Blocks: every CI in §13.
- EX-20 [H] Primary metric: the §13.4 tail mean is K-dependent for F1 (K = 600: 526; K = 3000: 4114) because F1 runs away at constant speed after the transient (R-10). Propose a metric that does not depend on K (for example time to radius r1, distance at a fixed time for each K, or drift rate), then freeze it. Blocks: EX-12.
- EX-21 [H] Stage-1 working title and provisional abstract ("Fixation Boat", 2026-11-19) are not frozen. Mismatches with the implemented model: "biophysically detailed" (the circuit is 10 LIF neurons, contract §12.3.4, `paradigm.py` emitter "lif"); reward "when the boat enters a predefined fixation region" (implemented reward is dense −Δe, §13.3 F3; no fixation region exists); "fixation" is not a contract term (the goal is drive to p*, §13.1); "HDP-dependent STDP" (F3 uses the HDP eligibility rule, `jaxfne/hdp_rule.py`; "STDP" not checked). Closes when: Hamm picks wording that matches the implemented model, or approves a new goal and scope, recorded in `goal_stack.md` or the manuscript plan. Blocks: manuscript Stage 2.
- EX-22 [A] Revise §13.7 for the critic round (Agent C, VERDICT REVISE, 9 findings; critique file blocked by a hook, text in the handback). Fix: (1) pin the window: pre-step `errors[480:600]` gives F1 = 526.32, post-step positions 481..600 give 527.86, same trajectory; (2) gate each contrast Δ_j, not per-strategy D_j; exempt F0; add an L2 weight-change gate before any CI; (3) name the baseline per contrast (F1 for H-on-F1, F3 at η = 0, F5 for the shuffle); (4) F1 falsifier names its metric (tail D_j ≤ R: 0 of 3 so far; transient min_k ≥ 1 < R: 2 of 3, R-11); (5) F4 ≡ F2 by construction, report as duplicate, and fix its description (scales input amplitude); F2b missing from §13.3 and §13.6; (6) shuffle permutation seed per start; (7) mark R-03/R-04/R-06 CIs "no CI" (n_eff = 1); (8) pin the dev-start draw call (`default_rng(2025).uniform(0, 2π, 20)`, used in R-12). Closes when: §13.7a records each finding resolved or rejected with a reason, and a fresh reviewer (not the writer) re-reads it.
- EX-24 [A] Directional positive control on the 10-neuron fixture with the Option A encoder (owner, 2026-11-19): a known-correct controller must bring the boat to p* from all 20 dev starts. Replaces EX-9 for the critical path. Closes when: a JSON with per-start final distance and fixation occupancy, re-derived from source. Blocked by: none.
- EX-25 [A] Build the 100-neuron LIF circuit spec (E/I 80/20, fixed-seed sparse recurrence, four motor groups, Option A encoder) and run zero-drive and stationary controls. Closes when: a spec file and control JSON exist, and the controls match the expected values. Blocked by: none.
  Progress 2026-11-19: spec `code/dev100_model.py`; R1, R3, R5 PASS; R2, R4 FAIL (O-8); P1 PASS 20/20, N1 0/20, T05 20/20 (`dev100_gate_record.md`). Files moved to jomission branch `_temp_plasticity_boat` (pushed at 280ebd0; the 13 files are under `_temp_plasticity_boat/artifacts/programme/`); the next step is the jomission todo `PB-DEV100-V2-VALIDATION` (BLOCKED on O-8 and the JaxFNE pin). Open: O-8 (criteria v2 or drop), O-9 (decoder sum vs mean). Not closed: the controls do not match the frozen expected values.
- EX-26 [H] Freeze the eight-strategy list and the metric (jomission `_temp_plasticity_boat/artifacts/programme/design/plans/plasticity_boat/protocol_v0.md` §3 to §5; resolves EX-20). Blocks: all confirmatory runs. Owner decision: confirm S1 to S8, or name changes.
- EX-27 [H] Supply the owner's 15-stage manuscript workflow. It is not in this repo or in jarness; stage numbers in jomission `_temp_plasticity_boat/artifacts/programme/manuscript/plasticity_boat/main.md` are provisional until it is supplied.
- EX-28 [A] Move every figure-source JSON from `_temp/` to a tracked path before citation. Closes when: `git ls-files` lists each source and each figure's legend cites it.
- EX-29 [H] Permission to download the AAAI-27 Author Kit (https://aaai.org/authorkit27/; the file size and name are not yet known). Needed for the reproducibility checklist text and the style file for the build gate.
- EX-31 [A] Plasticity Boat checks that run in jomission `_temp_plasticity_boat`, carried here so they are not lost: the 33-file tier-0 suite under `suite_lock` (not run; the sandbox denies the shared `suite.lock`), and the `release_artifacts` gate (not examined; dist absent). Closes when: each run's summary line is read and recorded in jomission `HANDOFF.md`, or the handoff marks it not run with the reason. Blocked by: none.
Note: EX-17 is now the F6 re-run; EX-18 closes after the last 8 dev starts (R-12) and the held-out run is not part of it.
- EX-17 [A] F6 rerun storing d[600] and the κ = 0 equivalence. Closes when: F6 mean is recomputed under the pinned window (errors[480:600]).
- EX-18 [A] F1 at θ = π and 3π/4, K = 600 (item 6 falsifier, 40 starts planned). Closes when: counts of starts with D_j(F1) ≤ R are reported.
- EX-19 [A] Decoder sweep over e in [0, 5] to check the "+x at every distance" claim (critic finding 7). Closes when: the crossover e is reported from the tonic and baseline values.
- EX-9 [A] Labelled directional-encoder scenario S3-dir (separate from the fixture; gain on the +x/−x channel by sign of p_x). K=200, starts ±5. Closes when: a JSON with per-start final distance and a corrective-direction count. Blocked by: none; explicitly not the M2 fixture.
- EX-10 [A] Labelled symmetric-gain scenario S1-sym (g[1] = 24). Prediction: zero net force, boat stays. Closes when: a 40-interval probe confirms or refutes it.
- EX-11 [H] Hamm decision on the fixture defect (contract §13.7): keep the fixture, or approve a labelled separate scenario. Blocks: confirmatory runs, §13 conclusions, integration.
- EX-12 [H] Hamm approval of §13.7 gates (after EX-8) before any held-out run. Blocks: the held-out confirmatory set (N = 40).
- EX-13 [A] Lane A F0 vs F1 K=600 discrepancy (527.86 vs 526.32): find the cause or record it unresolved with the two paths. Closes when: a 40-interval run of both code paths is compared bit-wise and the K=600 cause is named or marked unknown.
- EX-14 [A] Scale the best one or two strategies to S2. Deferred until EX-11 and EX-12 pass. Closes when: n ≥ 20 start-level results per cell.

## Deferred, each with its trigger

P-053 follow-up, trigger: a new hypothesis or coverage requirement. T-83 (2026-11-15, `pcl/port/P053_sparse_fixture_result.md`)
is NOT_REPRODUCED under criteria Amendment 1 only; V1's LAT half is never consumed and V1 adds information beyond V3
in seed 0 through TD, so no general sparse-weight equivalence is shown. Source-only discriminators (event order,
repeated addresses, ties, zero vs epsilon weights) are in `artifacts/assurance/issue_log/p053_p056_source_audit_2026-11-13.md`. AUD-02 reproduced 2026-11-15: a pinned receipt that is corrupted, status-flipped or deleted is flagged; a new
unparseable or status-less receipt is not a record by design (`_is_final_receipt`); no bypass, pinned by
`tests/test_record_integrity.py::test_corrupt_receipt_unpinned_is_not_a_record_but_pinned_is_caught`.

File-count budget (2026-11-20): `budget_files` raised 2798 -> 2813 (+15 exact, 0 slack: +14 from branch merges + 1 for preserved `scripts/p056_resource_smoke.py`; `budget_bytes` unchanged at 96.4 MB) under Hamm's 2026-11-20 branch consolidation authorizations. Trigger: the next needed new file. Action then: find removable files with evidence (match-cleanup), or ask Hamm for a `budget_files` ruling; never raise it unasked.

AT-10-R6 RSS growth before any run over 100 s; A8 gh-pages policy; architecture split on a measured bottleneck;
C5-C7 zero/NaN contract; the UNTESTED-exact refusal tail and O-67 skips; S27 population family; agent transport
(MCP); 0.5.4 items 1b and 0; the 2026-09-20 owner calls; Atlas ladder; the mechanism rule proposal; stored-only
fields when a caller needs one; the tracked `w3b_scan_run.log` at the next W3b refreeze.
Directions, open: augmentation runs; `pcl_stdp_h2`; structural HDP; primate primary-connection mode; theory F3-F5
and Lean proofs; Python 3.15 in CI once its wheels exist.

## Long-term quality and OCMA programme (planning; no new release authorization)

The unified scope, evidence gates, thinning protocol and proposed milestones 0.5.12–0.5.15 live only in `artifacts/programme/design/plans/ocma/ocma_jaxfne_0.5.10-0.5.15.md`. Keep the ordered 0.5.10 → 0.5.11 → 0.6.0 execution path above until Hamm approves a changed release sequence. At the 0.5.11 planning boundary, translate the proposed next milestone into smallest actionable rows with explicit checks; do not duplicate the roadmap here or treat planning as a signed protocol.

## OCMA cross-project assessment routing (planning only)

Dated scorecard: `artifacts/programme/design/plans/ocma/ocma_jaxfne_goal_match_2026-11-12.md` (JaxFNE 52/100, OCMA 55/100 provisional, not release scores). The top executable JaxFNE task remains T-3 PCL; B1 drive/parameter reconciliation precedes scored benchmarks. OCMA owns its local HOLD and paused R4-1; no OCMA file or gate is changed by this JaxFNE planning note. On completion of 0.5.10, use the unified plan to propose the next smallest checked todo rows; do not alter existing release acceptance or owner goal/fact stacks.
