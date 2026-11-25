# todo stack

### T-AI-DOC — Explain AI-assisted and autonomous data-inspired modeling direction

[A then H] Proposed documentation alignment, not a new scientific feature or release criterion. Writes: `README.md`, existing `docs/reference/project/scope_and_status.md` and canonical modeling/quickstart documentation only. Blocked by: owner ruling if product scope, scientific claims or public acceptance would change.

2026-11-23: governance audit done (`docs/reference/project/for_ai_agents.md` removed; changelog and the RBS/RBD inventory shown to carry no governance; guard `tests/test_agent_context_hygiene.py::test_public_docs_carry_no_agent_governance`). Left: the OCMA reference-audit section and the direction text below. Blocked by: O-AIDOC.

JaxFNE is the biophysical neural simulation/modeling tool for researcher-directed and AI-assisted/autonomous model construction, simulation, falsifiable comparison with neural data, and iterative discovery. Keep the tested chain explicit: model assumptions and units → simulation → observable spikes/population/fields → data comparison, uncertainty and residual structure → constrained mechanistic revision. State that JaxFNE does not independently validate an empirical hypothesis and that Jomission is a downstream scientific project rather than a built-in package workflow. Do not claim a working autonomous research agent. Check: every public capability maps to current API/docs/tests, no duplicate definition or unsupported promise, original source-field identity and JDNA reduction semantics preserved, native docs/link gates and independent reviewer PASS.


Ordered smallest to largest by effort, end to end. `→` is the next step, `⇒` a branch on the outcome.
`[A]` agent, `[H]` Hamm. Each item names the command that closes it. A `∥` item runs beside the path in its own
lane (at most 3 lanes, one writer per file set, clones at `E:/repos/_wt/jaxfne/<lane>`).
Done items are deleted; history is in git.

Problems: `open_stack.md`; issues: `assurance/issue_log/ISSUE_LOG.md`; rules B1-B8:
`control/harness/evidence_and_workflow_rules.md`; paper goal G-0: `programme/design/plans/arxiv_paper_plan.md`.

## Tier 0: Solmind 0.5.10 execution (Hamm's go 2026-11-21, S19)

0. T-P069 [A] P-069, left: the Izhikevich case (`enable_hdp=False` with `hdp_params`) refuses only if Hamm agrees
   that given params without the flag are an error (O-P069); the AdEx seed batch and homeostatic E/I refuse since
   2026-11-23. Closes when: `test_izhikevich_refuses_hdp_params_without_enable_hdp` is a plain passing test, or is
   removed on Hamm's other option with the inert behavior documented. Blocked by: O-P069.

0. T-R49-FIX [A] Repair the caption findings of `artifacts/programme/design/plans/solmind/caption_audit_r49_2026-11-23.md` (text only;
   generated captions through `scripts/generate_gallery.py`). Then the visual half of R49 (print-size render).
   Closes when: every listed page:line is fixed or answered in the audit's follow-up, the docs gate and
   `mkdocs build --strict` pass, and an outside reviewer accepts. Blocked by: none.
   2026-11-24: text half closed (20ef57cf, e1a1cbe5; outside reviews ACCEPT). Rows with no repo source (EPSP,
   V1/V4/PFC expansions, construct 03 "mm", MVC#1 placement, `tutorials/index.md` cards) are listed in the audit's
   "Follow-up 2026-11-24" for Hamm. Left: the visual half (print-size render of the 78 figures).

0. task-gpunondet [A] GPU run-to-run nondeterminism: on vlix (RTX A4000, jax 0.10.2) two `simulate` calls of one
   model differ in `sources` by up to 4.8e-7 in 1-2 of 1200 entries, in some processes only (R46 worker and reviewer,
   2026-11-24, fixture `tests/test_r46_h_inert_without_hdp.py`); CPU is bitwise. Check whether the same cause makes
   `tests/test_paradigm.py::test_m2_chunked_rollout_equals_one_continuous_simulate` (exact V_m equality) fail on GPU
   (2 of 11 runs paired with a second test, 0 of 12 alone). Siblings: exact-equality tests on GPU outputs. Also:
   `jaxfne/_runtime_config.py:116` comment says all gains default 0 while `K_HDP` defaults 1.0; test docstrings
   `test_probe_report_contract_v0212.py:3`, `test_probe_operators_eeg_meg_emm.py:262` still name `ProbeReport`.
   Closes when: the mechanism is named (op and flag) or the flake is pinned to a reproducing command, and each
   exact-equality GPU test either runs deterministic or states its tolerance with the measured bound.

1. T-PLAN-EXECUTE [A] Work the tasks of `artifacts/programme/design/plans/solmind/solmind_0510_plan.md` (R01-R72,
   gates S01-S23; reviewed, audit 3 ACCEPT) at most 3 lanes including reviewers; a row runs only when its own gates
   are clear; recompute the frozen checklist only from exact-SHA receipts; stop at READY_FOR_OWNER_GO for Gate 5.
   R01, R03, R70, R72 done 2026-11-21; R02 and R11 done 2026-11-22 (O-7 ruled, plan amendment 2026-11-22; O-77 closed); R10 done 2026-11-22 (O-76 and O-NUM closed). Next: rows whose gates clear (plan §8 status names each
   hold). Closes when: prepublication readiness reported on one candidate SHA with Gates 1-4 green, issue-log Open
   empty and the release todo section drained (R51-R54). Blocked by: R04 by S21, R17 by S23, R57 by P-047; other
   rows by their S gate.

## Tier 1: blocked on a ruling

- T-FIG05 [A] re-freeze equivalence-gate fig05 and its C3 receipt under change control (Hamm 2026-11-23, plan §10.3
  item 10); print stdout in `_run_gate` failures. The freeze platform is macOS with matplotlib 3.10.9
  (`equivalence_gate.py` `byte_identity_pinned`), so a render from Windows or Linux cannot be the freeze. Closes when:
  `tests/test_equivalence_gate_v20260815.py` passes on darwin (O-FIG05, `assurance/mac_gate_attribution.md`).
  Blocked by: Mac access (T-MAC; `ssh vmac` timed out 2026-11-23).
- T-MAC [nonblocking; O-NUM closed 2026-11-22] macOS validation. Run `scripts/internal/onum_measure.py`, the
  five darwin-skipped tests (forced to run) and `tests/test_adex_route.py` / `tests/test_lif_tensor_route.py` on a
  Mac, and record per-test maxima with the environment and SHA in `o_num_ruling_packet.md`. Until then macOS stays
  UNKNOWN in verified coverage. Closes when: that receipt exists. Blocked by: Mac access (SSH to vmac refused).
- T-MATCH [A] run `python scripts/run_test_gate.py broad` unsandboxed on a matching host (Linux, jax 0.10.2) at the
  integrating SHA. Closes when: it reads 0 failed and CI is green on that SHA.
## Tier 2: 1 to 2 hours each

2. T-D3 [A then H] (moved to 0.5.11, Hamm 2026-11-23) Mejias 2016 G6 — re-tiered 2026-11-12 from "1 to 2 hours" to needs-a-signed-prereg. G6 is not a
   dashboard metric: `SOURCE_P1.md:70` defines it as the paper's full numerics unknown (time constants, weights,
   noise, delays, dt, seeds), and `PREREG_P3.md:1` is still "draft, UNSIGNED" with `(fill)` estimands at `:10` and
   `:26`. Missing: digitized figure targets, a verdict scorer, a rate-route driver, a P4-passing manifest, receipts.
   Closes when: a filled P3 is signed and G6 is scored under it; the inventory is in the lane's records.

## Tier 3: half a day

- task-ephaptic (T-98) [A] Ephaptic operator: rename it a distance-weighted lateral voltage coupling, take it out of HDP
  semantics, fix alpha's units and lag per route. Closes when: a test pins the same alpha meaning on LIF and
  Izhikevich.
- task-source-scope (T-99) [A] Source scope: write S's domain as (X, input) and the observation-claim scope of LFP, CSD, EEG/MEG
  from Q; name kernel LFP a separate factorization. Closes when: the scope text is reviewed.
- task-hinv (T-100) [A] H invariance: separate continuous-time invariance (barrier, `methods.md` Eq. 7) from discrete
  clamping and from stability. Outside the core claim. Closes when: the text states each with its evidence.
- task-dmap (T-103) [A] Discrete-map semantics: the discrete map defines the system; each claimed rule gets a stated
  consistency order and a dt-refinement test. Closes when: the tests pass for the core rules.
- task-n2-plant (T-104) [A] N2 extension table: each extension with its same-backend identity test and a planted break that
  fails it. Closes when: every planted break turns its test red.
- task-repl-tfne (T-105) [A] The replication record is scoped as simulator validation until at least two replication
  lanes run through `jaxfne.tfne` and the provenance map with verified receipts (2026-11-21). Closes when: two lanes'
  `jaxfne.tfne` receipts are verified and the plan's scope line is updated, or the plan states simulator validation
  as final.
- task-receipts (T-106) [A] Receipt durability: copy or hash-anchor each scored lane's receipts in a tracked location (Brunel's
  are on C:). Closes when: each scored G3 record names a tracked receipt path or sha256 that resolves.

14. T-F5 [A] (`Model.tune` half moved to 0.5.11, Hamm 2026-11-23) Candidate batching for AdEx, LIF and `Model.tune` (checklist F5, 1.5 pts). Candidates stay serial
    (`CANDIDATE_BATCH_STATUS`, `jaxfne/_model_tune.py:150`) until a test proves batched results equal serial ones,
    including order dependence and reproducibility across seeds. Closes when: that equivalence test passes.
17. T-90 [A] P-056 kernel-LFP deep-layer power. Closes when: a settled column shows deep alpha-beta power or the
    cause is named. 2026-11-15 step 1 done: the pilot's estimator, band, layer map, rates and N are not retained
    (only the diff at tag `archive/kernel-lfp`), so the deficit is unreproduced as measured; the chain
    kernel_lfp -> Welch -> 8-25 Hz power is validated on a known source (`tests/test_kernel_lfp_band_power.py`: tone
    power = analytic, x4 for x2 drive, shuffle 0.7%, zero 1e-14; a wrong-frequency expectation would miss by 14-18%).
    Spec draft: `programme/design/capabilities/p056_diagnostic_spec_v1.md` (estimand, controls, sizing; unsigned). Resource smokes done 2026-11-15 (N 1000 reviewed ACCEPT; N 2000 within limits; receipts in spec 5b, 5d): say nothing about P-056. iSTDP-off control (spec 5e, 5g): `hdp_rule_params={"eta": 0.0}` keeps I-to-E weights at 0 and removes the E-rate change under PV or SOM drive that default iSTDP produces (E ratio 0.876 and 0.969, 5 seeds, far outside spread); resource-smoke network only. Independent review of 5d to 5g: ACCEPT 2026-11-15, six minor corrections applied; open: a fixed-weight control at the grown mean separates plasticity from nonzero weights (unrun). Blocked on owner inputs: population map, per-class N, weights, drive, deep-power threshold; no P-056 intervention until set.
    Next: rebuild the column pilot from the AdEx route with its metric fixed first (bands, L2/3 normalization,
    layer map), record per-layer rates and N, then discriminate candidates (rates, deep kernel amplitude,
    coherence radius) by single-variable interventions; keep the observed deficit as the baseline row.

## Tier 4: a day or more

- task-mixfam [A] Expansion after the core (Hamm 2026-11-19, "start narrowed core, we will expand as we go"):
  mixed-family circuits through one factorization; two families in one tensor are refused since c6ee232e (2026-10-13;
  `jaxfne/neuronal_tensor.py:2953-2959`). Starts after the core's stage 2 gaps close. Closes when: a two-family
  circuit runs through `field()`/`probe()` with a test, and the plan's headline is re-reviewed.

18. T-87 [A] F4 scored benchmark across N on CPU and the A4000 with recording controls (checklist B1-B3, 10 pts).
    Closes when: the receipt records all 6 metrics and is evaluated against the purchase gate.
20. T-C1 [A] (moved to 0.5.11, Hamm 2026-11-23) Augmentation ensemble study with provenance and the Fig. S7 panel (checklist C1, 4 pts). Closes when:
    Fig. S7 renders from pinned receipts. Blocked by: S17.
21. T-92 [A] 0.5.10 exit gates 1-5 (30 pts): broad gate, CI, replications audit and dashboard fresh on one SHA;
    version and changelog; second review receipt; then Hamm's tag and publish. Closes when:
    `python scripts/run_test_gate.py broad` and `python scripts/check_replications.py --check` exit 0 on one SHA
    and the review receipt is committed.

## 0.5.10 rows still open, carried with their checks

23. T-R28 [A] Hamm's 2026-10-28 rulings left: dashboard P1 then P2 (ruling text not in the repo,
    `programme/design/plans/solmind/solmind_0510_plan.md:315`); the jitter reader on larger sets; C4's arbitrary-angle S3 panel; C6
    `geometry="laminar"` with Brian2 and Jaxley baselines. Closes when: each has a commit with its test or receipt.
25. T-32 [A] 0.5.11 lanes picked 2026-11-08 (O-87; `etudes/papers/{lee2025,mazzoni2015}/NOTES.md`): Lee 2025 Fig 6
    and Mazzoni 2015 Fig 5 RWS proxy. Closes when: each has a signed P3 prereg.
26. T-54 [A then H] awesome-jax PR after 0.5.10 and 7 replicated papers. Closes when: the PR is open with Hamm's go.
27. T-11 [A] CAMINOS NWB excerpt for jnwb `tests/`. Closes when: `git -C D:/repos/jnwb grep -n caminos -- tests/`
    returns its test.
    T-RG [A] komposer `run_guard.py stop` cwd fallback. Closes when: that test passes from a foreign cwd.
27. T-94 [A] Ephaptic coupling (Hamm 2026-11-17 spec; placement ruled: membrane via `drive_bias`): rule
    `ephaptic_field` and `ephaptic_kernel` (`jaxfne/hdp_rules_ephaptic.py`, `tests/test_hdp_rules_ephaptic.py`),
    outside review ACCEPT 2026-11-17, on dev since c25182ea (broad gate 0 failed in a regular clone). Left: the
    exogenous-field contrast (imposed field through the drive schedule vs the rule, same kernel); the poster's
    hypotheses (shared variability, decoding) as a signed prereg. P-057 closed 2026-11-18 (backend mismatch; P-062).
    Closes when: the contrast test passes (plan R19).

## Owner queue (asked one at a time, in this order)

28. [H] OCMA round 4: re-rule the rater with no reachable second model family on this host. Cause observed
    2026-11-12: Group Policy sets `ExecutionPolicy=AllSigned` so `cursor-agent.ps1` cannot run; the direct
    `node.exe index.js` launch gets past that but Cursor refuses `--sandbox` on Windows; agy produced no output.
    Options: a macOS/Linux host with Cursor; opencode Muse as the second family; defer the round.
29. [H] T-72 NeuroAI 2027: deadline 2026-11-20 23:59 AoE lapsed; the repo holds no submission receipt (last
    record a1d12c5b, 2026-11-02: draft built, three owner decisions queued). Closes when: Hamm confirms lapsed or
    supplies the receipt.
30. [A] T-25 retired names in dated records: history allowlist (asked in jchat, jynx keeper).

## Beside the path (∥, at most 3 lanes)

31. ∥ [A] T-8 standing: each new score updates `dashboard/sources.json`, regenerates, pushes dev.
32. ∥ [A] T-10b jnwb proxy-claim wording gate or a dated drop; T-42 CI plan steps at the first rc; T-49 closed lane
    folders by kind.

## Crazy Sailor M2 (merged to dev 2026-11-20; contract §12)

Scope and decisions: `programme/design/plans/crazy_sailor/m0_contract.md` §12.9. Ordered by effort.

- CS-3 [A] Generic `controller.init/step` adapter (§12.3.1). Closes when: a test drives the M2 controller through the generic interface and reproduces the rollout trajectory on the fixture.
- CS-4 [A] 60 s closed-loop episode (3000 control intervals) with spikes, motor commands, positions and distance recorded. Closes when: one reproducible run records SHA, config and seed, with 3000 intervals.
- CS-5 [A] Dashboard page "The Crazy Sailor" from the CS-4 run. Closes when: it renders from that run's data with provenance. Publishing needs Hamm's authorization.
- CS-6 [A] GIFs of the CS-4 trajectory. Closes when: the GIFs render from that run. Blocked by: CS-4.
- CS-7 [A] Spike-level feedback validation. Measured by contract §13.1 L2–L3. Closes when: a declared regime shows a reproducible nonzero change in motor counts with distance, or the null is recorded.
- CS-8 [A] Goal-directed navigation. Measured by §13.1 L3 on the held-out set, which waits on EX-12. Closes when: the decoded force points toward `p*` over the declared episode set, against the F0 and F5 controls.
- CS-9 [H] Empirical calibration of `b_i`, `g_i`, `d0`, the E/I weights, and fixed vs learned weights (§12.6). Hamm supplies the values with sources; the agent does not choose them. Closes when: the values are recorded in the contract and the M2 tests are re-frozen.

## Crazy Sailor exploration (contract §13; exploratory, not confirmatory)

Goal and checklist: `programme/design/plans/crazy_sailor/m0_contract.md` §13.1, §13.5. Lanes run in parallel, at most three.

Done (results in contract §13.8 registry): Lane A (F2, F2b, F4 vs F1, not F0), Lane B (F3, F5), Lane C (F0 zero-encoder, F1, F6); each re-derived from its JSON.

- EX-15 [A] Seeds are not replicates: the M2 fixture has no stochastic input, so every seed gives one trajectory. Add seeded start jitter (§13.7). Closes when: EX-12 passes and the held-out set is generated. Blocks: every CI in §13.
- EX-20 [H] Primary metric: the §13.4 tail mean is K-dependent for F1 (K = 600: 526; K = 3000: 4114) because F1 runs away at constant speed after the transient (R-10). Propose a metric that does not depend on K (for example time to radius r1, distance at a fixed time for each K, or drift rate), then freeze it. Closes when: Hamm's metric is frozen in contract §13.4 (O-4). Blocks: EX-12.
- EX-21 [H] Stage-1 working title and provisional abstract ("Fixation Boat", 2026-11-19) are not frozen. Mismatches with the implemented model: "biophysically detailed" (the circuit is 10 LIF neurons, contract §12.3.4, `paradigm.py` emitter "lif"); reward "when the boat enters a predefined fixation region" (implemented reward is dense −Δe, §13.3 F3; no fixation region exists); "fixation" is not a contract term (the goal is drive to p*, §13.1); "HDP-dependent STDP" (F3 uses the HDP eligibility rule, `jaxfne/hdp_rule.py`; "STDP" not checked). Closes when: Hamm picks wording that matches the implemented model, or approves a new goal and scope, recorded in `goal_stack.md` or the manuscript plan. Blocks: manuscript Stage 2.
- EX-22 [A] Revise §13.7 for the critic round (Agent C, VERDICT REVISE, 9 findings; critique file blocked by a hook, text in the handback). Fix: (1) pin the window: pre-step `errors[480:600]` gives F1 = 526.32, post-step positions 481..600 give 527.86, same trajectory; (2) gate each contrast Δ_j, not per-strategy D_j; exempt F0; add an L2 weight-change gate before any CI; (3) name the baseline per contrast (F1 for H-on-F1, F3 at η = 0, F5 for the shuffle); (4) F1 falsifier names its metric (tail D_j ≤ R: 0 of 3 so far; transient min_k ≥ 1 < R: 2 of 3, R-11); (5) F4 ≡ F2 by construction, report as duplicate, and fix its description (scales input amplitude); F2b missing from §13.3 and §13.6; (6) shuffle permutation seed per start; (7) mark R-03/R-04/R-06 CIs "no CI" (n_eff = 1); (8) pin the dev-start draw call (`default_rng(2025).uniform(0, 2π, 20)`, used in R-12). Closes when: §13.7a records each finding resolved or rejected with a reason, and a fresh reviewer (not the writer) re-reads it.
- EX-24 [A] Directional positive control on the 10-neuron fixture with the Option A encoder (owner, 2026-11-19): a known-correct controller must bring the boat to p* from all 20 dev starts. Replaces EX-9 for the critical path. Closes when: a JSON with per-start final distance and fixation occupancy, re-derived from source. Blocked by: none.
- EX-25 [A] Build the 100-neuron LIF circuit spec (E/I 80/20, fixed-seed sparse recurrence, four motor groups, Option A encoder) and run zero-drive and stationary controls. Closes when: a spec file and control JSON exist, and the controls match the expected values. Waits on Hamm's O-8 and O-9 (`open_stack.md`).
  Progress 2026-11-19: spec `code/dev100_model.py`; R1, R3, R5 PASS; R2, R4 FAIL (O-8); P1 PASS 20/20, N1 0/20, T05 20/20 (`dev100_gate_record.md`). Files moved to jomission branch `_temp_plasticity_boat` (pushed at 280ebd0; the 13 files are under `_temp_plasticity_boat/artifacts/programme/`); the next step is the jomission todo `PB-DEV100-V2-VALIDATION` (BLOCKED on O-8 and the JaxFNE pin). Open: O-8 (criteria v2 or drop), O-9 (decoder sum vs mean). Not closed: the controls do not match the frozen expected values.
- EX-26 [H] Freeze the eight-strategy list and the metric (jomission `_temp_plasticity_boat/artifacts/programme/design/plans/plasticity_boat/protocol_v0.md` §3 to §5; resolves EX-20). Blocks: all confirmatory runs. Owner decision: confirm S1 to S8, or name changes. Closes when: `protocol_v0.md` §3 to §5 are marked frozen (O-6).
- EX-28 [A] Move every figure-source JSON from `_temp/` to a tracked path before citation. Closes when: `git ls-files` lists each source and each figure's legend cites it.
- EX-29 [H] Permission to download the AAAI-27 Author Kit (https://aaai.org/authorkit27/; the file size and name are not yet known). Needed for the reproducibility checklist text and the style file for the build gate. Closes when: Hamm's yes or no is recorded here.
- EX-31 [A] Plasticity Boat checks that run in jomission `_temp_plasticity_boat`, carried here so they are not lost: the 33-file tier-0 suite under `suite_lock` (not run; the sandbox denies the shared `suite.lock`), and the `release_artifacts` gate (not examined; dist absent). Closes when: each run's summary line is read and recorded in jomission `HANDOFF.md`, or the handoff marks it not run with the reason. Blocked by: none.
- EX-17 [A] F6 rerun storing d[600] and the κ = 0 equivalence. Closes when: F6 mean is recomputed under the pinned window (errors[480:600]).
- EX-19 [A] Decoder sweep over e in [0, 5] to check the "+x at every distance" claim (critic finding 7). Closes when: the crossover e is reported from the tonic and baseline values.
- EX-9 [A] Labelled directional-encoder scenario S3-dir (separate from the fixture; gain on the +x/−x channel by sign of p_x). K=200, starts ±5. Closes when: a JSON with per-start final distance and a corrective-direction count. Blocked by: none; explicitly not the M2 fixture.
- EX-10 [A] Labelled symmetric-gain scenario S1-sym (g[1] = 24). Prediction: zero net force, boat stays. Closes when: a 40-interval probe confirms or refutes it.
- EX-11 [H] Hamm decision on the fixture defect (contract §13.7): keep the fixture, or approve a labelled separate scenario. Closes when: Hamm's choice is recorded in contract §13.7 (O-3). Blocks: confirmatory runs, §13 conclusions, integration.
- EX-12 [H] Hamm approval of §13.7 gates (after EX-8) before any held-out run. Closes when: Hamm's approval is recorded in contract §13.7a. Blocks: the held-out confirmatory set (N = 40).
- EX-13 [A] Lane A F0 vs F1 K=600 discrepancy (527.86 vs 526.32): find the cause or record it unresolved with the two paths. Closes when: a 40-interval run of both code paths is compared bit-wise and the K=600 cause is named or marked unknown.
- EX-14 [A] Scale the best one or two strategies to S2. Closes when: n ≥ 20 start-level results per cell. Blocked by: EX-11, EX-12.

## Deferred, each with its trigger

experiment_a b0/b1/b3 receipts (P-066 review minor, z now in mm), trigger: their next regeneration; re-record them then.

P-053 follow-up, trigger: a new hypothesis or coverage requirement. T-83 (2026-11-15, `artifacts/etudes/papers/pcl/port/P053_sparse_fixture_result.md`)
is NOT_REPRODUCED under criteria Amendment 1 only; V1's LAT half is never consumed and V1 adds information beyond V3
in seed 0 through TD, so no general sparse-weight equivalence is shown. Source-only discriminators (event order,
repeated addresses, ties, zero vs epsilon weights) are in `artifacts/assurance/issue_log/p053_p056_source_audit_2026-11-13.md`. AUD-02 reproduced 2026-11-15: a pinned receipt that is corrupted, status-flipped or deleted is flagged; a new
unparseable or status-less receipt is not a record by design (`_is_final_receipt`); no bypass, pinned by
`tests/test_record_integrity.py::test_corrupt_receipt_unpinned_is_not_a_record_but_pinned_is_caught`.

File-count budget: `budget_files` 2825 (S16, Hamm 2026-11-21; tree 2813 files), `budget_bytes` 96.4 MB. Trigger: the next needed new file. Action then: find removable files with evidence (match-cleanup), or ask Hamm for a `budget_files` ruling; never raise it unasked.

AT-10-R6 RSS growth before any run over 100 s; A8 gh-pages policy; architecture split on a measured bottleneck;
C5-C7 zero/NaN contract; the UNTESTED-exact refusal tail and O-67 skips; S27 population family; agent transport
(MCP); 0.5.4 items 1b and 0; the 2026-09-20 owner calls; Atlas ladder; the mechanism rule proposal; stored-only
fields when a caller needs one; the tracked `w3b_scan_run.log` at the next W3b refreeze.
Directions, open: augmentation runs; `pcl_stdp_h2`; structural HDP; primate primary-connection mode; theory F3-F5
and Lean proofs; Python 3.15 in CI once its wheels exist.

## Long-term quality and OCMA programme (planning; no new release authorization)

The unified scope, evidence gates, thinning protocol and proposed milestones 0.5.12–0.5.15 live only in `artifacts/programme/design/plans/ocma/ocma_jaxfne_0.5.10-0.5.15.md`. Keep the ordered 0.5.10 → 0.5.11 → 0.6.0 execution path above until Hamm approves a changed release sequence. At the 0.5.11 planning boundary, translate the proposed next milestone into smallest actionable rows with explicit checks; do not duplicate the roadmap here or treat planning as a signed protocol.

## OCMA cross-project assessment routing (planning only)

Dated scorecard: `artifacts/programme/design/plans/ocma/ocma_jaxfne_goal_match_2026-11-12.md` (JaxFNE 52/100, OCMA 55/100 provisional, not release scores). The top executable JaxFNE task is T-PLAN-EXECUTE. OCMA owns its local HOLD and paused R4-1; this JaxFNE planning note changes no OCMA file or gate. On completion of 0.5.10, use the unified plan to propose the next smallest checked todo rows; do not alter existing release acceptance or owner goal/fact stacks.
