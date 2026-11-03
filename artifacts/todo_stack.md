# todo stack

One ordered path, end to end: the 0.5.9 release first (checklist and score in
`etudes/meta/plans/release_plan_0.5.9-0.6.0.md`), then 0.5.10, 0.5.11 and 0.6.0. Each step names its closing check
and where the path goes next; `→` is the next step, `⇒` a branch on the outcome. `[A]` agent, `[H]` Hamm: an `[H]`
node is a decision the path waits on, asked one at a time. Steps marked `∥` run beside the path in their own lane
(at most 3 lanes, one writer per file set, clones at `E:/repos/_wt/jaxfne/<lane>`, launched from a fixed commit).
Done steps are deleted; history is in git (the sectioned form before this rewrite: `git show 00eade8d:artifacts/todo_stack.md`).
Problems: `open_stack.md`; issues: `assurance/issue_log/ISSUE_LOG.md`; rules B1-B8:
`control/harness/evidence_and_workflow_rules.md`; paper goal G-0: `programme/design/plans/arxiv_paper_plan.md`.

## 0.5.9

3. T-30 [A] CAMINOS on the core route (9D, option A ruled 2026-10-31; the first port lane was rejected for replacing
   behavior tests with refusals). Core additions in `jaxfne/`, one lane at a time (shared files), each with a failing
   test first and a changelog fragment (lanes a d9fee4b9, b fb5ef158, c ed9058c2, d 2cda7117 landed; d carries the
   no-edge refusal, the schedule record and the equivalence tests).
   → E1 (Welch TOST ±5%, Hamm 2026-11-02) at the confirmatory n 12 (2026-11-03): not equivalent, 20 of 24 cells;
   rates equivalent in all 12; outside 0.4 f_pk, 0.7 band power and max E (unresolved) and 0.9 band power (−2.78%,
   interval −5.41 to −0.15%; two-sided p 0.083) (`RESULT_FIGURES.md`). Attribution of the 0.9 band-power
   cell (2026-11-03): both routes deterministic and unchanged across batches; seed SD 3.6-3.9%, so the margin is
   about 1.3 SD; no implementation difference found. Waits on Hamm: accept the core route or keep the oracle.
   ⇒ equivalent: rerun the Cell Reports sweeps on the GPU, the S2 entry for `cellrep_figures.py` closes. Checklist: 9D CAMINOS half. Closes when: `check_replications.py --check` lists no S2 entry for
   `cellrep_figures.py` and `RESULT_FIGURES.md` cites the core-route reruns.
4. T-3 [A] (P) PCL R3: replay the training files through the chained core scan, the oracle and the C++ reference
   (WSL `~/pcl_cpp`) and attribute each residual (implementation, parameters, stochastic, model). → R4 prereg declaring T4 up front (driven column (4,4),
   (0,0) reported beside it) → [H] sign (step 7) → R5 score, exact and approximate apart. The signed v5 T2 FAIL
   stays. Then T-22, T-25: PCL B3 on the intra-step operator; trained weights load as A' (`with_trained_weights`).
   Status 2026-10-31: R2 on `shapes.h5` from one seeded init (`run_r2_reference.py`, 922270fd); C++ null exact
   on weights (O-81); oracle vs C++ exact within rtol 1e-12 on both layers (CALIBRATION R2, receipts
   `F:/warehouse/pcl/r2_922270fd`); oracle receipt 2026-11-01 (111 195 spikes, weights exact again;
   `F:/warehouse/pcl/orc_receipt_bafa4797`). Status 2026-11-02: the
   training set is `F:/warehouse/pcl/data/training_natural/events5_re2/` (1999 npz, 3.8 GB, extracted from the
   Zenodo `PCL_Nat_training_natural_dataset.zip`); the C++ reads npz (`NetworkHandle.cpp:89-90`). R3 runs files 0-4
   chained from the R2 init: C++ arm `~/run_pcl_r3_cpp.sh` (WSL, out `F:/warehouse/pcl/r3_cpp_f0_4`); oracle and core
   arms through the `chain` subcommand (Muse packet PCL-R3-HARNESS, clone `pclr3` from d96dc49b). C++ arm done:
   two chains bit-identical (null), weights moved from init (`F:/warehouse/pcl/r3_cpp_f0_4/*.json`). Oracle arm
   exact on files 0-4 (max relative weight difference 7.4e-15, `F:/warehouse/pcl/r3_oracle_f0_4/`). Core arm costs
   about 0.17 s per camera-0 event, linear in events (500 events 98 s, 2000 events 348 s, learning on; 2026-11-02),
   so the core arm ran file 0 only: exact against C++ (max relative weight difference 5.3e-15; same spikes as the
   oracle; 11:53:46 wall). R3 done 2026-11-03 (CALIBRATION.md R3, a receipt per arm); P-053 does not appear on file 0.
   R4 prereg drafted 2026-11-03 (`pcl/prereg/P3_fig3_r4.md`, unsigned). R5 timing pilot (CALIBRATION R5 pilot):
   core 1026 s per file, equal to the oracle on a non-scored file. Hamm 2026-11-03: no-local, the 16 pilot tasks,
   and the core on all 3168 tasks (about 900 CPU h), in the R4b draft (`P3_fig3_r4b.md`, unsigned). 12-worker scaling on the pilot file (CALIBRATION):
   24.7 files per hour, outputs equal, so about 128 h wall for the core set. Next: [H] sign R4, then R5 from a frozen clone. Closes when: R5 is scored in `prereg/` with exact and approximate reproduction apart.
6. T-R28 [A] Hamm's rulings 2026-10-28 left: dashboard P1 then P2; P-052 beta by horizon scan against the central
   finite difference (each output change with a changelog fragment); the jitter reader reruns the frozen readout
   on larger sets first, `w_max` only if it stays weak; paper: C4 Euclidean radius cutoff with an arbitrary-angle
   S3 test, C6 `geometry="laminar"` with Brian2 and Jaxley baselines declared. Closes when: each listed piece has a
   commit with its test or receipt, cited here before the row is deleted.
7. [H] Signatures, one at a time as each draft is ready: PCL R4 (the other P3 drafts are signed and scored; later
   Destexhe runs record status and the signed prereg hash).
   Each: sign ⇒ scored run from a fixed clone → result file, dashboard, checklist 9D/9E; defer ⇒ the package moves
   to 0.5.10 with its reason (Gate 1). Closes when: Hamm has signed or deferred PCL R4.
8. T-70 [A] 9F: the rc gate end to end on the candidate (JUnit parity, figure lock, clean-checkout figure job;
    Release CI JUnit reused for the same tree). 9G: a fragment for every change since v0.5.8; the assembled section
    under the hygiene caps (26 of 30 ghost hits on 2026-10-31). Closes when: `run_test_gate.py rc` prints
    "RC GATE PASS" and its attestation names the candidate SHA; every `jaxfne/` commit since v0.5.8 has a fragment or
    is recorded as changing no behavior.
9. [H] Gate 1: every package 9A-9G closed or moved with its reason; Hamm rules each move. Closes when: the checklist
   in `etudes/meta/plans/release_plan_0.5.9-0.6.0.md` shows each package closed or moved with Hamm's dated ruling.
10. T-71 [A] Gates 2-4 on the release SHA: broad gate, CI, `check_replications.py --check` 0 unlisted, dashboard
    fresh; second review with its receipt; release day: version 0.5.9 in `pyproject.toml`,
    `assemble_changelog.py --version 0.5.9 --date <day>`, `release_body.py --check`. Closes when: each command
    exits 0 on the release SHA and the second review's receipt is committed.
11. [H] Gate 5: Hamm's go → tag, main, Release, PyPI. Closes when: 0.5.9 is on PyPI from the tagged release SHA.

## Beside the path (∥)

11a. ∥ T-72 [A then H] NeuroAI 2027 short paper, deadline 2026-11-20 23:59 AoE (paper plan amendment 2026-11-02;
    rules verified on the OpenReview form, non-archival, no dual-submission bar). Status 2026-11-02: draft in the
    private thesis repo, branch `_temp_neuroai2027`, `artifacts/notes/neuroai2027/`;
    3 pages main text, 2 figures, Table 1; three critic rounds, the round-3 fixes unreviewed. The submission build
    (pdfLaTeX, TeX Live 2026 in WSL, the kit's unmodified style; Hamm's choice) passes: 4 pages, 0 overfull boxes, all
    fonts embedded, no Type 3. Waits on [H]: the anonymous.4open.science mirror of the code repo (Hamm's choice;
    the link replaces `[MISSING: anonymized code link]`), then Hamm reads and submits. Closes when Hamm has submitted.
14. ∥ T-O68 [A] Local copies (O-68): rescue each cited file into the repo or `F:/warehouse` with a `state.md`
    pointer, delete one at a time (`cl-reader-cursor`, `jaxfne-wt-p022`, the four WSL copies, the non-git leftovers
    named in O-68), and the finished lane clones under `E:/repos/_wt/jaxfne/` after an alternates check. Closes when:
    every O-68 path is gone or rescued, and no surviving clone's `objects/info/alternates` names a deleted one.
    Status 2026-11-02 (Haiku inventory, no alternates under `E:/repos`): `p6run`, `pilots`, `potmem` (HEAD on dev,
    clean) and the empty `d28` deleted. Left: 14 clones with dirty files to read; 11 one-commit clones whose commit is
    on no jaxfne ref; `clogsweep` (2cd1dd27) and `pclr2study` (c7137b4b) are the sole holders of one commit each;
    `cl-reader-cursor` (`reader_exp/`, an R1 audit note) and `jaxfne-wt-p022` (a receipt) hold untracked files; the
    non-git leftovers are under `C:/workspace` (`jaxfne_rc057`, `jaxfne_receipts`, `jaxfne_reader_exp`,
    `jaxfne_pclr1_audit`); which four WSL copies O-68 means is not recorded.
16. ∥ T-46 [A] C9, C2: 8 gaps in `programme/paper/reproducibility_audit.md` section 7 (left: data and code statement,
    PCL natural-image and DVS128/N-MNIST licences, S16 generators). Close test (Hamm 2026-11-02): section 7 closes
    when the gaps touching 0.5.9 items close; the paper-only gaps (licences, data and code statement) move to the
    paper lane with a dated note.
17. ∥ T-30 [A] Conformance for the rest (`scripts/replication_known_violations.json`; the S1 entries are Hamm's
    exceptions): GSDR (HH in `construct()` dispatch, estimators), Destexhe (estimators, LEDGER, RESULT), pcl_h and
    Q-learning (phase records; pcl_h `sys.path` imports), PCL (closes at R4). Closes when: `check_replications.py
    --check` lists no violation without Hamm's exception.
18. ∥ T-8 [A] Standing: each new score updates `dashboard/sources.json`, regenerates, pushes dev; then the lane
    dashboard (skill `jaxfne-dashboard`). Standing: never closes; each score's commit cites the regenerated
    dashboard.
19. ∥ T-10b [A] jnwb cross-learning: a proxy-claim wording gate, or drop it. T-42 [A] CI plan steps observed at the
    first rc. T-49 [A] split closed lane folders by kind as each lane closes (`caminos/`, `izhikevich2003/`,
    `brunel2000/`). Closes when: T-10b has its gate or a dated drop note; T-42 lists the CI steps seen at the rc;
    each closed lane folder passes `tests/test_artifacts_tree_degree.py` with no R6 entry.

## Owner queue (asked one at a time, in this order)

24. [H] P-052 surrogate fitness for training (`artifacts/runs/perf/profiles/precision_058.md`).
25. [A] Retired names in dated records kept as history (Hamm 2026-11-02, O-74 to O-79 closed): jarness selfcheck needs
    a history allowlist for them (asked in jchat, jynx keeper).

## After 0.5.9

26. 0.5.10: T-62 Cell Reports Fig 1D, excluded for 0.5.9 (Hamm 2026-11-02): the raw Onorato data is on request
    from the lead contact; Hamm decides at 0.5.10 whether to request it. T-47 kernel-LFP commits 6eb3c6b8, 44cd0111, ff45364b, then the iSTDP out-degree and kernel LFP checks,
    then `coherence_radius_um` with its two limit tests; T-32 GSDR on `simulate_hh_batch`, seed and candidate
    batching for AdEx, LIF and `Model.tune` (`_model_tune.py:451`), core homes for lane analysis, Mejias G6, the
    LIF edge kernel on GPU; F2, F4, S16 per the release plan.
27. 0.5.10: T-48 Jansen-Rit 1995 neural-mass emitter, then P1.
28. After the 0.5.10 release and 7 replicated papers: T-54 [A then H] awesome-jax PR with Hamm's go.
29. 0.5.11 and 0.6.0: the release plan's later sections.

External and gated: T-11 CAMINOS NWB excerpt for jnwb `tests/` (waits on jnwb's path); T-O10 the jynx keeper
rules each harness proposal (jchat conference #29); T-RG komposer `run_guard.py stop` cwd fallback. Deferred, each
with its trigger: AT-10-R6 RSS growth before any run over 100 s; A8 gh-pages policy; architecture split on a
measured bottleneck; C5-C7 with a signed-zero/NaN contract; the UNTESTED-exact refusal tail and O-67 skips; S27
population family; agent transport (MCP); 0.5.4 items 1b and 0; 2026-09-20 owner calls (`compile_step_fn
**hdp_kwargs`, `validate_hdp_params`, P-003); Atlas ladder; the mechanism rule proposal; stored-only fields
(`LaminarColumnConfig.cell_colors`, `cell_signs`, `cy_m`, `HDPRuleContext.exc_mask`, `SanityDeltaConfig` asserts)
when a caller needs one; the tracked `protocol_w/w3b_parameter_domain/w3b_scan_run.log` at the next W3b refreeze.
Directions, open: augmentation runs; `pcl_stdp_h2`; structural HDP; primate primary-connection mode; theory F3-F5
and Lean proofs; Python 3.15 in CI once its wheels exist.
