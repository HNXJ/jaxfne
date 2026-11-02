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
   test first and a changelog fragment (lanes a d9fee4b9, b fb5ef158, c ed9058c2 landed):
   d. per-step edge gain between two cell types (lane `adexd` done, under review; 6B excluded by the T-65 ruling).
   → port `cellrep_figures.py` to `caminos_api` and check equivalence at full size against the frozen result,
   within its between-seed range (T-R28 ruling). ⇒ outside: attribute (parameters, drive, delay, random streams)
   before any rescoring. ⇒ inside: rerun the Cell Reports sweeps on the GPU, the S2 entry for `cellrep_figures.py`
   closes. Checklist: 9D CAMINOS half.
4. T-3 [A] (P) PCL R2: replay the trained network's training files through the chained scan and compare with the
   port and the C++ reference (WSL `~/pcl_cpp`, test binary 7 of 7, `data/events/shapes.h5`). → R3, attribute each
   residual (implementation, parameters, stochastic, model). → R4 prereg declaring T4 up front (driven column (4,4),
   (0,0) reported beside it) → [H] sign (step 7) → R5 score, exact and approximate apart. The signed v5 T2 FAIL
   stays. Then T-22, T-25: PCL B3 on the intra-step operator; trained weights load as A' (`with_trained_weights`).
   Status 2026-10-31: R2 on `shapes.h5` from one seeded init (`run_r2_reference.py`, 922270fd); C++ null exact
   on weights (O-81); oracle vs C++ exact within rtol 1e-12 on both layers (CALIBRATION R2, receipts
   `F:/warehouse/pcl/r2_922270fd`); oracle receipt 2026-11-01 (111 195 spikes, weights exact again;
   `F:/warehouse/pcl/orc_receipt_bafa4797`). Next: R3 on the training files, then R4 prereg.
5. T-69 [A] Destexhe pilots on the WSL GPU after step 12: `run_destexhe.py fig6b_cont --seed S --tag fig6b_cont
   --out <unscored dir> --kick-fraction 0.25 --kick-rate-hz 286` for seeds 0-3 (50-85 s each) → prereg draft (E1-E4,
   scoring seeds 9-13) → [H] sign (step 7). Status 2026-10-31: pilots done (E1-E3 jointly on seed 0 of 0-3; E4 off,
   48-49 Hz); `PREREG_P3_v3.md` (v2 misstated v1's E4 as gated) pinned with three owner decisions (E4 reported or gated, 4-of-5
   aggregation, sign or defer; outside the 0.5.9 checklist, S16). CALIBRATION step 6 erratum dc9b458c.
6. T-R28 [A] Hamm's rulings 2026-10-28 left: dashboard P1 then P2; P-052 beta by horizon scan against the central
   finite difference (each output change with a changelog fragment); the jitter reader reruns the frozen readout
   on larger sets first, `w_max` only if it stays weak; paper: C4 Euclidean radius cutoff with an arbitrary-angle
   S3 test, C6 `geometry="laminar"` with Brian2 and Jaxley baselines declared.
7. [H] Signatures, one at a time as each draft is ready: Potjans P3 v2 (signed 2026-11-02, `prereg/P3_v2_signed.md`;
   next seeds 4-8 with `--signed`, T 10 s, scale 0.1, 1 s segments), Vogels P3 v4 (signed), Destexhe, PCL R4 (GSDR P3
   fit signed 2026-11-01). B8 pilots done 2026-11-02, every known positive moves: Potjans (CALIBRATION 0b; lesion
   R-L5E e 40.9; baseline predicts FAIL on R-L23E, R-L5E, O-IE L5, CV L6E); Vogels at T 150 s (CALIBRATION, B8 T 150 s
   pilots; rho0 6 Hz R-E e 1.31, eta 0 O-DROP 0.99; baseline converged at 4.11 Hz, predicts R-E FAIL e 0.41).
   Vogels P3 v4 signed 2026-11-02 (`prereg/P3_v4_signed.md`; CONV with the slope test): next a scored driver
   (`vogels2011_api.py score`, seeds 4-8 only, signed) then seeds 4-8 on the GPU, ~40 min each.
   Each: sign ⇒ scored run from a fixed clone → result file, dashboard, checklist 9D/9E; defer ⇒ the package moves
   to 0.5.10 with its reason (Gate 1).
8. T-70 [A] 9F: the rc gate end to end on the candidate (JUnit parity, figure lock, clean-checkout figure job;
    Release CI JUnit reused for the same tree). 9G: a fragment for every change since v0.5.8; the assembled section
    under the hygiene caps (26 of 30 ghost hits on 2026-10-31).
9. [H] Gate 1: every package 9A-9G closed or moved with its reason; Hamm rules each move.
10. T-71 [A] Gates 2-4 on the release SHA: broad gate, CI, `check_replications.py --check` 0 unlisted, dashboard
    fresh; second review with its receipt; release day: version 0.5.9 in `pyproject.toml`,
    `assemble_changelog.py --version 0.5.9 --date <day>`, `release_body.py --check`.
11. [H] Gate 5: Hamm's go → tag, main, Release, PyPI.

## Beside the path (∥)

12. ∥ T-63b [A] CAMINOS PLoS (O-83 closed 2026-11-02: the 2B/3B SOM fails follow the stated 2 ms delay,
    `RESULT_PLOS_FIGS.md`): 2C-F, 3C-F: rerun 40 and 90 % saving rasters
    and population traces (the figP rows keep spectra only), mean-field traces beside them. Closes when the replica
    block scores them and O-83 holds an attribution.
13. ∥ T-59 [A] Cell Reports Fig S1 runner with its test; `replication_audit` reads clean in the report gate. Drive
    (Hamm 2026-11-02, O-73 closed): p·NE Poisson sources at 4 Hz, `ext="private"` (the default, as every other panel).
14. ∥ T-O68 [A] Local copies (O-68): rescue each cited file into the repo or `F:/warehouse` with a `state.md`
    pointer, delete one at a time (`cl-reader-cursor`, `jaxfne-wt-p022`, the four WSL copies, the non-git leftovers
    named in O-68), and the finished lane clones under `E:/repos/_wt/jaxfne/` after an alternates check.
15. ∥ T-45 [A] The object specializer `PV[model=IZH]` parses (`Base[NAME]` only today); `Base[model=X]` raises
    ValueError naming the supported keys (Hamm 2026-11-02, O-82 closed). Closes when: parse and refusal tests pass and
    S1 regenerates.
16. ∥ T-46 [A] C9, C2: 8 gaps in `programme/paper/reproducibility_audit.md` section 7 (left: data and code statement,
    PCL natural-image and DVS128/N-MNIST licences, S16 generators). Close test (Hamm 2026-11-02): section 7 closes
    when the gaps touching 0.5.9 items close; the paper-only gaps (licences, data and code statement) move to the
    paper lane with a dated note.
17. ∥ T-30 [A] Conformance for the rest (`scripts/replication_known_violations.json`; the S1 entries are Hamm's
    exceptions): GSDR (HH in `construct()` dispatch, estimators), Destexhe (estimators, LEDGER, RESULT), pcl_h and
    Q-learning (phase records; pcl_h `sys.path` imports), PCL (closes at R4). Closes when: `check_replications.py
    --check` lists no violation without Hamm's exception.
18. ∥ T-8 [A] Standing: each new score updates `dashboard/sources.json`, regenerates, pushes dev; then the lane
    dashboard (skill `jaxfne-dashboard`).
19. ∥ T-10b [A] jnwb cross-learning: a proxy-claim wording gate, or drop it. T-42 [A] CI plan steps observed at the
    first rc. T-49 [A] split closed lane folders by kind as each lane closes (`caminos/`, `izhikevich2003/`,
    `brunel2000/`).

## Owner queue (asked one at a time, in this order)

23. [H] T-62 Figure 1D (Onorato data): download permission (file name, source, size in `state.md`); which of panels
    B, C, D hold.
24. [H] P-052 surrogate fitness for training (`artifacts/runs/perf/profiles/precision_058.md`).
25. [A] Retired names in dated records kept as history (Hamm 2026-11-02, O-74 to O-79 closed): jarness selfcheck needs
    a history allowlist for them (asked in jchat, jynx keeper).

## After 0.5.9

26. 0.5.10: T-47 kernel-LFP commits 6eb3c6b8, 44cd0111, ff45364b, then the iSTDP out-degree and kernel LFP checks,
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
