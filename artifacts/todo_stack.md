# todo stack

Open work only. Order: priority (1 blockers, 2 majors, 3 rest), then effort, fastest first; prerequisites above
dependents. Items waiting on the owner join no lane (section 4). `[A]` agent, `[H]` Hamm. Priority: a blocker is the
next step of the top goal (`state.md` "Next action") or an open item the others wait on; a major is an open item
(`open_stack.md`) or a G-0 checklist item (`programme/design/plans/arxiv_paper_plan.md`); the rest is all else.
Size: S under 1 h, M 1-4 h, L a day or more or heavy compute. Figures marked `est.` are estimates.
Scope letters group rows that one Haiku 5.5 `worker` can take in its own lane, a clone at
`E:/repos/_wt/jaxfne/<lane>`, at most 3 lanes at a time, one writer per file set; the orchestrator verifies and
commits. Done rows are deleted. The detail of each row before this rewrite is in
`git show 6be045f7:artifacts/todo_stack.md` (each row names its old id).
Open problems: `open_stack.md`; issues: `assurance/issue_log/ISSUE_LOG.md`; standing rules B1-B8:
`control/harness/evidence_and_workflow_rules.md`; paper goal G-0: `programme/design/plans/arxiv_paper_plan.md`
(C, F and S ids); release plan: `etudes/meta/plans/release_plan_0.5.9-0.6.0.md`.

## Scopes

| Scope | Covers | May write | Tier |
|---|---|---|---|
| R | records, dashboard sources, issue log | `artifacts/`, `dashboard/sources.json` | worker medium |
| D | docs, examples, comments | `docs/`, `examples/`, `scripts/` text, `tests/` comments | worker medium |
| T | tests, gates, CI | `tests/`, `scripts/run_test_gate.py`, `.github/` | worker high |
| C | small core fixes, failing test first | `jaxfne/`, `tests/` | worker high |
| U | read-only audits | nothing (findings in the report) | worker xhigh plus high |
| E | replication throughput | `jaxfne/analysis/`, `scripts/check_replications.py`, `artifacts/etudes/` | worker high, orchestrator semantics |
| P | PCL replication | `artifacts/etudes/papers/pcl/`, `jaxfne/emitters_lif.py` | orchestrator with workers |

## 1. Blockers

- T-59 [A] (E) Cell Reports Fig S1 runner (Fig 5 runs through `run_figA.py`, verdicts in
  `etudes/papers/caminos/RESULT_FIGURES.md`). Closes when: an S1 runner with its test passes and the
  `replication_audit` family reads clean in `python scripts/run_test_gate.py report`. Blocked-by: T-O73.
  Effort: M, est. 2 h.
- T-63 [A] (E) CAMINOS GPU queues (WSL `~/figC_queue.sh` then `~/figP_queue.sh`, launched 2026-10-31; 1C and 5C
  scored): score Cell Reports 3A/3B and 4A/4B, then PLoS CB 2C-F, 3C-F and 4A bottom (`FIGA_PARAM_SET=default`).
  Closes when: the verdicts are in `RESULT_FIGURES.md` and the two replica blocks of `dashboard/sources.json`, and
  `python scripts/build_dashboard_data.py --check` passes. Blocked-by: none. Effort: L, about 6 h GPU.
- T-3 [A] (P) PCL replication, option (b) (Hamm 2026-10-27). Core lanes L1-L4 closed. Next: R2, the
  port's replay of the trained network's training files through the chained scan, compared with the port
  (the C++ reference builds headless in WSL and its test binary passes 7 of 7, from a copy at `~/pcl_cpp`
  with `data/events/shapes.h5`; the oracle comparison is not run yet); then R3 (attribute each residual: implementation, parameters,
  stochastic, model); R4 prereg, declaring T4 up front (F6: the driven column (4,4)); R5 score, exact and approximate as separate outcomes.
  The signed v5 T2 FAIL stays. Closes when: the R2 receipt exists in `artifacts/etudes/papers/pcl/`, and a signed
  prereg reproduces Fig. 3 or Hamm closes it. Blocked-by: none (WSL DNS closed, O-58). Effort: L, est. 1 day.
- T-R28 [A] Hamm's rulings 2026-10-28 still to apply (wave 1 and wave 2 landed: 9d338ba3 to b1d9dfe1):
  dashboard P1 then P2; 0.5.9 waits for 9A-9G.
  Science, each output-changing with a changelog fragment: P-052 next is a beta by horizon scan
  against the central finite difference.
  Protocol: Izhikevich P6 scores seeds 2-6, Q5 option B, margins as drafted, signed once the
  `instantaneous` receptor (landed b1d9dfe1) replaces the tau 1e-3 ms coupling in its model table; Potjans
  P3 v2 at 10% with in-degree held, e <= 0.20; Vogels P3 v2 as drafted (full N, e <= 0.20, O-DROP ratio
  >= 2); Destexhe and GSDR reserve scoring seeds 9-13 (pilots 0-3, GSDR fit 1-3), signatures after pilots
  (T-O70); the CAMINOS core-route rerun is an equivalence check against the frozen result (same prereg and
  seeds, within its between-seed range); PCL R4 scores the driven column (4,4) and reports (0,0) beside
  it; the jitter reader reruns the frozen readout on larger sets first, `w_max` only if it stays weak, receipts
  into the repo.
  Paper: no cap on references (C10 keeps the DOI check); C4 adds a Euclidean radius cutoff
  (rotation-invariant candidates, an arbitrary-angle S3 test); C6 uses `geometry="laminar"` and declares
  Brian2 and Jaxley baselines.
  Closes when: T-R28 is empty (O-72). Blocked-by: the Hamm signatures and the P-052 fitness call in section 4.
  Effort: L, est. 2 days.

## 2. Majors

- T-O68 [A] (R) Local copies (O-68, ruled 2026-10-28): rescue each cited file into the repo or `F:/warehouse`
  with a `state.md` pointer, list each deletion in the report, delete one at a time. Copies: `cl-reader-cursor`,
  `jaxfne-wt-p022`, the four WSL copies, and the non-git leftovers named in O-68. Closes when: each named copy is
  rescued and deleted (check: `ls E:/repos/` and the pointer in `state.md`). Blocked-by: none. Effort: M, est. 1-2 h.
- T-45 [A] C3, S1 (`docs/theory/doctrine/definition_map.md`; D17, D21, D26, D34 map to public symbols,
  0bc1e47a and 201750c9; D35 maps to `jaxfne.fields.apply_calibration`; D31, D32 map to
  `jaxfne.hdp_rule` public names, 6337c002; D28 maps to `jaxfne.hdp_rule.QuantityKind`, 1aad086a);
  the object specializer `PV[model=IZH]` does not parse (`Base[NAME]` only). Closes when: the map has no unsymboled row and
  `python scripts/publication_figures/supplement.py` regenerates S1, S2, S5. Blocked-by: none recorded.
  Effort: M, est. 2 h.
- T-46 [A] C9, C2: 8 gaps in `programme/paper/reproducibility_audit.md` section 7 (the `publication` job passed;
  fig03-05 RGBA and all bytes differ from the macOS freeze); left: the data and code statement (0.5.11), the
  PCL natural-image and DVS128/N-MNIST licences, S16 generators. Closes when: section 7 lists 0 gaps (close test
  under review, section 4). Blocked-by: the 0.5.11 data statement sits outside this release. Effort: M, est. 3 h.
- T-O70 [A then H] H3 preregs: Potjans and Vogels estimands, B8 pilots on seed 0 and a known positive
  (Potjans at 10% with in-degree held is about 3e7 synapses: the 1% and 2% pilots, 3.0e6 and 6.0e6
  synapses at 2.5 and 4.0 GB, extrapolate to roughly 16 GB, inferred; the earlier 3e8 and 150 GB was the
  full-scale count), then Hamm signs; GSDR and Destexhe pilots (scoring seeds 9-13). Closes when: Hamm
  signs or defers each (section 4). Blocked-by: the pilots run before any signature. Effort: L, est. 1-2 days,
  heavy compute.
- T-30 [A] (E) Replication conformance per paper (`scripts/replication_known_violations.json`, 26 entries;
  the 11 S1 entries are Hamm's exceptions, 2026-10-28):
  Izhikevich 2003 (net.m order through `construct`, closes at the signed P6), CAMINOS (estimators into
  `jaxfne.analysis`, P3 sweeps re-run on the core as an equivalence check), GSDR (HH in `construct()` dispatch,
  estimators), Destexhe (estimators, LEDGER, RESULT), pcl_h and Q-learning (phase records; pcl_h `sys.path`
  imports), PCL (closes at the R4 prereg). Closes when: `python scripts/check_replications.py --check` lists no
  violation without Hamm's exception. Blocked-by: none recorded. Effort: L, est. 2-3 days.

## 3. Rest

- T-8 [A] Standing: each new score updates `dashboard/sources.json`, regenerates, pushes dev; after each dev
  push refresh the lane dashboard (skill `jaxfne-dashboard`). Closes when: `python
  scripts/build_dashboard_data.py --check` passes on each push. Effort: S, per push.
- T-10b [A] jnwb cross-learning (`control/harness/jnwb_cross_learning.md`): the errors page landed
  (8f3ce57e) and the kernel freeze is dropped (Hamm 2026-10-28); left: a proxy-claim wording gate.
  Closes when: it lands with its test or is dropped. Blocked-by: none. Effort: M, est. 2 h.
- T-42 [A] (T) CI plan (`etudes/meta/plans/ci_plan.md`; B, C, D, F landed): `paper-lanes` is in
  `fast-build`'s needs (2026-10-28, observed green on 760bcec3); F's reuse
  requires one executed case per family in each JUnit leg (2026-10-28); the family mapping and the `gh` field
  names meet a live run at the first rc; B has its seven chains (jdna develop added 2026-10-28) and
  runs as a CI stage first; E's first nightly is its observation. Closes when: `ci_plan.md` lists
  every step landed and observed. Blocked-by: the first live CI run. Effort: M, est. 2 h plus the CI wait.
- T-49 [A] B6: split closed lane folders by kind once each lane closes (`etudes/papers/caminos/`,
  `izhikevich2003/`, `brunel2000/` exceed 7). Closes when: their exceptions leave
  `tests/test_artifacts_tree_degree.py`. Blocked-by: each lane closing (none recorded as closed). Effort: M, est. 3 h.
- T-22, T-25 [A] (P) After T-3 R2: PCL B3 on the intra-step operator with the core mapping; PCL trained
  weights load as A' (`with_trained_weights`). Closes when: both run on the core with tests.
  Blocked-by: T-3 R2. Effort: M, est. 4-8 h.
- T-47 [A] 0.5.10 fields: the `kernel-lfp` commits 6eb3c6b8, 44cd0111, ff45364b (hybrid kernel LFP; the three
  objects exist and none is an ancestor of dev HEAD, so they land first); then
  the iSTDP out-degree check and kernel LFP checks in the validation backbone; then `coherence_radius_um` in
  `jaxfne.fields.kernel_lfp` with its two limit tests. Closes when: on dev with tests. Effort: L, est. 1-2 days.
- T-32 [A] (E) One replication a day (Hamm 2026-10-19): GSDR fit on `simulate_hh_batch` with
  `gsdr_mcdp_transform`; T-32c seed and candidate batching for the AdEx and LIF routes and `Model.tune`
  (`_model_tune.py:451`); T-32d core homes for lane analysis (ISI ratio and CV, aperiodic fit, tuning fit);
  core gap Mejias G6; the LIF
  edge kernel on GPU (13.4 GiB at 25000 x 12500).
  Closes when: each lands with its test or record. Blocked-by: none recorded. Effort: L, est. 1 day per replication.
- T-48 [A] Jansen-Rit 1995 opens the EEG/MEG lane: a neural-mass emitter in the core, then P1.
  Closes when: P1 recorded. Blocked-by: none recorded. Effort: L, est. 2 days or more.

## 4. Waiting on owner

Each row names the decision needed. None joins a lane.

- T-O73 [H] (O-73) S1 drive: rule the drive first. The design under review holds it fixed (`ext_sources="fixed"`)
  while p varies; `SOURCE_P1.md` line 18 and commit `b8b38aea` give p·NE Poisson sources at 4 Hz. Then rule private
  vs shared sources (G3). Unblocks the S1 part of T-59. Closes when: Hamm rules and the S1 runner follows.
- T-64 [H] CAMINOS PLoS CB Fig 1D: the mean-field port of the printed Methods gives 12.5 Hz against 35 Hz
  (`etudes/papers/caminos/RESULT_MEANFIELD.md`). Decision: permission to download the authors' code (Zenodo
  10.5281/zenodo.20391405) as the oracle. Closes when: Hamm rules; 1D is rescored against that code.
- T-65 [H] CAMINOS Cell Reports 2A bottom, 2B bottom (t0) and 6B (E-to-E gain): the protocol values are not in
  the text. Closes when: Hamm gives the values or excludes the panels.
- T-62 [H] Figure 1D: permission to download the Onorato data. Needed: file name, source and size (`state.md`).
  Blocks Fig 1D only. `state.md` also says panels B, C and D are checked, with no record in the repo; Hamm confirms
  which panels hold.
- T-46 close test [H]: rule whether section 7 closes at 0 gaps (its data and code statement is slated for 0.5.11)
  or narrows to the 0.5.9 gaps.
- Signatures [H] (T-R28, T-O70): Izhikevich P6, Potjans P3 v2, Vogels P3 v2, then Destexhe and GSDR after pilots.
  Each: sign or defer.
- P-052 surrogate fitness for training (`state.md`, Broken or in flight; `artifacts/runs/perf/profiles/precision_058.md`):
  Hamm's call. Blocks no executable row; the beta scan in T-R28 proceeds.
- T-O74 to T-O79 [H] (O-74 to O-79, retired selfcheck names): rule whether the dated records keep the retired names as
  history (`artifacts/runs/benchmark/pilot_055.md`, `results_055.json`,
  `artifacts/control/harness/haiku55_evaluation_2026-10-09.md`, `artifacts/etudes/theory/theory_f1/README.md`,
  `theory_f4/README.md`) or get edited. Effort once ruled: S.
- T-54 [A then H] Submit jaxfne to awesome-jax (New Libraries, beside BrainPy), gated on the 0.5.10 full release
  and 7 replicated papers (Hamm 2026-10-27). Closes when: the PR to n2cholas/awesome-jax is open, with
  Hamm's go.

## 5. External and gated

- T-11 [A] G-3 CAMINOS NWB excerpt for jnwb `tests/` (waits on jnwb's path). Closes when: the excerpt's
  recovery test passes in jnwb `tests/`.
- T-O10 jynx keeper takes or refuses each harness proposal (jchat conference #29). Also proposed to the jynx keeper:
  jarness selfcheck skips files a workspace declares as history (record pins, receipt trees).
- T-RG komposer `run_guard.py stop` falls back to the payload cwd; fix in komposer, reinstall.
- Deferred, each with its trigger: AT-10-R6 RSS growth before any run over 100 s; A8 gh-pages policy;
  architecture split on a measured bottleneck; C5-C7 with a signed-zero/NaN contract; the UNTESTED-exact
  refusal tail and O-67 skips; S27 population family; agent transport (MCP); 0.5.4 items 1b and 0;
  2026-09-20 owner calls (`compile_step_fn **hdp_kwargs`, `validate_hdp_params`, P-003); Atlas ladder;
  the mechanism rule proposal; stored-only fields (`LaminarColumnConfig.cell_colors`, `cell_signs`, `cy_m`,
  `HDPRuleContext.exc_mask`, `SanityDeltaConfig` asserts) when a caller needs one; the tracked
  `protocol_w/w3b_parameter_domain/w3b_scan_run.log` at the next W3b refreeze.

Directions recorded, open (no row): augmentation runs (`programme/design/plans/augment_science_runs_plan.md`);
`pcl_stdp_h2` for new work; structural HDP; primate primary-connection mode; theory family F3, F4 or F5
and Lean proofs; Python 3.15 in CI once its wheels exist.
