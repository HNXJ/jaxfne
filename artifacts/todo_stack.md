# todo stack

Open work only, fastest first within each tier; prerequisites above dependents. `[A]` agent, `[H]` Hamm.
Size: S under 1 h, M 1-4 h, L a day or more or heavy compute. Scope letters group rows that one Haiku 5.5
`worker` can take in its own lane, a clone at `E:/repos/_wt/jaxfne/<lane>`, at most 3 lanes at a time, one
writer per file set; the orchestrator verifies and commits. Done rows are deleted. The detail of each row
before this rewrite is in `git show 6be045f7:artifacts/todo_stack.md` (each row names its old id).
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

## 1. Fast (S)

- T-35 [A] (D) The secondary metadata keys stay (Hamm 2026-10-28): `amplitude_status`, `run_status`,
  `model_status`, `field_model_status` (63, 38, 39, 13 files, none in `jaxfne/`), documented in one table
  in `docs/guides/readout/output_bundles.md` beside `claim_level` and `field_claim_level` (packet KEYS).
  Closes when: the table lands and `python scripts/run_test_gate.py docs` passes.
- T-56 [A] (C) Siblings of the wave-1 rulings (2026-10-28), each a ruling for Hamm or a fix by analogy:
  `spectrolaminar_psd_jax` uses the symmetric `jnp.hanning` (scipy's `hann` is periodic);
  `vis.core.binned_population_rate_hz` rounds an off-grid bin silently (documented for plots);
  `_model_tune.py:471` seeds candidates `seed + idx`, so scans with adjacent seeds share simulation seeds.
  Closes when: each is fixed with a test or ruled.
- T-58 [A] (T) `scripts/audit_results_draft.py` rewrites its `write_once` receipt
  (`artifacts/publication/results_reconstruction/results_audit_receipt.json`) on every run (two stale
  `docs\theory\publication` path strings still sit in the committed receipt) and crashes on a cp1252
  console. Closes when: the script writes only under `artifacts/_temp/` or checks the receipt, and
  prints ASCII.
- T-8 [A] Standing: each new score updates `dashboard/sources.json`, regenerates, pushes dev; after each dev
  push refresh the lane dashboard (skill `jaxfne-dashboard`). Closes when: `python
  scripts/build_dashboard_data.py --check` passes on each push.

## 2. Medium (M)

- T-42 [A] (T) CI plan (`etudes/meta/plans/ci_plan.md`; B, C, D, F landed): `paper-lanes` is in
  `fast-build`'s needs (2026-10-28, observed green on 760bcec3); F's reuse
  requires one executed case per family in each JUnit leg (2026-10-28); the family mapping and the `gh` field
  names meet a live run at the first rc; B has its seven chains (jdna develop added 2026-10-28) and
  runs as a CI stage first; E's first nightly is its observation. Closes when: `ci_plan.md` lists
  every step landed and observed.
- T-10b [A] jnwb cross-learning (`control/harness/jnwb_cross_learning.md`): the errors page landed
  (8f3ce57e) and the kernel freeze is dropped (Hamm 2026-10-28); left: a proxy-claim wording gate.
  Closes when: it lands with its test or is dropped.
- T-45 [A] C3, S1 (`docs/theory/doctrine/definition_map.md`; D17, D21, D26, D34 map to public symbols,
  0bc1e47a and 201750c9; D35 maps to `jaxfne.fields.apply_calibration`; D31, D32 map to
  `jaxfne.hdp_rule` public names, 6337c002): D28 has no symbol (KNOWN_GAPS; ruled public `QuantityKind`);
  the object specializer `PV[model=IZH]` does not parse (`Base[NAME]` only). Closes when: the map has no unsymboled row and
  `python scripts/publication_figures/supplement.py` regenerates S1, S2, S5.
- T-46 [A] C9, C2: 8 gaps in `programme/paper/reproducibility_audit.md` section 7 (the `publication` job passed;
  fig03-05 RGBA and all bytes differ from the macOS freeze); left: the data and code statement (0.5.11), the
  PCL natural-image and DVS128/N-MNIST licences, S16 generators. Closes when: section 7 lists 0 gaps.

## 3. Slow (L)

- T-3 [A] (P) PCL replication, option (b) (Hamm 2026-10-27). Core lanes L1-L4 closed. Next: R2, the
  port's replay of the trained network's training files through the chained scan, compared with the port
  (the C++ binary needs WSL installs, O-58); then R3 (attribute each residual: implementation, parameters,
  stochastic, model); R4 prereg, declaring T4 up front (F6: the driven column (4,4)); R5 score, exact and approximate as separate outcomes.
  The signed v5 T2 FAIL stays. Closes when: a signed prereg reproduces Fig. 3 or Hamm closes it.
- T-22, T-25 [A] (P) After T-3 R2: PCL B3 on the intra-step operator with the core mapping; PCL trained
  weights load as A' (`with_trained_weights`). Closes when: both run on the core with tests.
- T-O70 [A then H] H3 preregs: Potjans and Vogels estimands, B8 pilots on seed 0 and a known positive
  (Potjans at 10% with in-degree held is about 3e7 synapses: the 1% and 2% pilots, 3.0e6 and 6.0e6
  synapses at 2.5 and 4.0 GB, extrapolate to roughly 16 GB, inferred; the earlier 3e8 and 150 GB was the
  full-scale count), then Hamm signs; GSDR and Destexhe pilots (scoring seeds 9-13). Closes when: Hamm
  signs or defers each.
- T-32 [A] (E) One replication a day (Hamm 2026-10-19): GSDR fit on `simulate_hh_batch` with
  `gsdr_mcdp_transform`; T-32c seed and candidate batching for the AdEx and LIF routes and `Model.tune`
  (`_model_tune.py:451`); T-32d core homes for lane analysis (ISI ratio and CV, aperiodic fit, tuning fit);
  core gaps Potjans G-TAUSYN, G-POPDRIVE, G-WDIST, G-DDIST, Vogels G-SC, Mejias G6 and phi(0)=1; the LIF
  edge kernel on GPU (13.4 GiB at 25000 x 12500); Brunel sensitivity S2, S3 (S1 is in `RESULT_P3.md`).
  Closes when: each lands with its test or record.
- T-30 [A] (E) Replication conformance per paper (`scripts/replication_known_violations.json`, 26 entries;
  the 11 S1 entries are Hamm's exceptions, 2026-10-28):
  Izhikevich 2003 (net.m order through `construct`, closes at the signed P6), CAMINOS (estimators into
  `jaxfne.analysis`, P3 sweeps re-run on the core as an equivalence check), GSDR (HH in `construct()` dispatch,
  estimators), Destexhe (estimators, LEDGER, RESULT), pcl_h and Q-learning (phase records; pcl_h `sys.path`
  imports), PCL (closes at the R4 prereg). Closes when: `check_replications.py --check` lists no violation
  without Hamm's exception.
- T-47 [A] 0.5.10 fields: the `kernel-lfp` commits 6eb3c6b8, 44cd0111, ff45364b (hybrid kernel LFP); then
  the iSTDP out-degree check and kernel LFP checks in the validation backbone; then `coherence_radius_um` in
  `jaxfne.fields.kernel_lfp` with its two limit tests. Closes when: on dev with tests.
- T-48 [A] Jansen-Rit 1995 opens the EEG/MEG lane: a neural-mass emitter in the core, then P1.
  Closes when: P1 recorded.
- T-49 [A] B6: split closed lane folders by kind once each lane closes (`etudes/papers/caminos/`,
  `izhikevich2003/`, `brunel2000/` exceed 7). Closes when: their exceptions leave
  `tests/test_artifacts_tree_degree.py`.

## 4. Waiting on Hamm

- T-R28 [A] Hamm's rulings 2026-10-28 still to apply (wave 1 and wave 2 landed: 9d338ba3 to b1d9dfe1):
  dashboard P1 then P2; 0.5.9 waits for 9A-9G.
  Science, each output-changing with a changelog fragment: P-052 next is a beta by horizon scan
  against the central finite difference.
  Protocol: Izhikevich P6 scores seeds 2-6, Q5 option B, margins as drafted, signed once the
  `instantaneous` receptor (landed b1d9dfe1) replaces the tau 1e-3 ms coupling in its model table; Potjans
  P3 v2 at 10% with in-degree held, e <= 0.20; Vogels P3 v2 as drafted (full N, e <= 0.20, O-DROP ratio
  >= 2); Destexhe and GSDR reserve scoring seeds 9-13 (pilots 0-3, GSDR fit 1-3), signatures after pilots
  (O-70); the CAMINOS core-route rerun is an equivalence check against the frozen result (same prereg and
  seeds, within its between-seed range); PCL R4 scores the driven column (4,4) and reports (0,0) beside
  it; WSL: Hamm fixes DNS (O-58), then the agent installs the C++ build packages; the jitter reader reruns
  the frozen readout on larger sets first, `w_max` only if it stays weak, receipts into the repo.
  Paper: no cap on references (C10 keeps the DOI check); C1: CL-29 states 0.426 as the maximum over
  repetitions, CL-05 names its one post-registration revision, the other five conflicts and both orphans
  are text fixes against the ledger and receipts (agy drafting the edits); C4 adds a Euclidean radius
  cutoff (rotation-invariant candidates, an arbitrary-angle S3 test); C6 uses `geometry="laminar"` and
  declares Brian2 and Jaxley baselines; D28 becomes public `QuantityKind` (B, R, E, T).
  Local copies (O-68): rescue what a record cites into the repo or `F:/warehouse` with a `state.md`
  pointer, list each deletion for Hamm, delete one at a time. jarness selfcheck: propose to the jynx keeper
  that selfcheck skips files a workspace declares as history (record pins, receipt trees).
- T-O58 [H] WSL DNS. Closes when: `pip download jaxfne==0.5.7 --no-deps` works in WSL.
- Directions recorded, open: augmentation runs (`programme/design/plans/augment_science_runs_plan.md`);
  `pcl_stdp_h2` for new work; structural HDP; primate primary-connection mode; theory family F3, F4 or F5
  and Lean proofs; Python 3.15 in CI once its wheels exist.

## 5. External and gated

- T-11 [A] G-3 CAMINOS NWB excerpt for jnwb `tests/` (waits on jnwb's path). Closes when: the excerpt's
  recovery test passes in jnwb `tests/`.
- T-O10 jynx keeper takes or refuses each harness proposal (jchat conference #29).
- T-RG komposer `run_guard.py stop` falls back to the payload cwd; fix in komposer, reinstall.
- T-54 [A] Submit jaxfne to awesome-jax (New Libraries, beside BrainPy), gated on the 0.5.10 full release
  and 7 replicated papers (Hamm 2026-10-27). Closes when: the PR to n2cholas/awesome-jax is open, with
  Hamm's go.
- Deferred, each with its trigger: AT-10-R6 RSS growth before any run over 100 s; A8 gh-pages policy;
  architecture split on a measured bottleneck; C5-C7 with a signed-zero/NaN contract; the UNTESTED-exact
  refusal tail and O-67 skips; S27 population family; agent transport (MCP); 0.5.4 items 1b and 0;
  2026-09-20 owner calls (`compile_step_fn **hdp_kwargs`, `validate_hdp_params`, P-003); Atlas ladder;
  the mechanism rule proposal; stored-only fields (`LaminarColumnConfig.cell_colors`, `cell_signs`, `cy_m`,
  `HDPRuleContext.exc_mask`, `SanityDeltaConfig` asserts) when a caller needs one; the tracked
  `protocol_w/w3b_parameter_domain/w3b_scan_run.log` at the next W3b refreeze.
