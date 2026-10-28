# todo stack

Open work only, fastest first within each tier; prerequisites above dependents. `[A]` agent, `[H]` Hamm.
Size: S under 1 h, M 1-4 h, L a day or more or heavy compute. Lane letters group rows that one Haiku 5.5
`worker` can take in its own clone (`E:/repos/_wt/jaxfne/<lane>`), at most 3 lanes at a time, one writer
per file set; the orchestrator verifies and commits. Done rows are deleted. The detail of each row
before this rewrite is in `git show 6be045f7:artifacts/todo_stack.md` (each row names its old id).
Open problems: `open_stack.md`; issues: `assurance/issue_log/ISSUE_LOG.md`; standing rules B1-B8:
`control/harness/evidence_and_workflow_rules.md`; paper goal G-0: `programme/design/plans/arxiv_paper_plan.md`
(C, F and S ids); release plan: `etudes/meta/plans/release_plan_0.5.9-0.6.0.md`.

## Lanes

| Lane | Scope | May write | Tier |
|---|---|---|---|
| R | records, dashboard sources, issue log | `artifacts/`, `dashboard/sources.json` | worker medium |
| D | docs, examples, comments | `docs/`, `examples/`, `scripts/` text, `tests/` comments | worker medium |
| T | tests, gates, CI | `tests/`, `scripts/run_test_gate.py`, `.github/` | worker high |
| C | small core fixes, failing test first | `jaxfne/`, `tests/` | worker high |
| U | read-only audits | nothing (findings in the report) | worker xhigh plus high |
| E | replication throughput | `jaxfne/analysis/`, `scripts/check_replications.py`, `artifacts/etudes/` | worker high, orchestrator semantics |
| P | PCL replication | `artifacts/etudes/papers/pcl/`, `jaxfne/emitters_lif.py` | orchestrator with workers |

## 1. Fast (S)

- T-34 [A] (R) Records: `dashboard/sources.json` `stage_artifacts` cite pre-reorg paths; the dashboard shows
  GSDR at P1 with P3 records present; `hh-at01` has no folder; PCL `CALIBRATION.md:24` keeps the
  `no construct route (OPEN)` marker (old T-30, section 4). Closes when: `python scripts/build_dashboard_data.py
  --check` and `python scripts/check_replications.py --check` pass with each fixed.
- T-35 [A] (D) Owner items (old T-OWN): amplitude-status vocabulary in docs, release notes, examples and the
  docs/archive manifest; examples v031/v032 name absent canonical JSON targets; examples 03/05 fall back to
  V_m for EEG/MEG; except-pass handlers in examples; `_tutorial_runtime` defaults an empty env with `or`;
  `consolidate_hdp_controllability_etude.py:38,254` names the absent `hdp_mvc_etude.py`;
  `upload_testpypi.sh:20` names the absent RELEASE_CHECKLIST.md; `protocol_c/d/e/__init__.py` read
  "receipts.Checkout". Closes when: each named string is gone and `python scripts/run_test_gate.py docs` passes.
- T-36 [A] (D) About 60 comment paths under `tests/` do not resolve (e.g. `test_artifacts_root_layout.py:211`,
  `test_manifest_readout.py:91`, `test_izh_analysis.py:264`). Closes when: a path resolver over `tests/`
  comments reports 0.
- T-37 [A] (T) Test findings (old T-AUD): `test_release_ci_pytest_sweep_is_unfiltered`
  (`test_release_gate_hierarchy.py:378`) reads no `if:` trigger; the `smoke`/`fast`/`integration`
  auto-markers (`tests/conftest.py:55-85`) are selected nowhere; `test_script_repo_root_guard.py:230`
  accepts any "site-packages" substring; the E1 delayed-edge route has no test. Closes when: each check
  fails on a planted defect.
- T-38 [A] (C) Small core findings: `fields/diagnostics.py:111` projection warnings never raise; the multi-area
  helper scales W by 0.1 against its docstring (`emitters.py` near 4883, trace callers); the dev-gate warning
  "compiled 2 times! Invariant N_compile <= 1 violated"; GSDR HH notes (`held_steps` counts edge elements
  while the docstring says steps, int32 count scalars, `dt="0.1"` coerced, no full-kernel grad test).
  Closes when: each is fixed with a test that fails first, or recorded here with its reason.
- T-39 [A] (C) `simulate_batch` refuses the surrogate on the dense route that `simulate` realizes (additive).
  Closes when: `tests/test_surrogate_gradient.py` covers `simulate_batch` equal to looped `simulate`.
- T-40 [A] (C) `visualize_network_3d`: a label for relative coordinates (unknown unit strings already refuse).
  Closes when: a test shows relative rows labelled relative.
- T-8 [A] Standing: each new score updates `dashboard/sources.json`, regenerates, pushes dev; after each dev
  push refresh the lane dashboard (skill `jaxfne-dashboard`). Closes when: `python
  scripts/build_dashboard_data.py --check` passes on each push.

## 2. Medium (M)

- T-41 [A] (U) Audit the modules no lane read: `jaxfne/jdna/`, `jaxfne/tfne.py`, `jaxfne/optim/`,
  `jaxfne/analysis/`, `jaxfne/protocol_*/`; two efforts, union verified by the orchestrator. Closes when:
  each verified finding is fixed or has a row here.
- T-42 [A] (T) CI plan (`etudes/meta/plans/ci_plan.md`): C `paper-lanes` CI job, D code-tree stamp, F rc
  reuses Release CI JUnit (approved 2026-10-27), a `jaxfne.jdna` canary chain for B; E's first nightly is
  its observation. Closes when: `ci_plan.md` lists every step landed.
- T-43 [A] (U) `artifacts/` layout map: for each root folder list every reference (package, tests, scripts,
  frozen manifests, record pins), class it movable or pinned, and draft a layout with at most 7 unexempted
  folders. Moves wait on Hamm (frozen paths). Closes when: the map is in
  `etudes/meta/plans/artifacts_layout_plan.md` and Hamm rules on it.
- T-44 [A] (E) T-32e: script checklist rules E1, S2, F3, P4, R4 in `check_replications.py`. Closes when:
  each rule fails on a planted lane defect.
- T-10b [A] jnwb cross-learning (`control/harness/jnwb_cross_learning.md`): errors page, a proxy-claim
  wording gate, AST-hash freezing of validated kernels. Closes when: each lands with its test.
- T-45 [A] C3, S1 (`docs/theory/doctrine/definition_map.md`): D21, D34 have no symbol; D17, D26 are private
  names; the object specializer `PV[model=IZH]` does not parse (`Base[NAME]` only); D31, D32 need a code path
  exposing a_eff and the AdEx clip; the S2 generator waits on D35 (H). Closes when: the map has no
  unsymboled row and `python scripts/publication_figures/supplement.py` regenerates S1, S2, S5.
- T-46 [A] C9, C2: 8 gaps in `programme/paper/reproducibility_audit.md` section 7 (the `publication` job passed;
  fig03-05 RGBA and all bytes differ from the macOS freeze); left: the data and code statement (0.5.11), the
  PCL natural-image and DVS128/N-MNIST licences, S16 generators. Closes when: section 7 lists 0 gaps.

## 3. Slow (L)

- T-3 [A] (P) PCL replication, option (b) (Hamm 2026-10-27). Core lanes L1-L4 closed. Next: R2, the
  port's replay of the trained network's training files through the chained scan, compared with the port
  (the C++ binary needs WSL installs, O-58); then R3 (attribute each residual: implementation, parameters,
  stochastic, model); R4 prereg, declaring T4 up front (F6 column is Hamm's); R5 score, exact and approximate as separate outcomes.
  The signed v5 T2 FAIL stays. Closes when: a signed prereg reproduces Fig. 3 or Hamm closes it.
- T-22, T-25 [A] (P) After T-3 R2: PCL B3 on the intra-step operator with the core mapping; PCL trained
  weights load as A' (`with_trained_weights`). Closes when: both run on the core with tests.
- T-O70 [A then H] H3 preregs: Potjans and Vogels estimands, B8 pilots on seed 0 and a known positive
  (Potjans scale 0.1 needs about 150 GB RAM: CPU at the edge or a smaller scale), then Hamm signs; GSDR and
  Destexhe pilots (new reserved seeds; Destexhe 4-8 spent). Closes when: Hamm signs or defers each.
- T-32 [A] (E) One replication a day (Hamm 2026-10-19): GSDR fit on `simulate_hh_batch` with
  `gsdr_mcdp_transform`; T-32c seed and candidate batching for the AdEx and LIF routes and `Model.tune`
  (`_model_tune.py:451`); T-32d core homes for lane analysis (ISI ratio and CV, aperiodic fit, tuning fit);
  core gaps Potjans G-TAUSYN, G-POPDRIVE, G-WDIST, G-DDIST, Vogels G-SC, Mejias G6 and phi(0)=1; the LIF
  edge kernel on GPU (13.4 GiB at 25000 x 12500); Brunel sensitivity S2, S3 (S1 is in `RESULT_P3.md`).
  Closes when: each lands with its test or record.
- T-30 [A] (E) Replication conformance per paper (`scripts/replication_known_violations.json`, 31 findings):
  Izhikevich 2003 (net.m order through `construct`, closes at the signed P6), CAMINOS (estimators into
  `jaxfne.analysis`, P3 sweeps re-run on the core: Hamm's call), GSDR (HH in `construct()` dispatch,
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

- T-50 [H] Science calls: P-052 slope (scan beta for agreement with the finite difference, or a surrogate
  scaled to the upstroke); P-048 `u0` default; P-051 a named instantaneous receptor; HDP probe rules
  `synthetic_presyn_gain`, `eligibility_trace_gain` update per step without dt (`hdp_rule.py:318-372`);
  builders default seed 42 against `construct` 0; `adex_destexhe.py:181` rounds half-even; a state clamp on
  the plain Izhikevich path (T-19); same-layer zeroing for I/Inl/Ing (T-18); bfloat16 compute or storage.
  Each changes outputs. Closes when: each is ruled.
- T-51 [H] Protocol calls: P6 seeds and Q5 (`izhikevich2003/core/PREREG_P6.md`); Potjans and Vogels scale
  and bands; D2 Destexhe and GSDR preregs; CAMINOS S2 re-run; PCL F6 column and the WSL installs for the
  C++ binary; jitter reader follow-up (w_max, larger frozen sets). Closes when: each is ruled.
- T-52 [H] Paper calls (G-0): C10 cut 112 references to 80; C1 2 orphans and 7 manuscript-ledger conflicts
  (`programme/paper/claim_alignment_audit.md`); C6 benchmark protocol and a geometry for
  `cortical_column_localized_workflow.build_config` (raises at HEAD); C4 turn invariance (claim wording or a
  radius cutoff); D35 `apply_calibration`; D28 symbol (9A names). Closes when: each is ruled.
- T-53 [H] Repo calls: T-16 atlas test merge; T-17 the manuscript tree and five receipt-writing scripts;
  T-PR Dependabot #115, #116 and deleting the merged remote branches `caminos-p6`, `fix/rtd-mcc3-post-d1`,
  `izh2003`, `l2-caminos`, `pcl-port`; dashboard P1 and P2 after the P0 acceptance; the 0.5.9 release go.
  Closes when: each is ruled.
- T-O58 [H] WSL DNS. Closes when: `pip download jaxfne==0.5.7 --no-deps` works in WSL.
- T-O68 [H] Local copies (O-68). Closes when: each is at `E:/repos/<name>` or `E:/repos/_wt/<repo>/<task>`
  or deleted.
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
