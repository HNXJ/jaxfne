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

- T-35 [A] (D) Left after lane D (2026-10-27; fixed: dead test-comment paths, the TestPyPI pointer, the
  v031-v033 `update_canonical` branch writing to targets removed in 72eafbbd, the V_m fallback for EEG/MEG in
  examples 03 and 05; kept with reason: `_tutorial_runtime`'s `or`, the examples' ImportError clauses,
  `consolidate_hdp_controllability_etude.py`, which refuses when its local receipts are absent, as its docs
  say): metadata keys, two sets since 5ae19017: the package emits `claim_level` (20 files in `jaxfne/`) and
  `field_claim_level` (25), while `amplitude_status`, `model_status`, `run_status`, `field_model_status`
  (63, 53, 37, 24 files) live only outside `jaxfne/`; `physical_amplitude_claim_allowed` is a retired
  migration key (`_construct_extras.py:567`). The canonical set is Hamm's call (T-53). Closes when: it is
  ruled and swept, and `python scripts/run_test_gate.py docs` passes.
- T-37 [A] (T) Left after lane T (2026-10-27; fixed: the release sweep test reads job and step `if:`, the
  P-025 matcher needs the assert on the jaxfne module, the delayed-edge route has its test, the nightly
  tree-equality skip fails open under five planted defects): the `smoke` and `fast` auto-markers
  (`tests/conftest.py:77-85`) and the registered `integration` marker are selected by no gate, workflow or
  Makefile (recommend deleting them; Hamm's call in T-53, since local `-m fast` use is unknown). Closes when: the markers are deleted or selected.
- T-8 [A] Standing: each new score updates `dashboard/sources.json`, regenerates, pushes dev; after each dev
  push refresh the lane dashboard (skill `jaxfne-dashboard`). Closes when: `python
  scripts/build_dashboard_data.py --check` passes on each push.

## 2. Medium (M)

- T-41 [A] (C2) T-41 audit 2026-10-27 at c88be711 (two efforts over `jdna/`, `tfne.py`, `optim/`, `analysis/`,
  `protocol_*/`): lane C2 fixed 8 of 10 (jdna geometry and development keys, optim parameters never read,
  `learning_rate` 0, `allow_self` string, TFNE relation-origin delay; protocol D's drive window refuses a
  negative onset, 2026-10-28). Left: protocol E `support_only_on_O_H` (`e3_execution.py:268`), whose receipt `e3_execution_receipt.json` is frozen
  write-once, so a fix needs a new receipt version (Hamm, H-rule on frozen receipts). Estimator findings are
  in T-50. Closes when: both are fixed or ruled.
- T-55 [A] (C) Left of the T-41 findings verified 2026-10-27 (lane C3 fixed 2-9, item 1 fixed 2026-10-28,
  each with a failing test): (11) `analysis/spectral.py:80` float32 phase, absolute error 0.168 rad at
  1000 s; `spectrolaminar_psd_jax` is jitted with traced `freqs` and x64 off, so float64 changes its jit
  contract. Measured 2026-10-28 (sinusoids at 40 and 147.3 Hz, fs 1000 Hz, against a float64 DFT at the
  same freqs): max PSD error over peak 2.9e-4 at 10 s, 5.4e-3 at 100 s, 1.15e-1 at 1000 s. A float32 fix
  inside jit: `frac(r * 2^k)` is exact for r = f/fs, so the phase is the fractional part of the sum over
  the set bits of n; it changes every PSD at rounding level (published spectrolaminar outputs included),
  so it is Hamm's call in T-50. Hamm's calls in T-50 also: absence under a band lower bound of 0,
  `population_rates` dt in seconds, mask orientation. Closes when: (11) is ruled, then fixed with a test.
- T-42 [A] (T) CI plan (`etudes/meta/plans/ci_plan.md`; B, C, D, F landed): `paper-lanes` is in
  `fast-build`'s needs (2026-10-28, observed green on 760bcec3); F's reuse
  requires one executed case per family in each JUnit leg (2026-10-28); the family mapping and the `gh` field
  names meet a live run at the first rc; B has its seven chains (jdna develop added 2026-10-28) and
  runs as a CI stage first; E's first nightly is its observation. Closes when: `ci_plan.md` lists
  every step landed and observed.
- T-44 [A] (E) T-32e: rules E1, S2, P4, R4 landed in `check_replications.py` (d2553d55, 66231ddb, 5c4a52a0,
  8136beea). Left: F3 ("a fidelity fix carries a test that fails on the parent commit") is per commit, so a
  file scan cannot check it (a CI step over commits tagged as fidelity fixes is the candidate); the
  allowlist `scripts/replication_known_violations.json` (26 entries) shrinks to Hamm's exceptions; each S1
  entry names why its path is not static (temp copies, run-time arguments, `C:/workspace/pcl_diag`); S1
  follows the `_repo_root()` walk to `pyproject.toml` (2026-10-28, five entries cleared). Closes when: F3 has
  a check that fails on a planted fix without a failing test, and each allowlist entry is fixed or ruled.
- T-10b [A] jnwb cross-learning (`control/harness/jnwb_cross_learning.md`): errors page, a proxy-claim
  wording gate, AST-hash freezing of validated kernels. Survey 2026-10-28 for the freeze: one kernel is
  checked bit for bit against an independent oracle (`simulate_eig_izhikevich`, net_m branch,
  `tests/test_izh_amendments.py::test_b2b_core_net_m_matches_oracle_bit_for_bit`); AdEx, HH, LIF and local
  inhibition use tolerances, and the AdEx drive oracle calls `adex_step` itself, so the freeze would cover
  one function (low value; recommend dropping it). jaxfne exports two error classes (`TFNEError`,
  `EpistemicRefusal`); its refusals are `ValueError`s named in messages, so an errors page needs a scope.
  Closes when: each lands with its test or is dropped.
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
  Estimator calls from the T-41 audit: `fleiss_kappa_binary` (`analysis/metrics.py:411-426`) scores observed
  agreement as unanimity and expected agreement pairwise, so independent trains give about -1 while its
  docstring says 0; GSDR's kappa comes from it (`run_gsdr.py:295`; CALIBRATION 6.6 calls the estimator the
  same as the authors'), so textbook Fleiss or the paper's form; `welch_numpy` (`analysis/spectral.py:205-231`)
  uses a symmetric Hann window while its docstring says the scipy convention (PSD up to 15.7% apart); bins
  off the dt grid are rounded silently in `fano_factor`, `fleiss_kappa_binary` and `metrics.py:1014` while
  `wang_buzsaki_kappa` refuses them; optimizer seeds collide (`optim/core.py:1658` fold_in index at
  population above 10; `:1698` simulation seeds `seed + gen*pop + row` shared by runs whose seeds differ by a
  multiple of pop). From T-55: may a cell type be absent when its band lower bound is 0 (`jdna/genome.py:744`);
  `population_rates` takes dt in seconds while `mean_rate_hz` and `fano_factor` take `dt_ms`
  (`analysis/metrics.py:1082`), a public rename or a documented exception; the named matrix masks read
  `W[post, pre]` transposed (`_model_tune.py:1118-1145`): `E_to_I` selects I-to-E entries and
  `excitatory_to_all` selects E target rows plus every row with mostly positive input, so all of W when I
  cells receive mostly excitatory input; Suite No. 4 pairs `gGABA_w` with
  `E_to_I` and its tests and notebook depend on it (E_to_E and `all` are unaffected; E2b builds its own
  masks), so fix the masks (outputs change) or rename them; the float32 PSD phase fix in T-55 (11) (outputs
  change at rounding level; 11.5% of peak at 1000 s without it). Each changes outputs. Closes when: each is ruled.
- T-51 [H] Protocol calls: P6 seeds and Q5 (`izhikevich2003/core/PREREG_P6.md`); Potjans and Vogels scale
  and bands; D2 Destexhe and GSDR preregs; CAMINOS S2 re-run; PCL F6 column and the WSL installs for the
  C++ binary; jitter reader follow-up (w_max, larger frozen sets). Closes when: each is ruled.
- T-52 [H] Paper calls (G-0): C10 cut 112 references to 80; C1 2 orphans and 7 manuscript-ledger conflicts
  (`programme/paper/claim_alignment_audit.md`); C6 benchmark protocol and a geometry for
  `cortical_column_localized_workflow.build_config` (raises at HEAD); C4 turn invariance (claim wording or a
  radius cutoff); D35 `apply_calibration`; D28 symbol (9A names). Closes when: each is ruled.
- T-53 [H] Repo calls: T-16 atlas test merge; T-17 the manuscript tree and five receipt-writing scripts;
  T-PR Dependabot #115, #116 and deleting the merged remote branches `caminos-p6`, `fix/rtd-mcc3-post-d1`,
  `izh2003`, `l2-caminos`, `pcl-port`; dashboard P1 and P2 after the P0 acceptance; GSDR's stage (REPLICATIONS
  reads P1 with `PORT_SPEC.md` and an unsigned `PREREG_P3.md` on disk; Destexhe in the same state reads P3);
  deleting the unselected `smoke`, `fast` and `integration` markers (T-37); the canonical metadata key set,
  the package's `claim_level` family or the docs' `model_status` family (T-35); the 0.5.9 release go.
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
