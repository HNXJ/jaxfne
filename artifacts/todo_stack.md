# Remaining work

Top = next. Prerequisites sit above dependents. `[A]` agent can do it, `[H]` needs Hamm. Done items
are deleted (history: git; the pre-2026-09-27 stack and the sealed 0.5.0–0.5.5 sections are under
`artifacts/archive/0.5.x/`). Rules: `artifacts/AGENTS.md`. Facts: `artifacts/fact_stack.md` (Hamm
edits). Open issues: `artifacts/assurance/issue_log/ISSUE_LOG.md`. Other editors change this file: re-read
first, amend in place.
After 0.5.8, rows serve G-0 (`artifacts/programme/design/plans/arxiv_paper_plan.md`): each names its F, S or C id.
G-0 findings from lanes L1-L9 (2026-10-25/26):
- [A] C9, C2: 16 gaps in `artifacts/programme/paper/reproducibility_audit.md` section 7. Agent rows: a figure
  lock file, a clean-checkout figure job; the `publication` job in `release_ci.yml` (lane L11) awaits its first
  CI run, and off macOS it compares figure dimensions only. Closed 2026-10-26: the v0.5.7
  version DOI is in `docs/reference/project/zenodo_doi.md`; the 0.5.7 date in the release authority file
  matches the GitHub Release (2026-10-21 UTC; Zenodo dates its record 2026-10-22). [H] rows: the LICENSE name
  "H N", the data and code statement, external data licences (PCL, CAMINOS), the F1-F8 to frozen fig01-fig07
  mapping, S16 generators.
- [H] C10: `docs/reference/references.md` lists 112 entries, 94 figure-linked, against the 80 cap
  (`artifacts/programme/paper/references_inventory.md`: plasticity 8 and fitting 7 short; other 20, fields 7,
  tools 4 over; JAX itself uncited; E054 repeats E004). The cut and the category map are Hamm's call.
- [A] C4, F2: localized sampling keeps its edge set under translation (lane L8); a turn can change a few edges
  (worker receipt: an 11x11 grid quarter turn changes 2 of 484) because the bucket neighbourhood is axis-aligned.
  Exact turn invariance needs a radius cutoff on candidates, a change to the candidate law: settle it with the C4
  claim wording. The 100k tutorial notebook and the w8/w10 baselines keep pre-L8 draws as historical receipts.
- [A] C3, S1: `docs/theory/doctrine/definition_map.md` lists 5 definitions with no symbol and 3 realized only by
  private names (lane L10 moved D9, D15, D16, D20, D22, D30 to the public operation that realizes them, each with a
  test that reads the map). Several candidates each: D17 (`tfne.System`, `ExplicitModel.boundaries`,
  `Realization.s["boundaries"]`), D26 (two emitters sites, `HDPRuleUpdate.dH`), D35 (`apply_calibration`,
  `CalibrationSpec`). No symbol: D28 (`ValueTag` lacks B, R, E, T). New public names proposed: D21
  `resolved_edge_weight`, D34 `rbd_recurrent_gain` (both `jaxfne/emitters.py`); D31 and D32 need a code path
  that exposes a_eff and the AdEx clip. Public API choices: orchestrator, then Hamm.
- [H] C6, S6: benchmark protocol choices (`artifacts/programme/benchmarks/README.md`); brian2 and NEST not installed.
  `scripts/cortical_column_localized_workflow.build_config`, which the Izhikevich benchmark calls, raises at HEAD: the
  builder refuses `geometry='uniform3d'` with `layers` and `ei_profile='canonical'`. The repair picks a geometry.

## 1. Decisions waiting on Hamm (they unblock items below)

Ruled 2026-10-20: release plan D1 (a), archive the unreferenced and ungated scripts with an index
(15 left); D3 approved. Open, recommendation first: D2 (Destexhe and GSDR preregs, after
their pilots); a state clamp on the plain Izhikevich path (T-19, changes outputs); extending the same-layer zeroing rule
to I/Inl/Ing (T-18 remainder, a model change); the manuscript tree and other archive moves (T-17).
Earlier rulings sit in the rows they
govern (T-22 semantics defaults, T-25) and in `artifacts/etudes/meta/REPLICATIONS.md` (Izhikevich
2003 and CAMINOS P6 move to 0.5.7); finished rows are in git history.

- T-3 [A] PCL REOPENED (Hamm 2026-10-23: replication improvement authorized). The signed v5
  T2 FAIL stays immutable historical evidence; its status as the final assessment is withdrawn
  and a NEW PCL replication attempt opens, in stages:
  PCL-R1 audit implementation fidelity (resolve the three fidelity-audit findings: local
  inhibition reading, STDP persistence across files, the authors' selected neuron indices);
  PCL-R2 reproduce the reference implementation on identical inputs (behavior and estimators);
  PCL-R3 diagnose residual differences (attribute to implementation, parameters, stochastic
  variation, or model assumptions); PCL-R4 run the corrected replication (freeze the NEW
  protocol before confirmatory execution — B8); PCL-R5 score: exact PASS, approximate
  agreement, or FAIL on predeclared criteria.
  Approximate reproduction: per observable, normalized discrepancy
  e_j = d_j(yhat_j, y_ref_j)/s_j with d_j a declared distance and s_j a preregistered
  normalization scale; TWO independent outcomes - exact reproduction (the original criteria)
  and approximate reproduction (separately declared scientific tolerances including
  uncertainty and stochastic variability); approximate agreement is never labeled an exact
  pass; if neither succeeds, retain the negative result and quantify the remaining
  discrepancy. Constraints: fidelity corrections before parameter tuning; no post-hoc
  relaxation of the original thresholds; B2c core integration (4a07072e) is a prerequisite,
  not evidence of paper reproduction; the dashboard shows the historical failure alongside
  the latest independently scored outcome. Priority: after B1 stage 2, before lower-priority
  work. (Historical state below.) R1 CLOSED - FIDELITY AUDIT (2026-10-24; receipt
  `artifacts/etudes/papers/pcl/r1_fidelity_audit_2026-10-24.md`, hash-pinned; NOT PCL REPRODUCED;
  the signed v5 T2 FAIL immutable). Six findings, attributions and design dependencies: F1 local
  inhibition - implementation (reference-faithful inhibition); F2 STDP persistence - implementation
  (explicit training-file continuation); F3 serialization - implementation (canonical serialization);
  F4 scored state is not weights-only - implementation (complete continuation state: Y =
  F(W0, C0, I; theta), matching W0 without C0 does not preserve Y); F5 the core scan cannot replay
  the real file - implementation (replay capability and adaptation semantics); F6 column selection -
  observation/selection fidelity, same-run DeltaY_selection = 422 with (0,0) entirely silent
  (reference-faithful observation selection; correcting selection alone is inferred, not
  established, to resolve T2). Next authorized stage: the READ-ONLY PCL-R2/R4 design study
  (two alternatives, the oracle state-variable mapping to X/H/W/B/K, a continuation-equivalence
  test, generalization beyond PCL; stop before runtime changes). Historical state (records in
  `papers/pcl/prereg/` and `LEDGER.md`): v4 FAIL stands (`pcl-port` 5640ce18: T1, T3 pass; T2, T4
  fail). v5 signed by Hamm 2026-10-13 (`prereg/P3_fig3_v5.md`, 89e45191); scored 2026-10-13 (8f75acd0,
  `prereg/P3_fig3_v5_result.md`): FAIL on T2 (simple 10/16 points, SSE 1.50 × flat; complex 12/15,
  SSE 0.52 × flat, guard 0.5). The S6/S7 diagnostic is unscored. The three fidelity findings of
  2026-10-13 became R1's F1, F2 and F6. Then P6 (B1). Closes when: a signed prereg reproduces Fig. 3,
  or Hamm closes it, and the P6 attempt is recorded.

## 2. Next release: 0.5.8 (core-compliant replications)

0.5.7 published 2026-10-21 (tag v0.5.7 on main 03a86014, GitHub Release, PyPI; receipt
`artifacts/release/v0_5_7_release_receipt.json`). Plan: `artifacts/etudes/meta/release_plan_0.5.7-0.5.10.md`.

- 0.5.8 candidate (2026-10-26): version 0.5.8, changelog section (21 bullets), canonical atlas at
  0.5.8, P-037 closed, O-41 closed (RTD latest and stable at 03a86014 serve all 25 étude files).
  Left [A]: broad gate 0 failed, push dev, CI green on the SHA, rc gate, second review, receipt
  `artifacts/release/v0_5_8_release_receipt.json`. Then [H]: tag, main, GitHub Release, PyPI.

- Jitter reader (Hamm 2026-10-22): protocol (a) PASS on all four runs (A1/A2, seeds 0,1; w_pat -> 2.0
  bound, above w_car in A2; receipts `cl-reader-cursor/reader_exp/out/protoA_*.json`); frozen readout weak
  (0.33-0.67). Readout follow-up (w_max, larger frozen sets) is Hamm's call; lane paused until then.

- Brunel S1 (finite size, N=6250, in-degrees fixed) needs the fixed-in-degree lane option (release plan B5);
  S2 and S3 are recorded in `RESULT_P3.md` (2026-10-25).
- Dashboard three-verdict distinction (Hamm 2026-10-23 P0 feedback, bundle
  `C:/Users/nejath/Downloads/JAXFNE_SOURCES_AND_DASHBOARD_REVIEW_2026-10-23/`): LANDED 06a44180 -
  verdict_summary (paper/library/methods) derived from records, core-method rows out of the paper
  denominators, data_revision + stale warning, failed criteria ride the row. P1 (estimator/units/
  seeds/prereg exposure) and P2 (commands/runtime/memory) queued behind the P0 acceptance.
- Kernel LFP lane `cl-klfp` [A]: f57aae0f landed on dev as 85199ade (0.5.8; the plasticity scorer's
  iSTDP checks read per-edge traces). 6eb3c6b8, 44cd0111, ff45364b (hybrid kernel LFP) go to 0.5.10.
  Then the validation backbone continues (iSTDP out-degree check in plasticity, kernel LFP checks in fields).
- Dashboard/docs adoption from jnwb (Hamm 2026-10-24, all four: [A], not 0.5.8-blocking, moves to 0.5.9
  if unfinished): (a) docs stills get generated `.dark.png` twins (`#only-light`/`#only-dark`); (b) JSON
  Schema with 40-hex SHA patterns for `replications.json`, validated in `build_dashboard_data.py
  --check`; (c) sha256 of each committed receipt on its evidence item; (d) a test that loads
  `dashboard/index.html` + `app.js` in a real page. Joint rulings: both dashboards use the jaxfne verdict
  set and publish on every dev push; style files stay per repo, aligned by convention. Tree budget
  (Hamm 2026-10-24: the repo stays well below 100 MB, read as 100,000,000 bytes, so a budget
  amendment for the twins is fine): record the measured delta in the amendment file.
- Interactive figure theme (Hamm ruled for jnwb 2026-10-24: one HTML per figure that follows the site
  theme, no light/dark twins): jaxfne's interactive panels are one HTML each but built with a fixed
  `plotly_dark` template (`jaxfne/vis/atlas_suite.py`, `network_inspect.py`, `tutorial_panels.py`,
  `scripts/generate_docs_visuals.py`, `generate_doc_page_atlases.py`). [A] make them follow the
  reader's theme (iframe reads `prefers-color-scheme` / the parent's `data-theme`); closes when a
  panel renders light and dark from the same file. Not 0.5.8-blocking.
- Docs figures leftovers (0.5.9; docs build hygiene, outside the 0.5.8 engineering scope): (1) stop the
  `docs/_static` HTML builds and delete those tracked files (frees tree budget); (2) the four allowlisted
  iframes in `build_docs_assets.py` get a figure-site build or a still.

In order (Hamm 2026-10-22: work until the 0.5.8 todos and open issues are empty; open issues become
todos once their close is decided; checkpoint when both are empty; Muse Spark 1.3 code packets in
clones, verified before integration):
1. [A] B4 PARTIAL (R4 8136beea, P4 d5b07498, caminos + brunel E1 33e66089; findings 38 -> 31, all
   known, 0 unlisted; B1, B2, B3, B5 CLOSED 2026-10-22/24, evidence in git and the handoff).
   Remaining, each removing its known finding: izhikevich2003 E1 is BLOCKED on a construct path
   for the P3b arm's heterogeneous per-cell parameters (dt=0.5 + noise, default order; Hamm:
   a core capability or an exception on the finding); the S2 set (4 oracle-import findings) is
   protocol-gated: pcl closes at PCL-R4's new prereg, caminos by re-running its scored P3 sweeps
   on the verified core route (Hamm's call).
   Constraints kept from the closed items: B3 numerical contract (spikes/RNG exact; V_m
   atol 1e-4 mV; bitwise V_m tests retained as platform diagnostics); the signed PCL T2 failure,
   CAMINOS mixed verdict and Brunel conditional verdict are never promoted by integration
   success; the dashboard's three-verdict separation stays invariant.
7. [A] B8 pilots then preregs: Destexhe and GSDR (the GSDR fit with MCDP on via `mcdp_hdp_step`);
   then each lane's P3 prereg for signature; Jansen-Rit 1995 opens the EEG/MEG lane (a neural-mass
   emitter in the core, then P1).
8. [A] B6 split closed lane folders by kind (tree degree) once their lanes close.
Open-stack rows (each closes its `O-n`; the open stack empties when these close; audits 2026-10-25):
- T-O61 cosmetic [A] (wording approved by Hamm 2026-10-25): tier labels differ, REPLICATIONS against dashboard.
- [A] Check whether the manifests hash `artifacts/protocol_w/w3b_parameter_domain/w3b_scan_run.log`; if none does, record it.
- T-O10 [external] jynx keeper takes or refuses each harness proposal (jchat conference #29).
- T-O58 [H] WSL DNS (system setting); check: `pip download jaxfne==0.5.7 --no-deps` works in WSL.
- T-O68 [H] local copies and the layout rule (see O-68): each held copy is deleted or sits at
  `E:/repos/<name>` or `E:/repos/_wt/<repo>/<task>`.
- Moved to 0.5.9 (scope freeze 2026-10-25; evidence rules unchanged, the PCL T2 failure and the CAMINOS mixed verdict stand): CAMINOS S2 re-run, PCL-R2 to R5 and the PCL-R4 prereg, the Destexhe, GSDR, Potjans and Vogels preregs; izhikevich2003 E1 closure (per-edge uniform weights and
  per-cell Gaussian noise on the construct path, then a P6-style library-path prereg Hamm signs; Hamm 2026-10-25);
  the kernel LFP commits of `cl-klfp` (0.5.10). In 0.5.8: per-cell `cell_params`, the Vogels per-synapse fix.
- [H] Signatures: the 0.5.8 preregs from item 7 (Izhikevich v7 and v8.2 and CAMINOS P6 v2 are signed, 2026-10-13).
Blocked: T-11 (jnwb), T-16 and T-17 `[H]`, deferred list (section 5).

## 2a. Standing from 0.5.6 (Hamm 2026-10-12: cleaning, organizing, docs, dashboard)

- R2 restated (Hamm 2026-10-14): bytes -10% against the 0.5.5 tag (met: 86.8 MB
  tracked at 1bfdaa84 against 126.0 MB at the tag) plus files at the recorded budget with
  reasons on record (`budget_files` and `budget_bytes` in `scripts/tree_size_allowlist.json`; each
  raise is in `artifacts/release/budget_amendment_2026-10-23.md` or its commit). The regenerable
  docs assets (the 9 registry `copies` dests, atlas and étude HTML) are build products of
  `scripts/build_docs_assets.py --build` and gitignored. No further class qualifies: the gallery
  and tutorial-figure PNGs and every other tracked `docs/assets/` file are outside the build path,
  so untracking them would break embeds.
- T-8 [A] Each new score: `dashboard/sources.json` (lane SHA, stage artifacts), regenerate,
  push dev. Standing item. Closes when: `python scripts/build_dashboard_data.py --check` passes
  on each push.
- T-8b [A] After each dev push, refresh the lane dashboard artifact (skill `jaxfne-dashboard`).
- 0.5.10 candidate: `coherence_radius_um` readout in `jaxfne.fields.kernel_lfp` (currents sum in
  phase within R_c, random phase between patches), with the two limit tests (R_c = sheet radius
  gives the coherent sheet; R_c of one cell gives incoherent summation). After the kernel LFP lane.

## 3. 0.5.6–0.5.10 programme

Goals in plain words (0.5.x): a faster jaxfne that runs the full Atlas, from one neuron to twenty
areas, and a manuscript built on it; old results change only under an authorized correctness
repair; new abilities are switched on explicitly; every claim has evidence. Analysis and
visualization move to jnwb wherever jnwb has the operation (human, 2026-09-29), additively, with
existing outputs bit-identical; jaxfne tracks the latest jnwb (extra `jnwb>=0.2.9`).

Rules B1–B7 (adopted 2026-10-11):
- [A] B1 no paper counts as replicated without a recorded library-path attempt (index row).
- [A] B2 negative verdicts go in the claim ledger with their reason (one row per criterion).
- [H] B3 one emitter per release, authorized by Hamm, gated by a P4 test perturbing every
  accepted parameter.
- [A] B4 narrowing, one class per commit with its gate; generated assets regenerate at the docs
  build (plan `artifacts/etudes/meta/plans/narrowing_plan.md`).
- [A] B5 GPU: one job at a time, claimed in jchat. B6 each accepted binding becomes a row here the
  same session. B7 calibrate on synthetic nulls and known positives before a signature.
- [A] B8 pilot-then-sign (Hamm 2026-10-18): a lane runs its full pipeline on unscored pilot seeds
  and a known positive before the prereg is signed, and the prereg declares its sensitivity set;
  after signature at most one amendment, any further change is a new protocol version scored on
  fresh seeds.

Decisions in force (2026-09-23…2026-10-11, Atlas caveats AT-10-R5, AT-10-R6, AT-05-R2,
AT-01-R6/AT-00-R4): `artifacts/archive/0.5.x/decisions_and_caveats_2026-10-12.md`. Newest: T1
option A (meeting 3790d96e), manuscript sections 1–7 freeze on 0.5.5 evidence, 0.5.6–0.5.10 fill
section 8 citing a replication only at a signed-prereg verdict; JAX runs above about 1000 neurons
run on WSL2 CUDA.

- T-32 [A] Reproduction throughput (Hamm 2026-10-18: "we should be reproducing a lot more").
  Target (Hamm 2026-10-19): close about one replication per day, a scored verdict in any class
  on the dashboard; the dashboard and REPLICATIONS update as each lands.
  Inspection the same day (three read-only sweeps at 74acf7e0): 7 lanes over 12 days, one
  reproduced; the queue holds no unstarted paper; about 40–50% of lane commits are protocol
  paperwork (CAMINOS prereg v1–v6 + v7, Izhikevich v7 to v8.2); every run loop is serial (GSDR
  500 generations 58,552 s, 24 serial HH calls per generation; PCL about 55 min per clip on one
  core); 4 of 28 checklist rules are scripted. Landed: (T-32a) lane kit `scripts/lane_kit.py`
  and `scripts/new_lane.py`, destexhe2009 migrated; (T-32b) `simulate_hh_batch`; MCDP in the core
  (O-42). Work, in order: the GSDR fit on `simulate_hh_batch` and `gsdr_mcdp_transform`; (T-32c)
  seed and candidate batching for the AdEx and LIF routes and `Model.tune` (serial loop
  `_model_tune.py:451`); (T-32d) core homes for lane-local analysis (ISI ratio and CV, aperiodic
  fit, tuning fit); (T-32e) script checklist rules E1, S2, F3, P4, R4 in
  `check_replications.py`; (T-32f) Brunel 2000, Vogels et al. 2011, Potjans and Diesmann 2014
  and Mejias et al. 2016 hold B8 pilots on dev (Brunel Fig. 8 A-D at full scale: rates and
  global frequencies within 10% at B-D, CALIBRATION); next, each lane's P3 prereg for Hamm's
  signature (Brunel P3 v4 scored 2026-10-20: reproduced (conditional), `RESULT_P3.md`; its S1-S3 sensitivity set waits on release plan B5), and the core gaps the lanes name: Potjans G-MULT, G-TAUSYN, G-POPDRIVE, G-WDIST,
  G-DDIST; Vogels G-SC; Mejias G6 (constants unknown) and the phi(0)=1 transfer
  normalization; the LIF edge kernel on the GPU (13.4 GiB at 25000 steps x 12500 cells, out of
  memory beside the desktop; about 110 ms per step at dt=0.5 against about 50 on the CPU, so the
  Brunel dt=0.1 runs use the CPU). Each lane follows B8.
- T-33 [A] Hamm's rulings of 2026-10-18, still to carry out (records, API refusals, tree
  degree, tooling and the foil sweep landed on dev 2026-10-19):
  - GSDR and Destexhe 2009: B8 pilot, then signature, then fresh seeds; Destexhe's reserved
    seeds 4–8 are spent, so new reserved seeds.
  - PCL: Fig. 3 rescored on the core (full-network mapping of feed, complex pooling and
    lateral/top-down paths onto the LIF tensor route; the numba port stays the event-exact
    oracle); the next core protocol declares T4 up front.
- T-10b [A] jnwb cross-learning follow-ups (T-10 closed 2026-10-21: `release_body.py --tag
  v0.5.7 --check` passed and built the v0.5.7 release body): the errors page (the docs newline check holds: `docs/reference/changelog.md` and
  `docs/_static/atlas/manifest.json` are all CRLF at HEAD, 745 and 136 lines, 2026-10-20), a
  proxy-claim wording gate, AST-hash freezing of validated kernels
  (`artifacts/control/harness/jnwb_cross_learning.md`).
- T-11 [A] G-3: CAMINOS p = 0.07 seed 1 as an NWB excerpt plus provenance for jnwb `tests/`
  (waits on jnwb to name its tests/ path). Closes when: the excerpt's recovery test passes in jnwb `tests/`.
- T-16 [A] Merge of the `test_atlas_at*_05x` files not done (score about 60, 2026-10-17): it renames
  every test id path, which costs the JUnit parity comparison against earlier CI runs for fewer
  files. Protocol_e parametrization and the CI dedupe (`run_test_gate.py` examples and
  wheel-smoke gates) landed 2026-10-17. Closes when: Hamm keeps the files or a parity-aware merge lands.
- T-17 [H] P3 audit, remaining owner calls: the manuscript tree and other archive moves. Four
  unreferenced scripts were archived 2026-10-13 (table in `artifacts/README.md`). Five more have
  no reference but write tracked or frozen receipts that docs cite, so they stay:
  `hdp_k_w_ctrl_sweep.py`, `p2v_a1b_dynamic_search.py`, `p2v_a1b_prefreeze_evidence.py`,
  `p2v_a3_hdp_boundedness.py` (`p2v_a4_claim_language.py` archived 2026-10-20 under D1 (a)).
  Closes when: each has Hamm's call.
- T-19 [A] Deep audit and code critique (Muse, 2026-10-13, at aef3129e). Landed: 0df3da9d,
  68c61e92, 10d80c08 (NaN metrics, refusals, narrowed excepts), 5090af24 and c4dc0922 (D1 noise
  sqrt(dt_ref/dt)), and the `compute_fields` model check plus the last emitters.py broad except
  (this commit's successor of 73165ced), and the differentiable gAMPA / synaptic_gain refusal on
  edge-list models (fc6be3a1). Open:
  - Design (no wrong result shown): HDP kernel with about 78 parameters
    (`emitters.py:3155-3232`); duplicated scan bodies behind record flags (`emitters.py:1412-1474`);
    return arity that depends on a flag (`_model_simulate.py:381-383`, `simulate(return_state=)`);
    order-dependent construction stages (`_construct_core.py:292-293,1392-1401`); drive defaults
    in two places (`_config.py:1620-1626`, `emitters.py:60-65`); no state clamp on the plain
    Izhikevich path (`emitters.py:522-523`); `bridges.py` patches `jnp.clip` process-wide;
    frozen dataclasses mutated through `object.__setattr__`; mixed `_s`/`_ms` parameters in
    the HDP controller.
  Recorded 2026-10-20 (release plan A5), no change: the `bridges.py` clip shim is installed only
  when jaxley needs it, reversible (`uninstall_jax_clip_compat`) and reported in the runtime
  report; `object.__setattr__` sits in `__post_init__` normalization of frozen dataclasses, the
  standard idiom; the `_s`/`_ms` names and the `simulate(return_state=)` arity are public API
  (documented, A2); the HDP kernel size, the duplicated scan bodies and the construction order
  wait for a measured need (deferred item 3). Drive defaults read one table (A5, 2026-10-20).
  Open: a state clamp on the plain Izhikevich path changes outputs, so it is Hamm's call (section 1).
  Closes when: each item is fixed with a test or recorded here with Hamm's call.

hyx standardization, Hamm's rulings 2026-10-14 (form (h, y) = f(x, h') on A):
- T-22 [A] Same-step interactions (PCL local inhibition, CAMINOS zero delay, gap G12) are a
  declared intra-step fixed point: f applies R ordered rounds, with R and the order part of A and
  recorded in the manifest; delay 0 means "resolved inside the step". Closes when: the operator is
  in the core with tests, PCL B3 uses it, and the CAMINOS 0-delay sensitivity runs on it. Phase 1
  (AdEx zero-delay edges, `connections(intra_step_rounds=R)`) landed 2026-10-18. Semantics
  (Hamm 2026-10-18, O-35 defaults confirmed): (1) the declaration sits on the Configuration
  connection for the AdEx route, PCL keeps `NeuronalTensor` `local_inhibition`; (2) `event_scan`
  stays out of the operator; (3) iterate to quiescence with R as a cap that refuses when hit
  (Hamm 2026-10-18, reversing "fixed R": a fixed R dropped the delay-0 output of the last
  round's new spikers, so CAMINOS R = 2 and R = 4 diverged at step 29596); (4) round rule:
  OR-monotone re-step from the pre-step state. Phase 2 (Poisson drive delay 0 with `"same_step": True`;
  brian2 and the registered-HDP kernel refuse it) landed 2026-10-18. CAMINOS on the core
  (5235f807): a 0-step delay passes `same_step` and `intra_step_rounds` (R = 2 from a probe,
  CAMINOS CALIBRATION section 11); the R = 2 seeds 1-8 are void, and the quiescence rerun
  (cap 64, 3 rounds used on every seed) matches the 0.1 ms sensitivity: group lag means 6.92-7.18 ms, f_pk
  43-46 Hz (section 11 results). PCL B3
  (Muse sweep 2026-10-18, checked): its rule is an inhibitory depth-ordered cascade (a winner
  vetoes later candidates), already in the core as the LIF route's
  `local_inhibition_resolution="event_order"` (`emitters_lif.py` `_cascade_step`); the AdEx
  rounds only add excitation. What keeps B3 off the core is the full-network mapping (cover-table
  feed, complex pooling, lateral and top-down; `run_tuning.py` CORE_GAP), which is PCL lane work
  (O-24). The CAMINOS zero-delay runs are scored (CALIBRATION section 11: lag 6.92-7.18 ms, f_pk
  43-46 Hz, as the 0.1 ms sensitivity; R = 4 against R = 2 is superseded by the quiescence cap).
  Left: PCL B3 on the operator, with the PCL core mapping (release plan B2).
- T-25 [A] Plastic weights: W is state in h during a run; A holds W₀ and its ownership. A trained
  network is A' = A with W₀ ← W_T, recorded with provenance. Closes when: the tensor carries W₀
  provenance and PCL's trained weights load as A'. Landed 2026-10-18: `W0Provenance`,
  `with_trained_weights` (izhikevich route; lif/adex refused). Left: PCL's trained weights come
  from the numba port, which has no tensor bridge; they load as A' once PCL runs on the core
  (T-30, O-24).
- GSDR (lane 4f, on dev 2026-10-17 under `etudes/papers/gsdr2026/`): the saturated regime
  (κ ≈ 0.95 against 0.0156) survived three tested suspicions, CALIBRATION §6.4 to §6.6:
  inhibition ×1 to ×16 (only slows 21 to 13.5 Hz), OU noise ×1 to ×4, and the κ estimator
  (identical on both sides). §6.7 structural sweeps: saturation persists with the stimulus off
  and across 0.25 to 2× overall gain; only AMPA 0 breaks it. §6.8–6.9: the 5.2 Hz, κ 0.0
  outlier is an early collapse to silence, GPU-only (3 of 30 calls; inferred cause: segment_sum
  order), rare on CPU (2 of 40 at a 1e-2 mV kick, 0 of 40 noise seeds). §6.10: wiring matches;
  the one-compartment reduction S1 removes the reference's axial bottleneck (inferred ~1e4×
  more drive). Hamm 2026-10-17: the 4-compartment layout is deliberate; keep it (S1 retired).
  The multi-compartment route landed (jaxley 0.14.0 oracle within 8.3e-12 mV; gates start at
  0.2 as in the reference; `single` bit-identical). §6.11: at the one-compartment g* the
  reference cell fires faster (70.7 Hz, κ_fit 0.61), so g* says little about it. §6.12: the
  500-generation refit on `--morphology reference` stays saturated (best loss carried by the
  rate penalty; at its g* every E cell fires every cycle near 120 Hz, κ 0.60, on fit seed 1 and
  held-out seeds 4–8). §6.13 (isolating the source of κ, sol's review 2026-10-18): with AMPA ×0
  cells still fire near 20 Hz (y 2.14), and with the stimulus off the network locks at 45 Hz;
  with every synapse removed the OU background alone drives the noisy half of the cells near
  40 Hz (the authors' sigma draw, ported as written). Reading: the authors' loss bounds rates at
  5–50 Hz and does not target y's 2.14 Hz, so the rate estimand of the scored steps is
  `PREREG_P3.md`'s call (O-25). CPU and GPU agree on all 10 sweep trials (§6.13). The κ source
  is isolated as far as the port can go without the estimand ruling (O-25). The scored steps run under `PREREG_P3.md`, still a draft:
  Hamm signs it before a scored verdict (O-25). HH hardening landed; open low notes: `held_steps` counts edge elements while the
  docstring says steps, int32 count scalars, no full-kernel grad test, `dt="0.1"` coerced.
- T-30 [A] Replication conformance to the checklist (`etudes/meta/RESIDUAL_PROTOCOL.md`, section
  "Replication checklist"; audit 2026-10-17), per paper:
  - PCL Fig. 3: scored tuning runs on the numba oracle (`papers/pcl/port/run_tuning.py:85-94`,
    CORE_GAP) and `pcl_event_ref*` transcriptions drive scored results (E1, S2). Next: the core
    cascade through T-22, then rescore on the core, or Hamm's exception (O-24).
  - Izhikevich 2003: scored runs call `simulate_eig_izhikevich` directly and use an étude-local
    `kernel_net_m.py`; the `net_reference.py` transcription is a scored arm (E1, E2, S2). Next:
    `net.m` update order as a core option, runs through `construct`, the transcription in tests.
  - CAMINOS: P3 sweeps scored on the `caminos_jax.py` oracle (P6 v2 is core-compliant);
    `ref42_phase_delay` and `analyze_run` are étude-local (E3, S2). Next: estimators into
    `jaxfne.analysis`; P3 claims cite the P6 route.
  - GSDR: `run_gsdr.py` builds through `construct_hh_model`, outside `construct()` dispatch; local
    `detect_spikes`, rates, PSD and κ (E1, E3). Next: HH in `construct()` dispatch, estimators
    into `jaxfne.analysis`.
  - Destexhe 2009: core route; local `cv_isi`, `pairwise_cc`, `mean_rate_hz` (E3); no LEDGER or
    RESULT yet (R1).
  - pcl_h and Q-learning: core-compliant; no phase records (R1); pcl_h imports sibling-étude
    modules through `sys.path` and is not on the dashboard.
  - Records: SOURCE_P1 for CAMINOS and PCL are drafts (2026-10-18), Hamm or the keeper verifies
    them against the articles; `dashboard/sources.json` `stage_artifacts`
    cite pre-reorg paths; the dashboard shows GSDR at P1 with P3 records present; `hh-at01` has
    no folder.
  - Tree caps: `papers/pcl/`, `caminos/`, `izhikevich2003/` over 7 (section 5).
  Gate: `scripts/check_replications.py --check` (families replication_audit, etude_no_resim, run
  by the dev gate too) against `scripts/replication_known_violations.json` (31 findings,
  2026-10-25); a fix removes its entry. Closes when: every paper passes the checker without a listed violation, or each listed one
  carries Hamm's exception.
- CAMINOS lag (6.47 against 8.5 ms): Hamm attributes the gap to the paper's estimator rather than
  the model; class stays verified, the lag stays a recorded finding, no fit.
- Frozen, do not move: `artifacts/publication/frozen_manifest.json`,
  `artifacts/release/current_release_authorities.json`, hash-pinned skills and `AGENTS.md`;
  `artifacts/legacy/AGENTS.md` kept by Hamm 2026-10-11.

## 4. Directions recorded by Hamm, open

- Augmentation science runs (2026-10-01, `artifacts/programme/design/plans/augment_science_runs_plan.md`): an
  opt-in "balanced" preset with a measured asynchronous-irregular regime, canonical 1000n column
  bit-identical; runs 1k → 2k → 5k, then a 10k decision.
- PCL: `pcl_stdp_h2` for new work, `pcl_stdp` stays default for K1 and the replication; a
  learned-tuning A1 test needs a declared criterion (`artifacts/etudes/papers/pcl/README.md`).
  Lane 2 (tensor route): Stage C passed 2026-10-15, 20 of 20 clips exact (receipt
  `artifacts/etudes/papers/pcl/stage_c_tensor_route_20clip.json`; CALIBRATION.md Steps 8–9). Open in the
  record: the `no construct route (OPEN)` marker (CALIBRATION.md:24) is not marked closed; Step 9
  lists no next items. T-3 (Fig. 3) is separate and unchanged.
- Structural HDP (2026-09-29, `artifacts/programme/design/plans/structural_hdp_design.md`): existence over a
  declared candidate set, opt-in; API and hard-vs-graded existence open.
- Primate primary-connection mode (2026-09-29): label FF/FB defaults uncalibrated; a macaque-only
  evidence table (direct / anatomical-only / functional / unknown, cited), SLN in [0,1],
  heavy-tailed FLN with a distance rule; unknown edges refused.
- Theory programme (2026-10-08, `artifacts/programme/design/plans/theory_programme_proposal.md`): next family
  F3, F4 or F5 is Hamm's pick; T3 Lean proofs of L1 and L3 need Lean.
- Python 3.15 in CI and the classifier once 3.15.0 ships and pyyaml, h5py and zarr publish cp315
  wheels.

## 5. Deferred, each with its trigger

- (from the open stack, 2026-10-18) Over-7 lane folders excepted in
  `tests/test_artifacts_tree_degree.py` while their lanes are open: `etudes/papers/caminos/`,
  `etudes/papers/izhikevich2003/`. Trigger: a lane closes; then split by kind or give the
  exception a path-addressing reason.
1. AT-10-R6: RSS growth (+80–98 MB over 4 chunks of the 20-area run): resolve against the XLA cache
   before any run over 100 s (0.5.8 laminar models on GPU). WSL2 CUDA 20-area receipts
   (`scratch/wsl_cuda_2026-10-09/`, gitignored): one unexplained failed GPU run. Closes when:
   RSS is flat over chunks with the XLA cache cleared.
2. A8 gh-pages policy for D1 or Atlas figures (the dashboard is at `/dashboard/` only).
3. Architecture split (`emitters/`, `tfne.py`; `artifacts/audit/reviews/deep_review_89_2026-10-04.md` §5)
   only as a measured bottleneck.
4. C5–C7 numerical deferrals: need a signed-zero/NaN exactness contract.
5. UNTESTED-exact refusal tail; artifact-gated skips (O-67); post-0.4.14 aliases.
6. S27 population definition family, when CTX-01 cannot express one.
7. Agent-native step 10: MCP or another transport.
9. From 0.5.4: item 1b (S20.1 ordering, needs Hamm) and item 0 (`X[k]` frontier override).
10. 2026-09-20 audit owner calls: `compile_step_fn **hdp_kwargs` unknown keys;
    `validate_hdp_params` non-dict pass; frozen protocols recording JAX versions (P-003).
11. Field Approximation Atlas ladder (deep review 89 §7).
12. Rule proposal (deep review 89 §12): new mechanism ⇒ reference model, discriminator and
    reduction test; adoption is Hamm's.
13. Stored-only fields left after the 2026-10-25 audit (cosmetic): `LaminarColumnConfig.cell_colors`,
    `cell_signs`, `cy_m` (tutorial scaffold), `HDPRuleContext.exc_mask` (context offered to rules; built-in
    rules do not read it), `SanityDeltaConfig` validation by `assert` (stripped under `python -O`; the
    existing test expects `AssertionError`). The `physical_amplitude_calibrated=True` overwrite is the
    truth-gate clamp, tested as intended. Trigger: a caller needs one of them.

Tracks (standing directions): P performance (optimize profiled components only) · T testing
(cheapest gate per defect class) · S skills · I integration (one implementation, many entrances) ·
A architecture (factor only with measured benefit) · V validation (configured → realized →
executed → observed) · U inspection (provenance queries).
