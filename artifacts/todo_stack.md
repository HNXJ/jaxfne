# Remaining work

Top = next. Prerequisites sit above dependents. `[A]` agent can do it, `[H]` needs Hamm. Done items
are deleted (history: git; the pre-2026-09-27 stack and the sealed 0.5.0–0.5.5 sections are under
`artifacts/archive/0.5.x/`). Rules: `artifacts/AGENTS.md`. Facts: `artifacts/fact_stack.md` (Hamm
edits). Open issues: `artifacts/assurance/issue_log/ISSUE_LOG.md`. Other editors change this file: re-read
first, amend in place.

## 1. Decisions waiting on Hamm (they unblock items below)

Open: the manuscript tree and other archive moves (T-17). Decided 2026-10-13: T-12, T-13,
T-14, T-17 (calls in each item). Decided 2026-10-14: hyx standardization (T-22–T-25),
Destexhe Vpeak, GSDR diagnosis, CAMINOS lag (§3); T-1 (a) landed in e3a7a41b; T-2
bundle; Izhikevich 2003 and CAMINOS P6 move to 0.5.7 (reasons in
`artifacts/etudes/meta/REPLICATIONS.md`).

- T-3 [A] PCL Fig. 3 stays in progress (Hamm 2026-10-13: it must replicate). v4 FAIL stands
  (`pcl-port` 5640ce18: T1, T3 pass; T2, T4 fail). v5 signed by Hamm 2026-10-13 on `pcl-port`
  89e45191 (`prereg/P3_fig3_v5.md`; signature recorded in 640cb97c): T2 runs the authors'
  `plot_fqcy` procedure on the port and scores it per K/k point against Fig. 3c,d digitized
  (`reference/fig3_freq_digitized.json`, ε 0.04) with a shape guard; A1 even phases; T4 drops
  the sign requirement, and its outcome is known from v4. v5 scored 2026-10-13 (`pcl-port`
  8f75acd0, `prereg/P3_fig3_v5_result.md`): FAIL on T2. Simple 10/16 points, SSE 1.50 × flat
  (higher off-peak response at K/k 0.6–0.8 and 1.3); complex 12/15, SSE 0.52 × flat (guard 0.5;
  passes with all 10 phases, not scored). S6/S7 diagnostic (unscored, `prereg/
  P3_fig3_diagnostic.md`): simple fails under both; complex passes with the 62.5 ms window
  (0.48); measured periods follow (20 − i)/2 px, supporting A2. Hamm 2026-10-13: investigate port
  fidelity, port-side only (never run the authors' code). Fidelity audit (read-only): (1) the
  C++ runs cSTDP3 without local inhibition ("static" is not "local", SpikingNetwork.cpp:351);
  the port reads "static" as local (declared alias, untested); (2) the C++ keeps STDP on and
  weights across files during tuning, the port freezes and reloads; (3) the authors' writer
  reads neuron files 0–63, which by index is column (0,0), against the port's (4,4)
  (unresolved: (0,0) lies outside the stimulated band). In flight: an unscored tuning run with
  local inhibition off (`F:\warehouse\pcl\diag_runs\`). Then P6 (B1). Closes when: a signed prereg reproduces Fig. 3, or Hamm
  closes it, and the P6 attempt is recorded.

## 2. 0.5.6 release (G-5; Hamm 2026-10-12: cleaning, organizing, docs, dashboard)

- R2 restated (Hamm 2026-10-14): bytes -10% against the 0.5.5 tag (met: 86.8 MB
  tracked at 1bfdaa84 against 126.0 MB at the tag) plus files at the recorded budget with
  reasons on record (2365 tracked, `budget_files` 2365 in
  `scripts/tree_size_allowlist.json`; each raise from 2269, its reason and authorization are in
  this line's git history; last raise +11 for the PCL tensor-route lane: 4 jaxfne modules, 5
  tests, the PCL calibration record and its Stage C receipt, none regenerable). The
  regenerable docs assets are untracked already: the 9 registry `copies` dests under
  `docs/assets/etudes/` and all atlas/etude HTML are build products of
  `scripts/build_docs_assets.py --build` (the Read the Docs `pre_build` path) and gitignored.
  No further class qualifies: the 9 `docs/_static/gallery/*.png` (embedded by `docs/gallery.md`,
  rewritten only as an index by `generate_gallery.py --write`, never rasterized by the build
  path), the 12 `docs/_static/tutorial_figures/*.png` (no caller in the build path), and every
  other tracked `docs/assets/` file (its generator, if one exists, is not in the build path;
  `--html-only` skips PNG rasterization) would leave broken embeds if untracked, so all stay
  tracked.
- T-7 [A] R5: every paper row in `REPLICATIONS.md` carries a verdict or a reason (2026-10-13).
  Closes when: `python scripts/build_dashboard_data.py --check` passes on the pushed SHA.
- T-8 [A] Each new score: `dashboard/sources.json` (lane SHA, stage artifacts), regenerate,
  push dev. Standing item. Closes when: `python scripts/build_dashboard_data.py --check` passes
  on each push.
- T-26 [A] `scripts/check_junit_parity.py` downloads CI JUnit into
  `artifacts/attestations/ci-junit/` without clearing it, so a rerun on a new SHA fails with
  "file exists" from `gh run download` (seen 2026-10-14 on 65af8541; it fails closed, never
  mixes SHAs). It also needs `gh` on the gate host (WSL has none; the 0.5.6 gate used a
  `gh.exe` shim with `GH_REPO`). Closes when: the script clears its dest per SHA, with a test.
- T-28 [H] `jaxfne/tfne.py:211` `_RESERVED_LOWER = {"x", "y", "h", "s"}` is defined and never read
  (Muse cleanup, 2026-10-15). Wiring it into name validation changes which names are accepted;
  deleting it drops the stated reservation. Closes when: Hamm picks one.

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
  build (T-4; plan `artifacts/etudes/meta/narrowing_plan.md`).
- [A] B5 GPU: one job at a time, claimed in jchat. B6 each accepted binding becomes a row here the
  same session. B7 calibrate on synthetic nulls and known positives before a signature.

Decisions in force (2026-09-23…2026-10-11, Atlas caveats AT-10-R5, AT-10-R6, AT-05-R2,
AT-01-R6/AT-00-R4): `artifacts/archive/0.5.x/decisions_and_caveats_2026-10-12.md`. Newest: T1
option A (meeting 3790d96e), manuscript sections 1–7 freeze on 0.5.5 evidence, 0.5.6–0.5.10 fill
section 8 citing a replication only at a signed-prereg verdict; JAX runs above about 1000 neurons
run on WSL2 CUDA.

- T-10 [A] G-1 (meet 204dfdaa): port jnwb's `changelog.d`, `assemble_changelog.py`,
  `release_body.py` and claim check; release body as tables. Closes when: `release_body.py --tag <next>` passes.
  Then one newline policy for docs (check: `docs/changelog.md` has 0 LF-only lines), the errors
  page, a proxy-claim wording gate, AST-hash freezing of validated kernels
  (`artifacts/control/harness/jnwb_cross_learning.md`).
- T-11 [A] G-3: CAMINOS p = 0.07 seed 1 as an NWB excerpt plus provenance for jnwb `tests/`
  (waits on jnwb to name its tests/ path). Closes when: the excerpt's recovery test passes in jnwb `tests/`.
- T-12 [A] `cross_area_coherence` Welch path (the periodogram fallback is refused since
  Hamm's call, 2026-10-13): `nperseg = min(1024, T)` gives one segment for T < 1536 samples, so
  coherence is 1 at the native bins there too. Independent white noise, dt 0.5 ms
  (`C:/workspace/pytest_temp/coh_single_segment.py`): median coherence 0.81 at T = 1000, 0.40 at
  1536 (2 segments), 0.22 at 2048, 0.10 at 4000. AT-09 (`artifacts/atlas/at08_at09_054.py`, 1000 ms,
  T = 2000) reads it in 8–25 Hz with about 2 segments. Hamm 2026-10-13: shrink nperseg so Welch
  has at least 8 half-overlapping segments, refuse when that leaves nperseg < 64. Landed with
  tests; AT-09 C_12 falls from 0.50/0.45/0.45 to 0.105/0.110/0.111 (noise level, no arm
  difference), recorded in `artifacts/programme/receipts/engine_053_054/composition_054.md`; the AT-09 verdict stands.
  Closes when: the change is on dev with the broad gate green.
- T-16 [A] C7: parametrize the protocol_e spec tests; merge the 8 `test_atlas_at*_05x` files
  after a per-test coverage run. CI dedupe (examples list and wheel smoke in three places).
  Closes when: `python scripts/run_test_gate.py broad` passes with coverage unchanged.
- T-17 [H] P3 audit, remaining owner calls: the manuscript tree and other archive moves. Four
  unreferenced scripts were archived 2026-10-13 (table in `artifacts/README.md`). Five more have
  no reference but write tracked or frozen receipts that docs cite, so they stay:
  `hdp_k_w_ctrl_sweep.py`, `p2v_a1b_dynamic_search.py`, `p2v_a1b_prefreeze_evidence.py`,
  `p2v_a3_hdp_boundedness.py`, `p2v_a4_claim_language.py`. Closes when: each has Hamm's call.
- T-18 [A] Inhibitory-type lists that omit `I`, `Inl`, `Ing`, `SOM` (F6 review, 57f40ddf):
  `tutorial_utils.py:1581` (`simulate_laminar_trials` zeroes same-layer input only onto
  PV/SST/VIP, a model rule; applied at :1587), `tutorial_utils.py:813` (default `cell_signs`), `util.py:712`
  (diagnostic grouping). Callers checked (b0a12a36): every caller of `make_laminar_column_config`
  and `simulate_laminar_trials` (4 notebooks, `tests/test_tutorial_utils.py`, `tune_laminar_agsdr`)
  passes only E/PV/SST/VIP, so no realized run is affected at b0a12a36. A change to 1581/1587 is a model
  change for Hamm. Closes when: each site reads the emitter table's sign or carries a stated scope.
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
  Closes when: each item is fixed with a test or recorded here with Hamm's call.
- T-20 [A] Lint leftovers (2026-10-14). F841 suspects, a value computed and unused while a panel
  or check hard-codes it: `scripts/audit_methods_draft.py:126,132,147,151`,
  `scripts/evidence_figures/ed05_manifest_hashes.py:223`,
  `scripts/internal/publication/fig08_e2_ping.py:190`, `fig09_e2_ssa.py:180`,
  `scripts/run_delta_notebook_01.py:277`, `artifacts/archive/scripts/make_delta_test_01_report.py:470`
  (and its placeholder pass 0.95 at :484). `--check` parsed and ignored in
  `scripts/release/assert_release_freeze.py` and `scripts/sync_release_metadata.py` (both
  check-only). Closes when: each is used, removed or ruled harmless.

hyx standardization, Hamm's rulings 2026-10-14 (form (h, y) = f(x, h') on A):
- T-22 [A] Same-step interactions (PCL local inhibition, CAMINOS zero delay, gap G12) are a
  declared intra-step fixed point: f applies R ordered rounds, with R and the order part of A and
  recorded in the manifest; delay 0 means "resolved inside the step". Closes when: the operator is
  in the core with tests, PCL B3 uses it, and the CAMINOS 0-delay sensitivity runs on it.
- T-23 [A] One regime classifier on y in the core, reported in every manifest and usable as a fit
  constraint. Labels per population over the kept window, defaults overridable per study: silent
  (rate < 0.1 Hz or active fraction < 5 %); saturated (active fraction > 95 % and κ > 0.5, or rate
  > 100 Hz); runaway (rate rises > 2× between window halves); asynchronous (κ < 0.1, CV_ISI
  0.7–1.5); oscillatory (spectral peak excess above a declared floor). κ is the Wang–Buzsáki
  coherence on binary spike counts in declared bins (default 5 ms; Hamm 2026-10-14), distinct from
  the dt-resolution Pearson κ in `tutorial_utils.kappa_synchrony`. Seeds in a different regime
  are reported, never dropped; a mixed-regime result is a finding. Closes when: the classifier and
  its tests are on dev and the replication scorers report it.
- T-24 [A] Constant drive (bias current, external Poisson rate) is part of x: a declared input
  source with a schedule (constant is a special case); the emitter `drive` field becomes a
  shorthand that compiles to an x source, existing numerics unchanged. Closes when: both routes
  compile drive to x with a byte-identity test on the old path.
- T-25 [A] Plastic weights: W is state in h during a run; A holds W₀ and its ownership. A trained
  network is A' = A with W₀ ← W_T, recorded with provenance. Closes when: the tensor carries W₀
  provenance and PCL's trained weights load as A'.
- GSDR (lane 4f): Hamm's first suspicion for the saturated regime (κ ≈ 0.95 against 0.0156) is weak
  inhibition: sweep g = J_I/J_E at fixed drive before any further fit.
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

1. AT-10-R6: RSS growth (+80–98 MB over 4 chunks of the 20-area run): resolve against the XLA cache
   before any run over 100 s (0.5.8 laminar models on GPU). WSL2 CUDA 20-area receipts
   (`scratch/wsl_cuda_2026-10-09/`, gitignored): one unexplained failed GPU run. Closes when:
   RSS is flat over chunks with the XLA cache cleared.
2. A8 gh-pages policy for D1 or Atlas figures (the dashboard is at `/dashboard/` only).
3. Architecture split (`emitters/`, `tfne.py`; `artifacts/audit/reviews/deep_review_89_2026-10-04.md` §5)
   only as a measured bottleneck.
4. C5–C7 numerical deferrals: need a signed-zero/NaN exactness contract.
5. UNTESTED-exact refusal tail; placeholder notebooks and artifact-gated skips; post-0.4.14 aliases.
6. `scripts/` lint (P-001, ruff 171): first `audit_w3_broad_handlers.py` `_OVERRIDES` (running it
   rewrites `artifacts/audit/baselines/w3_broad_handler_tally.json`): re-key by content or retire.
7. S27 population definition family, when CTX-01 cannot express one.
8. Agent-native step 10: MCP or another transport.
9. From 0.5.4: item 1b (S20.1 ordering, needs Hamm) and item 0 (`X[k]` frontier override).
10. 2026-09-20 audit owner calls: `compile_step_fn **hdp_kwargs` unknown keys;
    `validate_hdp_params` non-dict pass; frozen protocols recording JAX versions (P-003).
11. Field Approximation Atlas ladder (deep review 89 §7).
12. Rule proposal (deep review 89 §12): new mechanism ⇒ reference model, discriminator and
    reduction test; adoption is Hamm's.
13. 0.5.7 (Hamm 2026-10-13): Izhikevich 2003 P3 prereg draft v7 (calibration passed at θ = 4,
    `artifacts/etudes/papers/izhikevich2003/calibration/calibration_v7_15f36087.json`) for signature,
    then S1 reference arm `net_reference.py` and the scored runs. CAMINOS P6 prereg (Fig. 1
    through `emitter(family="adex")`, `connections(sampling="bernoulli")`,
    `Simulation(poisson_drive=)`; F1–F6 landed) for signature and its equivalence test.

Tracks (standing directions): P performance (optimize profiled components only) · T testing
(cheapest gate per defect class) · S skills · I integration (one implementation, many entrances) ·
A architecture (factor only with measured benefit) · V validation (configured → realized →
executed → observed) · U inspection (provenance queries).
