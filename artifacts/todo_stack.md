# Remaining work

Top = next. Prerequisites sit above dependents. `[A]` agent can do it, `[H]` needs Hamm. Done items
are deleted (history: git; the pre-2026-09-27 stack and the sealed 0.5.0–0.5.5 sections are under
`artifacts/archive/0.5.x/`). Rules: `artifacts/AGENTS.md`. Facts: `artifacts/fact_stack.md` (Hamm
edits). Open issues: `artifacts/assurance/issue_log/ISSUE_LOG.md`. Other editors change this file: re-read
first, amend in place.

## 1. Decisions waiting on Hamm (they unblock items below)

Ruled 2026-10-20: release plan D1 (a), archive the unreferenced and ungated scripts with an index
(worker running); D3 approved. Open, recommendation first: D2 (Destexhe and GSDR preregs, after
their pilots); a state clamp on the plain Izhikevich path (T-19, changes outputs); extending the same-layer zeroing rule
to I/Inl/Ing (T-18 remainder, a model change); the manuscript tree and other archive moves (T-17).
Earlier rulings sit in the rows they
govern (T-22 semantics defaults, T-25) and in `artifacts/etudes/meta/REPLICATIONS.md` (Izhikevich
2003 and CAMINOS P6 move to 0.5.7); finished rows are in git history.

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

## 2. Next release: 0.5.7 (truthful core)

Plan: `artifacts/etudes/meta/release_plan_0.5.7-0.5.10.md` (Hamm 2026-10-19); exit gate in its
section 6. Packages A1-A7 are on dev or in the push now gating; what is left, fastest first:

| Order | Item | ETA | Who |
|---|---|---|---|
| 4 | Brunel S2 (5 s window) and S3 (dt 0.05 ms) sensitivity, seeds 4-5, unscored: running detached on the WSL CPU since 2026-10-20 (`brunel_s23_chain.sh`, out `C:/workspace/jaxfne_receipts/brunel2000_p3v4/unscored_S2`, `unscored_S3`); RESULT_P3 sensitivity section | about 5 h compute | dispatcher reads |
| 5 | Head-forward core checks: the two line-source LFP agreements of `tests/test_head_forward.py`; tolerances moved out of the check functions so a test can tighten them and see a fail | 1 h | Muse |
| 6 | A9: assemble the fragments, `release_body.py --tag v0.5.7` claim check (closes T-10), rc gate, second review, receipt under `artifacts/release/` | 2-3 h | dispatcher, worker gate runs |
| 7 | Tag, main, PyPI and RTD `stable` | after 6 | `[H]` go |

Moved to 0.5.8 with reason (plan section 1: missed work never blocks its release): Destexhe and
GSDR B8 pilots and their preregs (A8 part; days of runs, signature `[H]`); Brunel S1 (needs the
fixed-in-degree option, B5).

Then 0.5.8 (core-compliant replications), in order: B4 checklist rules in `check_replications.py`;
B1 estimators into `jaxfne.analysis`; B5 lane gaps (Brunel S1 option, Vogels G-SC, Potjans G-MULT
to G-DDIST); B3 batching; B2 routes through `construct`; Destexhe and GSDR pilots, then preregs (the GSDR fit
with MCDP on via `mcdp_hdp_step`, Hamm 2026-10-20); Jansen and Rit 1995 opens the EEG/MEG lane
(Hamm 2026-10-20; P0 in REPLICATIONS): a neural-mass emitter in the core, then P1.
Blocked: T-11 (jnwb), T-16 and T-17 `[H]`, deferred list (section 5).

## 2a. Standing from 0.5.6 (Hamm 2026-10-12: cleaning, organizing, docs, dashboard)

- R2 restated (Hamm 2026-10-14): bytes -10% against the 0.5.5 tag (met: 86.8 MB
  tracked at 1bfdaa84 against 126.0 MB at the tag) plus files at the recorded budget with
  reasons on record (2438 tracked, `budget_files` 2438 in
  `scripts/tree_size_allowlist.json`, 2026-10-18; each raise from 2269, its reason and
  authorization are in this line's git history and c4056347; 2372 -> 2400 for the Destexhe lane
  and the O-8 test, 2026-10-16; 2400 -> 2428 on 2026-10-17 for the GSDR lane, its diagnostics,
  the regime tests and the HH tests, each raise in its commit: fdd797ce, d0c1faae, 405378e8,
  c6f052af, bb55e631, 5615fb98, 844900c6; 2428 -> 2436 on 2026-10-18 for the 10 changelog
  fragments since v0.5.6, which the assembler deletes at release; 2436 -> 2438 on 2026-10-18
  for the GSDR regime diagnostic at the reference g* and its test). The
  regenerable docs assets are untracked already: the 9 registry `copies` dests under
  `docs/assets/etudes/` and all atlas/etude HTML are build products of
  `scripts/build_docs_assets.py --build` (the Read the Docs `pre_build` path) and gitignored.
  No further class qualifies: the 9 `docs/_static/gallery/*.png` (embedded by `docs/start/gallery.md`,
  rewritten only as an index by `generate_gallery.py --write`, never rasterized by the build
  path), the 12 `docs/_static/tutorial_figures/*.png` (no caller in the build path), and every
  other tracked `docs/assets/` file (its generator, if one exists, is not in the build path;
  `--html-only` skips PNG rasterization) would leave broken embeds if untracked, so all stay
  tracked.
- T-8 [A] Each new score: `dashboard/sources.json` (lane SHA, stage artifacts), regenerate,
  push dev. Standing item. Closes when: `python scripts/build_dashboard_data.py --check` passes
  on each push.

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
- T-10 [A] G-1 (meet 204dfdaa): port jnwb's `changelog.d`, `assemble_changelog.py`,
  `release_body.py` and claim check; release body as tables. Closes when: `release_body.py --tag <next>` passes.
  The port itself landed (924f2423, 78d293d9); only the closing check waits for a tag.
  Then the errors page (the docs newline check holds: `docs/reference/changelog.md` and
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
  by the dev gate too) against `scripts/replication_known_violations.json` (25 findings,
  2026-10-18); a fix removes its entry. Closes when: every paper passes the checker without a listed violation, or each listed one
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
- (from the open stack, 2026-10-18) Scratch clones and finished Muse sessions in the session
  scratchpad. Trigger: their lanes land; delete then (check alternates dependents first).
1. AT-10-R6: RSS growth (+80–98 MB over 4 chunks of the 20-area run): resolve against the XLA cache
   before any run over 100 s (0.5.8 laminar models on GPU). WSL2 CUDA 20-area receipts
   (`scratch/wsl_cuda_2026-10-09/`, gitignored): one unexplained failed GPU run. Closes when:
   RSS is flat over chunks with the XLA cache cleared.
2. A8 gh-pages policy for D1 or Atlas figures (the dashboard is at `/dashboard/` only).
3. Architecture split (`emitters/`, `tfne.py`; `artifacts/audit/reviews/deep_review_89_2026-10-04.md` §5)
   only as a measured bottleneck.
4. C5–C7 numerical deferrals: need a signed-zero/NaN exactness contract.
5. UNTESTED-exact refusal tail; placeholder notebooks and artifact-gated skips; post-0.4.14 aliases.
6. S27 population definition family, when CTX-01 cannot express one.
7. Agent-native step 10: MCP or another transport.
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
