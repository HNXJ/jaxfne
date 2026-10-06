# Remaining work

Remaining work only. Done work lives in git, receipts and results; the full
pre-2026-09-27 stack (rationale, sealed 0.5.0–0.5.4 stacks, AT-10 narrative)
is archived byte-for-byte at `artifacts/archive/0.5.x/todo_stack_2026-09-27.md`.

Rules: `artifacts/AGENTS.md`. Facts (human edit only): `artifacts/fact_stack.md`.
Open issues: `artifacts/issue_log/ISSUE_LOG.md`. Atlas source:
`artifacts/project_sources/8_atlas.md`; requirement rows:
`artifacts/programme/atlas_coverage.json`. Other editors also change this
file: re-read before editing, amend in place, keep diffs you did not make.

## Goals in plain words (0.5.x)

Want: a faster JaxFNE that runs the full Atlas, from one neuron to twenty
brain areas, and a manuscript built on it. Need: old results never change
unless an authorized correctness repair says so; new abilities are switched
on explicitly; every claim has evidence.

0.5.5 (current, last release of the programme): one Atlas showing the same
model language works from 1 neuron to 20 areas, and the manuscript. Needs:
the 20-area simulation (done: `AT-10-N20`); one shared list of measurements;
what survives each simplification; generated figures; every manuscript claim
tied to evidence.

Starting now (human, 2026-09-29): jaxfne analyses and visualizes its results
through jnwb wherever jnwb has the operation. jnwb: data/signal → processing
→ analysis → results. jaxfne: model/signal → implementation → simulation →
results. The move is gradual, and simulation results pass to jnwb through
one seam that is independent of the rest of jaxfne, ideally with no delay
(item 0d below). The seam is additive: existing outputs stay bit-identical.

Short list for the jnwb move (human, 2026-09-30), in order:
1. jnwb 0.2.8 carries lane D (`compute_psd(nperseg=)` and the four spike
   measures); jaxfne pins the jnwb release that has them. Lane D built,
   bundled, and independently verified by antigravity-jnwb (jchat #78,
   2026-09-30: 6100 passed, 22/22 gates). 2026-10-06 (human): track the latest
   jnwb; extra pinned `jnwb>=0.2.9` (0.2.9 on PyPI, installed, jnwb tests pass).
2. Spectra go through jnwb: jaxfne's spectrum equals `jnwb.compute_psd`
   with its own `nperseg` within float tolerance; any other change is
   declared and listed; frozen receipts are untouched.
3. The three `analysis/metrics.py` defects (Fano of the population sum,
   `burst_index` ignoring `bin_ms`, unbinned pairwise correlation) are
   logged, then fixed by delegating to jnwb or declared.
4. The rest of `artifacts/programme/jnwb_inventory.md` moves in order;
   `to_jnwb` joins the top-level surface when that settles.

## Decisions in force

- 2026-09-23 (human): the Atlas manuscript is new, in
  `artifacts/publication/atlas/`, write-once per figure; the frozen Figure
  1–7 snapshot stays untouched.
- 2026-09-25 (human): `y_schema.py` stays an artifact module; `build_atlas`
  keeps its simulate fallback, the Atlas generator uses a view-only entry;
  data bundles are in memory; AT-07-R4 (B as input) OUT_OF_SCOPE.
- 2026-09-25 (agent): `build_atlas` given signals records seed/duration only
  when passed; network-spec digests stay separate from `at_manifest.spec_digest`.
- 2026-09-26 (human): P-014 refuse + migrate (done); G_20 synthetic
  hierarchy, 50 neurons per area, noise + H01 pulses, 10 s phases, AT-04-R2
  assay (done: `artifacts/atlas/at10_n20_055.py`).
- 2026-09-27 (human): harness work before ATLAS 7 (items H1–H4 below); docs
  = one tutorial track with visuals (D1); docs stay in the repo, gh-pages
  publishing decided separately (A8).
- 2026-09-27 (agent): AT-10-N20 is a new registry id beside the toy AT-10,
  which the frozen agent benchmark pins.
- 2026-09-27 (agent): descriptive labels nothing reads stay labels
  (`connectivity` route labels such as `feedforward`, `feedback`, `kind`,
  `mode`, `e_to_all`; `network(kind=)`; declared probe `modes`), following
  the human decision to keep field proxy labels.
- 2026-09-28 (human): P-017, fix the paradigm semantics: inject only events
  that carry a stimulus, for their own duration, at the declared amplitude
  and targets; refuse a multi-condition `Paradigm`; `evoked_l4` reads its
  onset and amplitude and targets L4; rebuild pages 08/10 on it; changed
  `ParadigmCondition` runs go in the seal receipt. P-018, wire `drive()` on
  the plain route. Wire `edge_seed` (in-repo outputs stay bit-identical).
  Probe kernels refuse `position`/`reference`/`filter_spec` they do not
  apply. H1 (e): construct refuses layer tables and column layers the route
  drops; `default_cortical_column_config` stops declaring them (outputs
  bit-identical; its config hash changes and pinned hashes are regenerated).
- 2026-09-28 (agent): probe kernels accept `reference`/`filter_spec="none"`,
  the `Configuration.probe` rule. The plain route leaves cell-type fractions
  within 1e-9 of unit mass unnormalized: dividing by a mass of 1 ± ulp moves
  n·frac off a .5 rounding boundary (n=10, 0.35) and would change existing
  labels.
- 2026-09-28 (agent): receipts made while a declaration was dropped stay as
  made, and their configurations declare what ran: `mcc3_config()` keeps the
  default drives of its checkpoint and etude (as `_mcc3_model` under P-014),
  `tests/test_equiv01_table.py` the drives of its bounds (P-019);
  `artifacts/perf/matrix_051.json` stays the 0.5.0 receipt (see ATLAS 8).
- 2026-09-28 (human): H1 (e), `default_complete_configuration` drops
  `uniform3d()` and builds a laminar column, so its declared L2/3→core map
  builds edges; its outputs and config hash change and the seal receipt
  lists them. P-016 and P-020: bisect each to the commit that moved it,
  then re-freeze in a new receipt that names the cause, or revert a buggy
  commit; the human signs off each outcome; nothing is re-tuned. P-013: add
  declarative per-task `arm_definitions` (inputs only) with a schema test;
  frozen `results_055` stays; the human reviews the definitions before the
  next freeze. Docs style pass over `docs/` and README
  (`match-writing-style`): cut padding only, soften overclaims and list
  each for review, no truth-status headers on public pages.
- 2026-09-28 (agent): P-012, exempt paths `git check-ignore` reports and
  skip the v033 check on an empty directory. P-019, bound the jit/eager gap
  per drive regime on the P-018 path.
- 2026-09-28 (human): agy (Antigravity CLI) takes high-load, low-complexity
  lanes: quality sweeps of docs, markdown and code against the canonical
  style skills, and upkeep of `artifacts/skills/jaxfne-*`. Every diff is
  reviewed before it is integrated.
- 2026-09-29 (human): item 0d starts now. Spectra move by adding `nperseg`
  to `jnwb.compute_psd` (float-tolerance bound). The jnwb extra carries a
  `python_version >= "3.12"` marker; jaxfne stays `>=3.11`. jnwb gaps are
  fixed in a fresh clone of HNXJ/jnwb at `E:/repos/jnwb`.
- 2026-09-29 (human): jnwb 0.2.8 gets a new lane D for jaxfne's needs
  (`compute_psd(nperseg=)`, spike-train Fano factor, burst index, mean
  pairwise spike correlation, Fleiss kappa). `to_jnwb` turns Jaxley
  threshold-level spikes into upward crossings; bridge outputs stay as
  they are. `pynwb_compat.write_nwb` writes a `to_jnwb` view (Units table,
  proxies as unit-less TimeSeries, never volts). `to_jnwb` stays at
  `jaxfne.jnwb_view`, off the top-level surface, until the migration settles.
- 2026-10-04 (agent, owner-delegated) sign-off verdicts, all verified this session:
  - Receptor taus APPROVED: AMPA 2.0 ms (conventional modeling default; primary citation owed at manuscript item 13), GABA_A 5.0 ms PRIMARY-VERIFIED (Xiang, Huguenard & Prince 1998, J Physiol 506:715-730, adult rat visual-cortex LV pyramidal sIPSC monoexponential tau_D 5.0 ms; young-cell and interneuron decays are slower, recorded as caveat). `presets.py` GABA_A source string now names the paper; values unchanged; 43 receptor tests green.
  - P-016/P-020 re-freeze APPROVED: receipt cause (`865e74b`) confirmed; P-016 module 4 passed here; exact new values present in both frozen files (`0.00280088258438386`, kappa `0.0182925…`).
  - `arm_definitions` APPROVED: inputs-only prose per arm (sole "judged" wording in AT-01 is role description, no critic implementation); keys==arms enforced by `test_arm_definitions_match_arms`, carry path tested, both green here.
  - C5–C7 ruling: deferral STANDS. No post-0.5.1 perf bottleneck measurement demands matmul reassociation, div-to-mul rewrites, null-term elision, memoization, or layout changes (newest perf artifacts are matrix/spec/import-cost, 2026-09-24…27).
- 2026-10-04 (agent, owner-delegated): D0 DONE. All 17 flagged items cleared (timing claims linked to `w11_atlas*.jsonl` or softened to machine-dependent; stale versions updated to v0.5.x; file-size range unified; µm labels converted to relative fractions; untraceable receipt pointer qualified); protocol_c_wave, colab, gallery kept with reasons. Doc gates green (language, vocabulary, orphans, integrity) + ruff + smoke. Note: edit-tool reflow hit `jaxfne/_signals.py` (437 lines) and `jaxfne/validation.py` (53 lines) mid-sweep — caught by diff-stat, reverted, re-applied binary-safe (P-004/P-008 class; harness worked as designed).
- 2026-10-04 (human): reorg step 2 DROPPED. Reference audit found live consumers on all five dirs (tests assert `legacy` paths; smoke scripts read them; docs link them; generator writes `mcc3_10s_checkpoint`; doctrine cites the sweeps). Layout is load-bearing; row deleted.
- 2026-10-04 (agent, owner-delegated): sweep queue CLOSED (P-027…P-034 + minor batch). Minor batch applied: stimulus grid refusal, streaming named Izhikevich/syn consts + dt guard, STDP shared kernel (hand-computed test + suites green), bridges teardown, Net DeprecationWarning, paradigm joint/collision refusals, RESIDUAL_NORM_TOL, load strict flag, geometry/spectral consts; 8-test minor-batch file green. Dismissed with evidence: examples 00/08 run green, tutorial prints are contractual UX, metrics NaN already documented+pinned, preset int/float signs left (dtype risk), NMDA/GABA_B sources await primary (item 13). Commit 41f0d35b.
- 2026-10-04 (agent, owner-delegated): coverage R2→VALIDATED (results JSON + `test_atlas_at10_n20_055.py` 3 passed this session; separate runs from one W0, not state-continuation), R7→OUT_OF_SCOPE (candidate; B-input refused upstream, AT-07-R4 precedent already OUT_OF_SCOPE). Ledger CL-22…CL-26 added (phases, W(t) negative, balanced negative, scaffold boundary, null caveats). Packet verdicts for the rest: R1 needs manifest re-emit with k_d/tfne provenance (genome is g20-hierarchy-v2, NOT the canonical column — corrects earlier shorthand); R3 needs numeric PSD/C/spatial/boundedness; R5 needs a discriminating arm; R6 needs spec + run; AT-00-R3/R5 need code-only checks (spec-diff, column map). No frozen bytes touched.
- 2026-10-04 (agent): AT-00-R5 investigated, NOT forced — panel order (raster→lfp→h_dynamics→hdp) does not follow grammar order (dynamics→state/plasticity→field); reordering means regenerating all atlas pages. Recorded as analysis; grammar mapping is a manuscript-semantics decision.
- 2026-10-04 (agent): R1 manifest re-emit path found — `build_atlas(provenance={tfne_digest: genome sha256, k_d: 20})` plumbing already exists (`_discover_upstream_lineage`); runner never passed it. Detached regen launched (same seeds/config); verifies panels byte-identical before copying manifests. k_d=20 is G20_DEV_SEED (develop default); tfne_digest is the g20-hierarchy-v2.json sha256 — choice recorded.
- 2026-10-04 (agent, owner-delegated): R1→VALIDATED WITHOUT resim. Two regen runs proved current-code output differs from committed pages (seed/duration args fixed in run 2, yet baseline H AVAILABLE vs committed OMITTED + network_3d bytes differ: renderer drift since the pages were built). Full regen would rewrite evidence under a metadata errand — refused. Instead filled `k_d: 20` + genome sha into the 3 committed manifests only (6 lines, CRLF preserved, panel bytes/status/sha untouched; values certain from G20_DEV_SEED + write-once genome file). R1 evidence now the manifests themselves.
- 2026-10-04 (agent, owner-delegated): R3→VALIDATED. Delegated runner extension (psd per area via repo welch path nperseg=256, boundedness PASS against [0.5,35], rate spread), full `run_all` regen to scratch: all pre-existing keys bit-identical, +9 keys only, sole deletion wall_s (machine timing). Committed results copied with CRLF preserved (8362 added lines are the new numerics). R2 test green on the new file. No frozen bytes touched.
- 2026-10-04 (agent, owner-delegated): AT-00-R3→VALIDATED. Delegated build (worker packet, files reviewed verbatim): `scripts/check_atlas_inheritance.py` parses the S-section array block (`8_atlas.md:64`) into cumulative stage sets and diffs against registry `STAGE_COMPONENTS` + `inheritance_check`; `tests/test_atlas_inheritance_055.py` asserts equality (2 passed here). First regen attempt exposed my own invocation gap (seed/duration/dt not passed → null identity + H AVAILABLE vs committed OMITTED); corrected run 2 in flight.
- 2026-10-04 (agent, owner-delegated): AT-10-R4 (W(t) trajectories) is
  OUT_OF_SCOPE with reason: `record_weight_trace` is False by declared
  recording budget (20,000 x 68,620 floats ~5.5 GB per 10 s phase;
  `at10_n20_055.py:111-116`, cited `at_manifest.py:477`); `w_final` plus
  the assay ratios (`w_mean_ratio`, `w_unchanged`, window rates per H0 arm
  in `at10_n20_055.json`) answer the plasticity question the trace would
  serve. X(t)/H(t)/Phi(t) remain measured in panels and assay. Reversible:
  flip the flag and re-run with traces on to re-open.

## 0.5.5 stack

Legend: [A] an agent can run it now · [H] needs a human decision · [B] blocked.

NEXT (ordered, executable)
(none — main fully green @ 839a3d2b: Fast + Release + Nightly run 37247983354 success 2026-10-04. Next work is the science lanes.)

HUMAN DECISIONS
- [H] agy lane: agy 1.2.13 is installed (2026-09-30), but its sign-in is
  not visible to Claude Code's processes, so agy agents are reached in jchat
  and take instructions from the human directly (antigravity-jnwb).

SEAL NOTES (go into the seal receipt)
- H1 (e), 9958a72: `default_complete_configuration` is laminar; its outputs
  and config hash changed (0 -> 173 inter-area edges at defaults). Other
  callers kept their outputs.
- P-017: list the runs whose drive changed (commit f8960a0: stimulus-less
  marker events now silent; stimulus events use their own `duration_ms`).
- Geometry (fact: apply or refuse): `build_laminar_column(geometry='laminar',
  radius_mm=/height_mm=)` now writes the given values to
  `column_radius_mm`/`column_height_mm`; before, it ignored them. Defaults
  (`None`) declare nothing, so outputs without the arguments are unchanged.
- AT-02…AT-06 run records (`artifacts/publication/atlas/`) carry the
  pre-correction spec digest (AT-02…AT-05 recorded `n_contacts` 4, executed
  16; AT-06 recorded a common-average reference and an 8–25 Hz band-pass that
  were never applied); regenerate them with the other Atlas records. The
  frozen agent task set keeps the old digests
  (`tests/test_agent_bench_055.py` allows exactly these corrections).

ATLAS
7. Reduction/scale matrix: for each transition M_i → M_(i+1) (1N → 2N →
   population → 2A → 20A), which observations survive within the
   predeclared tolerance and which do not; failures stay in the matrix.
   Existing row: `at01_at06_052.run_reduction` (HH → reduced → population;
   covered by `tests/test_atlas_reduction_052.py`). Missing rows: population
   → 2A → 20A (AT-10 / `AT-10-N20`); predeclare each tolerance before the
   run, record failures as failures, feed E_reduction to item 8 and to
   coverage rows in item 9.
8. Performance/reduction map: T_compute and M_compute per AT beside
   E_reduction. `artifacts/perf/matrix_051.json` ran the
   `benchmark_050_baseline` models at the default drives; their declared
   drives run since P-018, so measure them anew here.
   RESULT 2026-10-06 (loaded, non-final timings; `matrix_055.json` committed):
   all 17 cells MEASURED incl. `at10_20area` coupled; 14 null-drive cells
   bit-exact vs 051; `mech_hdp` differs as declared; `chunk_ref` 174 -> 173 =
   P-010 repair (below); frozen 051/050 hashes unchanged. Warm times were
   0.86-3.7x the 051 values on a saturated machine, so they are not a
   regression claim: rerun on a quiet machine before the seal, and add
   `T_test` (the broad gate took 539-811 s loaded) to the receipt.
   Older run-state notes follow.
   Run state: spec `artifacts/perf/matrix_055_spec.json` (frozen before the
   result), runner `scripts/benchmark_055_matrix.py` (drive threading per
   cell), result `artifacts/perf/matrix_055.json` (17 cells; both still
   untracked). Run on a quiet machine (timing). Accept when: every
   null-drive cell's `output_checksum` equals `matrix_051.json` (only
   `mech_hdp` may differ: E_semantic = 0), each cell records its drive
   provenance, and `T_test` is measured. Open on the 2026-10-06 file:
   `chunk_ref` differs from 051 (174 -> 173 spikes, V_sum -6654899.0 ->
   -6652058.0): deterministic (3 reruns identical); bisected to 865e74bb
   (0.5.3 item 3, P-010 chain-noise fix), an authorized correctness repair:
   list it in the seal receipt and re-baseline `chunk_ref` for 055 as a
   declared repair, not E_semantic != 0 (the reference is the edge_list
   single-call path; `chunk_k4` unchanged); `at10_20area` was
   UNSUPPORTED because P-023 (83848805) refuses an undeclared synapse time
   constant; the harness now declares the pre-P-023 default `dT_ms=0.1` on
   each ring AreaConnection (in-process check: coupled, 4,747,700 edges,
   equal to 051; the cell has no output checksum, so no bit-compare exists);
   rerun it in the final matrix and list P-023 in the seal receipt;
   the file has no `T_test` and no per-cell drive provenance, so it may
   predate the drive threading: confirm with the worker. Items 9-14 take the
   T_compute/M_compute deltas from this file.

   AT-10-R5 RESULT 2026-10-06 (`artifacts/atlas/results/at10_r5_hspace_055.json`,
   run under load, 151 s; protocol per `artifacts/programme/at10_r5_protocol_proposal.md`,
   option (a): N20's own HP_HEBB and arms): existing rate-space verdicts
   reproduce the committed ones exactly (H0=0.0 NO_LASTING_EFFECT; kick 0.8
   and 1.3 STABILIZED). H space (descriptive only, see review): engaged arms
   RETURNING (ratio 0.08-0.11); H0=0.0 disabled arm RETURNING (H relaxes with
   tau_0 = 5 ms even without plasticity, matching NO_LASTING_EFFECT);
   HDP-disabled kick arms PERSISTING (ratio 0.99-1.21). ADVERSARIAL REVIEW
   (opencode review tier, 2026-10-06, code SOUND, claim overstated), open
   before any promotion: (1) HIGH: W-kick twins differ in W, not in initial
   H, and the disabled-kick arms keep a different frozen W for the whole run,
   so PERSISTING there is two different fixed networks at two steady H
   values, not a twin-valid stability test; only the H0 arms are valid twins.
   A valid persisting case needs a K_ctrl = 0 arm. (2) HIGH: the HDP kernel
   clips H to [H_min, H_max] and |w| to w_ceiling (`jaxfne/emitters.py:4181`),
   so "within_bounds: true" cannot fail: report it as a clip check, not as
   evidence of boundedness. (3) MEDIUM: the imposed H0 = 0.0 lies outside
   [0.1, 10] but the recorded trace is post-clip. (4) MEDIUM: the run uses
   N20's own parameters, not the approved 0.5.3-fixture choices, and no
   K_ctrl = 0 arm exists, so the A-vs-B acceptance test of the proposal was
   not executed. (5) the kick-arm ratios may depend on `early_end`: recompute
   with `early_end = 2000`. Row stays SUPPORTED. Caveat: 1 noise seed.
   FOLLOW-UP RESULT (`artifacts/atlas/results/at10_r5_twin_055.json`,
   `run_r5_twin_assay`, 57 s loaded): valid H twins (interior H0 = 0.5,
   same regime, same seed). Restoring (K_ctrl 0.15): RETURNING, ratio 0.086,
   H in [0.50, 2.09], clip not reached (genuinely interior, bounded and
   returning). No restoring (K_ctrl 0): PERSISTING, ratio 0.90 (0.88 at
   early_end 2000), and at least one neuron reaches the clip (clip_reached =
   True, H_max_obs = 10.0). Wording (second review, 2026-10-06): say "no
   restoring term; the offset persists; some neurons reach H_max"; do NOT say
   boundedness "is" the clip: the flag is any neuron at any step, the fraction
   pinned is unknown, and PERSISTING (d_late 0.449 ~ the 0.5 offset) holds
   without clipping. within_bounds True is tautological for a clipped trace.
   Known limits: clip_reached checks the perturbed arm only; the early_end
   sweep barely varies d_early (late window fixed at -200, no ref-vs-ref noise
   control); one seed, one H0, loaded machine. Addresses review items 1, 3;
   item 2 is flagged per pair; item 4 (approved 0.5.3-fixture values) is
   replaced by the K_ctrl contrast on N20's own parameters, which needs the
   human's acceptance. ACCEPTED by Hamm 2026-10-06: AT-10-R5 VALIDATED
   (evidence `at10_r5_twin_055.json`), with the wording limits above.
   AT-10-R6 RESULT 2026-10-06 (`artifacts/atlas/results/at10_r6_long_055.json`,
   `run_r6_long`, loaded machine): AT-10-N20, HDP engaged, 100 s in 4
   continuation chunks of 25 s (178 s wall; 36-48 s per chunk). Engine change
   `deab8a1d`: Poisson noise composes with continuation, per-chunk seed from
   (seed, chunk_index); chunk 0 bit-identical to a plain call, so the noise
   realization differs from the 10 s R3 runs (new run, not a reproduction).
   Observed: mean rate 8.93-9.00 Hz in every chunk (no drift), clip never
   reached, mean |w| ratio 0.998-1.000, H max per area 1.65-2.41. RSS grew
   1757 -> 1855 MB over 4 chunks (+98 MB): not explained, check before a longer
   run. Spikes are never stored (per-chunk reductions only). One seed. Row
   SUPPORTED; VALIDATED needs your acceptance. Bug found
   on first live run: continuation needs `recurrent_backend="edge_list"` set
   explicitly (fixed in the runner).
   Second review 2026-10-06: engine change SOUND (chunk 0 bit-identical, default
   and non-Poisson paths unchanged, 4-field legacy state safe); evidence supports
   SUPPORTED, not VALIDATED. Wording limits: chunk-mean rates only; chunk-0
   identity tested at toy scale (12 ms) only. Replicates (noise seeds 12, 13;
   `at10_r6_replicates_055.json`, loaded machine): mean rate 8.93-9.01 Hz in
   every chunk, clip never reached, mean |w| ratio 0.998-1.001; so 3 seeds agree.
   `chunk_index` test for `run_continuation_strided` added (`082fcc47`). Open:
   RSS grows +80-98 MB over 4 chunks in all 3 runs (leak vs XLA cache
   unresolved; check before any run longer than 100 s).
   RESULT FILES 2026-10-06 (`scripts/emit_atlas_results_055.py`; Hamm chose a
   result JSON per SUPPORTED row; rows stay SUPPORTED until reviewed and
   accepted): `at01_055.json` (AT-01-R1/R2/R3/R7), `at_reduction_055.json`
   (AT-00-R4, AT-01-R6), `at05_055.json`, `at07_055.json`. Findings that block
   VALIDATED: (a) AT-01 and REDUCTION record 3 of 4 predeclared HH->reduced
   verdicts FAILED (v_peak diff 25.9 mV vs 20, v_rest 10.8 vs 10, first-spike
   10 ms vs 2; rate passes), so R6 "which quantities survive" = rate yes,
   amplitude and timing no; recorded, never tuned. (b) AT-05-R2 is NOT met, and the
   check is VACUOUS (verified 2026-10-06, n=8 seed 7): every per-source
   contribution at every contact is >= 0 (frac_neg 0.0), so
   |sum_i phi_i| = sum_i |phi_i| and `coherent_over_n_single` is 1.0 by
   algebra whatever the synchrony. The runner's "toy size / fully coherent"
   explanation is wrong; a larger or partially coherent arm cannot change it.
   A real test compares the N-neuron population field with N x a field from
   ONE neuron simulated alone (add an n=1 arm; A_Phi(N) vs N*A_Phi(1), by
   executed kappa). Which comparison defines Phi_N != N Phi_1 is Hamm's call. (c) AT-07 states "no stability assay, no adaptation
   phenotype"; AT-07-R3 would have to cite the N20 twin assay
   (`at10_r5_twin_055.json`) or get a stability arm of its own: Hamm's call.
   (d) result `_meta.jaxfne_version` reads 0.5.0 (version string not yet bumped).
   Coverage state (`python scripts/check_atlas_coverage.py`, VALID, 48 rows:
   30 VALIDATED, 8 SUPPORTED, 4 PLANNED, 6 OUT_OF_SCOPE). Rows still short of
   VALIDATED for 0.5.5: PLANNED AT-00-R5 (figure column order; item 10),
   AT-00-R6 (manuscript progression; item 14), AT-10-R5 (perturbation/control
   assay: bounded != returning != homeostatically stabilized; needs a new
   S10 run on `AT-10-N20`; protocol proposed in
   `artifacts/programme/at10_r5_protocol_proposal.md`, choices approved
   2026-10-06; next: build the runner, run after the matrix window), AT-10-R6 (reduced fast model, very long T; takes
   its envelope from item 8); SUPPORTED AT-00-R4, AT-01-R1/R2/R3/R6/R7,
   AT-05-R2, AT-07-R3 (promote with evidence at the seal; AT-00-R4 and
   AT-01-R6 follow item 7).

MANUSCRIPT (ends 0.5.5)
9. Atlas coverage matrix: every section, figure, simulation and claim in the
   Atlas source → todo item → produced artifact; 100% mapped or marked out of
   scope by the human; mechanical check over `atlas_coverage.json`. AT-10-R*
   rows point at `AT-10-N20` evidence (results, figures, caveats: selection
   status, control-window dependence, no-stimulus null).
10. Figures via the existing generator seam (`scripts/publication_figures/`
    pattern), consuming generated Atlas data only; columns structure →
    dynamics → state/plasticity → source → field → observation → computation.
11. Claim ledger: each claim with its V(claim) and evidence path; package
    capability claims apart from scientific-result claims. Scaffold:
    `artifacts/programme/claim_ledger_055.md` (48 rows from
    `atlas_coverage.json`: 26 capability, 18 result, 4 unset = the PLANNED
    rows; every evidence path exists). Left for the human: confirm each
    `kind`, fill wording strength, add manuscript claims that are not Atlas
    requirements, and fill the 4 PLANNED rows' evidence as they land.
12. Methods from manifests: units, calibration level, tolerances, seeds and
    versions quoted from generated artifacts.
13. Citations primary-verified (H10) before inclusion.
14. Manuscript text complete and consistent with the claim ledger;
    submission is a separate human action.

ACCEPTANCE (0.5.5 seal = end of programme)
- Existing canonical configurations bit-identical to the frozen baseline,
  except outputs changed by authorized repairs (P-014: suite2 presets,
  experiment_a, protocol_e now run their declared drives), each listed in
  the seal receipt. Run the release-tier baseline check before the seal
  (`python scripts/run_test_gate.py release`, then `rc`; receipt template
  `artifacts/release/v0_5_0_release_receipt.json`; authorities
  `artifacts/release/current_release_authorities.json`). Tag, main merge and
  release are separate human acts.
- AT-01…AT-10 (and AT-10-N20) regenerate from manifests; Atlas and figures
  reproduce.
- Coverage matrix 100% (item 9).
- Programme deltas vs v0.4.25 measured: T_simulation, M_peak, T_test,
  T_agent_task, E_semantic = 0, C_scientific retained.
- Every manuscript claim traces to a ledger row with PASS evidence; negative
  and failed results reported as such.

## Compaction track (audited 2026-10-06)

DONE 2026-10-06 (Hamm approved "C1-C5 plus unused scripts"; broad gate 4906
passed): C1 (mcc3 `metrics.json` 54.8 MB, `weights.csv` 34 MB, `H.csv` 7.4 MB
and the archive copy removed; `mcc3_10s_metrics_digest.json` keeps the fields
`rerun_etude_figures.py` compares, its `full_metrics_sha256` equals the one in
`mcc3_10s_manifest.json`; full data recoverable from commit 6a99cc12 or by
rerunning `scripts/mcc3_10s_scientific_checkpoint.py`), C2 (`visualize_bundle`),
C4 (12 scripts with zero references; kept `p2v_a4_claim_language.py` and
`audit_semantic_negatives.py` for the manuscript claim checks). SKIPPED with
reason: C3 `artifacts/legacy` (185 KB; referenced by
`scripts/generate_surface_contract.py`, `run_tutorial_smoke.py`,
`report_hygiene_check.py`), C5 `artifacts/.lab` (referenced by
`scripts/lab_balance.py` and `docs/doctrine/rbs_rbd_hdp_inventory.md`).
Not run: `rerun_etude_figures.py --etude hdp_mcc3` end to end (heavy); only
the digest's fields were checked against what it reads. REMAINING: C6, C7, C8
below; the history still holds the removed bytes (`.git` size unchanged until
a history rewrite, which needs explicit authorization).

Audit text (C1-C8 as audited):

Measured, not estimated: 2428 tracked files, 271 MB, ~262k readable lines.
BYTES: ~58% (158 MB) can leave git at near-zero loss; ~68% if the generated
HTML moves out too. READABLE LINES: only ~3-5% (code and tests are not
bloated); tests ~1-2% at near-zero loss, ~5-8% at low loss. Frozen
(do not touch): `artifacts/publication/frozen_manifest.json` + its 46 files,
`artifacts/release/current_release_authorities.json`, harness-hashed skills
and `AGENTS.md`. Order, lowest risk first (each item: remove, run the broad
gate, then push):
C1. `artifacts/mcc3_10s_checkpoint/` (96.8 MB; `metrics.json` 54.8 MB,
    `weights.csv` 34 MB, `H.csv` 7.4 MB) and the superseded copy
    `artifacts/archive/0.5.x/refreeze_2026-09-30/.../mcc3_10s_metrics.json`
    (54.8 MB): to a release asset or LFS; keep `manifest.json`, `rates.csv`,
    `pre_post.png`. Consumers checked: no test; `scripts/rerun_etude_figures.py:514`
    reads `metrics.json`; prose in `refreeze_p016_p020_2026-09-30.md`,
    `memory.md`; `docs/_static/etudes/hdp_controllability_reachability/manifest.json`.
    Fix the script to fetch the asset or fail clearly.
C2. `artifacts/archive/visualize_bundle/` (~7 MB, 53 same-hash duplicate
    blobs repo-wide): delete (regenerable from `jaxfne/vis`).
C3. `artifacts/legacy/` (185 KB): delete all but
    `internal_docs/agent_context/claude/CLAUDE.md`
    (`tests/test_agent_context_hygiene.py:18` needs it); update
    `scripts/generate_surface_contract.py` ARCHIVE list.
C4. 19 unreferenced scripts (3.1k lines; e.g. `make_delta_test_01_report.py`,
    `generate_mechanism_tutorials.py`, `repair_notebooks.py`; 0 references
    outside themselves, verified for 3): move to `scripts/_unused/`, delete
    after one release.
C5. `artifacts/.lab/` (108 JSON, 394 KB): confirm generated, then delete or
    regenerate on demand.
C6. Generated HTML (~27 MB: `docs/assets/interactive/*.html`, tutorial
    `etude{5,6}_network_3d.html`, `artifacts/publication/atlas` HTML 24.6 MB
    not in the frozen manifest): release asset or build at doc-build time;
    needs `tests/test_v037_interactive_column_docs.py` and doc links updated.
C7. Tests: parametrize protocol_e e0..e5 spec tests (7 files, 571 lines) and
    merge the 8 `test_atlas_at*_05x` files (638 lines); then triage the
    suite_no/vis/notebook-static groups (~5-8% of test lines at low loss).
    First get per-test coverage (`pytest --cov` with contexts) and check every
    candidate against `artifacts/publication/inventory_test_meaning_ledger.json`
    (it states load reductions must keep its invariant-class mappings); the
    98 `_vNNN` snapshot files and 45 legacy-keyword files stay until then.
C8. Agent context: `todo_stack.md` (449 lines) and `memory.md` (351 lines)
    into a short hot file + cold archive (AGENTS.md is hash-pinned: rehash).
Not duplicates (keep): `jaxfne/w3_stability_analysis.py` vs
`w3a_stability_analysis.py` (own tests and receipts).

## Open work outside the release stacks (after 0.5.5 unless a trigger fires)

0. Literature reproduction études, HDP off/on (human, 2026-09-29): plan in
   `artifacts/etudes/literature_reproduction_plan.md`. P0 table done
   (`literature_p0_candidates.md`); P1 partly done
   (`literature_p1_verification.md`: 5 sources verified, row 3 parameters read
   from the authors' code, row 10 equations unread). Targets confirmed. Next:
   PDF for row 10, P2 GAP list for row 3 (no AdEx emitter), then P3 sign-off. Starts after the
   0.5.5 seal, with its own stack.
0a. Controlled model augmentation (human, 2026-09-30): the canonical order is
   N, then G, then Theta_C, then Theta_X, then W0, then H0. Transforms act on
   a NeuronalTensor before construct. Scaling N keeps w/sqrt(N). Stochastic
   variation requires its own seed K_V.
   - AUG-2 field map (human, 2026-09-30): Theta_X is `static.g_mech` plus
     `dT_ms`. Theta_C is `delay_ms` plus `AreaConnection.probability`. W_0 is
     `plastic.w_mech`. H_0 is `plastic.H`. Each has a factor plus bounded
     jitter under K_V, with sign kept.
   - Deviation (highly recommended, 90): H_0 is additive (offset plus an
     absolute jitter, signed). H defaults to 0.0 and is a signed relative
     state, so a factor would be a no-op on every default tensor.
   - AUG-1, AUG-2 and AUG-3 are on dev (7b29c1b1, 8ed400e3, e1924e51).
     `jaxfne.augment` is public, and its limits are listed in
     `docs/api/augment.md`.
   - AUG-4 (the small suite, `tests/test_augment_suite.py`) is on dev as
     1052abaf. All three compositions commute on realized quantities.
   - Science runs (human, 2026-10-01): the plan is in
     `artifacts/programme/augment_science_runs_plan.md`.
     - The canonical 1000n column stays unchanged, with its receipts intact.
       The docs state that it runs in a synchronous-regular regime: CV_ISI
       0.036, r_sc 0.61, PV below threshold at 0.84 Hz, SST uninhibited at
       48 Hz, and E locked to its own drive.
     - A separate opt-in "balanced" preset must have E respond to recurrent
       input, with an asynchronous-irregular regime that is measured, not
       assumed.
     - The augmentation science runs on that preset, in phases:
       1k -> 2k -> 5k first, then a decision on 10k.
     - Order: (1) a docs note on the canonical regime (landed cbdf2b29);
       (2) the balanced preset; (3) the phased runs.
     - Preset format (human, 2026-10-01): extend the NeuronalTensor preset
       format so a preset carries per-cell-type drive and background noise;
       absent fields keep the canonical column bit-identical. Reason: weights
       alone cannot reach AI; an opencode attempt needed poisson amplitude 90
       and w 67.5, and E stayed drive-locked at 11 Hz even when disconnected.
       agy builds on branch balanced-preset; opencode validates; Claude
       integrates.
     - Pairing (human, 2026-10-02): agy and opencode work the open issues and
       todos together and validate each other's bundles (handoff folder
       D:/cowork/jnwb-bus/); Claude integrates. opencode also fixes the two
       Python 3.11 release-test failures that block the docs-style PR to main.
0b. Structural HDP (human, 2026-09-29): existence over a declared candidate
   set; design note `artifacts/programme/structural_hdp_design.md`. Build
   after the 0.5.5 seal, additive and opt-in; API and hard-vs-graded
   existence still open.
0c. Primate primary-connection mode (human, 2026-09-29): an opt-in named mode
   for bottom-up, top-down, intralaminar and interlaminar connections, with
   every edge carrying its evidence status. Current defaults
   (`builders.py` `p_feedforward=0.3`, `p_feedback=0.2`, uniform weight
   ranges; FF L2/3 E to L4, FB L6 to L1/L5) are uncalibrated scalars; label
   them so now (docs and metadata only). Grade: recommended (60-80).
   - Step 1 (minimal): evidence table, source layer x source E/I x target
     layer x target E/I, each edge tagged {direct, anatomical-only,
     functional, unknown} with species, area, method, primary citation.
     Macaque only; a rodent or cat value never fills a gap. Unknown edges
     are refused or declared, not defaulted.
   - Step 2: mode declares FF/FB/lateral by continuous SLN in [0,1], not a
     binary label; interareal weights heavy-tailed (FLN) with an
     exponential distance rule; intra-area L2/3 horizontals patchy; L1
     interneuron-only (no E1); local block is the full layer matrix (many
     |i-j|>1 entries nonzero), area-specific (V1 is not PFC).
   - Rules: population fraction != connection probability != synaptic
     weight; every number primary-verified (chat-derived figures such as
     1,615 pathways, 66% density, 2-7 mm, 84% L1 inhibitory are
     unverified); refuse to realize a mode edge that the model cannot
     consume (stored is not consumed).
   - Link: spectrolaminar étude (item 0) depends on this laminar detail.
0d. jnwb as the analysis and visualization layer (human, 2026-09-29; goal
   above). Started 2026-09-29, beside the 0.5.5 stack. jaxfne keeps only the views jnwb
   cannot express (network, column and field geometry, HDP diagnostics).
   - Inventory and seam rules: `artifacts/programme/jnwb_inventory.md`. A
     jnwb gap is fixed in jnwb, not duplicated here.
   - Seam built: `jaxfne/jnwb_view.py` (`to_jnwb`, `to_jnwb_trials`),
     `pynwb_compat.write_nwb`/`read_nwb`, extra `jnwb` in both CI
     workflows. Promote `to_jnwb` to the top-level surface once the
     migration settles.
   - jnwb side, lane D of jnwb 0.2.8: `compute_psd(nperseg=)`, then the four
     gap analyses (`analysis/metrics.py`).
   - Step 3 (option A, human 2026-09-30): `vis.core.welch_psd` hands
     finite float64 input to `jnwb.compute_psd(nperseg=)` when the installed
     jnwb has it (bit-identical); float32, non-finite and 1-sample input stay
     local. When jnwb 0.2.8 is released, raise the extra's floor to
     `jnwb>=0.2.8` so CI runs the bit-identity test instead of skipping it
     (review of cd41725, finding 4). Follow-up: a jnwb item to keep float32
     as float32 (option C), then
     delegate all of it. Then migrate the rest in inventory order. Old
     entries delegate or are deprecated; frozen receipts stay untouched.
1. A8 gh-pages publishing policy. Trigger: publishing D1 or Atlas figures
   to the public site.
2. Architecture candidates (`emitters.py` variant split, entry
   fragmentation, dual manifests, same-named builders); enter only as a
   measured bottleneck. Specified split shape (deep review 89, §5, logged
   `artifacts/audit/deep_review_89_2026-10-04.md`): `emitters/` =
   base/izhikevich/receptors/recurrent/multicompartment/registry behind an
   unchanged import surface; `tfne.py` = parser/normalization/realization/
   indexing behind one compiler surface. No broad aesthetic refactor.
3. C5–C7 numerical deferrals; need a signed-zero/NaN exactness contract.
4. UNTESTED-exact refusal tail; PLACEHOLDER_NOTEBOOKS and artifact-gated
   skips; post-0.4.14 compatibility aliases.
5. P-001 `scripts/` legacy lint cleanup (ruff: 171 findings, 2026-10-04 sweep).
   Real defects first: `audit_w3_broad_handlers.py` `_OVERRIDES` repeats two
   keys (the later silently wins), holds a 3-tuple and a 1-tuple where
   `(class, note)` is unpacked, and is keyed on line numbers that have drifted;
   running it rewrites the tracked `artifacts/audit/w3_broad_handler_tally.json`
   (137 lines differ). Sweep 2026-10-04 pins it down: dup keys
   `(_construct_connectivity.py,592)` and `(_model_evaluate.py,297)` (F601 x2);
   actual handlers now at `_model.py:648`, `_construct_connectivity.py:756,878`,
   `neuronal_tensor.py:1235`, `_model_simulate.py:889`. Re-key by handler
   content/regex, dedupe, re-run tally — or retire it with the tally.
6. S27 population/`P_{l,c}` definition family. Trigger: a population
   definition CTX-01 cannot express.
7. Agent-native step 10: MCP or other transport.
8. Deferred from 0.5.4: item 1b (S20.1 ordering; individual human
    authorization required) and item 0 (`X[k]` frontier override, language
    decision).
9. Deferred owner decisions from the 2026-09-20 deep audit:
    `compile_step_fn **hdp_kwargs` unknown-key policy; `validate_hdp_params`
    non-dict non-strict silent pass; frozen protocols record JAX/lib
    versions (P-003).
10. Field Approximation Atlas ladder (post-0.5.5 programme, deep review 89
    §7): multicompartment → line source → point source → dipole →
    population → proxy, each rung with the observable it preserves and the
    tolerance at which the coarser rung takes over. Starts after the seal;
    PDE solve ≠ calibrated LFP stays the boundary until a rung proves it.
11. Post-0.5.5 development rule (deep review 89 §12): new mechanism ⇒
    reference model + discriminator + reduction test (not merely unit
    tests); API converges toward the two grammars (execution +
    scientific), everything else into advanced namespaces. Adoption is a
    human decision; until then the research firewall pattern holds.

## Tracks (standing directions, not tasks)

P performance (optimize only profiled components) · T testing (cheapest gate
per defect class) · S skills (task-shaped, benchmarked) · I integration (one
semantic implementation, many entrances) · A architecture (factor only with
measured benefit) · V validation (configured → realized → executed →
observed per capability) · U inspection (origin/provenance queries).
Full text in the archive.
