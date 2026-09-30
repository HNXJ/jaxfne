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
   2026-09-30: 6100 passed, 22/22 gates); waits on kickoff integration.
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

## 0.5.5 stack

Legend: [A] an agent can run it now · [H] needs a human decision · [B] blocked.

NEXT (ordered, executable)
1. [H] P-013 review the `arm_definitions` in `agent_tasks_055.json`
   (inputs only; outcome clauses removed after critic review), then close
   P-013. `freeze_task` does not emit them: a re-freeze must carry them over.
2. [H] P-024 oddball deviant drives exactly like the standard: deviant gain
   parameter or refusal.
3. [H] P-016 and P-020: bisects done (`artifacts/programme/bisect_p016_receipt.md`,
   `bisect_p020_receipt.md`, mover 865e74b); re-freeze or revert awaits sign-off.
4. [A] Fact audit, entrances (P-022): a hand-spelled Configuration now
   equals the NeuronalTensor entrance on spikes/V_m/sources, for one area
   and for two areas of different sizes with a delayed area connection
   (normalization, delay and sizes each shown to matter), and for a
   developed JDNA genome. Remaining: carrying `Layer.geometry` through the
   bridge (then field outputs can be compared too); P-023 defaults.
5. [A] D0b remaining lanes: `scripts/` comments and the markdown outside
   `docs/` (188 files; `artifacts/skills/` done, `scripts/` claimed by
   opencode-jaxfne; evidence records such as `artifacts/programme/` stay as written). By opencode actor + critic, or by agy agents
   instructed by the human in jchat.

HUMAN DECISIONS
- [H] D0 review: 23 softened overclaims and 17 flagged items in
  `artifacts/audit/docs_style_pass_2026-09-28.md`.
- [H] Sign off each re-freeze (P-016, P-020): bisect receipts
  `artifacts/programme/bisect_p016_receipt.md` and `bisect_p020_receipt.md`
  name `865e74b`.
- [H] Review the `arm_definitions` before the next freeze [B: item 1].
- [H] Stale version notes, edit or leave: `_signals.py:1445,1453`,
  `validation.py:1230`, `experimental_hpc/physical_field_solver_v040.py:55`.
- [H] `units.py` unwired: wire or remove.
- [H] Artifacts reorg step 2: repoint `legacy`, `subagents`,
  `hdp_k_w_ctrl_sweep`, `hdp_v2_rho_sweep`, `mcc3_10s_checkpoint` into
  `archive/`. The protocol_* and `private_acceptance` folders stay while
  frozen receipts cite them.
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
   Existing row: `at01_at06_052.run_reduction` (HH → reduced → population).
8. Performance/reduction map: T_compute and M_compute per AT beside
   E_reduction. `artifacts/perf/matrix_051.json` ran the
   `benchmark_050_baseline` models at the default drives; their declared
   drives run since P-018, so measure them anew here.

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
    capability claims apart from scientific-result claims.
12. Methods from manifests: units, calibration level, tolerances, seeds and
    versions quoted from generated artifacts.
13. Citations primary-verified (H10) before inclusion.
14. Manuscript text complete and consistent with the claim ledger;
    submission is a separate human action.

ACCEPTANCE (0.5.5 seal = end of programme)
- Existing canonical configurations bit-identical to the frozen baseline,
  except outputs changed by authorized repairs (P-014: suite2 presets,
  experiment_a, protocol_e now run their declared drives), each listed in
  the seal receipt. Run the release-tier baseline check before the seal.
- AT-01…AT-10 (and AT-10-N20) regenerate from manifests; Atlas and figures
  reproduce.
- Coverage matrix 100% (item 9).
- Programme deltas vs v0.4.25 measured: T_simulation, M_peak, T_test,
  T_agent_task, E_semantic = 0, C_scientific retained.
- Every manuscript claim traces to a ledger row with PASS evidence; negative
  and failed results reported as such.

## Open work outside the release stacks (after 0.5.5 unless a trigger fires)

0. Literature reproduction études, HDP off/on (human, 2026-09-29): plan in
   `artifacts/etudes/literature_reproduction_plan.md`. P0 table done
   (`literature_p0_candidates.md`); P1 partly done
   (`literature_p1_verification.md`: 5 sources verified, row 3 parameters read
   from the authors' code, row 10 equations unread). Targets confirmed. Next:
   PDF for row 10, P2 GAP list for row 3 (no AdEx emitter), then P3 sign-off. Starts after the
   0.5.5 seal, with its own stack.
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
2. `units.py` unwired: wire or remove (owner decision).
3. Architecture candidates (`emitters.py` variant split, entry
   fragmentation, dual manifests, same-named builders); enter only as a
   measured bottleneck.
4. C5–C7 numerical deferrals; need a signed-zero/NaN exactness contract.
5. UNTESTED-exact refusal tail; PLACEHOLDER_NOTEBOOKS and artifact-gated
   skips; post-0.4.14 compatibility aliases.
6. P-001 `scripts/` legacy lint cleanup (ruff: 176 findings, 2026-09-30).
   Real defects first: `audit_w3_broad_handlers.py` `_OVERRIDES` repeats two
   keys (the later silently wins), holds a 3-tuple and a 1-tuple where
   `(class, note)` is unpacked, and is keyed on line numbers that have drifted;
   running it rewrites the tracked `artifacts/audit/w3_broad_handler_tally.json`
   (137 lines differ). Re-key it or retire it with the tally.
7. S27 population/`P_{l,c}` definition family. Trigger: a population
   definition CTX-01 cannot express.
8. Agent-native step 10: MCP or other transport.
9. Deferred from 0.5.4: item 1b (S20.1 ordering; individual human
    authorization required) and item 0 (`X[k]` frontier override, language
    decision).
10. Deferred owner decisions from the 2026-09-20 deep audit:
    `compile_step_fn **hdp_kwargs` unknown-key policy; `validate_hdp_params`
    non-dict non-strict silent pass; frozen protocols record JAX/lib
    versions (P-003).

## Tracks (standing directions, not tasks)

P performance (optimize only profiled components) · T testing (cheapest gate
per defect class) · S skills (task-shaped, benchmarked) · I integration (one
semantic implementation, many entrances) · A architecture (factor only with
measured benefit) · V validation (configured → realized → executed →
observed per capability) · U inspection (origin/provenance queries).
Full text in the archive.
