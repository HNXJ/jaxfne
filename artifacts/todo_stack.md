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
  apply.

## 0.5.5 stack

HARNESS (human, 2026-09-27)
- H1 Consumption gate, remainder (Simulation, RuntimeConfig,
  RuntimeConfiguration, probe `n_contacts`, `set_emitter` rule kwargs and
  `cell_params` keys are covered by `tests/test_consumption_gate_055.py`).
  Declarative keys of `field`, `probe`, `drive`, `inter_column_connectivity`,
  `connections`, `connectivity`, `emitter` and plain-route `network(layers)`
  are restricted to realized values (human decision 2026-09-27; canonical
  proxy labels boundary/gauge/noise_policy kept); a later `network()`,
  `emitter()` or Poisson `field()` that construct would drop is refused, as
  are `runtime()` keys nothing reads, `runtime(vmap=True)`, `areas()` that
  differ from the columns, a non-neutral `plasticity()`, `network()` keys
  nothing reads (`connectivity`, `cell_type_fractions`, `layers`),
  homeostatic emitter keys on another family and `p_connect` outside [0, 1].
  construct re-checks the fields and probes of a configuration built
  directly, and compares Poisson fields as simulate solves them. Left,
  executable: (e) `layer_fractions()` changes nothing without `column()`
  (default and custom tables give identical labels and positions at n=60);
  (f) the plain route reads cell-type fractions unnormalized
  (`{"E": 1, "PV": 1}` at n=2 built two E cells; the population route
  normalizes); (g) the homeostatic_ei route reads only `network(n=)` and
  the emitter rules (`_construct_homeostatic_ei_model`): list and refuse
  the declarations it drops; (d) P-018: apply `drive()` on the plain route
  (human 2026-09-28); regenerate `artifacts/mcc3_10s_checkpoint/*` and
  `artifacts/perf/matrix_051*.json` if their outputs change; (a) wire
  `connectivity(edge_seed=)` and `build_laminar_column(edge_seed=)`
  (stored, never read; in-repo callers pass the runtime seed, so outputs
  must stay bit-identical); (b) the probe kernels (`spk_probe`,
  `vm_probe`, `source_probe`, `lfp_proxy_probe`) refuse the
  `position`/`reference`/`filter_spec` they never apply; rewrite the 0.5.2
  pin `tests/test_probe_electrode_052.py`.
- P-017 fix (human 2026-09-28): `stimulus_schedule` injects only events
  with a stimulus, for the event's duration, at its declared amplitude and
  targets; simulate refuses a multi-condition `Paradigm`;
  `evoked_l4_drive_paradigm` reads `l4_onset_ms`/`l4_amplitude` and targets
  L4. Re-pin the tests whose runs change and list those runs in the seal
  receipt; then D1b.
- Seal note: AT-02…AT-06 run records (`artifacts/publication/atlas/`) carry
  the pre-correction spec digest (AT-02…AT-05 recorded `n_contacts` 4,
  executed 16; AT-06 recorded a common-average reference and an 8–25 Hz
  band-pass that were never applied); regenerate them with the other Atlas
  records. The frozen agent task set keeps the old digests
  (`tests/test_agent_bench_055.py` allows exactly these corrections).

DOCS (human, 2026-09-27)
- D1b Pages 08/10 (`08_jaxfne_suite_no_2_evoked_l4_drive.md`,
  `10_v0313_omission_oddball.md`) claim figures and results their notebooks
  never produce: the evoked-L4 notebook passes a `Paradigm` that simulate
  ignores (P-017), and the omission notebook runs one plain simulation and
  plots two time steps under condition titles. Rebuild them on the P-017
  fix.
- D2 Études pages ("studios"): embed each page's own figure beside its
  text, as the tutorial track does (`scripts/generate_docs_visuals.py`
  pattern: still tab + interactive iframe).

ATLAS
7. Reduction/scale matrix: for each transition M_i → M_(i+1) (1N → 2N →
   population → 2A → 20A), which observations survive within the
   predeclared tolerance and which do not; failures stay in the matrix.
   Existing row: `at01_at06_052.run_reduction` (HH → reduced → population).
8. Performance/reduction map: T_compute and M_compute per AT beside
   E_reduction.

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

1. A8 gh-pages publishing policy. Trigger: publishing D1 or Atlas figures
   to the public site.
2. `units.py` and `pynwb_compat` unwired: wire or remove (owner decision).
3. Architecture candidates (`emitters.py` variant split, entry
   fragmentation, dual manifests, same-named builders); enter only as a
   measured bottleneck.
4. C5–C7 numerical deferrals; need a signed-zero/NaN exactness contract.
5. UNTESTED-exact refusal tail; PLACEHOLDER_NOTEBOOKS and artifact-gated
   skips; post-0.4.14 compatibility aliases.
6. P-001 `scripts/` legacy lint cleanup.
7. S27 population/`P_{l,c}` definition family. Trigger: a population
   definition CTX-01 cannot express.
8. Agent-native step 10: MCP or other transport.
9. P-016: slow pin `test_population_restoring_etude_regression_metrics`
   fails at HEAD; bisect to its cause before any re-freeze.
10. Deferred from 0.5.4: item 1b (S20.1 ordering; individual human
    authorization required) and item 0 (`X[k]` frontier override, language
    decision).
11. Deferred owner decisions from the 2026-09-20 deep audit:
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
